from krita import *
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, 
    QLabel, QDialog, QLineEdit, QComboBox, QFrame, 
    QSpinBox, QScrollArea, QSizePolicy, QFormLayout, QDialogButtonBox,
    QSlider, QAbstractItemView, QSplitter, QStyle, QGridLayout
)
from PyQt5.QtCore import (
    QThread, pyqtSignal, Qt, QEvent, QObject, QSize, QTimer
)
from PyQt5.QtGui import (
    QTextCursor, QFont, QIcon, QPixmap, QImage, QColor, QPalette
)
from typing import Optional, List, Tuple, Any
from PIL import Image
import io
import logging
import random
import time

from .gemini_api import GeminiClient
from .krita_api import (
    get_smart_context, update_preview_layer, apply_preview_layer, 
    delete_preview_layer, get_canvas_dimensions, get_nearest_supported_ar
)
from .utils import NanobananaSettings

# --- Utils ---
def pil2pixmap(im):
    """Converts PIL Image to QPixmap, handling BGR/RGB channel swapping for Qt."""
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    
    # Qt QImage.Format_ARGB32 expects data in BGRA order on Little Endian (Windows/Linux x64)
    # So we swap R and B channels from PIL's RGBA
    r, g, b, a = im.split()
    im_bgra = Image.merge("RGBA", (b, g, r, a))
    
    data = im_bgra.tobytes("raw", "RGBA")
    qim = QImage(data, im.size[0], im.size[1], QImage.Format_ARGB32)
    return QPixmap.fromImage(qim)

# --- Custom Widgets ---

class ClickableLabel(QLabel):
    clicked = pyqtSignal()
    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)

class ChatBubble(QFrame):
    """
    A message bubble widget. 
    Can hold text and/or images.
    """
    image_clicked_signal = pyqtSignal(object, int) # (QPixmap, seed/id)

    def __init__(self, role="model", text="", parent=None):
        super().__init__(parent)
        self.role = role
        self.setFrameShape(QFrame.NoFrame)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        
        # Main Layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(8)
        
        # Text Label
        self.lbl_text = QLabel(text)
        self.lbl_text.setWordWrap(True)
        self.lbl_text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        
        # Styling based on role
        font = QFont("Segoe UI", 10) 
        self.lbl_text.setFont(font)
        
        if role == "user":
            self.setStyleSheet("""
                ChatBubble { 
                    background-color: #1976D2; 
                    border-radius: 12px; 
                    border-bottom-right-radius: 2px;
                    margin-left: 40px;
                }
                QLabel { color: white; background: transparent; }
            """)
            self.layout.setAlignment(Qt.AlignRight)
        elif role == "model":
            self.setStyleSheet("""
                ChatBubble { 
                    background-color: #3E3E3E; 
                    border-radius: 12px; 
                    border-bottom-left-radius: 2px;
                    margin-right: 40px;
                }
                QLabel { color: #E0E0E0; background: transparent; }
            """)
            self.layout.setAlignment(Qt.AlignLeft)
        else: # System/Error
            self.setStyleSheet("""
                ChatBubble { 
                    background-color: #3E2723; 
                    border: 1px solid #D32F2F;
                    border-radius: 8px; 
                }
                QLabel { color: #FFCDD2; background: transparent; font-style: italic; }
            """)

        self.layout.addWidget(self.lbl_text)
        
        # Grid for images
        self.image_grid_widget = QWidget()
        self.image_grid = QGridLayout(self.image_grid_widget)
        self.image_grid.setContentsMargins(0, 0, 0, 0)
        self.image_grid.setSpacing(4)
        self.layout.addWidget(self.image_grid_widget)
        self.image_count = 0

    def append_text(self, chunk: str):
        current = self.lbl_text.text()
        self.lbl_text.setText(current + chunk)

    def add_image(self, pixmap: QPixmap, image_data: Image.Image, seed: int):
        # Calculate grid position (3 columns)
        row = self.image_count // 3
        col = self.image_count % 3
        
        # Create wrapper
        container = QWidget()
        l = QVBoxLayout(container)
        l.setContentsMargins(0,0,0,0)
        l.setSpacing(2)
        
        lbl_img = ClickableLabel()
        lbl_img.setCursor(Qt.PointingHandCursor)
        
        # Scale down for thumbnail (160px)
        thumb_pixmap = pixmap.scaledToWidth(160, Qt.SmoothTransformation)
        lbl_img.setPixmap(thumb_pixmap)
        
        # Connect click safely using a closure helper
        def make_handler(img, s):
            return lambda: self.image_clicked_signal.emit(img, s)
            
        lbl_img.clicked.connect(make_handler(image_data, seed))
        
        l.addWidget(lbl_img)
        
        # Caption
        lbl_cap = QLabel(f"Var {seed}")
        lbl_cap.setStyleSheet("color: #888; font-size: 10px;")
        lbl_cap.setAlignment(Qt.AlignCenter)
        l.addWidget(lbl_cap)
        
        self.image_grid.addWidget(container, row, col)
        self.image_count += 1

# --- Workers ---

class MagicWorker(QThread):
    result_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, client: GeminiClient, prompt: str, images: List[Image.Image]):
        super().__init__()
        self.client = client
        self.prompt = prompt
        self.images = images

    def run(self):
        try:
            result = self.client.enhance_prompt(self.prompt, self.images)
            self.result_signal.emit(result)
        except Exception as e:
            self.error_signal.emit(str(e))

class GenerationWorker(QThread):
    text_chunk_signal = pyqtSignal(str)
    image_received_signal = pyqtSignal(object, int, int) 
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(str)

    def __init__(self, client: GeminiClient, prompt: str, images: List[Image.Image], seed: int, model: str, aspect_ratio: str, batch_count: int):
        super().__init__()
        self.client = client
        self.prompt = prompt
        self.images = images
        self.seed = seed
        self.model = model
        self.aspect_ratio = aspect_ratio
        self.batch_count = batch_count

    def run(self) -> None:
        if not self.client: return
        
        for i in range(self.batch_count):
            try:
                self.progress_signal.emit(f"Generating {i+1}/{self.batch_count}...")
                
                # Completely random seed for each generation in batch
                sid = random.randint(0, 2**31)
                
                # Throttle to prevent API empty responses
                if i > 0: time.sleep(1.5) 
                
                # First image usually updates context (Chat History), subsequent are variations (Stateless)
                if i == 0:
                    stream = self.client.send_prompt(self.prompt, self.images, sid, self.model, self.aspect_ratio)
                else:
                    stream = self.client.generate_variation(self.prompt, self.images, sid, self.model, self.aspect_ratio)
                
                for text, img in stream:
                    if text and i == 0: 
                        self.text_chunk_signal.emit(text)
                    if img: 
                        self.image_received_signal.emit(img, sid, i)
                        
            except Exception as e:
                logging.error(f"Worker Generation Error (Batch {i+1}): {e}", exc_info=True)
                # Report error to chat but continue batch
                self.text_chunk_signal.emit(f"\n[System: Var {i+1} failed: {e}]")
        
        self.finished_signal.emit()

# --- Main Widget ---

class NanobananaChatWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.settings = NanobananaSettings()
        self.client: Optional[GeminiClient] = None
        
        # State
        self.batch_results = [] # Track results for applying
        self.last_geometry: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self.current_ai_bubble: Optional[ChatBubble] = None
        
        # Selection State
        self.selected_image: Optional[Image.Image] = None
        self.selected_seed: Optional[int] = None
        
        self.setup_api()
        self.setup_ui()

    def setup_api(self) -> None:
        key = self.settings.get("api_key") or ""
        if key:
            try:
                self.client = GeminiClient(key)
            except Exception as e:
                self.add_message("sys", f"API Error: {e}")
        else:
            self.client = None
            self.add_message("sys", "Welcome! Please set your API Key in Settings (Gear Icon).")

    def setup_ui(self) -> None:
        self.setStyleSheet("""
            QWidget { background: #2B2B2B; color: #E0E0E0; font-family: 'Segoe UI', sans-serif; font-size: 14px; }
            QScrollArea { border: none; background: transparent; }
            QTextEdit { 
                background: #1E1E1E; 
                border: 1px solid #444; 
                border-radius: 8px; 
                padding: 8px;
                font-size: 14px;
            }
            QTextEdit:focus { border: 1px solid #1976D2; }
            QPushButton { 
                background: #424242; 
                border: 1px solid #555; 
                border-radius: 6px; 
                padding: 6px; 
            }
            QPushButton:hover { background: #505050; border-color: #777; }
            QPushButton:pressed { background: #303030; }
        """)
        
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0,0,0,0)

        # 1. Top Toolbar
        toolbar = QFrame()
        toolbar.setStyleSheet("background: #333; border-bottom: 1px solid #444;")
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(8, 8, 8, 8)
        
        self.btn_settings = QPushButton("⚙️")
        self.btn_settings.setFixedWidth(36)
        self.btn_settings.clicked.connect(self.show_settings)
        
        self.btn_reset = QPushButton("🗑️")
        self.btn_reset.setFixedWidth(36)
        self.btn_reset.clicked.connect(self.reset_session)
        
        self.btn_undo = QPushButton("↩️")
        self.btn_undo.setFixedWidth(36)
        self.btn_undo.setToolTip("Undo Last Turn")
        self.btn_undo.clicked.connect(self.undo_last)

        # Removed btn_retry from here

        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet("color: #888; font-style: italic; margin-left: 10px;")
        
        tb_layout.addWidget(self.btn_settings)
        tb_layout.addWidget(self.lbl_status)
        tb_layout.addStretch()
        tb_layout.addWidget(self.btn_undo)
        tb_layout.addWidget(self.btn_reset)
        
        main_layout.addWidget(toolbar)

        # 2. Chat Area (Scrollable)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.chat_layout = QVBoxLayout(self.scroll_content)
        self.chat_layout.addStretch() # Push messages to bottom
        self.chat_layout.setSpacing(12)
        self.chat_layout.setContentsMargins(12, 12, 12, 12)
        
        self.scroll.setWidget(self.scroll_content)
        main_layout.addWidget(self.scroll, 1) # Expands

        # 3. Preview Bar (Hidden by default)
        self.preview_bar = QFrame()
        self.preview_bar.setStyleSheet("background: #252525; border-top: 1px solid #444;")
        self.preview_bar.hide()
        pb_layout = QHBoxLayout(self.preview_bar)
        pb_layout.setContentsMargins(8, 8, 8, 8)
        
        self.preview_lbl = QLabel("Preview Active")
        self.btn_apply = QPushButton("✅ Apply")
        self.btn_apply.setStyleSheet("background: #2E7D32; color: white; font-weight: bold;")
        self.btn_apply.clicked.connect(self.apply_layer)
        
        self.btn_discard = QPushButton("❌ Discard")
        self.btn_discard.setStyleSheet("background: #C62828; color: white; font-weight: bold;")
        self.btn_discard.clicked.connect(self.discard_layer)
        
        pb_layout.addWidget(self.preview_lbl)
        pb_layout.addStretch()
        pb_layout.addWidget(self.btn_discard)
        pb_layout.addWidget(self.btn_apply)
        
        main_layout.addWidget(self.preview_bar)

        # 4. Input Area
        input_container = QFrame()
        input_container.setStyleSheet("background: #333; border-top: 1px solid #444;")
        ic_layout = QVBoxLayout(input_container)
        ic_layout.setContentsMargins(10, 10, 10, 10)
        
        # Batch & Info Row
        info_row = QHBoxLayout()
        self.sl_batch = QSlider(Qt.Horizontal)
        self.sl_batch.setRange(1, 4)
        self.sl_batch.setValue(self.settings.get("batch_size", 1))
        self.sl_batch.setFixedWidth(100)
        self.sl_batch.valueChanged.connect(lambda v: self.lbl_batch.setText(f"Batch: {v}"))
        
        self.lbl_batch = QLabel(f"Batch: {self.sl_batch.value()}")
        
        info_row.addWidget(self.lbl_batch)
        info_row.addWidget(self.sl_batch)
        info_row.addStretch()
        ic_layout.addLayout(info_row)
        
        # Text & Send Row
        ts_row = QHBoxLayout()
        
        self.btn_magic = QPushButton("✨")
        self.btn_magic.setFixedSize(50, 50)
        self.btn_magic.setToolTip("Magic Prompt")
        self.btn_magic.setStyleSheet("background: #7B1FA2; color: white; font-size: 22px; border-radius: 25px;")
        self.btn_magic.clicked.connect(self.start_magic)

        self.btn_retry = QPushButton("🔄")
        self.btn_retry.setFixedSize(50, 50)
        self.btn_retry.setToolTip("Retry Last Prompt")
        self.btn_retry.setStyleSheet("background: #EF6C00; color: white; font-size: 22px; border-radius: 25px;")
        self.btn_retry.clicked.connect(self.retry_last)
        
        self.inp = QTextEdit()
        self.inp.setPlaceholderText("Type a prompt... (Enter to send)")
        self.inp.setFixedHeight(50)
        self.inp.installEventFilter(self)
        
        self.btn_send = QPushButton("🚀")
        self.btn_send.setFixedSize(50, 50)
        self.btn_send.setStyleSheet("background: #1976D2; color: white; font-size: 20px; border-radius: 25px;")
        self.btn_send.clicked.connect(self.start_gen)
        
        ts_row.addWidget(self.btn_magic)
        ts_row.addWidget(self.btn_retry)
        ts_row.addWidget(self.inp)
        ts_row.addWidget(self.btn_send)
        ic_layout.addLayout(ts_row)
        
        main_layout.addWidget(input_container)

    def eventFilter(self, o: QObject, e: QEvent) -> bool:
        if o is self.inp and e.type() == QEvent.KeyPress:
            if e.key() in (Qt.Key_Return, Qt.Key_Enter) and e.modifiers() != Qt.ShiftModifier:
                self.start_gen()
                return True
        return super().eventFilter(o, e)

    def add_message(self, role: str, text: str) -> ChatBubble:
        bubble = ChatBubble(role, text)
        # Connect the bubble's image click signal to the main handler
        bubble.image_clicked_signal.connect(self.on_image_clicked)
        self.chat_layout.addWidget(bubble)
        
        # Auto-scroll
        QTimer.singleShot(100, lambda: self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum()))
        return bubble

    def start_magic(self):
        txt = self.inp.toPlainText().strip()
        if not self.client: return
        # Allow magic even if text is empty (might just want to describe the image)
        
        self.btn_magic.setEnabled(False)
        self.btn_send.setEnabled(False)
        self.btn_retry.setEnabled(False)
        self.inp.setEnabled(False)
        self.lbl_status.setText("✨ Doing Magic...")
        
        # Get Context
        images, geom, desc, ar = get_smart_context()
        
        self.magic_worker = MagicWorker(self.client, txt, images)
        self.magic_worker.result_signal.connect(self.apply_magic)
        self.magic_worker.error_signal.connect(self.on_magic_err)
        self.magic_worker.start()

    def apply_magic(self, data: dict):
        # Use the full JSON structure as the prompt
        import json
        if data:
            try:
                final_prompt = data.get("final_prompt", "Magic Image")
                # Use full JSON for generation, but simplified text for display
                json_prompt = json.dumps(data)
                
                # Auto-start generation with specific params
                self.start_gen(api_prompt=json_prompt, visible_text=final_prompt)
                return # start_gen handles UI state and logic from here
                
            except Exception as e:
                self.add_message("sys", f"Error processing magic data: {e}")
        else:
             self.add_message("sys", "Magic returned empty data.")
        
        self.on_finish() # Re-enable UI if we didn't start gen

    def on_magic_err(self, err: str):
        self.add_message("sys", f"Magic Failed: {err}")
        self.on_finish()

    def start_gen(self, api_prompt: str = None, visible_text: str = None) -> None:
        # Handle signal bool arg if called via clicked
        if isinstance(api_prompt, bool): api_prompt = None
        
        txt = api_prompt if api_prompt else self.inp.toPlainText().strip()
        if not txt or not self.client: return
        
        disp_text = visible_text if visible_text else txt
        
        # 1. Cleanup prev state
        self.discard_layer() 
        self.inp.clear()
        self.inp.setEnabled(False)
        self.btn_send.setEnabled(False)
        self.btn_retry.setEnabled(False)
        self.btn_magic.setEnabled(False)
        self.selected_image = None
        self.selected_seed = None
        self.batch_results = [] # Clear results
        
        # 2. Add User Bubble
        self.add_message("user", disp_text)
        
        # 3. Prepare Context
        self.lbl_status.setText("Analyzing Canvas...")
        QApplication.processEvents()
        
        images, geom, desc, calc_ar = get_smart_context()
        self.last_geometry = geom
        
        self.lbl_status.setText(f"Initializing...")

        # 4. Add AI Placeholder Bubble
        self.current_ai_bubble = self.add_message("model", "")
        
        # 5. Start Worker
        batch = self.sl_batch.value()
        self.settings.set("batch_size", batch)
        
        # Use Settings AR or Calculated AR
        set_ar = self.settings.get("aspect_ratio", "Canvas (Native)")
        ar_req = calc_ar if "Native" in set_ar else set_ar.split()[0]
        
        seed = random.randint(0, 2**31)

        self.worker = GenerationWorker(
            self.client, txt, images, seed, self.settings.get("model"), ar_req, batch
        )
        self.worker.text_chunk_signal.connect(self.handle_text)
        self.worker.image_received_signal.connect(self.handle_image)
        self.worker.finished_signal.connect(self.on_finish)
        self.worker.progress_signal.connect(self.update_status)
        self.worker.error_signal.connect(self.on_err)
        self.worker.start()

    def update_status(self, status: str):
        self.lbl_status.setText(status)

    def handle_text(self, chunk: str):
        if self.current_ai_bubble:
            self.current_ai_bubble.append_text(chunk)
            # Scroll to bottom
            sb = self.scroll.verticalScrollBar()
            sb.setValue(sb.maximum())

    def handle_image(self, img: Image.Image, seed: int, idx: int):
        self.batch_results.append(img)
        if self.current_ai_bubble:
            # Add thumbnail to the bubble
            self.current_ai_bubble.add_image(pil2pixmap(img), img, seed)
        
        # Auto-select the LAST received image as current preview
        self.on_image_clicked(img, seed)

    def on_image_clicked(self, img: Image.Image, seed: int):
        """
        Handles selection of an image from the chat history.
        Updates the virtual preview layer and state.
        """
        self.selected_image = img
        self.selected_seed = seed
        
        # Update Preview on Canvas
        x, y, w, h = self.last_geometry
        update_preview_layer(img, x, y, w, h)
        
        # Update UI
        self.preview_bar.show()
        self.preview_lbl.setText(f"Preview: Seed {seed}")
        self.lbl_status.setText(f"Selected: {seed}")

    def on_finish(self):
        self.inp.setEnabled(True)
        self.btn_send.setEnabled(True)
        self.btn_retry.setEnabled(True)
        self.btn_magic.setEnabled(True)
        self.lbl_status.setText("Ready")
        self.inp.setFocus()
        
        # Check if we got images
        if not self.batch_results and self.current_ai_bubble:
            self.add_message("sys", "⚠️ No images generated. Model might have refused the prompt or is overloaded.")

    def on_err(self, err: str):
        # Only for catastrophic setup errors (not individual batch errors which are handled in loop)
        self.inp.setEnabled(True)
        self.btn_send.setEnabled(True)
        self.btn_retry.setEnabled(True)
        self.btn_magic.setEnabled(True)
        self.lbl_status.setText("Error")
        self.add_message("sys", f"Error: {err}")

    def apply_layer(self):
        if not self.selected_image: return
        
        # Apply the CURRENTLY SELECTED preview as a permanent layer
        # We use the seed to name it uniquely
        apply_preview_layer(f"AI Gen {self.selected_seed}")
        
        self.add_message("sys", f"Applied 'AI Gen {self.selected_seed}' to canvas.")
        
        # Keep preview bar open in case user wants to select another one and apply it too!
        # But we need to update the preview layer logic: 
        # 'apply_preview_layer' renames the node. So next 'update_preview_layer' creates a new one.
        # This works perfectly for multi-save workflow.

    def discard_layer(self):
        delete_preview_layer()
        self.preview_bar.hide()
        self.selected_image = None

    def remove_last_interaction(self):
        """Removes the last AI bubble and the last User bubble from UI."""
        # Items in layout: [Stretch, Bubble1, Bubble2, ...]
        # We want to remove the last 2 widgets.
        
        count = self.chat_layout.count()
        removed_count = 0
        
        # Iterate backwards, skipping the stretch item (index 0 usually, but let's be safe)
        # actually stretch is added first, so it's at index 0.
        
        while count > 1 and removed_count < 2:
            item = self.chat_layout.takeAt(count - 1)
            if item.widget():
                item.widget().deleteLater()
                removed_count += 1
            count = self.chat_layout.count()

    def undo_last(self):
        if self.client and self.client.undo_last_turn():
            self.remove_last_interaction()
            self.add_message("sys", "↺ Last turn undone.")
            self.discard_layer()

    def retry_last(self):
        """
        1. Identify last user text (from UI or history?)
           Better from UI to be safe.
        2. Undo last turn (Backend + UI).
        3. Put text in input.
        4. Click send.
        """
        # Find last user message text
        last_user_text = ""
        
        # Scan backwards for a user bubble
        for i in range(self.chat_layout.count() - 1, -1, -1):
            w = self.chat_layout.itemAt(i).widget()
            if isinstance(w, ChatBubble) and w.role == "user":
                last_user_text = w.lbl_text.text()
                break
        
        if not last_user_text:
            self.add_message("sys", "Nothing to retry.")
            return

        # Undo
        if self.client:
            self.client.undo_last_turn() # Backend
        self.remove_last_interaction() # UI
        
        # Setup Retry
        self.inp.setText(last_user_text)
        self.start_gen()

    def reset_session(self):
        if self.client: self.client.reset_session()
        # Clear UI
        while self.chat_layout.count() > 1: # Keep the stretch item
            item = self.chat_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.discard_layer()
        self.add_message("sys", "Session Reset.")

    def show_settings(self):
        d = QDialog(self)
        d.setWindowTitle("Settings")
        d.setStyleSheet("background: #333; color: #eee;")
        f = QFormLayout(d)
        
        k = QLineEdit(self.settings.get("api_key",""))
        k.setEchoMode(QLineEdit.Password)
        f.addRow("API Key:", k)
        
        m = QComboBox()
        m.addItems(["gemini-2.5-flash-image", "gemini-2.5-pro", "gemini-3-pro-preview"])
        m.setCurrentText(self.settings.get("model", "gemini-2.5-flash-image"))
        f.addRow("Model:", m)
        
        ar = QComboBox()
        ar.addItems(["Canvas (Native)", "1:1", "2:3", "3:2", "3:4", "4:3", "16:9", "21:9"])
        ar.setCurrentText(self.settings.get("aspect_ratio", "Canvas (Native)"))
        f.addRow("AR Strategy:", ar)
        
        btn = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        btn.accepted.connect(lambda: [
            self.settings.set("api_key", k.text()), 
            self.settings.set("model", m.currentText()),
            self.settings.set("aspect_ratio", ar.currentText()),
            self.setup_api(), 
            d.accept()
        ])
        btn.rejected.connect(d.reject)
        f.addRow(btn)
        d.exec_()

class NanobananaDocker(DockWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🍌 Nanobanana")
        self.setWidget(NanobananaChatWidget())
    def canvasChanged(self, c): pass

Krita.instance().addDockWidgetFactory(DockWidgetFactory("nanobanana", DockWidgetFactoryBase.DockRight, NanobananaDocker))