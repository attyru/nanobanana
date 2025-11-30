import sys
import types
from unittest.mock import MagicMock
import pytest
from PIL import Image

# --- Mock PyQt5 ---
try:
    import PyQt5
except ImportError:
    def create_fake_module(name):
        m = types.ModuleType(name)
        sys.modules[name] = m
        return m

    mock_qt = create_fake_module('PyQt5')
    
    mock_core = create_fake_module('PyQt5.QtCore')
    mock_core.QThread = MagicMock
    mock_core.pyqtSignal = MagicMock
    mock_core.Qt = MagicMock
    mock_core.Qt.Key_Return = 1
    mock_core.Qt.Key_Enter = 2
    mock_core.Qt.ShiftModifier = 4
    mock_core.Qt.PointingHandCursor = 8
    mock_core.Qt.AlignLeft = 1
    mock_core.QEvent = MagicMock
    mock_core.QEvent.KeyPress = 6
    mock_core.QByteArray = MagicMock
    mock_core.QSize = MagicMock
    mock_core.QObject = MagicMock # Added QObject
    
    # Add QStandardPaths for utils.py
    mock_core.QStandardPaths = MagicMock
    mock_core.QStandardPaths.AppDataLocation = 1
    mock_core.QStandardPaths.writableLocation = MagicMock(return_value="/tmp")
    
    mock_widgets = create_fake_module('PyQt5.QtWidgets')
    mock_widgets.QWidget = MagicMock
    mock_widgets.QVBoxLayout = MagicMock
    mock_widgets.QHBoxLayout = MagicMock
    mock_widgets.QPushButton = MagicMock
    mock_widgets.QTextEdit = MagicMock
    mock_widgets.QLabel = MagicMock
    mock_widgets.QDialog = MagicMock
    mock_widgets.QLineEdit = MagicMock
    mock_widgets.QComboBox = MagicMock
    mock_widgets.QCheckBox = MagicMock
    mock_widgets.QFrame = MagicMock
    mock_widgets.QSpinBox = MagicMock
    mock_widgets.QDialogButtonBox = MagicMock
    mock_widgets.QMessageBox = MagicMock
    mock_widgets.QApplication = MagicMock
    # New additions for v3.1 UI
    mock_widgets.QScrollArea = MagicMock
    mock_widgets.QSizePolicy = MagicMock
    mock_widgets.QFormLayout = MagicMock
    mock_widgets.QButtonGroup = MagicMock
    mock_widgets.QRadioButton = MagicMock
    mock_widgets.QSplitter = MagicMock
    mock_widgets.QStyle = MagicMock
    mock_widgets.QGridLayout = MagicMock
    
    # Additions for Thumbnail UI
    mock_widgets.QListWidget = MagicMock
    mock_widgets.QListWidgetItem = MagicMock
    mock_widgets.QSlider = MagicMock
    mock_widgets.QAbstractItemView = MagicMock
    
    mock_gui = create_fake_module('PyQt5.QtGui')
    mock_gui.QTextCursor = MagicMock
    mock_gui.QPixmap = MagicMock
    mock_gui.QImage = MagicMock
    mock_gui.QFont = MagicMock 
    mock_gui.QIcon = MagicMock
    mock_gui.QColor = MagicMock
    mock_gui.QPalette = MagicMock

    mock_core.QTimer = MagicMock # Added QFont

    # Link submodules to parent package
    mock_qt.QtCore = mock_core
    mock_qt.QtWidgets = mock_widgets
    mock_qt.QtGui = mock_gui

# --- Mock Krita ---
if 'krita' not in sys.modules:
    mock_krita = types.ModuleType('krita')
    mock_krita.Krita = MagicMock()
    mock_krita.DockWidget = MagicMock
    mock_krita.DockWidgetFactory = MagicMock
    mock_krita.DockWidgetFactoryBase = MagicMock
    mock_krita.DockWidgetFactoryBase.DockRight = 1
    sys.modules['krita'] = mock_krita

@pytest.fixture
def mock_genai_client(mocker):
    """Mocks the google.genai.Client"""
    return mocker.patch('gemini_api.genai.Client')

@pytest.fixture
def sample_image():
    """Creates a small dummy PIL image"""
    return Image.new('RGB', (100, 100), color='red')
