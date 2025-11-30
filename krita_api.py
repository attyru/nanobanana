from krita import *
import numpy as np
from PIL import Image
import logging
from PyQt5.QtCore import QByteArray
from typing import Optional, List, Tuple, Union

MAX_DIMENSION = 1536 

def resize_image_maintain_aspect(image: Image.Image, max_dim: int = MAX_DIMENSION) -> Optional[Image.Image]:
    """
    Downscales image to fit max_dim, maintaining aspect ratio (No Cropping).
    
    Args:
        image: PIL Image object.
        max_dim: Maximum width or height allowed.
        
    Returns:
        Resized PIL Image or None if input is invalid.
    """
    if not image: return None
    w, h = image.size
    if w <= max_dim and h <= max_dim: return image
    ratio = min(max_dim / w, max_dim / h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

def _bgra_to_pil(pixel_data: bytes, w: int, h: int) -> Optional[Image.Image]:
    """Converts raw BGRA byte data (from Krita) to an RGBA PIL Image."""
    try:
        arr = np.frombuffer(pixel_data, dtype=np.uint8)
        if arr.size != w * h * 4: 
            logging.error(f"Buffer size mismatch. Expected {w*h*4}, got {arr.size}")
            return None
        
        # Reshape to (Height, Width, Channels)
        arr = arr.reshape((h, w, 4))
        
        # Krita uses BGRA, PIL uses RGBA. Swap channels.
        # arr[:, :, [2, 1, 0, 3]] swaps B and R.
        return Image.fromarray(arr[:, :, [2, 1, 0, 3]].copy(), 'RGBA')
    except Exception as e:
        logging.error(f"Conversion error: {e}")
        return None

def get_nearest_supported_ar(width: int, height: int) -> str:
    """
    Maps arbitrary canvas dimensions to the closest Gemini-supported Aspect Ratio.
    
    Supported by Imagen 3 / Gemini 2.5 Flash Image:
    1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9
    """
    if width == 0 or height == 0: return "1:1"
    target_ratio = width / height
    
    # Official Gemini Image Generation Aspect Ratios & Resolutions
    supported_ratios = {
        "1:1": 1.0,
        "2:3": 0.666,
        "3:2": 1.500,
        "3:4": 0.750,
        "4:3": 1.333,
        "4:5": 0.800,
        "5:4": 1.250,
        "9:16": 0.5625,
        "16:9": 1.777,
        "21:9": 2.333
    }
    
    # Find key with minimum absolute difference
    best_ar = min(supported_ratios.keys(), key=lambda k: abs(supported_ratios[k] - target_ratio))
            
    logging.debug(f"AR Mapping: {width}x{height} (r={target_ratio:.2f}) -> {best_ar}")
    return best_ar

def get_smart_context() -> Tuple[List[Image.Image], Tuple[int, int, int, int], str, str]:
    """
    Extracts context for generation based on Krita state with Smart Detection.
    
    Logic:
    - Hides "Nanobanana Preview" layer to avoid feedback loops.
    - Grabs content based on Selection or Full Canvas.
    - Smart Detect: If content is Empty/White -> Txt2Img (returns empty list).
    - If content exists -> Img2Img (returns list with image).

    Returns:
        Tuple containing:
        - List of PIL Images (empty if Txt2Img, [img] if Img2Img)
        - Geometry Tuple (x, y, w, h) for placement
        - Description string (for UI)
        - Aspect Ratio string (for API)
    """
    doc = Krita.instance().activeDocument()
    if not doc: 
        return ([], (0,0,0,0), "No Doc", "1:1")

    # 1. Check for Multi-Layer override (Composition)
    window = Krita.instance().activeWindow()
    if window and window.activeView():
        nodes = window.activeView().selectedNodes()
        valid_nodes = [n for n in nodes if n.type() in ["paintlayer", "grouplayer", "vectorlayer"]]
        
        if len(valid_nodes) > 1:
            logging.info(f"Context: {len(valid_nodes)} Selected Layers")
            images = []
            for node in valid_nodes:
                if node.name() == "Nanobanana Preview": continue # Skip preview
                b = node.bounds()
                if b.isEmpty(): continue
                d = node.pixelData(b.x(), b.y(), b.width(), b.height())
                i = _bgra_to_pil(d, b.width(), b.height())
                if i and not _is_image_empty(i): 
                    images.append(resize_image_maintain_aspect(i))
            
            canvas_ar = get_nearest_supported_ar(doc.width(), doc.height())
            return (images, (0, 0, doc.width(), doc.height()), f"{len(images)} Layers (Smart)", canvas_ar)

    # 2. Unified Geometry Logic (Selection OR Full Canvas)
    sel = doc.selection()
    
    if sel:
        # Case A: Manual Selection
        x, y, w, h = sel.x(), sel.y(), sel.width(), sel.height()
        desc = "Region"
    else:
        # Case B: Full Canvas
        x, y, w, h = 0, 0, doc.width(), doc.height()
        desc = "Canvas"

    # --- Hide Preview Layer before grabbing context ---
    preview_node = doc.nodeByName("Nanobanana Preview")
    was_visible = False
    if preview_node:
        was_visible = preview_node.visible()
        if was_visible:
            preview_node.setVisible(False)
            doc.refreshProjection() # Force update to hide it from pixelData
    
    processed_imgs = []
    final_desc = desc
    
    try:
        # 3. Calculate AR & Extract Pixels
        ar_string = get_nearest_supported_ar(w, h)
        
        data = doc.pixelData(x, y, w, h)
        img = _bgra_to_pil(data, w, h)
        
        if img:
            if _is_image_empty(img):
                logging.info("Smart Detect: Canvas/Selection is empty/white -> Mode: Txt2Img")
                final_desc += " (Empty)"
            else:
                logging.info("Smart Detect: Content found -> Mode: Img2Img")
                processed_imgs.append(resize_image_maintain_aspect(img))
                final_desc += " (Ref)"
                
    finally:
        # Restore Preview Layer
        if preview_node and was_visible:
            preview_node.setVisible(True)
            doc.refreshProjection()
        
    return (processed_imgs, (x, y, w, h), final_desc, ar_string)

def _is_image_empty(img: Image.Image) -> bool:
    """
    Checks if an image is effectively empty (transparent or solid white).
    Used for Smart Txt2Img detection.
    """
    if not img: return True
    
    # 1. Check Alpha Channel (Transparency)
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        alpha = img.getchannel('A')
        # Get min/max alpha values
        min_a, max_a = alpha.getextrema()
        if max_a == 0: return True # Fully transparent
        
    # 2. Check content on White Background (standard Krita new doc)
    # Convert to RGB to check color values
    rgb = img.convert('RGB')
    
    # Check if all pixels are white (255, 255, 255)
    # We can check extrema. If min is 255 for all channels, it's pure white.
    extrema = rgb.getextrema() # [(Rmin, Rmax), (Gmin, Gmax), (Bmin, Bmax)]
    
    is_white = all(min_val >= 250 for min_val, max_val in extrema) # Tolerance for compression artifacts/slight off-white
    if is_white: return True
    
    return False

def delete_preview_layer() -> None:
    """Removes the preview layer if it exists (e.g., on Discard)."""
    try:
        doc = Krita.instance().activeDocument()
        if not doc: return
        node = doc.nodeByName("Nanobanana Preview")
        if node:
            node.remove()
            doc.refreshProjection()
    except Exception: pass

def update_preview_layer(pil_image: Image.Image, x: int = 0, y: int = 0, target_w: Optional[int] = None, target_h: Optional[int] = None, layer_name: str = "Nanobanana Preview") -> bool:
    """
    Updates (or creates) a preview layer with the generated image.
    
    Args:
        pil_image: The image to paste.
        x, y: Coordinates on canvas.
        target_w, target_h: Optional dimensions to resize image to (interpolation).
        layer_name: Name of the Krita layer.
    """
    try:
        doc = Krita.instance().activeDocument()
        if not doc: return False
        
        node = doc.nodeByName(layer_name)
        
        # Interpolation Logic
        if target_w and target_h:
            if abs(pil_image.width - target_w) > 1 or abs(pil_image.height - target_h) > 1:
                logging.info(f"Interpolating: {pil_image.size} -> {target_w}x{target_h}")
                pil_image = pil_image.resize((target_w, target_h), Image.Resampling.LANCZOS)

        if pil_image.mode != 'RGBA': pil_image = pil_image.convert('RGBA')
        w, h = pil_image.size
        
        # Create BGRA buffer for Krita
        arr = np.array(pil_image)
        bgra = np.zeros((h, w, 4), dtype=np.uint8)
        # RGBA -> BGRA
        bgra[:, :, 0] = arr[:, :, 2] # B
        bgra[:, :, 1] = arr[:, :, 1] # G
        bgra[:, :, 2] = arr[:, :, 0] # R
        bgra[:, :, 3] = arr[:, :, 3] # A
        
        pixel_bytes = QByteArray(bgra.tobytes())
        
        if not node:
            root = doc.rootNode()
            node = doc.createNode(layer_name, "paintlayer")
            root.addChildNode(node, None)
        
        node.setPixelData(pixel_bytes, x, y, w, h)
        node.setOpacity(255)
        doc.setActiveNode(node)
        doc.refreshProjection()
        return True
    except Exception as e:
        logging.error(f"Preview error: {e}", exc_info=True)
        return False

def apply_preview_layer(final_name: str) -> None:
    """Renames the preview layer to finalize it."""
    try:
        doc = Krita.instance().activeDocument()
        node = doc.nodeByName("Nanobanana Preview")
        if node: node.setName(final_name)
    except Exception: pass

def get_canvas_dimensions() -> Tuple[int, int]:
    """Safe accessor for canvas dimensions."""
    try:
        doc = Krita.instance().activeDocument()
        if doc: return doc.width(), doc.height()
    except Exception: pass
    return 0, 0
