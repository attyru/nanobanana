from krita_api import resize_image_maintain_aspect, get_smart_context, update_preview_layer, get_nearest_supported_ar
from PIL import Image
import numpy as np

# Implicitly uses mocked krita from conftest
import krita 
from unittest.mock import MagicMock

class TestKritaApi:
    
    def test_resize_logic(self):
        img = Image.new('RGB', (2000, 1000))
        resized = resize_image_maintain_aspect(img, max_dim=1000)
        assert resized.size == (1000, 500)
        
    def test_ar_calculation(self):
        assert get_nearest_supported_ar(100, 100) == "1:1"
        assert get_nearest_supported_ar(1600, 900) == "16:9"
        assert get_nearest_supported_ar(720, 1280) == "9:16" 
        assert get_nearest_supported_ar(3000, 2000) == "3:2" 
        assert get_nearest_supported_ar(800, 1000) == "4:5"
        assert get_nearest_supported_ar(3440, 1440) == "21:9"

    def test_smart_context_no_doc(self):
        krita.Krita.instance.return_value.activeDocument.return_value = None
        imgs, geom, desc, ar = get_smart_context()
        assert len(imgs) == 0
        assert desc == "No Doc"
        assert geom == (0,0,0,0)

    def test_smart_context_selection_priority_with_content(self):
        mock_doc = MagicMock()
        mock_doc.width.return_value = 1000
        mock_doc.height.return_value = 1000
        
        mock_sel = MagicMock()
        mock_sel.x.return_value = 10
        mock_sel.y.return_value = 10
        mock_sel.width.return_value = 100
        mock_sel.height.return_value = 100
        mock_doc.selection.return_value = mock_sel
        
        # Create Non-Empty Data (Grey, fully opaque)
        # BGRA: 100, 100, 100, 255
        pixel_count = 100 * 100
        # Create a pattern that isn't just zeros
        fake_data = bytes([100, 100, 100, 255] * pixel_count)
        
        mock_doc.pixelData.return_value = fake_data
        
        krita.Krita.instance.return_value.activeDocument.return_value = mock_doc
        
        imgs, geom, desc, ar = get_smart_context()
        
        assert "Region" in desc
        assert "(Ref)" in desc # Should detect content
        assert geom == (10, 10, 100, 100)
        assert len(imgs) == 1

    def test_smart_context_full_canvas_empty(self):
        """Test that empty canvas triggers Txt2Img mode (empty images list)"""
        mock_doc = MagicMock()
        mock_doc.width.return_value = 200
        mock_doc.height.return_value = 200
        mock_doc.selection.return_value = None 
        
        mock_win = MagicMock()
        mock_view = MagicMock()
        mock_view.selectedNodes.return_value = []
        mock_win.activeView.return_value = mock_view
        krita.Krita.instance.return_value.activeWindow.return_value = mock_win
        
        # Empty (Transparent)
        fake_data = bytes([0] * (200 * 200 * 4))
        mock_doc.pixelData.return_value = fake_data
        
        krita.Krita.instance.return_value.activeDocument.return_value = mock_doc
        
        imgs, geom, desc, ar = get_smart_context()
        
        assert "Canvas" in desc
        assert "(Empty)" in desc
        assert geom == (0, 0, 200, 200)
        assert len(imgs) == 0 # Should be empty for Smart Txt2Img

    def test_update_preview_layer(self):
        mock_doc = MagicMock()
        mock_root = MagicMock()
        mock_doc.rootNode.return_value = mock_root
        mock_doc.nodeByName.return_value = None 
        
        mock_node = MagicMock()
        mock_doc.createNode.return_value = mock_node
        
        krita.Krita.instance.return_value.activeDocument.return_value = mock_doc
        
        img = Image.new('RGBA', (50, 50), color='blue')
        
        # Test interpolation: fit 50x50 into 100x100 hole
        success = update_preview_layer(img, x=0, y=0, target_w=100, target_h=100)
        
        assert success is True
        mock_doc.createNode.assert_called_with("Nanobanana Preview", "paintlayer")
        
        # Ensure data was set for the TARGET size (100x100)
        args, _ = mock_node.setPixelData.call_args
        assert args[1] == 0 # x
        assert args[2] == 0 # y
        assert args[3] == 100 # w
        assert args[4] == 100 # h