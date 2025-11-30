import pytest
from utils import NanobananaSettings
import json

class TestNanobananaSettings:
    
    def test_load_save_settings(self, tmp_path, mocker):
        mocker.patch('PyQt5.QtCore.QStandardPaths.writableLocation', return_value=str(tmp_path))
        
        settings = NanobananaSettings()
        
        # Check defaults
        assert settings.get("model") == "gemini-2.5-flash-image"
        
        # Test new keys
        settings.set("batch_size", 4)
        settings.set("aspect_ratio", "16:9")
        
        # Verify file on disk
        expected_file = tmp_path / "nanobanana" / "settings.json"
        assert expected_file.exists()
        
        with open(expected_file, 'r') as f:
            data = json.load(f)
            assert data["batch_size"] == 4
            assert data["aspect_ratio"] == "16:9"
            
    def test_invalid_json_handling(self, tmp_path, mocker):
        mocker.patch('PyQt5.QtCore.QStandardPaths.writableLocation', return_value=str(tmp_path))
        
        settings_dir = tmp_path / "nanobanana"
        settings_dir.mkdir()
        settings_file = settings_dir / "settings.json"
        
        with open(settings_file, 'w') as f:
            f.write("{ invalid json")
            
        settings = NanobananaSettings()
        
        assert settings.get("model") == "gemini-2.5-flash-image"
        
    def test_set_invalid_key_type(self, tmp_path, mocker):
        mocker.patch('PyQt5.QtCore.QStandardPaths.writableLocation', return_value=str(tmp_path))
        settings = NanobananaSettings()
        
        with pytest.raises(ValueError):
            settings.set(123, "value")
            
    def test_set_non_serializable_value(self, tmp_path, mocker):
        mocker.patch('PyQt5.QtCore.QStandardPaths.writableLocation', return_value=str(tmp_path))
        settings = NanobananaSettings()
        
        class Unserializable: pass
            
        with pytest.raises(ValueError):
            settings.set("key", Unserializable())