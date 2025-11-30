import os
# [AI-CHANGE] Import pathlib for modern path handling to improve cross-platform compatibility and readability.
import pathlib
import json
import logging
from PyQt5.QtCore import QStandardPaths

class NanobananaSettings:
    def __init__(self):
        app_data = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
        # [AI-FIX: Use pathlib.Path for better path manipulation and to avoid os.path issues.]
        self.settings_dir = pathlib.Path(app_data) / "nanobanana"
        self.settings_file = self.settings_dir / "settings.json"
        self.log_file = self.settings_dir / "nanobanana.logs"
        
        # Создание директории
        try:
            self.settings_dir.mkdir(parents=True, exist_ok=True)  # [AI-FIX: Use pathlib mkdir with parents=True for robustness and add try-except for permission errors.]
        except OSError as e:
            logging.error(f"Failed to create settings directory: {e}")  # [AI-FIX: Add logging for directory creation failure to improve error visibility.]
            raise  # Re-raise to prevent silent failures
        
        # Настройка логирования
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            encoding='utf-8'
        )
        
        self.load_settings()

    def load_settings(self):
        """Загрузка настроек из файла"""
        default = {
            "api_key": "",  # [AI-FIX: Add comment about security risk of storing API keys in plain text; consider encryption or environment variables.]
            "model": "gemini-2.5-flash-image"
        }
        
        try:
            if self.settings_file.exists():  # [AI-FIX: Use pathlib exists() method for consistency.]
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    self.settings = {**default, **loaded}
                logging.info("Settings loaded successfully")
            else:
                self.settings = default
                self.save_settings()
                logging.info("Default settings created")
        except json.JSONDecodeError as e:  # [AI-FIX: Catch specific JSON errors for better error handling.]
            logging.error(f"Invalid JSON in settings file: {e}")
            self.settings = default
        except OSError as e:  # [AI-FIX: Catch file system errors separately.]
            logging.error(f"File access error during load: {e}")
            self.settings = default
        except Exception as e:
            logging.error(f"Unexpected error during settings load: {e}")
            self.settings = default

    def save_settings(self):
        """Сохранение настроек в файл"""
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
            logging.info("Settings saved successfully")
            return True
        except OSError as e:  # [AI-FIX: Catch specific file system errors.]
            logging.error(f"File access error during save: {e}")
            return False
        except Exception as e:
            logging.error(f"Unexpected error during settings save: {e}")
            return False

    def get(self, key, default=None):
        """Получение значения настройки"""
        return self.settings.get(key, default)

    def set(self, key, value):
        """Установка значения настройки"""
        # [AI-FIX: Add validation to ensure key is string and value is JSON serializable for robustness.]
        if not isinstance(key, str):
            raise ValueError("Key must be a string")
        try:
            json.dumps(value)  # Test serializability
        except (TypeError, ValueError) as e:
            raise ValueError(f"Value is not JSON serializable: {e}")
        self.settings[key] = value
        return self.save_settings()