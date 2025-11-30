from krita import DockWidgetFactory, DockWidgetFactoryBase, Krita
from .nanobanana import NanobananaDocker
import logging
import os
import sys

# --- DEBUG LOGGING SETUP ---
# Log file will be in the same directory as this script
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nanobanana_debug.log')

# Remove existing handlers to prevent duplicate logs on plugin reload
root_logger = logging.getLogger()
for h in root_logger.handlers[:]:
    root_logger.removeHandler(h)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format='%(asctime)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s',
    filemode='a',  # Append mode for beta testing history
    force=True
)

# Hook into sys.excepthook to log crashes
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

logging.info("=== Nanobanana Plugin Initialized ===")
logging.info(f"Python Version: {sys.version}")
logging.info(f"Log File Path: {LOG_FILE}")

# Register the Docker
Krita.instance().addDockWidgetFactory(
    DockWidgetFactory("nanobanana", DockWidgetFactoryBase.DockRight, NanobananaDocker)
)
