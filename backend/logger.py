import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger(name="app_logger"):
    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # Prevent adding multiple handlers if setup is called multiple times
    if not logger.handlers:
        # Rotating File Handler (10 MB per file, max 5 backups)
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, "app.log"), 
            maxBytes=10 * 1024 * 1024,
            backupCount=5
        )
        
        # Console Handler
        console_handler = logging.StreamHandler()
        
        # Standard format
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    return logger

log = setup_logger()
