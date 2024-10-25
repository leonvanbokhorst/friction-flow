import logging
import logging.handlers
import os
from datetime import datetime

# Constants
MAX_LOG_FILE_SIZE: int = 10 * 1024 * 1024  # 10 MB
MAX_LOG_BACKUP_COUNT: int = 9  # 10 files total

def setup_logging() -> None:
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"nfs_{current_time}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # File Handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=MAX_LOG_FILE_SIZE, backupCount=MAX_LOG_BACKUP_COUNT
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
