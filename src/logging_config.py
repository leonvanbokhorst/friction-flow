import logging
import logging.handlers
import os
from datetime import datetime
import glob
import appdirs
from config import APP_NAME, IS_DEVELOPMENT

# Constants
MAX_LOG_FILE_SIZE: int = 10 * 1024 * 1024  # 10 MB
MAX_LOG_FILES: int = 10  # Total number of log files to keep


def setup_logging() -> None:
    """
    Set up logging configuration for the application.

    This function initializes the logging system, creating log directories,
    setting up file and console handlers, and cleaning up old log files.

    The log directory is determined based on whether the application is running
    in development mode or not.

    Returns:
        None
    """
    if IS_DEVELOPMENT:
        log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    else:
        log_dir = appdirs.user_log_dir(APP_NAME)

    os.makedirs(log_dir, exist_ok=True)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{APP_NAME}_{current_time}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    logger.addHandler(_create_file_handler(log_file))
    logger.addHandler(_create_console_handler())

    _cleanup_old_logs(log_dir)

    logging.info(f"Logging initialized. Log file: {log_file}")


def _create_file_handler(log_file: str) -> logging.Handler:
    """
    Create and configure a file handler for logging.

    Args:
        log_file (str): The path to the log file.

    Returns:
        logging.Handler: A configured RotatingFileHandler instance.
    """
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=MAX_LOG_FILE_SIZE, backupCount=MAX_LOG_FILES - 1
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    return file_handler


def _create_console_handler() -> logging.Handler:
    """
    Create and configure a console handler for logging.

    Returns:
        logging.Handler: A configured StreamHandler instance.
    """
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    return console_handler


def _cleanup_old_logs(log_dir: str) -> None:
    """
    Remove old log files if the total number exceeds MAX_LOG_FILES.

    Args:
        log_dir (str): The directory containing the log files.

    Returns:
        None
    """
    log_files = glob.glob(os.path.join(log_dir, f"{APP_NAME}_*.log*"))
    log_files.sort(key=os.path.getmtime, reverse=True)
    for old_file in log_files[MAX_LOG_FILES:]:
        try:
            os.remove(old_file)
        except OSError as e:
            logging.error(f"Error deleting old log file {old_file}: {e}")
