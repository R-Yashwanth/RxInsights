import logging
import sys
from pathlib import Path
from datetime import datetime

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Creates filename like: pharma_rag_2024_01_15_10_30_45.log
LOG_FILE = LOG_DIR / f"pharma_rag_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"

LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """
    Creates and returns a production-ready logger for any module.

    Why this function?
    → Every file calls get_logger(__name__) to get its own logger
    → __name__ automatically becomes the module name
    → Example: ingestion.pdf_loader, retrieval.hybrid_search

    Args:
        name : name of the module calling this function
               always pass __name__ from the calling module

    Returns:
        logging.Logger : configured logger instance

    Usage:
        from utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("PDF loaded successfully")
        logger.error("Failed to load PDF")
    """

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
