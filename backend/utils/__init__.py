# =============================================================================
# PHARMA RAG - UTILS PACKAGE
# =============================================================================
# Why this file?
# → Makes 'utils' folder a Python package
# → Exposes most used imports at package level
# → Other files can import directly from utils
# =============================================================================

from utils.config import config
from utils.logger import get_logger
from utils.helpers import (
    get_pdf_files,
    ensure_directory,
    format_documents,
    deduplicate_documents,
    filter_documents_by_score,
    validate_query,
    Timer,
    save_json,
    load_json,
)

__all__ = [
    "config",
    "get_logger",
    "get_pdf_files",
    "ensure_directory",
    "format_documents",
    "deduplicate_documents",
    "filter_documents_by_score",
    "validate_query",
    "Timer",
    "save_json",
    "load_json",
]
