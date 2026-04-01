import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document

from utils.logger import get_logger
from utils.config import config

logger = get_logger(__name__)


# =============================================================================
# FILE & DIRECTORY HELPERS
# =============================================================================

def get_pdf_files(pdf_dir: Optional[Path] = None) -> List[Path]:
    """
    Returns list of all PDF files from the given directory.

    Why this function?
    → Used by pdf_loader to find all PDFs automatically
    → Validates directory exists before processing
    → Logs exactly which files were found

    Args:
        pdf_dir : path to PDF directory
                  defaults to config.data.pdf_dir if not provided

    Returns:
        List[Path] : list of PDF file paths

    Raises:
        FileNotFoundError : if directory does not exist
        ValueError        : if no PDFs found in directory
    """
    # Use config default if no directory provided
    if pdf_dir is None:
        pdf_dir = config.data.pdf_dir

    # Convert to Path object if string passed
    pdf_dir = Path(pdf_dir)

    # Validate directory exists
    if not pdf_dir.exists():
        raise FileNotFoundError(
            f"❌ PDF directory not found: {pdf_dir}\n"
            f"   Please create it and add your drug PDFs."
        )

    # Find all PDF files recursively
    pdf_files = list(pdf_dir.glob("*.pdf"))

    # Validate at least one PDF exists
    if not pdf_files:
        raise ValueError(
            f"❌ No PDF files found in: {pdf_dir}\n"
            f"   Please add your drug PDFs to this folder."
        )

    logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")

    # Log each file found
    for pdf in pdf_files:
        logger.debug(f"  → {pdf.name}")

    return sorted(pdf_files)


def ensure_directory(path: Path) -> Path:
    """
    Creates a directory if it does not exist.

    Why this function?
    → Used before saving any files — prevents FileNotFoundError
    → Safe to call even if directory already exists

    Args:
        path : directory path to create

    Returns:
        Path : the same path after ensuring it exists
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Directory ensured: {path}")
    return path


# =============================================================================
# DOCUMENT HELPERS
# =============================================================================

def format_documents(documents: List[Document]) -> str:
    """
    Formats a list of LangChain Documents into a single clean string.

    Why this function?
    → LLM needs all chunks as ONE string — not a list
    → Adds clear separator between chunks
    → Includes source metadata so LLM knows which drug each chunk is from

    Args:
        documents : list of LangChain Document objects

    Returns:
        str : formatted string of all document contents

    Example output:
        [Source: jardiance.pdf]
        Jardiance is indicated for heart failure...

        ---

        [Source: farxiga.pdf]
        Farxiga reduces cardiovascular death risk...
    """
    if not documents:
        logger.warning("No documents to format — empty list received")
        return ""

    formatted_chunks = []

    for i, doc in enumerate(documents):
        # Extract source filename from metadata
        # Why .get: metadata might not always have source key
        source = doc.metadata.get("source", "Unknown")
        page   = doc.metadata.get("page", "N/A")

        # Format each chunk with its source info
        chunk = (
            f"[Source: {source} | Page: {page}]\n"
            f"{doc.page_content.strip()}"
        )
        formatted_chunks.append(chunk)

    # Join all chunks with clear separator
    formatted = "\n\n---\n\n".join(formatted_chunks)

    logger.debug(f"Formatted {len(documents)} document chunks")

    return formatted


def deduplicate_documents(documents: List[Document]) -> List[Document]:
    """
    Removes duplicate chunks from retrieved documents.

    Why this function?
    → Hybrid search (BM25 + Semantic) can return same chunk twice
    → Duplicates waste LLM tokens and confuse the answer
    → Uses page_content as unique identifier

    Args:
        documents : list of LangChain Document objects (may have duplicates)

    Returns:
        List[Document] : deduplicated list of documents
    """
    seen_contents  = set()
    unique_docs    = []

    for doc in documents:
        # Use stripped content as unique key
        content_key = doc.page_content.strip()

        if content_key not in seen_contents:
            seen_contents.add(content_key)
            unique_docs.append(doc)

    removed = len(documents) - len(unique_docs)

    if removed > 0:
        logger.info(f"Deduplication removed {removed} duplicate chunks")

    logger.debug(f"Unique documents after dedup: {len(unique_docs)}")

    return unique_docs


def filter_documents_by_score(
    documents_with_scores : List[tuple],
    threshold             : Optional[float] = None
) -> List[Document]:
    """
    Filters out documents below the relevance score threshold.

    Why this function?
    → After reranking, low score chunks are not useful
    → Sending garbage chunks to LLM hurts answer quality
    → Uses config threshold by default — can be overridden

    Args:
        documents_with_scores : list of (Document, score) tuples
        threshold             : minimum score to keep
                                defaults to config.retrieval.score_threshold

    Returns:
        List[Document] : only documents above threshold
    """
    if threshold is None:
        threshold = config.retrieval.score_threshold

    filtered = []

    for doc, score in documents_with_scores:
        if score >= threshold:
            filtered.append(doc)
            logger.debug(f"✅ Kept chunk (score: {score:.3f}) | {doc.metadata.get('source', 'Unknown')}")
        else:
            logger.debug(f"❌ Dropped chunk (score: {score:.3f}) | {doc.metadata.get('source', 'Unknown')}")

    logger.info(
        f"Score filtering: {len(documents_with_scores)} chunks → "
        f"{len(filtered)} kept (threshold: {threshold})"
    )

    return filtered


# =============================================================================
# TIMING HELPERS
# =============================================================================

class Timer:
    """
    Simple context manager to measure execution time of any block.

    Why this class?
    → Helps identify slow steps in the RAG pipeline
    → Works with Python 'with' statement — clean and simple
    → Logs time automatically when block finishes

    Usage:
        with Timer("PDF Loading"):
            load_pdfs()
        # Logs: "PDF Loading completed in 3.45s"
    """

    def __init__(self, operation_name: str):
        """
        Args:
            operation_name : name of the operation being timed
        """
        self.operation_name = operation_name
        self.start_time     = None
        self.elapsed        = None

    def __enter__(self):
        """Starts the timer when entering 'with' block."""
        self.start_time = time.time()
        logger.debug(f"⏱ Starting: {self.operation_name}")
        return self

    def __exit__(self, *args):
        """Stops timer and logs elapsed time when exiting 'with' block."""
        self.elapsed = time.time() - self.start_time
        logger.info(f"⏱ {self.operation_name} completed in {self.elapsed:.2f}s")


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def validate_query(query: str) -> str:
    """
    Validates and cleans the user query before processing.

    Why this function?
    → Empty queries crash the pipeline
    → Extra whitespace confuses embedding models
    → Very short queries give poor results
    → Catches bad input early — before hitting LLM

    Args:
        query : raw user query string

    Returns:
        str : cleaned and validated query

    Raises:
        ValueError : if query is empty
    """
    # Strip extra whitespace (unchanged)
    query = query.strip()

    # Check for empty query (unchanged)
    if not query:
        raise ValueError("❌ Query cannot be empty.")

    # -------------------------------------------------------------------------
    # CHANGE: Removed the 'len(query) < 5' check.
    #
    # WHAT YOU WROTE:
    #   if len(query) < 5:
    #       raise ValueError("Query too short...")
    #
    # WHY IT WAS WRONG:
    #   Short strings like "hi", "hey", "bye" (2–3 chars) are valid small-talk.
    #   validate_query is called FIRST in run_query and run_query_stream,
    #   BEFORE the small-talk guardrail gets a chance to intercept.
    #   So "hi" raised ValueError → FastAPI returned HTTP 400 → Streamlit
    #   showed "Response ended prematurely" even though it should have
    #   returned a friendly greeting response.
    #
    # WHAT WAS CHANGED:
    #   Removed the length check entirely.
    #   The small-talk guardrail (is_small_talk) and out-of-domain guardrail
    #   (is_out_of_domain) in rag_pipeline.py are the correct place to
    #   handle short/irrelevant queries gracefully ✅
    # -------------------------------------------------------------------------

    logger.debug(f"Query validated: '{query}'")

    return query


# =============================================================================
# JSON HELPERS
# =============================================================================

def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """
    Saves a dictionary as a JSON file.

    Why this function?
    → Used to save pipeline results, logs, metadata
    → Ensures directory exists before saving
    → Pretty prints JSON — human readable

    Args:
        data     : dictionary to save
        filepath : where to save the JSON file
    """
    filepath = Path(filepath)
    ensure_directory(filepath.parent)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.debug(f"Saved JSON to: {filepath}")


def load_json(filepath: Path) -> Dict[str, Any]:
    """
    Loads a JSON file and returns as dictionary.

    Why this function?
    → Used to load saved pipeline configs or metadata
    → Validates file exists before loading
    → Clear error message if file missing

    Args:
        filepath : path to JSON file

    Returns:
        Dict : loaded JSON data

    Raises:
        FileNotFoundError : if file does not exist
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"❌ JSON file not found: {filepath}"
        )

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.debug(f"Loaded JSON from: {filepath}")

    return data
