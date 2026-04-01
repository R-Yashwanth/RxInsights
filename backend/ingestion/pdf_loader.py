# =============================================================================
# PHARMA RAG - PDF LOADER
# =============================================================================
# Why this file?
# → Loads all drug PDF files from disk
# → Cleans and preprocesses raw PDF text
# → Extracts drug name from PDF content — not filename
# → Adds rich metadata to each page
# → Outputs LangChain Document objects — ready for chunking
#
# Why PyMuPDF (fitz)?
# → Best PDF parser for complex layouts like drug labels
# → Handles multi-column text, tables, headers correctly
# → Much faster than PyPDF2 or pdfplumber
# → Free and open source
# =============================================================================

import re
import fitz  # PyMuPDF
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from utils.logger import get_logger
from utils.config import config
from utils.helpers import get_pdf_files, Timer

# Module level logger
logger = get_logger(__name__)


# =============================================================================
# TEXT CLEANING
# =============================================================================

def clean_text(text: str) -> str:
    """
    Cleans raw extracted PDF text for better chunking and retrieval.

    Why clean text?
    → Raw PDF text has extra whitespace, broken lines, special characters
    → Clean text = better semantic chunks = better retrieval

    Cleaning steps:
    1. Remove non-UTF8 characters
    2. Fix broken hyphenated words
    3. Remove excessive newlines
    4. Remove excessive spaces
    5. Strip leading/trailing whitespace

    Args:
        text : raw text extracted from PDF page

    Returns:
        str : cleaned text ready for chunking
    """
    if not text:
        return ""

    # Step 1: Remove non-UTF8 characters
    text = text.encode("utf-8", errors="ignore").decode("utf-8")

    # Step 2: Fix hyphenated line breaks
    # "cardio-\nvascular" → "cardiovascular"
    text = re.sub(r"-\n", "", text)

    # Step 3: Replace multiple newlines with double newline
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Step 4: Replace multiple spaces with single space
    text = re.sub(r" {2,}", " ", text)

    # Step 5: Strip leading and trailing whitespace
    text = text.strip()

    return text


# =============================================================================
# DRUG NAME EXTRACTION
# =============================================================================

def extract_drug_name(pdf_path: Path) -> str:
    """
    Extracts drug name from first page of PDF content.

    Why from content not filename?
    → DailyMed PDFs have UUID filenames — not drug names
    → Reading content is more reliable — works with ANY filename
    → Production level approach

    How DailyMed PDFs are structured:
    → First line always: "DRUGNAME- generic name"
    → Example: "ENTRESTO- sacubitril and valsartan"
    → Example: "JARDIANCE- empagliflozin tablet"

    Extraction strategy:
    1. Read first page of PDF
    2. Find first line matching "DRUGNAME-" pattern
    3. Extract name before dash
    4. Fallback to filename if extraction fails

    Args:
        pdf_path : full Path object to PDF file

    Returns:
        str : clean drug name e.g. "Entresto", "Jardiance"
    """
    try:
        # Open PDF and read first page only
        pdf_document = fitz.open(str(pdf_path))
        first_page   = pdf_document[0]
        text         = first_page.get_text("text")
        pdf_document.close()

        # Split into clean lines
        lines = [
            line.strip()
            for line in text.split("\n")
            if line.strip()
        ]

        # Strategy 1: Match "DRUGNAME-" pattern
        # DailyMed always formats: "DRUGNAME- generic name"
        for line in lines[:10]:
            match = re.match(r"^([A-Z]+)\s*-", line)
            if match:
                name = match.group(1).strip().title()
                logger.debug(
                    f"Drug name extracted from content: "
                    f"'{name}' ← '{line[:50]}'"
                )
                return name

        # Strategy 2: First ALL CAPS word
        # Brand names are always uppercase in FDA labels
        for line in lines[:10]:
            match = re.match(r"^([A-Z]{3,})", line)
            if match:
                name = match.group(1).strip().title()
                logger.debug(
                    f"Drug name extracted from caps: "
                    f"'{name}' ← '{line[:50]}'"
                )
                return name

        # Strategy 3: Fallback to filename
        logger.warning(
            f"Could not extract drug name from content — "
            f"falling back to filename: {pdf_path.name}"
        )
        name = pdf_path.stem
        name = re.sub(r"[^a-zA-Z\s]", " ", name)
        name = " ".join(name.split()).title()
        return name

    except Exception as e:
        logger.error(
            f"Drug name extraction failed for {pdf_path.name}: {e}"
        )
        return pdf_path.stem.title()
    

def extract_generic_name(pdf_path: Path, brand_name: str) -> str:
    """
    Extracts generic drug name from first page of PDF.

    Why extract generic name?
    → Users often search by generic name
    → "empagliflozin" instead of "Jardiance"
    → Need to map generic → brand for filtering
    → Production level — no hardcoding ✅

    How DailyMed formats it:
    → "JARDIANCE- empagliflozin tablet"
                  ↑ generic name here

    Args:
        pdf_path   : full path to PDF
        brand_name : already extracted brand name

    Returns:
        str : generic name or empty string
    """
    try:
        pdf_document = fitz.open(str(pdf_path))
        first_page   = pdf_document[0]
        text         = first_page.get_text("text")
        pdf_document.close()

        lines = [
            line.strip()
            for line in text.split("\n")
            if line.strip()
        ]

        for line in lines[:10]:
            # Pattern: "BRANDNAME- generic_name tablet/capsule"
            # Example: "JARDIANCE- empagliflozin tablet"
            match = re.match(
                rf"^{brand_name.upper()}\s*[-–]\s*([a-z\s]+?)(?:\s+tablet|\s+capsule|\s+injection|\s+solution|,|\n|$)",
                line,
                re.IGNORECASE
            )
            if match:
                generic = match.group(1).strip().lower()
                logger.debug(
                    f"Generic name extracted: "
                    f"'{generic}' ← '{line[:50]}'"
                )
                return generic

        logger.debug(f"No generic name found for {brand_name}")
        return ""

    except Exception as e:
        logger.error(f"Generic name extraction failed: {e}")
        return ""



# =============================================================================
# METADATA BUILDER
# =============================================================================

def build_metadata(
    pdf_path    : Path,
    page_num    : int,
    drug_name   : str,
    generic_name: str,
    char_count  : int
) -> dict:
    """
    Builds rich metadata dictionary for each PDF page.

    Added generic_name:
    → Enables generic name search
    → Maps empagliflozin → Jardiance
    → No hardcoding needed ✅
    """
    return {
        "source"      : pdf_path.name,
        "drug_name"   : drug_name,
        "generic_name": generic_name,    # ← new field
        "page"        : page_num + 1,
        "file_path"   : str(pdf_path),
        "char_count"  : char_count,
    }


# =============================================================================
# SINGLE PDF LOADER
# =============================================================================

def load_single_pdf(pdf_path: Path) -> List[Document]:
    """
    Loads a single PDF file and returns list of LangChain Documents.

    Why one Document per page?
    → SemanticChunker works better with page-level text
    → Preserves page metadata
    → Easier to debug — trace answer back to exact page

    Args:
        pdf_path : Path to the PDF file

    Returns:
        List[Document] : one Document per page with metadata

    Raises:
        FileNotFoundError : if PDF file does not exist
        RuntimeError      : if PDF cannot be opened or read
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"❌ PDF not found: {pdf_path}")

    # Extract drug name from PDF content
    drug_name    = extract_drug_name(pdf_path)
    generic_name = extract_generic_name(pdf_path, drug_name)

    # CHANGE: Removed duplicate logger.info call.
    # WHAT YOU WROTE:
    #   logger.info(f"Loading PDF: {pdf_path.name} (Brand: {drug_name} | Generic: {generic_name})")
    #   ...
    #   logger.info(f"Loading PDF: {pdf_path.name} (Drug: {drug_name})")
    #
    # WHY IT WAS WRONG:
    #   There were TWO logger.info calls logging "Loading PDF" for the same file.
    #   The second one (inside the function body at line 288) printed with less
    #   information (missing generic name) and was redundant.
    #   → Every PDF printed two log lines instead of one ❌
    #
    # WHAT WAS CHANGED:
    #   Kept only the first, more informative log line. Removed the second.
    logger.info(
        f"Loading PDF: {pdf_path.name} "
        f"(Brand: {drug_name} | Generic: {generic_name})"
    )

    documents = []

    try:
        # Open PDF with PyMuPDF
        pdf_document = fitz.open(str(pdf_path))

        for page_num in range(len(pdf_document)):
            # Extract text from page
            page     = pdf_document[page_num]
            raw_text = page.get_text("text")

            # Clean extracted text
            clean    = clean_text(raw_text)

            # Skip pages with no meaningful content
            # Pages with less than 50 chars = blank or header only
            if len(clean) < 50:
                logger.debug(
                    f"Skipping page {page_num + 1} "
                    f"(too short: {len(clean)} chars)"
                )
                continue

            # Build metadata for this page
            metadata = build_metadata(
                pdf_path     = pdf_path,
                page_num     = page_num,
                drug_name    = drug_name,
                generic_name = generic_name,
                char_count   = len(clean)
            )

            # Create LangChain Document
            doc = Document(
                page_content = clean,
                metadata     = metadata
            )

            documents.append(doc)

        # Close PDF after reading
        pdf_document.close()

        logger.info(
            f"✅ Loaded {pdf_path.name}: "
            f"{len(documents)} pages extracted"
        )

    except Exception as e:
        raise RuntimeError(
            f"❌ Failed to load PDF: {pdf_path.name}\n"
            f"   Error: {str(e)}"
        )

    return documents


# =============================================================================
# ALL PDFs LOADER
# =============================================================================

def load_all_pdfs(pdf_dir: Path = None) -> List[Document]:
    """
    Loads all PDF files from directory and returns combined Documents.

    Why this function?
    → Main entry point for ingestion pipeline
    → Loads all 10 drug PDFs in one call
    → Aggregates all pages into single list for chunking
    → Reports summary of what was loaded

    Args:
        pdf_dir : directory containing PDF files
                  defaults to config.data.pdf_dir

    Returns:
        List[Document] : all pages from all PDFs combined
    """
    if pdf_dir is None:
        pdf_dir = config.data.pdf_dir

    pdf_dir  = Path(pdf_dir)
    all_docs = []
    failed   = []

    # Get list of all PDF files
    pdf_files = get_pdf_files(pdf_dir)

    logger.info(f"Starting ingestion of {len(pdf_files)} PDF files...")

    with Timer("PDF Loading"):
        for pdf_path in pdf_files:
            try:
                docs = load_single_pdf(pdf_path)
                all_docs.extend(docs)

            except Exception as e:
                # Log failure but continue with other PDFs
                logger.error(f"Failed to load {pdf_path.name}: {e}")
                failed.append(pdf_path.name)

    # Final summary
    logger.info("=" * 50)
    logger.info("✅ PDF Loading Summary:")
    logger.info(f"   Total PDFs attempted : {len(pdf_files)}")
    logger.info(f"   Successfully loaded  : {len(pdf_files) - len(failed)}")
    logger.info(f"   Failed               : {len(failed)}")
    logger.info(f"   Total pages loaded   : {len(all_docs)}")

    if failed:
        logger.warning(f"   Failed PDFs: {failed}")

    logger.info("=" * 50)

    return all_docs
