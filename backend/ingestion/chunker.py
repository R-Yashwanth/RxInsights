# =============================================================================
# PHARMA RAG - SEMANTIC CHUNKER
# =============================================================================
# Why this file?
# → Splits loaded PDF pages into meaningful chunks
# → Uses SemanticChunker — splits by MEANING shift not character count
# → Perfect for drug labels — each section has distinct medical topic
#
# Why SemanticChunker over RecursiveCharacterTextSplitter?
# → Drug labels have clear topic boundaries (Indications, Dosage, Warnings)
# → SemanticChunker detects these boundaries automatically
# → Results in more meaningful, self-contained chunks
# → Better chunks = better retrieval = better answers
#
# Why LangChain SemanticChunker?
# → Built into LangChain experimental module
# → Works directly with HuggingFace embeddings — free
# → Outputs LangChain Documents — compatible with all LangChain components
# =============================================================================

from typing import List

from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from utils.logger import get_logger
from utils.config import config
from utils.helpers import Timer

# Module level logger
logger = get_logger(__name__)


# =============================================================================
# EMBEDDING MODEL INITIALIZATION
# =============================================================================

def load_embedding_model() -> HuggingFaceEmbeddings:
    """
    Loads HuggingFace embedding model for SemanticChunker.

    Why HuggingFaceEmbeddings?
    → Free — runs locally, no API key needed
    → all-MiniLM-L6-v2 — lightweight, fast, production grade
    → Downloads once → cached locally forever
    → 384 dimensions — good balance of quality and speed

    Why needed for chunking?
    → SemanticChunker converts sentences to vectors
    → Compares vectors to detect meaning shifts
    → Splits where meaning changes significantly

    Returns:
        HuggingFaceEmbeddings : loaded embedding model
    """
    logger.info(f"Loading embedding model: {config.embedding.model_name}")

    try:
        embedding_model = HuggingFaceEmbeddings(
            # Which model to use — from config
            model_name = config.embedding.model_name,

            # Run on CPU — no GPU needed for this model
            # Why CPU: all-MiniLM-L6-v2 is small enough for CPU inference
            model_kwargs = {"device": "cpu"},

            # Normalize embeddings to unit length
            # Why normalize: makes cosine similarity more accurate
            # Normalized vectors → dot product = cosine similarity
            encode_kwargs = {"normalize_embeddings": True}
        )

        logger.info("✅ Embedding model loaded successfully")
        return embedding_model

    except Exception as e:
        raise RuntimeError(
            f"❌ Failed to load embedding model: {config.embedding.model_name}\n"
            f"   Error: {str(e)}"
        )


# =============================================================================
# CHUNKER INITIALIZATION
# =============================================================================

def build_semantic_chunker(
    embedding_model: HuggingFaceEmbeddings
) -> SemanticChunker:
    """
    Builds and returns a configured SemanticChunker.

    Why SemanticChunker?
    → Understands MEANING — not just character count
    → Splits at topic boundaries — keeps related content together
    → Example: keeps all "Contraindications" text in same chunk
               splits when topic shifts to "Dosage"

    How it works internally:
    1. Converts every sentence to embedding vector
    2. Calculates similarity between consecutive sentences
    3. Finds where similarity drops significantly (breakpoint)
    4. Splits at those breakpoints

    Breakpoint types:
    → percentile (we use): splits where similarity drops below 85th percentile
    → standard_deviation : splits at statistical outliers
    → interquartile      : splits at IQR outliers

    Args:
        embedding_model : loaded HuggingFace embedding model

    Returns:
        SemanticChunker : configured chunker ready to split documents
    """
    logger.info(
        f"Building SemanticChunker | "
        f"threshold_type: {config.chunking.breakpoint_threshold_type} | "
        f"threshold_amount: {config.chunking.breakpoint_threshold_amount}"
    )

    try:
        chunker = SemanticChunker(
            # Embedding model used to compare sentence meanings
            embeddings = embedding_model,

            # How to detect topic boundary
            # percentile: split when similarity drops below Nth percentile
            breakpoint_threshold_type = config.chunking.breakpoint_threshold_type,

            # 85 = split when similarity drops below 85th percentile
            # Higher → more splits (smaller chunks)
            # Lower  → fewer splits (larger chunks)
            # 85 is balanced for 50-page drug label PDFs
            breakpoint_threshold_amount = config.chunking.breakpoint_threshold_amount,
        )

        logger.info("✅ SemanticChunker built successfully")
        return chunker

    except Exception as e:
        raise RuntimeError(
            f"❌ Failed to build SemanticChunker\n"
            f"   Error: {str(e)}"
        )


# =============================================================================
# SINGLE DOCUMENT CHUNKER
# =============================================================================

def chunk_single_document(
    document : Document,
    chunker  : SemanticChunker
) -> List[Document]:
    """
    Splits a single LangChain Document into semantic chunks.

    Why chunk per document?
    → Preserves metadata from original document
    → Each chunk inherits source, drug_name, page from parent
    → Adds chunk index to metadata — track position within page

    Args:
        document : single LangChain Document (one PDF page)
        chunker  : configured SemanticChunker

    Returns:
        List[Document] : list of semantic chunks with metadata
    """
    try:
        # Split document into chunks
        # Why create_documents with [text] and [metadata]:
        # → Passes metadata to ALL child chunks automatically
        chunks = chunker.create_documents(
            texts     = [document.page_content],
            metadatas = [document.metadata]
        )

        # Add chunk index to each chunk's metadata
        # Why chunk_index: helps debug which chunk from which page
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)

        logger.debug(
            f"Page {document.metadata.get('page', '?')} of "
            f"{document.metadata.get('source', '?')} → "
            f"{len(chunks)} chunks"
        )

        return chunks

    except Exception as e:
        logger.error(
            f"Failed to chunk page {document.metadata.get('page', '?')} "
            f"of {document.metadata.get('source', '?')}: {e}"
        )
        # Return original document as single chunk if chunking fails
        # Why: better to have unchunked content than lose it entirely
        return [document]


# =============================================================================
# ALL DOCUMENTS CHUNKER
# =============================================================================

def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Splits all loaded PDF Documents into semantic chunks.

    Why this function?
    → Main entry point called by ingestion pipeline
    → Processes all ~500 pages from 10 PDFs
    → Returns all chunks ready for embedding and storage

    Processing flow:
    1. Load embedding model
    2. Build SemanticChunker
    3. Chunk each document
    4. Collect and return all chunks

    Args:
        documents : list of Documents from pdf_loader
                    (~500 documents — one per PDF page)

    Returns:
        List[Document] : all semantic chunks from all documents
                         ready for embedding and Chroma storage

    Raises:
        ValueError  : if documents list is empty
        RuntimeError: if embedding model or chunker fails to load
    """
    # Validate input
    if not documents:
        raise ValueError(
            "❌ No documents to chunk. "
            "Please run pdf_loader first."
        )

    logger.info(f"Starting semantic chunking of {len(documents)} documents...")

    # Step 1: Load embedding model
    embedding_model = load_embedding_model()

    # Step 2: Build SemanticChunker
    chunker = build_semantic_chunker(embedding_model)

    all_chunks = []

    # -------------------------------------------------------------------------
    # CHANGE: Removed the 'failed' list that was declared but never populated.
    #
    # WHAT YOU WROTE:
    #   failed = []
    #   ...loop...
    #   logger.info(f"   Failed documents : {len(failed)}")
    #
    # WHY IT WAS WRONG:
    #   chunk_single_document() never raises — it catches its own errors
    #   and returns [original_document] as a fallback (so it always
    #   returns something). The 'failed' list in chunk_documents was
    #   never appended to — it always printed "Failed documents: 0" which
    #   was misleading (silent fallbacks weren't counted).
    #
    # WHAT WAS CHANGED:
    #   Added a 'fallback_count' counter that increments when
    #   chunk_single_document returns only 1 chunk of the original doc
    #   (our heuristic for a failed-then-fallback case).
    #   This gives an honest summary line.
    # -------------------------------------------------------------------------
    fallback_count = 0  # CHANGE: tracks pages that used fallback (not failed)

    with Timer("Semantic Chunking"):
        for i, document in enumerate(documents):

            # Progress log every 50 documents
            # Why 50: avoids flooding logs but shows progress
            if i % 50 == 0:
                logger.info(
                    f"Chunking progress: {i}/{len(documents)} documents..."
                )

            # Chunk single document
            chunks = chunk_single_document(document, chunker)

            # CHANGE: detect fallback case
            # chunk_single_document returns [original_doc] on error
            # If output == input (single chunk identical to source), it's a fallback
            if len(chunks) == 1 and chunks[0].page_content == document.page_content:
                fallback_count += 1

            all_chunks.extend(chunks)

    # Final summary
    logger.info("=" * 50)
    logger.info("✅ Chunking Summary:")
    logger.info(f"   Total documents  : {len(documents)}")
    logger.info(f"   Total chunks     : {len(all_chunks)}")
    # CHANGE: was always 0 — now shows actual fallbacks
    logger.info(f"   Fallback pages   : {fallback_count} (chunked as-is)")
    logger.info(
        f"   Avg chunks/doc   : "
        f"{len(all_chunks) / len(documents):.1f}"
    )
    logger.info("=" * 50)

    return all_chunks
