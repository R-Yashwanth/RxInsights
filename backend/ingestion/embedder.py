# =============================================================================
# PHARMA RAG - EMBEDDER
# =============================================================================
# Why this file?
# → Converts semantic chunks into vector embeddings
# → Stores embeddings in Chroma vector store persistently
# → Loads existing Chroma store for retrieval
# → Bridge between chunking and retrieval pipeline
#
# Why Chroma?
# → Local — no server setup needed
# → Persistent — survives restarts, no re-embedding needed
# → Free — no cloud costs
# → Native LangChain integration — works with all retrievers
#
# Why HuggingFaceEmbeddings?
# → Same model used in chunker — consistency is critical
# → If chunking uses model A but retrieval uses model B
#   → vectors are incompatible → wrong results ❌
# → Free, runs locally, no API key needed
# =============================================================================

from pathlib import Path
from typing import List, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from utils.logger import get_logger
from utils.config import config
from utils.helpers import Timer, ensure_directory

# Module level logger
logger = get_logger(__name__)


# =============================================================================
# EMBEDDING MODEL
# =============================================================================

def load_embedding_model() -> HuggingFaceEmbeddings:
    """
    Loads HuggingFace embedding model for vector generation.

    CRITICAL: Must use SAME model as chunker.py
    → Chunker uses this model to detect topic boundaries
    → Embedder uses this model to convert chunks to vectors
    → Retrieval uses this model to convert query to vector
    → All three MUST use identical model — otherwise vectors incompatible

    Why all-MiniLM-L6-v2?
    → 384 dimensions — compact but powerful
    → Trained on large corpus — understands medical terminology
    → Fast inference on CPU — no GPU needed
    → Free and open source

    Returns:
        HuggingFaceEmbeddings : loaded embedding model
    """
    logger.info(f"Loading embedding model: {config.embedding.model_name}")

    try:
        embedding_model = HuggingFaceEmbeddings(
            # Model name from config — same as chunker
            model_name = config.embedding.model_name,

            # Run on CPU — sufficient for this model size
            model_kwargs = {"device": "cpu"},

            # Normalize vectors — consistent with chunker
            # Why: normalized vectors → cosine similarity = dot product
            # → faster and more accurate similarity search
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
# CHROMA VECTOR STORE — CREATE & STORE
# =============================================================================

def embed_and_store(chunks: List[Document]) -> Chroma:
    """
    Converts chunks to vectors and stores them in Chroma.

    Why this function?
    → Takes chunks from chunker → embeds → stores in Chroma
    → Persists to disk — no need to re-embed on restart
    → Only needs to run ONCE during initial ingestion

    Processing flow:
    1. Validate chunks exist
    2. Load embedding model
    3. Ensure Chroma directory exists
    4. Create Chroma store from documents
    5. Persist to disk
    6. Return Chroma store for immediate use

    Args:
        chunks : list of semantic chunks from chunker.py

    Returns:
        Chroma : populated and persisted vector store

    Raises:
        ValueError  : if chunks list is empty
        RuntimeError: if embedding or storage fails
    """
    # Validate input
    if not chunks:
        raise ValueError(
            "❌ No chunks to embed. "
            "Please run chunker first."
        )

    logger.info(f"Starting embedding of {len(chunks)} chunks...")

    # Step 1: Load embedding model
    embedding_model = load_embedding_model()

    # Step 2: Ensure Chroma directory exists
    persist_dir = Path(config.vectorstore.persist_dir)
    ensure_directory(persist_dir)

    logger.info(f"Chroma persist directory: {persist_dir}")

    try:
        with Timer("Embedding + Chroma Storage"):

            # Step 3: Create Chroma vector store from chunks
            # Why from_documents:
            # → Takes List[Document] directly
            # → Embeds each chunk automatically
            # → Stores vectors + metadata + content in Chroma
            # → All in one call — clean and simple
            vectorstore = Chroma.from_documents(
                # All semantic chunks to embed
                documents = chunks,

                # Embedding model — converts text to vectors
                embedding = embedding_model,

                # Collection name — like a table name in Chroma
                collection_name = config.vectorstore.collection_name,

                # Where to save on disk
                # Why persist_directory: survives restarts
                # Without this → stored in memory → lost on restart ❌
                persist_directory = str(persist_dir),
            )

        logger.info("=" * 50)
        logger.info("✅ Embedding Complete:")
        logger.info(f"   Chunks embedded     : {len(chunks)}")
        logger.info(f"   Collection name     : {config.vectorstore.collection_name}")
        logger.info(f"   Persisted to        : {persist_dir}")
        logger.info("=" * 50)

        return vectorstore

    except Exception as e:
        raise RuntimeError(
            f"❌ Failed to embed and store chunks\n"
            f"   Error: {str(e)}"
        )


# =============================================================================
# CHROMA VECTOR STORE — LOAD EXISTING
# =============================================================================

def load_vectorstore(
    embedding_model: Optional[HuggingFaceEmbeddings] = None
) -> Chroma:
    """
    Loads existing Chroma vector store from disk.

    Why this function?
    → Ingestion runs ONCE — embeddings saved to disk
    → Every query just LOADS existing store — no re-embedding
    → Much faster than re-embedding 2000+ chunks every time
    → Called by retrieval pipeline on every query

    Args:
        embedding_model : optional pre-loaded embedding model
                          if None → loads fresh model
                          Why optional: avoids reloading if already loaded

    Returns:
        Chroma : loaded vector store ready for retrieval

    Raises:
        FileNotFoundError : if Chroma store not found on disk
        RuntimeError      : if loading fails
    """
    persist_dir = Path(config.vectorstore.persist_dir)

    # Validate Chroma store exists
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"❌ Chroma store not found at: {persist_dir}\n"
            f"   Please run ingestion pipeline first:\n"
            f"   POST /ingest endpoint or run embedder directly."
        )

    logger.info(f"Loading Chroma store from: {persist_dir}")

    # Load embedding model if not provided
    if embedding_model is None:
        embedding_model = load_embedding_model()

    try:
        # Load existing Chroma store
        # Why same collection_name and persist_directory:
        # → Must match exactly what was used during embed_and_store
        # → Wrong name → loads empty collection → no results ❌
        vectorstore = Chroma(
            collection_name   = config.vectorstore.collection_name,
            embedding_function= embedding_model,
            persist_directory = str(persist_dir),
        )

        # Get count of stored chunks
        # Why: confirms store loaded correctly
        chunk_count = vectorstore._collection.count()

        logger.info("✅ Chroma store loaded successfully")
        logger.info(f"   Collection  : {config.vectorstore.collection_name}")
        logger.info(f"   Total chunks: {chunk_count}")
        logger.info(f"   Location    : {persist_dir}")

        return vectorstore

    except Exception as e:
        raise RuntimeError(
            f"❌ Failed to load Chroma store\n"
            f"   Error: {str(e)}"
        )


# =============================================================================
# VECTORSTORE STATUS CHECK
# =============================================================================

def is_vectorstore_ready() -> bool:
    """
    Checks if Chroma vector store exists and has data.

    Why this function?
    → Called by FastAPI on startup AND by /status endpoint
    → If store not ready → block queries → tell user to ingest first
    → Prevents confusing empty results

    # -------------------------------------------------------------------------
    # CHANGE: Replaced full model loading with lightweight filesystem check.
    #
    # WHAT YOU WROTE:
    #   embedding_model = load_embedding_model()     ← loads HuggingFace model
    #   vectorstore     = load_vectorstore(embedding_model)  ← opens Chroma DB
    #   chunk_count     = vectorstore._collection.count()
    #
    # WHY IT WAS SLOW:
    #   This function is called by GET /status, which Streamlit's frontend
    #   calls on EVERY rerun (FAQ click, clear chat, typing, etc).
    #   load_embedding_model() takes ~1-3 seconds to load HuggingFace model.
    #   load_vectorstore() takes ~0.5-1 second to open Chroma.
    #   So every UI interaction wasted 2-4 seconds just to check "does DB exist?"
    #   This was the BIGGEST cause of UI slowness.
    #
    # WHAT WAS CHANGED:
    #   Just check if the Chroma SQLite file exists on disk and has size > 0.
    #   Chroma stores data in chroma.sqlite3 — if it exists and is non-empty,
    #   the vectorstore is ready. No model loading needed.
    #   This takes <1ms instead of 2-4 seconds.
    # -------------------------------------------------------------------------

    Returns:
        bool : True if store exists and has data, False otherwise
    """
    persist_dir = Path(config.vectorstore.persist_dir)

    # Check directory exists
    if not persist_dir.exists():
        logger.warning("Chroma store directory not found")
        return False

    try:
        # CHANGE: lightweight file check instead of loading full model+store
        # Chroma stores its data in chroma.sqlite3 inside persist_dir
        sqlite_file = persist_dir / "chroma.sqlite3"

        if not sqlite_file.exists():
            logger.warning("Chroma sqlite file not found")
            return False

        # If file exists and has meaningful size (>4KB = not empty DB)
        file_size = sqlite_file.stat().st_size
        if file_size < 4096:
            logger.warning("Chroma store exists but appears empty")
            return False

        logger.debug(f"✅ Vectorstore ready (sqlite: {file_size} bytes)")
        return True

    except Exception as e:
        logger.error(f"Vectorstore check failed: {e}")
        return False
