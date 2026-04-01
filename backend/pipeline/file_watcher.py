# =============================================================================
# RXINSIGHT — AUTO-INGESTION FILE WATCHER
# =============================================================================
#
# Monitors the data folder for new PDF files and automatically ingests them
# into the vectorstore. No server restart needed.
#
# HOW IT WORKS:
#   1. On startup, loads a manifest of already-processed files
#   2. Scans data folder for PDFs not in the manifest
#   3. If new PDFs found → loads, chunks, embeds, adds to vectorstore
#   4. Updates drug dictionary so new drugs appear in frontend immediately
#   5. Runs as a background daemon thread — zero impact on request handling
#
# WHY POLLING (not watchdog):
#   - Zero extra dependencies (watchdog requires pip install)
#   - Polling every 30 seconds is sufficient for enterprise use
#   - Works reliably across Windows/Linux/Mac
#   - Simple, debuggable, no filesystem event race conditions
#
# USAGE:
#   from pipeline.file_watcher import start_file_watcher
#   start_file_watcher()  # Call once on FastAPI startup
# =============================================================================

import json
import time
import threading
from pathlib import Path
from typing import List, Set
from datetime import datetime

from ingestion import (
    load_single_pdf,
    chunk_documents,
    load_embedding_model,
)
from langchain_chroma import Chroma
from utils.logger import get_logger
from utils.config import config
from utils.helpers import Timer

logger = get_logger(__name__)

# Manifest file — tracks which PDFs have been ingested
MANIFEST_FILE = Path(config.vectorstore.persist_dir) / "ingested_manifest.json"

# How often to check for new files (seconds)
POLL_INTERVAL = 30


# =============================================================================
# MANIFEST — Track Processed Files
# =============================================================================

def _load_manifest() -> dict:
    """Loads the manifest of processed files from disk."""
    if MANIFEST_FILE.exists():
        try:
            with open(MANIFEST_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load manifest: {e}")
    return {"files": {}}


def _save_manifest(manifest: dict):
    """Saves the manifest to disk."""
    try:
        MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(MANIFEST_FILE, "w") as f:
            json.dump(manifest, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save manifest: {e}")


def _get_processed_files(manifest: dict) -> Set[str]:
    """Returns set of already-processed filenames."""
    return set(manifest.get("files", {}).keys())


# =============================================================================
# INCREMENTAL INGESTION — Process Only New Files
# =============================================================================

def ingest_new_files(new_files: List[Path]) -> int:
    """
    Ingests only the specified new PDF files into the existing vectorstore.

    This is INCREMENTAL — it adds to the existing vectorstore, not replace it.
    Uses Chroma's add_documents() to append new chunks.

    Args:
        new_files : list of PDF paths to ingest

    Returns:
        int : number of new chunks added
    """
    if not new_files:
        return 0

    logger.info(f"📥 Auto-ingesting {len(new_files)} new PDF(s)...")

    # Step 1: Load new PDFs
    all_docs = []
    for pdf_path in new_files:
        try:
            docs = load_single_pdf(pdf_path)
            all_docs.extend(docs)
            logger.info(f"  ✅ Loaded: {pdf_path.name} ({len(docs)} pages)")
        except Exception as e:
            logger.error(f"  ❌ Failed to load {pdf_path.name}: {e}")

    if not all_docs:
        logger.warning("No documents loaded from new files")
        return 0

    # Step 2: Chunk new documents
    logger.info(f"  Chunking {len(all_docs)} pages...")
    chunks = chunk_documents(all_docs)
    logger.info(f"  → {len(chunks)} chunks created")

    # Step 3: Add to existing vectorstore (APPEND, not replace)
    logger.info("  Embedding and adding to vectorstore...")
    embedding_model = load_embedding_model()
    persist_dir = str(Path(config.vectorstore.persist_dir))

    with Timer("Incremental Embedding"):
        vectorstore = Chroma(
            collection_name    = config.vectorstore.collection_name,
            embedding_function = embedding_model,
            persist_directory  = persist_dir,
        )
        # Add new chunks to existing store (incremental)
        vectorstore.add_documents(chunks)

    logger.info(f"  ✅ Added {len(chunks)} chunks to vectorstore")

    # Step 4: Refresh pipeline state (drug dictionary, BM25, etc.)
    try:
        from pipeline.rag_pipeline import refresh_pipeline_state
        refresh_pipeline_state()
        logger.info("  ✅ Pipeline state refreshed")
    except Exception as e:
        logger.warning(f"  ⚠️ Pipeline refresh failed: {e} — will refresh on next query")

    return len(chunks)


# =============================================================================
# FILE WATCHER — Background Polling Thread
# =============================================================================

def _watch_loop():
    """
    Background loop that checks for new PDFs every POLL_INTERVAL seconds.
    Runs as a daemon thread — exits when main process exits.
    """
    logger.info(f"👁️ File watcher started — checking every {POLL_INTERVAL}s")
    data_dir = Path(config.data.pdf_dir)

    while True:
        try:
            time.sleep(POLL_INTERVAL)

            if not data_dir.exists():
                continue

            # Get all PDFs in data folder
            current_files = {f.name for f in data_dir.glob("*.pdf")}

            # Load manifest of already-processed files
            manifest = _load_manifest()
            processed = _get_processed_files(manifest)

            # Find new files
            new_file_names = current_files - processed
            if not new_file_names:
                continue

            logger.info(f"📂 Detected {len(new_file_names)} new PDF(s): {new_file_names}")

            # Ingest new files
            new_paths = [data_dir / name for name in new_file_names]
            chunks_added = ingest_new_files(new_paths)

            # Update manifest
            for name in new_file_names:
                manifest["files"][name] = {
                    "ingested_at": datetime.now().isoformat(),
                    "chunks_added": chunks_added // len(new_file_names) if new_file_names else 0,
                }
            _save_manifest(manifest)

            logger.info(f"✅ Auto-ingestion complete | {chunks_added} chunks added")

        except Exception as e:
            logger.error(f"File watcher error: {e}")
            # Continue watching — don't let one error kill the watcher


def build_initial_manifest():
    """
    Builds manifest from currently ingested files on first startup.
    This prevents re-ingesting files that were already in the vectorstore
    before the watcher was added.
    """
    manifest = _load_manifest()
    if manifest.get("files"):
        return  # Manifest already exists, skip

    data_dir = Path(config.data.pdf_dir)
    if not data_dir.exists():
        return

    current_files = list(data_dir.glob("*.pdf"))
    if not current_files:
        return

    # If vectorstore exists, assume all current files are already ingested
    from ingestion import is_vectorstore_ready
    if is_vectorstore_ready():
        logger.info(f"📋 Building initial manifest for {len(current_files)} existing PDFs")
        for pdf_path in current_files:
            manifest["files"][pdf_path.name] = {
                "ingested_at": datetime.now().isoformat(),
                "chunks_added": "existing",
            }
        _save_manifest(manifest)
        logger.info("✅ Initial manifest created")


def start_file_watcher():
    """
    Starts the file watcher as a background daemon thread.
    Call this once during FastAPI startup.

    The thread automatically dies when the main process exits (daemon=True).
    """
    # Build manifest for existing files first
    build_initial_manifest()

    # Start background watcher
    watcher_thread = threading.Thread(
        target=_watch_loop,
        name="RxInsight-FileWatcher",
        daemon=True,  # Dies when main process exits
    )
    watcher_thread.start()
    logger.info("🟢 Auto-ingestion file watcher is active")
