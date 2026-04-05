# =============================================================================
# RXINSIGHT — STREAMLIT CLOUD ENTRY POINT
# =============================================================================
#
# This is the main entry point for Streamlit Cloud deployment.
#
# ARCHITECTURE:
#   Instead of starting a FastAPI server as a subprocess (which doesn't work
#   reliably on Streamlit Cloud), this entry point:
#   1. Adds the backend directory to Python's import path
#   2. Auto-ingests PDFs on first startup if vectorstore doesn't exist
#   3. Starts the file watcher for auto-ingesting new PDFs
#   4. Runs the Streamlit frontend directly
#
#   The frontend calls backend Python functions DIRECTLY — no HTTP middleman.
#   FastAPI (backend/main.py) is still available for local/API development.
#
# =============================================================================

import sys
import os
from pathlib import Path

# =============================================================================
# STEP 1: Set up Python path so backend modules can be imported
# =============================================================================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
sys.path.insert(0, BACKEND_DIR)

# =============================================================================
# STEP 2: Load environment variables
# =============================================================================
# config.py handles: .env → Streamlit secrets → defaults
# config.py handles: relative path resolution (always relative to backend/)

# =============================================================================
# STEP 3: Auto-ingest on first startup if vectorstore doesn't exist
# =============================================================================
import streamlit as st

if "startup_complete" not in st.session_state:
    st.session_state.startup_complete = False

if not st.session_state.startup_complete:
    try:
        from ingestion import is_vectorstore_ready
        from utils.logger import get_logger

        logger = get_logger("startup")

        if not is_vectorstore_ready():
            logger.info("🚀 First startup — vectorstore not found, running ingestion...")
            with st.spinner("🔄 Building knowledge base from PDFs... This may take a few minutes on first startup."):
                from pipeline import run_ingestion
                result = run_ingestion()
                logger.info(f"✅ Ingestion complete: {result}")
        else:
            logger.info("✅ Vectorstore already exists — skipping ingestion")

        # Start file watcher for auto-ingesting new PDFs
        try:
            from pipeline.file_watcher import start_file_watcher
            start_file_watcher()
        except Exception as e:
            logger.warning(f"File watcher startup failed: {e}")

        # Preload pipeline models
        try:
            from pipeline.rag_pipeline import get_pipeline_state
            get_pipeline_state()
            logger.info("✅ Pipeline models preloaded")
        except Exception as e:
            logger.warning(f"Model preloading failed: {e}")

        st.session_state.startup_complete = True

    except Exception as e:
        st.error(f"Startup failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

# =============================================================================
# STEP 4: Run the frontend
# =============================================================================
frontend_path = os.path.join(ROOT_DIR, "frontend", "app.py")
with open(frontend_path, "r", encoding="utf-8") as f:
    exec(f.read())
