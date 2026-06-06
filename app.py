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
# STEP 3: Run the frontend
# =============================================================================
frontend_path = os.path.join(ROOT_DIR, "frontend", "app.py")
with open(frontend_path, "r", encoding="utf-8") as f:
    exec(f.read())
