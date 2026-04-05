# =============================================================================
# PHARMA RAG - PIPELINE PACKAGE
# =============================================================================
# Why this file?
# → Makes 'pipeline' folder a Python package
# → Exposes pipeline functions at package level
# → main.py imports cleanly from here
# =============================================================================

from pipeline.rag_pipeline import (
    run_ingestion,
    run_query,
    get_pipeline_state,
    generate_answer,
    run_query_stream
)

__all__ = [
    "run_ingestion",
    "run_query",
    "get_pipeline_state",
    "generate_answer",
    "run_query_stream",
]
