# =============================================================================
# PHARMA RAG - INGESTION PACKAGE
# =============================================================================
# Why this file?
# → Makes 'ingestion' folder a Python package
# → Exposes all ingestion functions at package level
# → Clean single import for pipeline
# =============================================================================

from ingestion.pdf_loader import (
    load_all_pdfs,
    load_single_pdf,
    clean_text,
    extract_drug_name,
)

from ingestion.chunker import (
    chunk_documents,
    chunk_single_document,
    load_embedding_model as chunker_embedding_model,
    build_semantic_chunker,
)

from ingestion.embedder import (
    embed_and_store,
    load_vectorstore,
    is_vectorstore_ready,
    load_embedding_model as embedder_embedding_model,
)

__all__ = [
    # pdf_loader
    "load_all_pdfs",
    "load_single_pdf",
    "clean_text",
    "extract_drug_name",

    # chunker
    "chunk_documents",
    "chunk_single_document",
    "build_semantic_chunker",
    "chunker_embedding_model",

    # embedder
    "embed_and_store",
    "load_vectorstore",
    "is_vectorstore_ready",
    "embedder_embedding_model",
]
