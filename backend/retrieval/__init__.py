# =============================================================================
# PHARMA RAG - RETRIEVAL PACKAGE
# =============================================================================
# Why this file?
# → Makes 'retrieval' folder a Python package
# → Exposes all retrieval functions at package level
# → Clean single import for pipeline
# =============================================================================
from retrieval.hybrid_search import (
    hybrid_search,
    build_hybrid_retriever,
    build_bm25_retriever,
    build_semantic_retriever,
)

from retrieval.reranker import (
    rerank_documents,
    score_documents,
    load_reranker_model,
)

__all__ = [

    # hybrid_search
    "hybrid_search",
    "build_hybrid_retriever",
    "build_bm25_retriever",
    "build_semantic_retriever",

    # reranker
    "rerank_documents",
    "score_documents",
    "load_reranker_model",

]
