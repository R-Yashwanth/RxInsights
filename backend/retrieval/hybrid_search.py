# =============================================================================
# PHARMA RAG - HYBRID SEARCH
# =============================================================================
# Why this file?
# → Core retrieval step in Advanced RAG pipeline
# → Combines BM25 (keyword) + Semantic (meaning) search
# → Each sub-query from decomposer hits hybrid search separately
# → Results combined and deduplicated
#
# Why Hybrid Search over just Semantic?
# → Semantic alone misses exact medical terms
#   "empagliflozin" might not match "Jardiance" semantically
# → BM25 alone misses meaning — finds keywords but not context
# → Combined → best of both worlds
#
# Why LangChain EnsembleRetriever?
# → Native LangChain component — combines retrievers cleanly
# → Weighted combination — 60% semantic, 40% BM25
# → Returns ranked results automatically
# → Works directly with Chroma and BM25Retriever
#
# Why LangChain BM25Retriever?
# → Built into LangChain — no extra setup
# → Works on List[Document] directly
# → Free, fast, no API needed
# =============================================================================

from typing import List
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_core.documents import Document

from utils.logger import get_logger
from utils.config import config
from utils.helpers import Timer, deduplicate_documents

# Module level logger
logger = get_logger(__name__)


# =============================================================================
# BM25 RETRIEVER
# =============================================================================

def build_bm25_retriever(
    documents   : List[Document],
    filter_dict : dict = None,
) -> BM25Retriever:
    """
    Builds BM25 keyword retriever.

    Why filter documents for BM25?
    → BM25 searches in memory
    → Pre-filter documents list
    → Only pass relevant drug docs ✅

    Args:
        documents   : all chunks
        filter_dict : optional metadata filter

    Returns:
        BM25Retriever : configured keyword retriever
    """
    logger.debug("Building BM25 retriever...")

    try:
        # Pre-filter documents for BM25
        # Why: BM25 doesn't support metadata filter natively
        # So we filter the list before building index
        if filter_dict:
            filtered_docs = [
                doc for doc in documents
                if all(
                    doc.metadata.get(k) == v
                    for k, v in filter_dict.items()
                )
            ]
            logger.info(
                f"BM25 filtered: {len(documents)} → "
                f"{len(filtered_docs)} chunks"
            )
        else:
            filtered_docs = documents
            logger.info(f"BM25 using all {len(documents)} chunks")

        # Fallback if filter returns empty
        # Why: wrong drug name → no chunks → use all
        if not filtered_docs:
            logger.warning(
                "Filter returned empty — "
                "falling back to all documents"
            )
            filtered_docs = documents

        bm25_retriever = BM25Retriever.from_documents(
            documents = filtered_docs,
            k         = config.retrieval.top_k,
        )

        logger.debug("✅ BM25 retriever built")
        return bm25_retriever

    except Exception as e:
        raise RuntimeError(
            f"❌ Failed to build BM25 retriever\n"
            f"   Error: {str(e)}"
        )

# =============================================================================
# SEMANTIC RETRIEVER
# =============================================================================

def build_semantic_retriever(
    vectorstore : Chroma,
    filter_dict : dict = None,
) -> object:
    """
    Builds semantic similarity retriever from Chroma.

    Why metadata filter?
    → Search only relevant drug chunks
    → Query about Jardiance → search 300 chunks not 2984
    → Faster + more accurate ✅

    Args:
        vectorstore : loaded Chroma vector store
        filter_dict : optional metadata filter
                      e.g. {"drug_name": "Jardiance"}
                      None = search all chunks

    Returns:
        VectorStoreRetriever : configured semantic retriever
    """
    logger.debug("Building semantic retriever from Chroma...")

    try:
        # Build search kwargs
        search_kwargs = {"k": config.retrieval.top_k}

        # Add metadata filter if provided
        # Why conditional: general queries need all chunks
        if filter_dict:
            search_kwargs["filter"] = filter_dict
            logger.info(f"Metadata filter applied: {filter_dict}")
        else:
            logger.info("No metadata filter — searching all chunks")

        semantic_retriever = vectorstore.as_retriever(
            search_type   = "similarity",
            search_kwargs = search_kwargs,
        )

        logger.debug("✅ Semantic retriever built")
        return semantic_retriever

    except Exception as e:
        raise RuntimeError(
            f"❌ Failed to build semantic retriever\n"
            f"   Error: {str(e)}"
        )


# =============================================================================
# ENSEMBLE (HYBRID) RETRIEVER
# =============================================================================

def build_hybrid_retriever(
    documents   : List[Document],
    vectorstore : Chroma,
    filter_dict : dict = None,
) -> EnsembleRetriever:
    """
    Builds EnsembleRetriever combining BM25 + Semantic.

    Args:
        documents   : all chunks for BM25
        vectorstore : Chroma for semantic
        filter_dict : optional metadata filter
    """
    logger.info(
        f"Building hybrid retriever | "
        f"filter: {filter_dict} | "
        f"BM25: {config.retrieval.bm25_weight} | "
        f"Semantic: {config.retrieval.semantic_weight}"
    )

    # Pass filter to both retrievers
    bm25_retriever     = build_bm25_retriever(documents, filter_dict)
    semantic_retriever = build_semantic_retriever(vectorstore, filter_dict)

    try:
        hybrid_retriever = EnsembleRetriever(
            retrievers = [bm25_retriever, semantic_retriever],
            weights    = [
                config.retrieval.bm25_weight,
                config.retrieval.semantic_weight,
            ],
        )

        logger.info("✅ Hybrid retriever built")
        return hybrid_retriever

    except Exception as e:
        raise RuntimeError(
            f"❌ Failed to build hybrid retriever\n"
            f"   Error: {str(e)}"
        )

# =============================================================================
# SINGLE QUERY SEARCH
# =============================================================================

def search_single_query(
    query           : str,
    hybrid_retriever: EnsembleRetriever,
) -> List[Document]:
    """
    Runs hybrid search for a single query.

    Why this function?
    → Clean wrapper around retriever.invoke()
    → Handles errors per query — one failure doesn't stop others
    → Logs results clearly

    Args:
        query            : single sub-query string
        hybrid_retriever : built EnsembleRetriever

    Returns:
        List[Document] : retrieved chunks for this query
    """
    logger.debug(f"Searching: '{query}'")

    try:
        # Run hybrid search
        # Why invoke: newer LangChain standard
        documents = hybrid_retriever.invoke(query)

        logger.debug(
            f"Query: '{query[:50]}...' → "
            f"{len(documents)} chunks retrieved"
        )

        return documents

    except Exception as e:
        logger.error(f"Search failed for query '{query}': {e}")
        return []


# =============================================================================
# MAIN HYBRID SEARCH
# =============================================================================

def hybrid_search(
    sub_queries : List[str],
    documents   : List[Document],
    vectorstore : Chroma,
    filter_dict : dict = None,
) -> List[Document]:
    """
    Runs hybrid search with optional metadata filtering.

    Args:
        sub_queries : list of queries
        documents   : all chunks for BM25
        vectorstore : Chroma for semantic
        filter_dict : optional metadata filter
                      e.g. {"drug_name": "Jardiance"}

    Returns:
        List[Document] : deduplicated chunks
    """
    if not sub_queries:
        raise ValueError("❌ No sub-queries provided")

    if not documents:
        raise ValueError("❌ No documents for BM25")

    logger.info(
        f"Hybrid search | "
        f"queries: {len(sub_queries)} | "
        f"filter: {filter_dict}"
    )

    # Build retriever once — pass filter
    hybrid_retriever = build_hybrid_retriever(
        documents   = documents,
        vectorstore = vectorstore,
        filter_dict = filter_dict,
    )

    all_chunks = []

    with Timer("Hybrid Search"):
        for i, query in enumerate(sub_queries, 1):
            logger.info(
                f"Searching {i}/{len(sub_queries)}: '{query}'"
            )
            chunks = search_single_query(query, hybrid_retriever)
            all_chunks.extend(chunks)

    logger.info(f"Raw chunks: {len(all_chunks)}")

    unique_chunks = deduplicate_documents(all_chunks)

    logger.info("=" * 50)
    logger.info("🔍 Hybrid Search Summary:")
    logger.info(f"   Raw chunks    : {len(all_chunks)}")
    logger.info(f"   After dedup   : {len(unique_chunks)}")
    logger.info(f"   Filter used   : {filter_dict}")
    logger.info("=" * 50)

    return unique_chunks
