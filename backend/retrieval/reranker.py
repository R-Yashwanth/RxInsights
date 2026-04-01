# =============================================================================
# PHARMA RAG - RERANKER
# =============================================================================
# Why this file?
# → First step of Post-Retrieval stage in Advanced RAG
# → Hybrid search returns ~20 chunks — not all equally relevant
# → Reranker scores each chunk against query precisely
# → Sorts chunks by relevance — best chunks rise to top
#
# Why Cross-Encoder Reranker?
# → Hybrid search uses BI-encoder — query and chunk encoded separately
#   → Fast but less accurate
# → Cross-encoder reads query + chunk TOGETHER
#   → Slower but much more accurate
# → Perfect for post-retrieval — small set of chunks to score
#
# Why HuggingFace cross-encoder/ms-marco-MiniLM-L-6-v2?
# → Free — runs locally, no API key needed
# → Trained on MS MARCO — largest Q&A dataset
# → Specifically trained for passage relevance scoring
# → Production grade — used in enterprise search systems
# → Small enough to run on CPU efficiently
# =============================================================================

from typing import List, Tuple

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

from utils.logger import get_logger
from utils.config import config
from utils.helpers import Timer, filter_documents_by_score

# Module level logger
logger = get_logger(__name__)


# =============================================================================
# CROSS-ENCODER MODEL
# =============================================================================

def load_reranker_model() -> CrossEncoder:
    """
    Loads HuggingFace CrossEncoder model for reranking.

    Why CrossEncoder not BiEncoder?
    BiEncoder (used in retrieval):
    → Encodes query separately  → vector A
    → Encodes chunk separately  → vector B
    → Compares vectors          → similarity score
    → Fast but approximate

    CrossEncoder (used in reranking):
    → Takes [query + chunk] TOGETHER as input
    → Reads both simultaneously — understands relationship
    → Outputs single relevance score
    → Slower but highly accurate

    Why ms-marco-MiniLM-L-6-v2?
    → Trained on MS MARCO — 8.8M query-passage pairs
    → Optimized for question-answer relevance scoring
    → MiniLM — lightweight, fast on CPU
    → L-6 — 6 layers — good balance of speed and accuracy

    Returns:
        CrossEncoder : loaded reranker model
    """
    logger.info(f"Loading reranker model: {config.reranker.model_name}")

    try:
        reranker = CrossEncoder(
            # Model from config
            model_name = config.reranker.model_name,

            # Max input length
            # Why 512: standard BERT input limit
            # query + chunk combined must fit in 512 tokens
            max_length = 512,
        )

        logger.info("✅ Reranker model loaded successfully")
        return reranker

    except Exception as e:
        raise RuntimeError(
            f"❌ Failed to load reranker model: {config.reranker.model_name}\n"
            f"   Error: {str(e)}"
        )


# =============================================================================
# SCORE DOCUMENTS
# =============================================================================

def score_documents(
    query     : str,
    documents : List[Document],
    reranker  : CrossEncoder,
) -> List[Tuple[Document, float]]:
    """
    Scores each document against query using CrossEncoder.

    Why score all documents?
    → Hybrid search ranking is approximate
    → CrossEncoder gives precise relevance score for each chunk
    → Enables accurate filtering and sorting

    How CrossEncoder scoring works:
    → Input:  [query, chunk_text] as pair
    → Output: single float score (higher = more relevant)
    → Reads both texts together — understands context
    → Example:
       Query: "empagliflozin adverse reactions HFrEF"
       Chunk: "In EMPEROR trial, empagliflozin reduced
               hospitalization for heart failure..."
       Score: 0.94 ← highly relevant ✅

    Args:
        query     : rewritten user query (not sub-queries)
        documents : chunks from hybrid search
        reranker  : loaded CrossEncoder model

    Returns:
        List[Tuple[Document, float]] : (document, score) pairs
                                       unsorted at this stage
    """
    if not documents:
        logger.warning("No documents to score — empty list received")
        return []

    logger.info(f"Scoring {len(documents)} chunks with CrossEncoder...")

    try:
        # Build input pairs for CrossEncoder
        # Why pairs: CrossEncoder needs [query, chunk] together
        # All pairs scored in ONE batch call — efficient
        input_pairs = [
            [query, doc.page_content]
            for doc in documents
        ]

        with Timer("CrossEncoder Scoring"):
            # Score all pairs in one batch
            # Why batch: much faster than scoring one by one
            # CrossEncoder handles batching internally
            scores = reranker.predict(input_pairs)

        # Combine documents with their scores
        documents_with_scores = [
            (doc, float(score))
            for doc, score in zip(documents, scores)
        ]

        logger.info(f"✅ Scored {len(documents_with_scores)} chunks")

        # Log score distribution for debugging
        score_values = [s for _, s in documents_with_scores]
        logger.debug(f"Score range: {min(score_values):.3f} - {max(score_values):.3f}")
        logger.debug(f"Score mean : {sum(score_values)/len(score_values):.3f}")

        return documents_with_scores

    except Exception as e:
        raise RuntimeError(
            f"❌ CrossEncoder scoring failed\n"
            f"   Error: {str(e)}"
        )


# =============================================================================
# SORT AND FILTER
# =============================================================================

def sort_by_score(
    documents_with_scores: List[Tuple[Document, float]]
) -> List[Tuple[Document, float]]:
    """
    Sorts documents by relevance score — highest first.

    Why sort?
    → After scoring, documents are in random order
    → Need highest scoring chunks at top
    → Top chunks go to LLM — quality matters

    Args:
        documents_with_scores : list of (Document, score) tuples

    Returns:
        List[Tuple[Document, float]] : sorted by score descending
    """
    sorted_docs = sorted(
        documents_with_scores,
        # Sort by score (second element of tuple)
        key     = lambda x: x[1],
        # Highest score first
        reverse = True,
    )

    logger.debug("Documents sorted by relevance score (descending)")

    # Log top 5 scores for visibility
    for i, (doc, score) in enumerate(sorted_docs[:5], 1):
        source = doc.metadata.get("source", "Unknown")
        page   = doc.metadata.get("page", "?")
        logger.debug(
            f"  Rank {i}: score={score:.3f} | "
            f"{source} | page {page}"
        )

    return sorted_docs


# =============================================================================
# MAIN RERANKER
# =============================================================================

def rerank_documents(
    query     : str,
    documents : List[Document],
    # -------------------------------------------------------------------------
    # CHANGE: Added optional 'model' parameter.
    #
    # WHAT YOU WROTE:
    #   def rerank_documents(query, documents):
    #       ...
    #       reranker = load_reranker_model()  ← ALWAYS loads a NEW model
    #
    # WHY IT WAS WRONG:
    #   rag_pipeline.py caches the reranker in global _reranker_model and
    #   passes it via get_pipeline_state(). But rerank_documents() ignored
    #   that and called load_reranker_model() on EVERY query.
    #   → CrossEncoder loads from disk each time (~1–2 seconds wasted) ❌
    #   → Global cache in pipeline was completely useless ❌
    #   → Optimization 7 (model preloading) was broken ❌
    #
    # WHAT WAS CHANGED:
    #   Added optional 'model' parameter (default None).
    #   If the caller passes the cached model → use it directly ✅
    #   If None → load fresh (backward compatible for direct calls) ✅
    # -------------------------------------------------------------------------
    model     : CrossEncoder = None,
) -> List[Document]:
    """
    Reranks retrieved chunks by relevance to query.

    Why this function?
    → Main entry point called by RAG pipeline
    → Takes hybrid search results → returns top relevant chunks
    → Applies scoring, sorting, filtering in sequence

    Processing flow:
    1. Load CrossEncoder model (or use cached one)
    2. Score all chunks against query
    3. Sort by score (highest first)
    4. Filter below score threshold
    5. Keep only top FINAL_TOP_K chunks
    6. Return final chunks for LLM

    Args:
        query     : rewritten query (for accurate scoring)
        documents : chunks from hybrid_search.py (~20 chunks)
        model     : optional pre-loaded CrossEncoder (pass global
                    _reranker_model for Optimization 7)

    Returns:
        List[Document] : top relevant chunks ready for LLM
                         (max FINAL_TOP_K chunks)

    Raises:
        ValueError  : if query or documents empty
        RuntimeError: if reranker model fails
    """
    # Validate inputs
    if not query or not query.strip():
        raise ValueError("❌ Query cannot be empty for reranking")

    if not documents:
        logger.warning("No documents to rerank — returning empty list")
        return []

    logger.info(
        f"Reranking {len(documents)} chunks | "
        f"threshold: {config.retrieval.score_threshold} | "
        f"final_top_k: {config.retrieval.final_top_k}"
    )

    # Step 1: Use passed model OR load fresh
    # CHANGE: was always 'reranker = load_reranker_model()' with no option
    if model is not None:
        # Caller passed cached model → use it (Optimization 7) ✅
        reranker = model
        logger.debug("Using pre-loaded reranker model (cached)")
    else:
        # Fallback: load fresh (for direct calls without pipeline) ✅
        logger.debug("No model passed — loading fresh reranker")
        reranker = load_reranker_model()

    # Step 2: Score all documents (unchanged)
    documents_with_scores = score_documents(query, documents, reranker)

    # Step 3: Sort by score descending (unchanged)
    sorted_docs = sort_by_score(documents_with_scores)

    # Step 4: Filter below threshold (unchanged)
    # Why filter: low score chunks confuse LLM → worse answers
    filtered_docs = filter_documents_by_score(
        sorted_docs,
        config.retrieval.score_threshold
    )

    # Step 5: Keep only top FINAL_TOP_K (unchanged)
    # Why limit: LLM context window + token cost
    # 3 high quality chunks >> 10 mediocre chunks
    final_docs = filtered_docs[:config.retrieval.final_top_k]

    # Final summary
    logger.info("=" * 50)
    logger.info("📊 Reranking Summary:")
    logger.info(f"   Input chunks     : {len(documents)}")
    logger.info(f"   After threshold  : {len(filtered_docs)}")
    logger.info(f"   Final chunks     : {len(final_docs)}")
    logger.info("=" * 50)

    # -------------------------------------------------------------------------
    # CHANGE: Return empty list instead of forcing a fallback chunk.
    #
    # WHAT YOU WROTE:
    #   if not final_docs and sorted_docs:
    #       return [sorted_docs[0][0]]  ← always returned SOMETHING
    #
    # WHY IT WAS WRONG:
    #   When ALL chunks score below threshold, it means the query has
    #   NOTHING to do with drugs. Forcing a random chunk (e.g. Opdivo page 38)
    #   made the LLM generate a response with misleading sources.
    #   You can't catch every out-of-domain query with a keyword list —
    #   there are infinite topics (sports, cooking, politics, etc).
    #
    # THE SMART FIX:
    #   If reranker says "none of these chunks are relevant" → return empty [].
    #   The pipeline (rag_pipeline.py) checks: if reranked_chunks is empty →
    #   respond with "I can only help with pharmaceutical questions."
    #   This works for ANY topic automatically — no keyword list needed.
    #   The reranker IS your out-of-domain detector. ✅
    # -------------------------------------------------------------------------
    if not final_docs:
        logger.warning(
            "All chunks below threshold — "
            "query likely out-of-domain"
        )
        return []  # ← CHANGED: was 'return [sorted_docs[0][0]]'

    return final_docs
