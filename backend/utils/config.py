import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv(override=True)

def _get_env(key: str, default: str = None, required: bool = False) -> str:
    """
    Safely reads an environment variable.

    Args:
        key      : environment variable name
        default  : fallback value if key not found
        required : if True, raises error when key is missing

    Returns:
        value of the environment variable

    Raises:
        ValueError: if required=True and key is not found in .env
    """
    value = os.getenv(key, default)

    if required and value is None:
        raise ValueError(
            f"❌ Missing required environment variable: '{key}'\n"
            f"   Please add it to your .env file."
        )

    return value


def _get_int(key: str, default: int) -> int:
    """
    Reads an environment variable and converts it to integer.

    Args:
        key     : environment variable name
        default : fallback integer value

    Returns:
        integer value of the environment variable
    """
    value = _get_env(key, str(default))
    try:
        return int(value)
    except ValueError:
        raise ValueError(
            f"❌ Environment variable '{key}' must be an integer. "
            f"Got: '{value}'"
        )


def _get_float(key: str, default: float) -> float:
    """
    Reads an environment variable and converts it to float.

    Args:
        key     : environment variable name
        default : fallback float value

    Returns:
        float value of the environment variable
    """
    value = _get_env(key, str(default))
    try:
        return float(value)
    except ValueError:
        raise ValueError(
            f"❌ Environment variable '{key}' must be a float. "
            f"Got: '{value}'"
        )


def _get_bool(key: str, default: bool) -> bool:
    """
    Reads an environment variable and converts it to boolean.

    Args:
        key     : environment variable name
        default : fallback boolean value

    Returns:
        boolean value of the environment variable
    """
    value = _get_env(key, str(default)).lower()
    return value in ("true", "1", "yes")


# CONFIGURATION DATACLASSES
@dataclass
class LLMConfig:
    """
    Groq LLM configuration.
    Why Groq: Free tier, 128K context, fastest Llama inference available.
    """
    api_key : str = field(default_factory=lambda: _get_env("GROQ_API_KEY", required=True))
    model   : str = field(default_factory=lambda: _get_env("GROQ_MODEL", "llama-3.3-70b-versatile"))


@dataclass
class EmbeddingConfig:
    """
    HuggingFace Embedding model configuration.
    Why all-MiniLM-L6-v2: Free, runs locally, production-grade, 384 dimensions.
    No API key needed — downloads once, cached locally forever.
    """
    model_name : str = field(default_factory=lambda: _get_env("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))


@dataclass
class VectorStoreConfig:
    """
    Chroma vector store configuration.
    Why Chroma: Simple, persistent, local — no server setup needed.
    """
    persist_dir     : str = field(default_factory=lambda: _get_env("CHROMA_PERSIST_DIR", "./vectorstore/chroma_db"))
    collection_name : str = field(default_factory=lambda: _get_env("COLLECTION_NAME", "pharma_rag"))


@dataclass
class ChunkingConfig:
    """
    SemanticChunker configuration.
    Why SemanticChunker: Splits by meaning shift — smarter than character-based.
    Why percentile: Most stable breakpoint detection for structured drug label PDFs.
    """
    breakpoint_threshold_type   : str = field(default_factory=lambda: _get_env("BREAKPOINT_THRESHOLD_TYPE", "percentile"))
    breakpoint_threshold_amount : int = field(default_factory=lambda: _get_int("BREAKPOINT_THRESHOLD_AMOUNT", 85))


@dataclass
class RetrievalConfig:
    """
    Retrieval pipeline configuration.
    TOP_K        : cast wide net first — retrieve 10 chunks initially
    SCORE_THRESHOLD : drop chunks below 70% relevance
    FINAL_TOP_K  : only 3 clean chunks go to LLM — saves tokens
    BM25_WEIGHT  : 40% keyword search weight
    SEMANTIC_WEIGHT : 60% semantic search weight
    """
    top_k             : int   = field(default_factory=lambda: _get_int("TOP_K", 10))
    score_threshold   : float = field(default_factory=lambda: _get_float("SCORE_THRESHOLD", 0.70))
    final_top_k       : int   = field(default_factory=lambda: _get_int("FINAL_TOP_K", 3))
    bm25_weight       : float = field(default_factory=lambda: _get_float("BM25_WEIGHT", 0.4))
    semantic_weight   : float = field(default_factory=lambda: _get_float("SEMANTIC_WEIGHT", 0.6))


@dataclass
class RerankerConfig:
    """
    HuggingFace Cross-Encoder reranker configuration.
    Why ms-marco-MiniLM-L-6-v2: Free, runs locally, trained specifically
    for passage reranking — perfect for question-answer pairs.
    """
    model_name : str = field(default_factory=lambda: _get_env("RERANKER_MODEL", "cross-encoder/ms-marco-TinyBERT-L-2-v2"))


@dataclass
class APIConfig:
    """
    FastAPI server configuration.
    Why 0.0.0.0: accepts requests from any machine — needed for deployment.
    Why reload=True: auto restarts on code change during development.
    """
    host   : str  = field(default_factory=lambda: _get_env("API_HOST", "0.0.0.0"))  # nosec B104
    port   : int  = field(default_factory=lambda: _get_int("API_PORT", 8000))
    reload : bool = field(default_factory=lambda: _get_bool("API_RELOAD", True))


@dataclass
class DataConfig:
    """
    Data directory configuration.
    Why Path: cross-platform path handling — works on Windows, Mac, Linux.
    """
    pdf_dir : Path = field(default_factory=lambda: Path(_get_env("PDF_DIR", "./data/pdfs")))

# MASTER CONFIG CLASS
@dataclass
class Config:
    """
    Master configuration — combines all config groups.
    Every module imports this single object.

    Usage:
        from utils.config import config
        print(config.llm.model)
        print(config.retrieval.top_k)
    """
    llm          : LLMConfig          = field(default_factory=LLMConfig)
    embedding    : EmbeddingConfig    = field(default_factory=EmbeddingConfig)
    vectorstore  : VectorStoreConfig  = field(default_factory=VectorStoreConfig)
    chunking     : ChunkingConfig     = field(default_factory=ChunkingConfig)
    retrieval    : RetrievalConfig    = field(default_factory=RetrievalConfig)
    reranker     : RerankerConfig     = field(default_factory=RerankerConfig)
    api          : APIConfig          = field(default_factory=APIConfig)
    data         : DataConfig         = field(default_factory=DataConfig)

config = Config()
