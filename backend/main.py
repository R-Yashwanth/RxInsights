# =============================================================================
# RXINSIGHT - FASTAPI ENTRY POINT
# =============================================================================

import uvicorn
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from pipeline import run_ingestion, run_query, run_query_stream
from ingestion import is_vectorstore_ready
from utils.logger import get_logger
from utils.config import config

logger = get_logger(__name__)


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title       = "RxInsight API",
    description = (
        "RxInsight — Enterprise pharmaceutical intelligence platform. "
        "Powered by LangChain, Groq, HuggingFace, and Chroma."
    ),
    version  = "2.0.0",
    docs_url = "/docs",
    redoc_url= "/redoc",
)


# =============================================================================
# CORS
# =============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# =============================================================================
# STARTUP EVENT — Preload Models
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Preloads all models when FastAPI starts.

    Why on startup?
    → First query should be fast
    → Models loaded before any request comes in
    → Zero loading time per query ✅
    """
    logger.info("🚀 RxInsight starting up...")

    if is_vectorstore_ready():
        logger.info("Preloading pipeline models...")
        try:
            from pipeline.rag_pipeline import get_pipeline_state
            get_pipeline_state()
            logger.info("✅ All models preloaded successfully")
        except Exception as e:
            logger.error(f"Model preloading failed: {e}")
    else:
        logger.warning(
            "⚠️ Vectorstore not ready — "
            "models will load on first query"
        )

    # Start auto-ingestion file watcher
    # Monitors data folder for new PDFs and ingests them automatically
    try:
        from pipeline.file_watcher import start_file_watcher
        start_file_watcher()
    except Exception as e:
        logger.error(f"File watcher startup failed: {e}")


# =============================================================================
# REQUEST / RESPONSE MODELS
# =============================================================================

class QueryRequest(BaseModel):
    """Request model for POST /query endpoint."""
    query: str = Field(
        example     = "What are the side effects of Jardiance?",
        description = "Natural language question about pharmaceutical drugs",
        max_length  = 500,
    )
    chat_history: Optional[List[Dict]] = []


class QueryResponse(BaseModel):
    """Response model for POST /query endpoint."""
    answer          : str
    rewritten_query : str
    sub_queries     : list
    chunks_used     : int
    sources         : list


class IngestResponse(BaseModel):
    """Response model for POST /ingest endpoint."""
    status         : str
    pages_loaded   : int
    chunks_created : int
    message        : str


class StatusResponse(BaseModel):
    """Response model for GET /status endpoint."""
    vectorstore_ready : bool
    message           : str


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get(
    "/",
    summary     = "Health Check",
    description = "Check if API is running",
)
def health_check():
    """Basic health check — returns 200 if running."""
    logger.info("Health check called")
    return {
        "status" : "running",
        "api"    : "Pharma RAG API",
        "version": "1.0.0",
        "docs"   : "/docs",
    }


@app.get(
    "/status",
    response_model = StatusResponse,
    summary        = "Vectorstore Status",
)
def get_status():
    """Checks if vectorstore is ready for queries."""
    logger.info("Status check called")
    ready = is_vectorstore_ready()
    return StatusResponse(
        vectorstore_ready = ready,
        message = (
            "Vectorstore ready — you can run queries!"
            if ready else
            "Vectorstore not ready — please run POST /ingest first"
        )
    )


@app.post(
    "/ingest",
    response_model = IngestResponse,
    summary        = "Run Ingestion Pipeline",
)
def ingest():
    """Triggers complete ingestion pipeline."""
    logger.info("Ingestion endpoint called")
    try:
        result = run_ingestion()
        return IngestResponse(
            status         = result["status"],
            pages_loaded   = result["pages_loaded"],
            chunks_created = result["chunks_created"],
            message        = result["message"],
        )
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(
            status_code = 500,
            detail      = f"Ingestion failed: {str(e)}"
        )


@app.post(
    "/query",
    response_model = QueryResponse,
    summary        = "Run RAG Query",
)
def query(request: QueryRequest):
    """
    Runs Advanced RAG pipeline for user query.

    Why non streaming here?
    → Used for programmatic access
    → Returns complete response at once
    → Use /query/stream for streaming ✅
    """
    logger.info(f"Query endpoint: '{request.query}'")
    try:
        result = run_query(
            user_query   = request.query,
            chat_history = request.chat_history,
        )
        return QueryResponse(
            answer          = result["answer"],
            rewritten_query = result["rewritten_query"],
            sub_queries     = result["sub_queries"],
            chunks_used     = result["chunks_used"],
            sources         = result["sources"],
        )
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(
            status_code = 500,
            detail      = f"Query failed: {str(e)}"
        )


@app.post(
    "/query/stream",
    summary = "Run RAG Query with Streaming",
)
def query_stream(request: QueryRequest):
    """
    Streams RAG answer word by word.

    Why streaming?
    → User sees response immediately
    → Feels like ChatGPT typing effect ✅
    → Much better UX than waiting

    How it works:
    → FastAPI StreamingResponse
    → Yields text chunks as they generate
    → Streamlit reads and displays each chunk

    # -------------------------------------------------------------------------
    # CHANGE: Added try/except INSIDE the generator (generate function).
    #
    # WHAT YOU WROTE:
    #   def generate():
    #       for chunk in run_query_stream(...):  ← no error handling inside
    #           yield chunk
    #
    #   The outer try/except only catches errors BEFORE the generator starts.
    #   Once StreamingResponse starts streaming, the outer try/except is
    #   already done — any error inside the generator silently closes
    #   the HTTP stream, and Streamlit shows "Response ended prematurely".
    #
    # WHAT WAS CHANGED:
    #   Moved try/except INSIDE generate() so errors are caught mid-stream
    #   and yielded as an error message instead of closing the connection.
    # -------------------------------------------------------------------------
    """
    logger.info(f"Stream query: '{request.query}'")

    # ORIGINAL: outer try/except — only catches setup errors, NOT stream errors
    # CHANGED : generator now handles its own errors internally (see below)
    def generate():
        try:
            # Same as you wrote — iterate chunks from run_query_stream
            for chunk in run_query_stream(
                user_query   = request.query,
                chat_history = request.chat_history,
            ):
                yield chunk

        # CHANGE: catch errors mid-stream and send them as text
        # WHY: if this except block is missing, stream closes silently
        #      → Streamlit shows "Response ended prematurely" ← YOUR BUG
        except Exception as e:
            logger.error(f"Stream generator error: {e}")
            yield f"Error: {str(e)}"

    return StreamingResponse(
        generate(),
        # Why text/plain: simple format for streaming (unchanged)
        media_type = "text/plain",
    )


@app.get(
    "/drugs",
    summary = "Get Available Drugs",
)
def get_drugs():
    """
    Returns list of available drugs from Chroma.

    Why this endpoint?
    → Frontend uses this to show drug list
    → FAQs generated from actual drug data ✅
    → No hardcoding in frontend ✅
    → Auto updates when new drugs ingested ✅
    """
    logger.info("Drugs endpoint called")

    try:
        from ingestion.embedder import load_vectorstore

        vectorstore = load_vectorstore()
        all_data    = vectorstore.get()
        metadatas   = all_data.get("metadatas", [])

        # Extract unique drugs with generic names
        drugs = {}
        for meta in metadatas:
            drug_name    = meta.get("drug_name", "")
            generic_name = meta.get("generic_name", "")

            if drug_name and drug_name not in drugs:
                drugs[drug_name] = {
                    "brand_name"  : drug_name,
                    "generic_name": generic_name,
                }

        drug_list = list(drugs.values())

        logger.info(f"Returning {len(drug_list)} drugs")
        return {
            "drugs" : drug_list,
            "total" : len(drug_list),
        }

    except Exception as e:
        logger.error(f"Failed to get drugs: {e}")
        raise HTTPException(
            status_code = 500,
            detail      = f"Failed to get drugs: {str(e)}"
        )


# =============================================================================
# RUN SERVER
# =============================================================================

if __name__ == "__main__":
    logger.info("Starting Pharma RAG API...")
    logger.info(
        f"Docs at: http://localhost:{config.api.port}/docs"
    )
    uvicorn.run(
        "main:app",
        host   = config.api.host,
        port   = config.api.port,
        reload = config.api.reload,
    )
