# =============================================================================
# RXINSIGHT - STREAMLIT FRONTEND
# =============================================================================
# Architecture:
# → Direct Python calls to backend — no HTTP/FastAPI middleman
# → Works on Streamlit Cloud without running a separate server
# → FastAPI (backend/main.py) still available for API/local development
#
# Features:
# → Streaming responses (ChatGPT effect)
# → Dynamic FAQs from drug data
# → Chat history for follow up questions
# → Sources display
# → Knowledge base status
# → Auto-ingestion: new PDFs in data/ appear automatically
#
# Performance:
# → Cached status/drug checks via @st.cache_data
# → Pipeline models preloaded once at startup (via app.py)
# → No HTTP overhead — direct function calls
# =============================================================================

import json
import os
import sys
from pathlib import Path
import streamlit as st

# =============================================================================
# PATH INJECTION (Safety for Streamlit Cloud)
# =============================================================================
# This ensures the frontend can find the 'backend' folder even if run directly.
# 1. Resolve Root and Backend directories
FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(FRONTEND_DIR)
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")

# 2. Add backend to sys.path so 'from ingestion import...' works
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# 3. Check if we are running without the root launcher
if "startup_complete" not in st.session_state or not st.session_state.startup_complete:
    try:
        # Import backend functions (now that path is fixed)
        from ingestion import is_vectorstore_ready
        from pipeline import run_ingestion
        from pipeline.rag_pipeline import get_pipeline_state
        from pipeline.file_watcher import start_file_watcher

        # Run ingestion if needed
        if not is_vectorstore_ready():
            with st.spinner("🔄 Building knowledge base from PDFs... This may take a few minutes on first startup."):
                run_ingestion()
        
        # Start background file watcher
        try:
            start_file_watcher()
        except Exception:
             pass 

        # Pre-load models into memory
        get_pipeline_state()
        
        st.session_state.startup_complete = True
    except Exception as e:
        # If anything fails, we'll try again on next rerun
        pass


# =============================================================================
# BACKEND IMPORTS — Direct Python calls (no HTTP)
# =============================================================================

from ingestion import is_vectorstore_ready
from ingestion.embedder import load_vectorstore
from pipeline.rag_pipeline import (
    run_query,
    run_query_stream,
    get_pipeline_state,
    build_drug_dictionary,
)
from utils.logger import get_logger

logger = get_logger("frontend")


# =============================================================================
# BACKEND CALLS — Cached for Performance
# =============================================================================

@st.cache_data(ttl=60, show_spinner=False)
def call_status() -> dict:
    """Checks if vectorstore is ready. Cached for 60 seconds."""
    try:
        ready = is_vectorstore_ready()
        return {
            "vectorstore_ready": ready,
            "message": "ready" if ready else "not ready",
        }
    except Exception as e:
        return {"vectorstore_ready": False, "error": str(e)}


@st.cache_data(ttl=60, show_spinner=False)
def call_drugs() -> list:
    """
    Fetches available drugs directly from the vectorstore.
    Cached for 60 seconds — refreshes when new PDFs are auto-ingested.
    """
    try:
        vectorstore = load_vectorstore()
        all_data = vectorstore.get()
        metadatas = all_data.get("metadatas", [])

        drugs = {}
        for meta in metadatas:
            drug_name = meta.get("drug_name", "")
            generic_name = meta.get("generic_name", "")
            if drug_name and drug_name not in drugs:
                drugs[drug_name] = {
                    "brand_name": drug_name,
                    "generic_name": generic_name,
                }

        return list(drugs.values())
    except Exception as e:
        logger.error(f"Failed to get drugs: {e}")
        return []


def call_query(query: str, history: list) -> dict:
    """Calls query pipeline directly — no HTTP."""
    try:
        return run_query(
            user_query=query,
            chat_history=history,
        )
    except Exception as e:
        return {"error": str(e)}


def stream_query(query: str, history: list):
    """
    Streams answer directly from the pipeline.

    Why streaming?
    → User sees response immediately ✅
    → ChatGPT typing effect ✅
    → Better UX than waiting

    Yields:
        str : chunks of answer text
    """
    try:
        for chunk in run_query_stream(
            user_query=query,
            chat_history=history,
        ):
            yield chunk
    except Exception as e:
        yield f"Error: {str(e)}"


# =============================================================================
# FAQ GENERATOR
# =============================================================================

def generate_faqs(drugs: list) -> list:
    """
    Generates FAQ questions from drug list.

    Why generate not hardcode?
    → Drug list comes from Chroma ✅
    → Auto updates with new drugs ✅
    → Covers all available drugs ✅
    """
    faqs = []
    templates = [
        "What is {drug} used for?",
        "What are the side effects of {drug}?",
        "What is the recommended dose of {drug}?",
    ]

    for drug in drugs:
        brand = drug.get("brand_name", "")
        if brand:
            for template in templates:
                faqs.append(template.format(drug=brand))

    return faqs


# =============================================================================
# MAIN UI
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # Page config
    # -------------------------------------------------------------------------
    st.set_page_config(
        page_title="RxInsight",
        page_icon="💊",
        layout="wide",
    )

    # -------------------------------------------------------------------------
    # Session state
    # -------------------------------------------------------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "input_query" not in st.session_state:
        st.session_state.input_query = ""

    if "drugs" not in st.session_state:
        st.session_state.drugs = []

    if "faqs" not in st.session_state:
        st.session_state.faqs = []

    # -------------------------------------------------------------------------
    # SIDEBAR
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.title("RxInsight")
        st.markdown("*AI-powered drug information*")
        st.divider()

        # Status check — cached for 60s
        status = call_status()

        if status.get("vectorstore_ready"):
            st.success("✅ Knowledge base ready")
        else:
            st.error("❌ Knowledge base not ready")
            error_msg = status.get("error", "")
            if error_msg:
                st.warning(f"Error: {error_msg}")
            else:
                st.warning("Knowledge base is not available. Please check that PDFs exist in the data folder.")
            st.stop()

        st.divider()

        # FAQ Section — dynamic from drug data
        st.subheader("Frequently Asked Questions")
        st.caption("Click any question to ask it")

        drugs = call_drugs()
        dynamic_faqs = generate_faqs(drugs)

        # Use dynamic FAQs if available, otherwise show generic ones
        if dynamic_faqs:
            display_faqs = dynamic_faqs[:12]  # Limit to 12 FAQs
        else:
            display_faqs = [
                "What drugs are available?",
                "Tell me about available medications",
            ]

        for faq in display_faqs:
            if st.button(
                faq,
                key=f"faq_{faq}",
                use_container_width=True,
            ):
                st.session_state.input_query = faq
                st.rerun()

        st.divider()

        # Available drugs list
        st.subheader("Available Drugs")
        if drugs:
            for drug in drugs:
                brand = drug.get("brand_name", "")
                generic = drug.get("generic_name", "")
                if generic:
                    st.markdown(f"• **{brand}** *({generic})*")
                else:
                    st.markdown(f"• **{brand}**")
        else:
            st.caption("No drugs ingested yet")

        st.divider()

        # Clear chat button
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.input_query = ""
            st.rerun()

    # -------------------------------------------------------------------------
    # MAIN AREA
    # -------------------------------------------------------------------------
    st.title("RxInsight — Drug Intelligence Assistant")
    st.caption(
        "Ask questions about any available drug — "
        "side effects, dosage, interactions and more"
    )
    st.divider()

    # -------------------------------------------------------------------------
    # CHAT HISTORY
    # -------------------------------------------------------------------------
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources for assistant messages
            if (
                message["role"] == "assistant"
                and message.get("sources")
            ):
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.markdown(f"• {source}")

    # -------------------------------------------------------------------------
    # CHAT INPUT
    # -------------------------------------------------------------------------
    default_input = st.session_state.input_query
    if default_input:
        st.session_state.input_query = ""

    query = st.chat_input(
        "Ask about any drug — side effects, dosage, interactions..."
    )

    # Use FAQ pre-filled query if available
    if default_input:
        query = default_input

    # -------------------------------------------------------------------------
    # PROCESS QUERY
    # -------------------------------------------------------------------------
    if query:
        # Add user message (ONLY place this happens — no duplicate)
        st.session_state.messages.append({
            "role": "user",
            "content": query,
        })

        with st.chat_message("user"):
            st.markdown(query)

        # Prepare chat history for pipeline
        history = [
            {
                "role": m["role"],
                "content": m["content"],
            }
            for m in st.session_state.messages[-20:]
        ]

        # Set spinner state
        st.session_state["waiting_for_response"] = True
        st.rerun()

    # Handle spinner and response streaming after rerun
    if st.session_state.get("waiting_for_response", False) and st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        query = st.session_state.messages[-1]["content"]
        history = [
            {
                "role": m["role"],
                "content": m["content"],
            }
            for m in st.session_state.messages[-21:-1]
        ]
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_answer = ""
            placeholder.markdown("🔍 <i>Thinking...</i>", unsafe_allow_html=True)
            for chunk in stream_query(query, history):
                full_answer += chunk
                display_text = full_answer.split("__SOURCES__")[0]
                clean_answer = (
                    display_text
                    .replace("Answer:", "")
                    .replace("🤖", "")
                    .strip()
                )
                if clean_answer:
                    placeholder.markdown(clean_answer + "▌")

            sources = []
            if "__SOURCES__" in full_answer:
                parts = full_answer.split("__SOURCES__", 1)
                answer_text = parts[0]
                try:
                    sources = json.loads(parts[1])
                except Exception:
                    sources = []
            else:
                answer_text = full_answer
            clean_answer = (
                answer_text
                .replace("Answer:", "")
                .replace("🤖", "")
                .strip()
            )
            placeholder.markdown(clean_answer)
            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.markdown(f"• {source}")
        st.session_state.messages.append({
            "role": "assistant",
            "content": clean_answer,
            "sources": sources,
        })
        st.session_state.input_query = ""
        st.session_state["waiting_for_response"] = False
        st.rerun()


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    main()