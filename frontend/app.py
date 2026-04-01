# =============================================================================
# RXINSIGHT - STREAMLIT FRONTEND
# =============================================================================
# Why Streamlit?
# → Fast to build — no HTML/CSS needed
# → Free deployment on Streamlit Cloud
# → Built in chat interface
# → Easy integration with FastAPI
#
# Features:
# → Streaming responses (ChatGPT effect)
# → Dynamic FAQs from drug data
# → Chat history for follow up questions
# → Sources display
# → Knowledge base status
#
# -------------------------------------------------------------------------
# CHANGES MADE FOR SPEED:
#
# 1. REMOVED 'from torch import chunk':
#    Was loading ~500MB PyTorch library on every startup for nothing.
#
# 2. CACHED call_status() with @st.cache_data(ttl=60):
#    YOUR CODE called call_status() on EVERY Streamlit rerun.
#    Every FAQ click, clear chat, or keystroke triggered a full HTTP call
#    to /status, which called is_vectorstore_ready(), which loaded the
#    embedding model + vectorstore from scratch EVERY TIME.
#    → Now cached for 60 seconds. Status doesn't change that often.
#
# 3. FIXED FAQ double-message bug:
#    YOUR CODE: FAQ button handler appended message to st.session_state,
#    then called st.rerun(). After rerun, the process query section
#    appended the SAME message again → duplicate user messages in chat.
#    → Now FAQ button only sets input_query, doesn't append message.
#      The process query section handles the append once.
#
# 4. CACHED call_drugs() with @st.cache_data(ttl=300):
#    Drug list doesn't change unless you re-ingest.
#    No need to call API on every rerun.
#
# 5. FIXED clear chat instant:
#    Clear chat no longer triggers call_status() or call_drugs()
#    because those are now cached. Rerun is instant.
# -------------------------------------------------------------------------
# =============================================================================

import requests
import streamlit as st
# CHANGE: Removed 'from torch import chunk' — see note #1 above.


# =============================================================================
# CONFIGURATION
# =============================================================================

import os
API_URL = os.getenv("API_URL", "http://localhost:8000")



# =============================================================================
# API CALLS — NOW CACHED
# =============================================================================


# =============================================================================
# API CALLS — NOW CACHED
# =============================================================================

# -------------------------------------------------------------------------
# CHANGE: Added @st.cache_data(ttl=60)
#
# WHAT YOU WROTE:
#   def call_status() -> dict:
#       response = requests.get(f"{API_URL}/status", timeout=10)
#       return response.json()
#
# WHY IT WAS SLOW:
#   Streamlit reruns the ENTIRE main() function on every interaction:
#   - Click FAQ button → rerun → call_status() → HTTP call → backend
#     loads embedding model + vectorstore → 2-5 seconds wasted
#   - Click clear chat → rerun → same 2-5 seconds wasted
#   - Type in chat box → rerun → same 2-5 seconds wasted
#   This was THE MAIN reason everything felt slow.
#
# WHAT WAS CHANGED:
#   @st.cache_data(ttl=60) caches the result for 60 seconds.
#   First call: normal HTTP request.
#   Next 60 seconds: returns cached dict instantly (0ms).
#   Vectorstore status doesn't change unless you re-ingest.
# -------------------------------------------------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def call_status() -> dict:
    """Checks if vectorstore is ready. Cached for 60 seconds."""
    try:
        response = requests.get(
            f"{API_URL}/status",
            timeout = 10,
        )
        return response.json()
    except Exception as e:
        return {"vectorstore_ready": False, "error": str(e)}


# -------------------------------------------------------------------------
# CHANGE: Added @st.cache_data(ttl=300)
#
# WHAT YOU WROTE:
#   def call_drugs() -> list:
#       response = requests.get(f"{API_URL}/drugs", timeout=10)
#       ...
#
# WHY IT WAS SLOW:
#   Same as call_status — called on every rerun.
#   Drug list only changes after re-ingestion, so caching for
#   5 minutes (300 seconds) is safe.
#
# WHAT WAS CHANGED:
#   @st.cache_data(ttl=300) — cached for 5 minutes.
# -------------------------------------------------------------------------
# CHANGE: Reduced TTL from 300s to 60s for auto-ingestion compatibility.
# When new PDFs are added, new drugs should appear within ~1 minute.
@st.cache_data(ttl=60, show_spinner=False)
def call_drugs() -> list:
    """
    Fetches available drugs from FastAPI. Cached for 60 seconds.

    Why from API?
    → Not hardcoded ✅
    → Auto updates when new drugs ingested ✅
    → Returns brand + generic names ✅
    """
    try:
        response = requests.get(
            f"{API_URL}/drugs",
            timeout = 10,
        )
        data = response.json()
        return data.get("drugs", [])
    except Exception as e:
        return []


def call_query(query: str, history: list) -> dict:
    """Calls non-streaming query endpoint."""
    try:
        response = requests.post(
            f"{API_URL}/query",
            json    = {
                "query"       : query,
                "chat_history": history,
            },
            timeout = 60,
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def stream_query(query: str, history: list):
    """
    Streams answer from FastAPI word by word.

    Why streaming?
    → User sees response immediately ✅
    → ChatGPT typing effect ✅
    → Better UX than waiting 8 sec

    Yields:
        str : chunks of answer text
    """
    try:
        with requests.post(
            f"{API_URL}/query/stream",
            json    = {
                "query"       : query,
                "chat_history": history,
            },
            stream  = True,
            timeout = 60,
        ) as response:
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    yield chunk.decode("utf-8")
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

    FAQ pattern per drug:
    → What is {drug} used for?
    → What are the side effects of {drug}?
    → What is the recommended dose of {drug}?

    Args:
        drugs : list of drug dicts from /drugs endpoint

    Returns:
        list : FAQ question strings
    """
    faqs     = []
    # FAQ templates — generic medical questions
    # Not drug specific — works for any drug ✅
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
        page_title = "RxInsight",
        page_icon  = "💊",
        layout     = "wide",
    )

    # -------------------------------------------------------------------------
    # Session state
    # Why session_state?
    # → Persists data across Streamlit reruns
    # → Maintains chat history
    # → Stores drug list
    # -------------------------------------------------------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "input_query" not in st.session_state:
        st.session_state.input_query = ""

    if "drugs" not in st.session_state:
        st.session_state.drugs = []

    if "faqs" not in st.session_state:
        st.session_state.faqs = []

    HARDCODED_FAQS = [
        "What is Jardiance used for?",
        "What are the side effects of Jardiance?",
        "What is the dosage of Jardiance?",
        "What is Entresto used for?",
        "What are the side effects of Entresto?",
        "Compare Jardiance vs Entresto",
        "Drug interactions of Jardiance",
        "Contraindications of Entresto",
        "How to take Jardiance?",
        "Warnings of Entresto"
    ]

    # -------------------------------------------------------------------------
    # SIDEBAR
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.title("RxInsight")
        st.markdown("*AI-powered drug information*")
        st.divider()

        # Status check — NOW CACHED (see @st.cache_data above)
        # CHANGE: This used to take 2-5 seconds on every rerun.
        # Now returns instantly from cache after first call.
        status = call_status()


        if status.get("vectorstore_ready"):
            st.success("✅ Knowledge base ready")
        else:
            st.error("❌ Knowledge base not ready")
            st.warning("Please upload PDFs and run ingestion.")

            # File uploader for PDFs
            uploaded_files = st.file_uploader(
                "Upload PDF files for ingestion",
                type=["pdf"],
                accept_multiple_files=True,
                key="pdf_uploader",
            )

            # Save uploaded files to a temp directory
            import tempfile, shutil
            temp_dir = tempfile.mkdtemp()
            pdf_paths = []
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    file_path = f"{temp_dir}/{uploaded_file.name}"
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    pdf_paths.append(file_path)

            # Ingest button
            if st.button("Run Ingestion", type="primary", use_container_width=True, disabled=not pdf_paths):
                # Send files to backend ingestion endpoint
                with st.spinner("Ingesting PDFs and building knowledge base..."):
                    files = [("files", (open(path, "rb"))) for path in pdf_paths]
                    try:
                        response = requests.post(f"{API_URL}/ingest", files=files, timeout=600)
                        if response.status_code == 200:
                            st.success("Ingestion complete! Reloading...")
                            st.cache_data.clear()
                            st.rerun()
                        else:
                            st.error(f"Ingestion failed: {response.text}")
                    except Exception as e:
                        st.error(f"Ingestion error: {e}")
            st.stop()

        st.divider()

        # FAQ Section
        st.subheader("Frequently Asked Questions")
        st.caption("Click any question to ask it")

        for faq in HARDCODED_FAQS:
            if st.button(
                faq,
                key=f"faq_{faq}",
                use_container_width=True,
            ):
                # ---------------------------------------------------------
                # CHANGE: Fixed FAQ double-message bug.
                #
                # WHAT YOU WROTE:
                #   st.session_state.messages.append({
                #       "role": "user",
                #       "content": faq
                #   })
                #   st.session_state.input_query = faq
                #   st.rerun()
                #
                # WHY IT WAS WRONG:
                #   1. Button handler appends message to messages list
                #   2. st.rerun() reruns main()
                #   3. After rerun, default_input = faq → query = faq
                #   4. Process query section (line ~325) appends SAME
                #      message to messages list AGAIN
                #   → User message appeared TWICE in chat history ❌
                #   → Also slowed things down — processed query twice
                #
                # WHAT WAS CHANGED:
                #   Only set input_query — DON'T append message here.
                #   Let the process query section handle the single append.
                # ---------------------------------------------------------
                st.session_state.input_query = faq
                st.rerun()

        st.divider()

        # Available drugs list — NOW CACHED via call_drugs()
        st.subheader("Available Drugs")
        drugs = call_drugs()
        for drug in drugs:
            brand   = drug.get("brand_name", "")
            generic = drug.get("generic_name", "")
            if generic:
                st.markdown(f"• **{brand}** *({generic})*")
            else:
                st.markdown(f"• **{brand}**")

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

    # Spinner/Thinking message while waiting for LLM response
    # (Handled below in the streaming section, so do not duplicate here)

    # -------------------------------------------------------------------------
    # CHAT INPUT
    # -------------------------------------------------------------------------
    # Handle pre-filled input from FAQ clicks
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
        # Add user message (ONLY place this happens now — no duplicate)
        st.session_state.messages.append({
            "role"   : "user",
            "content": query,
        })

        with st.chat_message("user"):
            st.markdown(query)

        # Prepare chat history for API
        history = [
            {
                "role"   : m["role"],
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
                "role"   : m["role"],
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
            import json as _json
            sources = []
            if "__SOURCES__" in full_answer:
                parts = full_answer.split("__SOURCES__", 1)
                answer_text = parts[0]
                try:
                    sources = _json.loads(parts[1])
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