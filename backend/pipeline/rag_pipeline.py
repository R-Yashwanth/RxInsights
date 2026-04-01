# =============================================================================
# PHARMA RAG - OPTIMIZED RAG PIPELINE
# =============================================================================
# Optimizations:
# 1. TinyBERT reranker    → 5x faster than MiniLM-L6
# 2. No query rewriter    → Llama 3.3 70B understands natively
# 3. No compressor        → Reranker filters well enough
# 4. Rule based router    → Zero LLM call for query analysis
# 5. Metadata filtering   → Search drug chunks only
# 6. Response caching     → Repeated queries instant
# 7. Model preloading     → Load once at startup
# 8. Sync search          → asyncio.run removed (causes lag)
# 9. Streaming response   → Word by word output
# Guardrails:
# G1. Small talk          → Instant response no RAG
# G2. Out-of-domain       → Graceful decline
# G3. Context guardrail   → No hallucination via prompt
# =============================================================================

import json
import re
import hashlib
import random
from pathlib import Path
from typing import Optional, List, Dict, Generator

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_core.documents import Document

from ingestion import (
    load_all_pdfs,
    chunk_documents,
    embed_and_store,
    load_vectorstore,
    is_vectorstore_ready,
)

from retrieval import (
    hybrid_search,
    rerank_documents,
    load_reranker_model,
)

from utils.logger import get_logger
from utils.config import config
from utils.helpers import (
    validate_query,
    format_documents,
    Timer,
)

logger = get_logger(__name__)


# =============================================================================
# GLOBAL STATE — Load Once Reuse Forever
# =============================================================================

_vectorstore     : Optional[Chroma]         = None
_documents       : Optional[List[Document]] = None
_reranker_model                             = None
_llm                                        = None  # ← cached LLM
_llm_stream                                 = None  # ← cached streaming LLM
_drug_dictionary : Dict[str, str]           = {}
_config_data     : Dict                     = {}
_response_cache  : Dict[str, dict]          = {}


# =============================================================================
# GUARDRAIL 1: SMALL TALK DETECTOR
# =============================================================================

SMALL_TALK_PATTERNS = [
    "hi", "hello", "hey", "sup",
    "how are you", "how r u", "how are you doing",
    "good morning", "good evening", "good afternoon",
    "thanks", "thank you", "thx",
    "what's up", "whats up",
    "nice to meet you",
    "bye", "goodbye", "see you",
    "you there", "are you there"
]

SMALL_TALK_RESPONSES = {
    "greeting": [
        "Hello! I'm your pharmaceutical assistant. How can I help you with medications today?",
        "Hi there! I can help with pharmaceutical information. What would you like to know?",
    ],
    "how_are_you": [
        "I'm functioning well, thank you! Ready to help with your pharmaceutical questions.",
        "All good! How can I assist you with medications today?",
    ],
    "thanks": [
        "You're welcome! Feel free to ask more pharmaceutical questions.",
        "Happy to help! Let me know if you need anything else.",
    ],
    "goodbye": [
        "Goodbye! Feel free to return if you have more pharmaceutical questions.",
        "Take care! I'll be here when you need medication information.",
    ],
    "default": [
        "I'm your pharmaceutical assistant. How can I help with medication questions?",
    ]
}


def _word_match(pattern: str, text: str) -> bool:
    """
    Checks if pattern exists as a whole word/phrase in text.

    # -------------------------------------------------------------------------
    # CHANGE: Added word-boundary matching helper.
    #
    # WHAT YOU WROTE:
    #   any(p in query_lower for p in SMALL_TALK_PATTERNS)
    #   → simple substring check: 'hi' in 'at what age this drug is preferrable'
    #   → 'this' contains 'hi' → matched as small talk! ❌
    #   → Same bug: 'thanks' matched in 'thanksgiving'
    #   → Same bug: 'hey' matched in 'they'
    #
    # WHAT WAS CHANGED:
    #   Uses regex \\b (word boundary) so 'hi' only matches the standalone
    #   word 'hi', not 'this', 'him', 'history', etc. ✅
    # -------------------------------------------------------------------------
    """
    return bool(re.search(r'\b' + re.escape(pattern) + r'\b', text))


def is_small_talk(query: str) -> bool:
    """Detects simple conversational queries using word-boundary matching."""
    query_lower = query.lower().strip()
    # Exact match for very short greetings
    if query_lower in ["hi", "hello", "hey", "sup"]:
        return True
    # Word-boundary match for patterns (CHANGED from substring match)
    return any(_word_match(p, query_lower) for p in SMALL_TALK_PATTERNS)


def get_small_talk_response(query: str) -> str:
    """Returns appropriate response for small talk."""
    q = query.lower().strip()
    # CHANGED: all checks now use _word_match instead of 'in' substring
    if any(_word_match(g, q) for g in ["hi", "hello", "hey", "sup", "what's up"]):
        return random.choice(SMALL_TALK_RESPONSES["greeting"])
    elif any(_word_match(h, q) for h in ["how are you", "how r u"]):
        return random.choice(SMALL_TALK_RESPONSES["how_are_you"])
    elif any(_word_match(t, q) for t in ["thanks", "thank you", "thx"]):
        return random.choice(SMALL_TALK_RESPONSES["thanks"])
    elif any(_word_match(b, q) for b in ["bye", "goodbye", "see you"]):
        return random.choice(SMALL_TALK_RESPONSES["goodbye"])
    return random.choice(SMALL_TALK_RESPONSES["default"])


# =============================================================================
# GUARDRAIL 2: OUT-OF-DOMAIN DETECTOR
# =============================================================================

OUT_OF_DOMAIN_KEYWORDS = [
    # Politics
    # CHANGE: Added short-form keywords like "pm", "cm"
    # WHAT YOU WROTE: only had "prime minister", "pm of india"
    # WHY: "who is our pm" didn't match — needs standalone "pm" too
    "prime minister", "pm of india", "pm", "cm", "modi", "rahul gandhi",
    "election", "parliament", "lok sabha", "rajya sabha",
    "minister", "political", "bjp", "congress",
    "trump", "biden", "president", "white house",
    # Sports
    "cricket", "football", "soccer", "basketball", "tennis",
    "world cup", "ipl", "match", "score", "tournament",
    "virat kohli", "dhoni", "messi", "ronaldo",
    # Entertainment
    "movie", "film", "actor", "actress", "bollywood", "hollywood",
    "netflix", "prime video", "song", "music", "celebrity",
    "oscar", "award", "series", "tv show",
    # Tech (non-pharma)
    "python", "javascript", "code", "programming", "software",
    "computer", "laptop", "phone", "iphone", "android",
    "website", "app", "algorithm",
    # General
    "capital of", "country", "weather", "climate",
    "earthquake", "flood", "disaster",
]

OUT_OF_DOMAIN_RESPONSES = [
    "I'm a pharmaceutical assistant focused on medication information. I can help with drug details, dosages, side effects, and comparisons. What medication can I help you with?",
    "I specialize in pharmaceutical information. If you have questions about drugs or treatments, I'm here to help!",
    "My expertise is pharmaceutical topics. For medication questions, I'm at your service!",
]


def is_out_of_domain(query: str) -> bool:
    """Detects queries outside pharmaceutical domain."""
    q = query.lower().strip()
    # CHANGED: was 'any(kw in q for kw ...)' — same substring bug as small talk
    # e.g. 'app' matched 'application', 'code' matched 'barcode'
    return any(_word_match(kw, q) for kw in OUT_OF_DOMAIN_KEYWORDS)


def get_out_of_domain_response() -> str:
    """Returns helpful decline message."""
    return random.choice(OUT_OF_DOMAIN_RESPONSES)

# =============================================================================
# FOLLOW-UP QUERY RESOLVER — LLM-Based (Enterprise Grade)
# =============================================================================
# CHANGE: Replaced hardcoded pattern lists with LLM-based query rewriting.
#
# WHAT YOU HAD BEFORE (v1 — hardcoded, fragile):
#   _FOLLOWUP_PATTERNS = ["this drug", "that drug", ...]
#   _PRONOUN_PATTERNS = ["its", "it's"]
#   → Checked each pattern one by one via string/regex matching
#   → Missed: "their", "those", "what about", "tell me more", etc.
#   → Required manually adding every possible pronoun — not scalable ❌
#
# WHY HARDCODING FAILS:
#   Users express follow-ups in infinite ways:
#   - "what were their side effects"
#   - "any warnings I should know about"
#   - "how about elderly patients"
#   - "can children take it"
#   - "compared to Entresto how is it"
#   You can never enumerate all patterns with a list.
#
# THE ENTERPRISE FIX:
#   Use the LLM itself (already cached in _llm) with a tiny prompt to
#   rewrite the query into a self-contained question.
#   Input:  ~100 tokens (history + query)
#   Output: ~20 tokens (rewritten query)
#   Latency: ~100-200ms on Groq (fastest LLM provider)
#
#   Only fires when:
#   1. Query does NOT contain a known drug name (skip if direct question)
#   2. Chat history exists (skip if first question)
#   → Zero extra latency for direct questions like "What is Jardiance?" ✅
#
# THIS IS THE STANDARD APPROACH:
#   Microsoft Copilot, Perplexity, Google's RAG — all use LLM-based
#   query rewriting. It's the industry standard for enterprise RAG.
# =============================================================================

# Prompt for query rewriting — kept minimal for speed
_REWRITE_PROMPT = """Given the conversation history, rewrite the follow-up question into a standalone question.
- Replace all pronouns (it, its, their, this, that, etc.) with the actual entity from the conversation
- The rewritten question must be fully self-contained — understandable without any history
- Output ONLY the rewritten question, nothing else
- If the question is already self-contained, return it as-is

Conversation:
{history}

Follow-up question: {question}

Rewritten question:"""


def resolve_followup_query(
    query: str,
    chat_history: List[Dict]
) -> str:
    """
    Enterprise-grade follow-up query resolver using LLM.

    Uses the cached Groq LLM to rewrite follow-up queries into
    self-contained questions. Only fires when needed (no drug name
    in query + chat history exists).

    Args:
        query        : raw user query (e.g. "what are their side effects?")
        chat_history : list of previous messages

    Returns:
        str : rewritten query (e.g. "What are the side effects of Jardiance?")
              or original query if rewriting not needed
    """
    # Skip if no chat history — first question, nothing to resolve
    if not chat_history:
        return query

    # Skip if query already contains a known drug name — direct question
    query_lower = query.lower()
    global _drug_dictionary
    if _drug_dictionary:
        for keyword in _drug_dictionary:
            if _word_match(keyword, query_lower):
                logger.debug(f"Query already has drug name '{keyword}' — skip rewrite")
                return query

    # Use cached LLM to rewrite the query
    global _llm
    if _llm is None:
        logger.warning("LLM not loaded — skipping query rewrite")
        return query

    try:
        # Format recent history (last 4 messages = 2 turns, enough for context)
        history_lines = []
        for msg in chat_history[-4:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            # Truncate long assistant messages to save tokens
            content = msg["content"][:200]
            history_lines.append(f"{role}: {content}")
        history_text = "\n".join(history_lines)

        # Call LLM with minimal prompt
        prompt = _REWRITE_PROMPT.format(
            history=history_text,
            question=query
        )

        with Timer("Query Rewrite (LLM)"):
            rewritten = _llm.invoke(prompt)

            # Extract text from AIMessage if needed
            if hasattr(rewritten, 'content'):
                rewritten = rewritten.content

            rewritten = rewritten.strip().strip('"').strip("'")

        # Validate: rewritten should not be empty or too long
        if not rewritten or len(rewritten) > 300:
            logger.warning("Query rewrite produced bad output — using original")
            return query

        if rewritten.lower() != query.lower():
            logger.info(f"🔄 Query rewritten: '{query}' → '{rewritten}'")
        else:
            logger.debug("Query unchanged after rewrite")

        return rewritten

    except Exception as e:
        # If rewrite fails, use original query — never block the pipeline
        logger.warning(f"Query rewrite failed: {e} — using original query")
        return query



# =============================================================================
# DRUG CONFIG LOADER
# =============================================================================

def load_drug_config() -> Dict:
    """
    Loads drug_config.json for comparison keywords.
    No drug names hardcoded — only generic English words.
    """
    config_path = Path(__file__).parent.parent / "drug_config.json"
    try:
        with open(config_path, "r") as f:
            data = json.load(f)
        logger.info("✅ Drug config loaded")
        return data
    except Exception as e:
        logger.error(f"Failed to load drug_config.json: {e}")
        return {
            "comparison_keywords": [
                "vs", "versus", "compare",
                "difference", "better", "between"
            ],
            "section_keywords": {}
        }


# =============================================================================
# DRUG DICTIONARY — Dynamic from Chroma
# =============================================================================

def build_drug_dictionary(vectorstore: Chroma) -> Dict[str, str]:
    """
    Builds drug search dictionary dynamically from Chroma metadata.

    Why dynamic?
    → Brand + generic names from actual PDF data ✅
    → Add new drug → re-ingest → auto updates ✅
    → Zero hardcoding ✅
    """
    dictionary = {}
    try:
        all_data   = vectorstore.get()
        metadatas  = all_data.get("metadatas", [])
        seen_drugs = set()

        for meta in metadatas:
            drug_name    = meta.get("drug_name", "")
            generic_name = meta.get("generic_name", "")

            if drug_name and drug_name not in seen_drugs:
                seen_drugs.add(drug_name)
                dictionary[drug_name.lower()] = drug_name
                if generic_name:
                    dictionary[generic_name.lower()] = drug_name

        logger.info(
            f"✅ Drug dictionary: "
            f"{len(seen_drugs)} drugs | "
            f"{len(dictionary)} keywords"
        )
    except Exception as e:
        logger.error(f"Failed to build drug dictionary: {e}")
    return dictionary


# =============================================================================
# RULE BASED QUERY ROUTER — Zero LLM Call
# =============================================================================

def detect_drugs(query: str) -> List[str]:
    """Detects drug names using dynamic dictionary lookup."""
    global _drug_dictionary
    if not _drug_dictionary:
        return []

    query_lower = query.lower()
    detected    = []
    for keyword, drug_name in _drug_dictionary.items():
        if keyword in query_lower:
            if drug_name not in detected:
                detected.append(drug_name)

    if detected:
        logger.info(f"Detected drugs: {detected}")
    else:
        logger.info("No drug detected — searching all")
    return detected


def is_comparison_query(query: str) -> bool:
    """Detects comparison queries using config keywords."""
    global _config_data
    keywords = _config_data.get("comparison_keywords", [])
    return any(kw in query.lower() for kw in keywords)


def route_query(query: str) -> dict:
    """
    Routes query to correct retrieval strategy.
    Zero LLM call — pure rule based ✅

    Returns:
        dict: {drugs, filter, is_comparison}
    """
    detected_drugs = detect_drugs(query)
    is_comparison  = is_comparison_query(query)

    # Comparison with multiple drugs → no filter
    if is_comparison and len(detected_drugs) > 1:
        logger.info("Comparison query — no filter")
        return {
            "drugs"        : detected_drugs,
            "filter"       : None,
            "is_comparison": True,
        }

    # Single drug → filter to that drug
    if len(detected_drugs) == 1:
        logger.info(f"Single drug — filter: {detected_drugs[0]}")
        return {
            "drugs"        : detected_drugs,
            "filter"       : {"drug_name": detected_drugs[0]},
            "is_comparison": False,
        }

    # No drug → search all
    logger.info("General query — no filter")
    return {
        "drugs"        : [],
        "filter"       : None,
        "is_comparison": False,
    }


# =============================================================================
# RESPONSE CACHE — Optimization 6
# =============================================================================

def get_cache_key(query: str) -> str:
    """SHA256 hash of lowercase query for caching."""
    return hashlib.sha256(
        query.lower().strip().encode()
    ).hexdigest()


def get_cached_response(query: str) -> Optional[dict]:
    """Returns cached response if available."""
    key = get_cache_key(query)
    if key in _response_cache:
        logger.info(f"✅ Cache HIT: '{query[:50]}'")
        return _response_cache[key]
    logger.debug(f"Cache MISS: '{query[:50]}'")
    return None


def cache_response(query: str, response: dict) -> None:
    """Caches response. Max 100 entries — evicts oldest."""
    if len(_response_cache) >= 100:
        oldest = next(iter(_response_cache))
        del _response_cache[oldest]
    key                  = get_cache_key(query)
    _response_cache[key] = response
    logger.info(f"Cached: '{query[:50]}'")


# =============================================================================
# MODEL PRELOADING — Optimization 7
# =============================================================================

def get_pipeline_state() -> tuple:
    """
    Returns preloaded pipeline state.
    All models loaded ONCE — reused forever.

    Fix: Also caches LLM instances to avoid
    creating new ChatGroq on every query ✅
    """
    global _vectorstore, _documents, _reranker_model
    global _drug_dictionary, _config_data
    global _llm, _llm_stream

    # 1. TinyBERT reranker — Optimization 1
    if _reranker_model is None:
        logger.info("Preloading TinyBERT reranker...")
        _reranker_model = load_reranker_model()
        logger.info("✅ Reranker loaded")

    # 2. Chroma vectorstore
    if _vectorstore is None:
        logger.info("Preloading vectorstore...")
        _vectorstore = load_vectorstore()
        logger.info("✅ Vectorstore loaded")

    # 3. Drug dictionary — Optimization 4+5
    if not _drug_dictionary:
        logger.info("Building drug dictionary...")
        _drug_dictionary = build_drug_dictionary(_vectorstore)

    # 4. Config data
    if not _config_data:
        _config_data = load_drug_config()

    # 5. BM25 documents
    if _documents is None:
        logger.info("Preloading BM25 documents...")
        raw        = _vectorstore.get()
        _documents = [
            Document(
                page_content = content,
                metadata     = meta
            )
            for content, meta in zip(
                raw["documents"],
                raw["metadatas"]
            )
        ]
        logger.info(f"✅ {len(_documents)} BM25 chunks loaded")

    # 6. Cache LLM instances
    # Why cache? Creating ChatGroq is expensive
    # Reuse same instance ✅
    if _llm is None:
        logger.info("Preloading Groq LLM...")
        _llm = ChatGroq(
            api_key     = config.llm.api_key,
            model       = config.llm.model,
            temperature = 0.1,
            max_tokens  = 1000,
        )
        logger.info("✅ LLM loaded")

    if _llm_stream is None:
        logger.info("Preloading streaming LLM...")
        _llm_stream = ChatGroq(
            api_key     = config.llm.api_key,
            model       = config.llm.model,
            temperature = 0.1,
            max_tokens  = 1000,
            streaming   = True,
        )
        logger.info("✅ Streaming LLM loaded")

    return _vectorstore, _documents, _reranker_model


# =============================================================================
# ANSWER PROMPT
# =============================================================================

# -------------------------------------------------------------------------
# CHANGE: Rewrote prompt to be more conversational and follow-up aware.
#
# WHAT YOU WROTE:
#   - "Determine if question is a follow-up: YES → use chat history"
#   - "Clear and concise (5 sentences)"
#   - Very clinical, robotic tone
#
# WHY IT WAS WRONG:
#   1. The follow-up instruction was too vague — the model didn't
#      understand that "this drug" = the drug from the previous answer.
#   2. "5 sentences" made every answer feel like a bullet-point list.
#   3. No instruction to be friendly/natural — sounded like a textbook.
#
# WHAT WAS CHANGED:
#   1. Explicit follow-up rules: "resolve pronouns like 'it', 'this drug'
#      using chat history" — now understands references.
#   2. Conversational tone: "respond like a knowledgeable friend"
#   3. Flexible length: short for simple questions, longer for complex.
#   4. Greeting-like opening removed — jumps straight to the answer.
# -------------------------------------------------------------------------
ANSWER_PROMPT_TEMPLATE = """
You are a friendly and knowledgeable pharmaceutical assistant.
Respond naturally — like a helpful pharmacist having a conversation, not a textbook.

## STRICT RULES
- Use ONLY the provided context below
- NO external knowledge or assumptions
- NO hallucinated drug names, dosages, or side effects
- If answer not in context → say: "I don't have enough information about that in my documents."
- Never provide medical advice — only factual information
- Never mention source documents, page numbers, or "the context"

## FOLLOW-UP QUESTIONS (CRITICAL)
Look at the chat history carefully:
- If the user says "this drug", "it", "the same one", "that medication" → they mean the drug from the PREVIOUS conversation
- If the user asks a question that only makes sense in context of a previous exchange → use chat history to understand what they mean
- Example: Previous Q: "What is Jardiance?" → Current Q: "What are its side effects?" → "its" = Jardiance
- Always resolve pronouns and references using chat history before answering

## ANSWER STYLE
- Be warm and conversational — not robotic
- Give thorough, detailed answers — aim for 7-10 lines of explanation
- Explain WHY, not just WHAT — help the user understand the reasoning
- Use bullet points when listing multiple items (side effects, dosages, etc)
- After bullet points, add a brief summary or important note
- Don't be overly brief — the user wants to learn, not just get a one-liner

---
{chat_history}

Context:
{context}

Question: {question}

Answer:
""".strip()


# =============================================================================
# FORMAT CHAT HISTORY — Helper
# =============================================================================

def format_chat_history(chat_history: List[Dict]) -> str:
    """
    Formats chat history for prompt.
    Why separate function? Reused in both
    generate_answer and generate_answer_stream ✅
    """
    if not chat_history:
        return ""

    recent = chat_history[-6:]
    lines  = []
    for msg in recent:
        role    = msg.get("role", "")
        content = msg.get("content", "")[:200]
        if role == "user":
            lines.append(f"Previous Question: {content}")
        elif role == "assistant":
            lines.append(f"Previous Answer: {content}")

    return (
        "Previous conversation:\n" + "\n".join(lines)
        if lines else ""
    )


# =============================================================================
# ANSWER GENERATION — Uses Cached LLM
# =============================================================================

def generate_answer(
    question     : str,
    context      : str,
    chat_history : List[Dict] = [],
) -> str:
    """
    Generates answer using cached Groq LLM.

    Fix: Uses global _llm — not creating
    new ChatGroq instance every query ✅
    """
    global _llm
    logger.info("Generating answer...")

    try:
        prompt = PromptTemplate(
            template        = ANSWER_PROMPT_TEMPLATE,
            input_variables = ["context", "question", "chat_history"]
        )

        chain = prompt | _llm

        with Timer("Answer Generation"):
            result = chain.invoke({
                "context"      : context,
                "question"     : question,
                "chat_history" : format_chat_history(chat_history),
            })

        answer = result.content.strip()
        logger.info("✅ Answer generated")
        return answer

    except Exception as e:
        raise RuntimeError(f"❌ Answer generation failed: {str(e)}")


# =============================================================================
# STREAMING ANSWER — Uses Cached Streaming LLM
# =============================================================================

def generate_answer_stream(
    question     : str,
    context      : str,
    chat_history : List[Dict] = [],
) -> Generator:
    """
    Streams answer using cached streaming LLM.

    Fix: Uses global _llm_stream — not creating
    new ChatGroq instance every query ✅
    Optimization 9 — Streaming ✅

    # -------------------------------------------------------------------------
    # CHANGE: Re-raise exception after logging instead of just yielding a string.
    #
    # WHAT YOU WROTE:
    #   except Exception as e:
    #       logger.error(f"Streaming failed: {e}")
    #       yield "Error generating response. Please try again."
    #
    # WHY IT CAUSED "Response ended prematurely":
    #   When Groq throws an error (rate limit, token limit, network timeout),
    #   the yield inside except never actually reached Streamlit because the
    #   generator frame was already in a broken state — LangChain's chain.stream()
    #   raises INSIDE the for-loop, which exits the generator scope abnormally.
    #   Python generators that raise inside yield don't always deliver the final
    #   yield to the consumer. The stream connection closed with no data.
    #
    # WHAT WAS CHANGED:
    #   Re-raise the exception so it propagates to run_query_stream's
    #   generator, which is wrapped in a try/except (in main.py generate())
    #   and yields the error message reliably FROM OUTSIDE the broken frame.
    # -------------------------------------------------------------------------
    """
    global _llm_stream
    logger.info("Streaming answer...")

    try:
        prompt = PromptTemplate(
            template        = ANSWER_PROMPT_TEMPLATE,
            input_variables = ["context", "question", "chat_history"]
        )

        chain = prompt | _llm_stream

        # Same as you wrote — stream chunks from Groq LLM
        for chunk in chain.stream({
            "context"      : context,
            "question"     : question,
            "chat_history" : format_chat_history(chat_history),
        }):
            yield chunk.content

    except Exception as e:
        # CHANGE: log and re-raise instead of yielding inside broken generator
        # The caller (run_query_stream → main.py generate()) catches and yields
        # the error text reliably ✅
        logger.error(f"Streaming failed: {e}")
        raise  # ← CHANGED: was 'yield "Error generating response..."'


# =============================================================================
# PIPELINE STATE REFRESH — Called by Auto-Ingestion File Watcher
# =============================================================================

def refresh_pipeline_state():
    """
    Refreshes vectorstore, documents, and drug dictionary after auto-ingestion.

    This is called by file_watcher.py after new PDFs are added to the
    vectorstore. It ONLY reloads the data-dependent parts:
    - Vectorstore (new chunks available for search)
    - BM25 documents (new chunks for keyword search)
    - Drug dictionary (new drug names for routing)
    - Response cache (clear stale cached answers)

    It does NOT reload:
    - LLM (unchanged — stays cached)
    - Streaming LLM (unchanged — stays cached)
    - Reranker model (unchanged — stays cached)

    This makes the refresh fast (~1 sec) while keeping expensive
    models in memory.
    """
    global _vectorstore, _documents, _drug_dictionary, _response_cache

    logger.info("🔄 Refreshing pipeline state after auto-ingestion...")

    try:
        # Reload vectorstore (picks up new chunks)
        _vectorstore = load_vectorstore()
        logger.info("  ✅ Vectorstore reloaded")

        # Reload BM25 documents
        _documents = _vectorstore.get()
        docs_list = []
        for i, content in enumerate(_documents.get("documents", [])):
            meta = _documents["metadatas"][i] if _documents.get("metadatas") else {}
            docs_list.append(Document(page_content=content, metadata=meta))
        _documents = docs_list
        logger.info(f"  ✅ Documents reloaded: {len(_documents)} chunks")

        # Rebuild drug dictionary (new drug names)
        _drug_dictionary = build_drug_dictionary(_vectorstore)
        logger.info(f"  ✅ Drug dictionary rebuilt: {len(_drug_dictionary)} keywords")

        # Clear response cache (old answers may be incomplete)
        _response_cache.clear()
        logger.info("  ✅ Response cache cleared")

    except Exception as e:
        logger.error(f"Pipeline refresh failed: {e}")
        raise


# =============================================================================
# INGESTION PIPELINE
# =============================================================================

def run_ingestion() -> dict:
    """Runs complete ingestion pipeline."""
    global _vectorstore, _documents, _reranker_model
    global _drug_dictionary, _config_data
    global _response_cache, _llm, _llm_stream

    logger.info("🚀 Starting Ingestion Pipeline")

    with Timer("Full Ingestion"):
        logger.info("Step 1/3: Loading PDFs...")
        documents = load_all_pdfs()

        logger.info("Step 2/3: Semantic chunking...")
        chunks = chunk_documents(documents)

        logger.info("Step 3/3: Embedding + storing...")
        embed_and_store(chunks)

    # Reset all state
    _vectorstore     = None
    _documents       = None
    _reranker_model  = None
    _drug_dictionary = {}
    _config_data     = {}
    _llm             = None
    _llm_stream      = None
    _response_cache.clear()

    summary = {
        "status"        : "success",
        "pages_loaded"  : len(documents),
        "chunks_created": len(chunks),
        "message"       : "Ingestion complete. Ready for queries."
    }

    logger.info(f"✅ Ingestion Complete: {summary}")
    return summary


# =============================================================================
# QUERY PIPELINE — All Optimizations + Guardrails
# =============================================================================

def run_query(
    user_query   : str,
    chat_history : List[Dict] = [],
) -> dict:
    """
    Optimized RAG pipeline with guardrails.

    Order of checks (fastest first):
    1. Validate query
    2. Cache check         → instant ✅
    3. Small talk check    → instant ✅
    4. Out-of-domain check → instant ✅
    5. RAG pipeline        → full processing
    """
    logger.info(f"🔍 Query: '{user_query}'")

    # Step 1: Validate
    user_query = validate_query(user_query)

    # CHANGE: Resolve follow-up references BEFORE cache/search
    # "this drug" / "it" → actual drug name from chat history
    user_query = resolve_followup_query(user_query, chat_history)

    # Step 2: Cache check FIRST
    # Most common optimization — check before anything else
    cached = get_cached_response(user_query)
    if cached:
        logger.info("✅ Cache HIT")
        return cached

    # Step 3: Guardrail — Small talk
    if is_small_talk(user_query):
        logger.info("✅ Small talk")
        return {
            "answer"          : get_small_talk_response(user_query),
            "rewritten_query" : user_query,
            "sub_queries"     : [user_query],
            "chunks_used"     : 0,
            "sources"         : [],
        }

    # Step 4: Guardrail — Out-of-domain
    if is_out_of_domain(user_query):
        logger.info("⚠️ Out-of-domain")
        return {
            "answer"          : get_out_of_domain_response(),
            "rewritten_query" : user_query,
            "sub_queries"     : [user_query],
            "chunks_used"     : 0,
            "sources"         : [],
        }

    # Step 5: Check vectorstore
    if not is_vectorstore_ready():
        raise ValueError("❌ Vectorstore not ready.")

    # Step 6: Get preloaded models
    vectorstore, documents, reranker = get_pipeline_state()

    with Timer("RAG Pipeline"):

        # Rule based routing — zero LLM ✅
        route = route_query(user_query)

        # Hybrid search with metadata filter
        # Optimization 5 — filter before search ✅
        logger.info("Step 1/3: Hybrid search...")
        retrieved_chunks = hybrid_search(
            sub_queries = [user_query],
            documents   = documents,
            vectorstore = vectorstore,
            filter_dict = route["filter"],
        )

        # TinyBERT reranking with cached model
        # Optimization 1 — TinyBERT ✅
        # Optimization 7 — cached reranker ✅
        logger.info("Step 2/3: Reranking...")
        reranked_chunks = rerank_documents(
            query     = user_query,
            documents = retrieved_chunks,
            # CHANGE: Pass the pre-loaded reranker from get_pipeline_state()
            # WHAT YOU WROTE:
            #   rerank_documents(query=user_query, documents=retrieved_chunks)
            #   ← No model passed, so rerank_documents loaded a new one every query
            # WHAT WAS CHANGED:
            #   Pass global _reranker_model → reranker.py skips load_reranker_model()
            #   → Optimization 7 (model caching) now actually works ✅
            model     = reranker,
        )

        # -----------------------------------------------------------------
        # CHANGE: Smart out-of-domain detection via reranker scores.
        #
        # WHAT YOU WROTE:
        #   Always proceeded to generate_answer even with irrelevant chunks.
        #
        # THE SMART FIX:
        #   If reranker returns empty list, ALL chunks scored below threshold.
        #   This means the query has nothing to do with drugs.
        #   No keyword list can cover every possible off-topic question
        #   (sports, cooking, politics, math, etc.).
        #   But the reranker CAN — it compares the query against actual
        #   drug content and says "none of this is relevant."
        #   The reranker IS your universal out-of-domain detector. ✅
        # -----------------------------------------------------------------
        if not reranked_chunks:
            # Check if query contains a known drug name
            found_drug = None
            query_lower = user_query.lower()
            global _drug_dictionary
            if _drug_dictionary:
                for keyword in _drug_dictionary:
                    if keyword in query_lower:
                        found_drug = keyword
                        break
            if found_drug:
                # Fallback: retrieve at least one chunk for the drug
                logger.info(f"⚠️ Reranker returned 0 chunks, but found drug '{found_drug}' in query — forcing context")
                # Find any chunk mentioning the drug
                fallback_chunks = [doc for doc in documents if found_drug in doc.metadata.get('drug_name', '').lower()]
                if not fallback_chunks:
                    fallback_chunks = documents[:1]  # fallback to first chunk if none found
                reranked_chunks = fallback_chunks
            else:
                logger.info("⚠️ Reranker returned 0 chunks — out-of-domain")
                return {
                    "answer"          : get_out_of_domain_response(),
                    "rewritten_query" : user_query,
                    "sub_queries"     : [user_query],
                    "chunks_used"     : 0,
                    "sources"         : [],
                }

        # Generate answer with cached LLM
        # Optimization 7 — cached LLM ✅
        logger.info("Step 3/3: Answer...")
        context = format_documents(reranked_chunks)
        answer  = generate_answer(
            question     = user_query,
            context      = context,
            chat_history = chat_history,
        )

    sources = list(set([
        f"{doc.metadata.get('drug_name', 'Unknown')} "
        f"(page {doc.metadata.get('page', '?')})"
        for doc in reranked_chunks
    ]))

    response = {
        "answer"          : answer,
        "rewritten_query" : user_query,
        "sub_queries"     : [user_query],
        "chunks_used"     : len(reranked_chunks),
        "sources"         : sources,
    }

    # Cache for next time
    cache_response(user_query, response)

    logger.info(f"✅ Done | sources: {sources}")
    return response


# =============================================================================
# STREAMING QUERY PIPELINE — Optimization 9
# =============================================================================

def run_query_stream(
    user_query   : str,
    chat_history : List[Dict] = [],
) -> Generator:
    """
    Streaming RAG pipeline with guardrails.

    Same order as run_query — fastest checks first.
    Streams answer word by word ✅
    """
    user_query = validate_query(user_query)

    # CHANGE: Resolve follow-up references BEFORE guardrails/search
    user_query = resolve_followup_query(user_query, chat_history)

    # Guardrail: Small talk → instant stream
    if is_small_talk(user_query):
        yield get_small_talk_response(user_query)
        return

    # Guardrail: Out-of-domain → instant stream
    if is_out_of_domain(user_query):
        yield get_out_of_domain_response()
        return

    # Check vectorstore
    if not is_vectorstore_ready():
        yield "❌ Vectorstore not ready. Please run ingestion first."
        return

    # Get preloaded models
    vectorstore, documents, reranker = get_pipeline_state()

    # Route
    route = route_query(user_query)

    # Hybrid search
    retrieved_chunks = hybrid_search(
        sub_queries = [user_query],
        documents   = documents,
        vectorstore = vectorstore,
        filter_dict = route["filter"],
    )

    # Rerank with cached model
    reranked_chunks = rerank_documents(
        query     = user_query,
        documents = retrieved_chunks,
        # CHANGE: Same fix as run_query above—pass cached reranker
        # WHAT YOU WROTE:
        #   rerank_documents(query=user_query, documents=retrieved_chunks)
        # WHAT WAS CHANGED:
        #   Pass 'reranker' so the model isn't reloaded on every stream request
        model     = reranker,
    )

    # Same smart out-of-domain check as run_query (see comment there)
    # If reranker says "nothing is relevant" → don't waste an LLM call
    if not reranked_chunks:
        # Check if query contains a known drug name
        found_drug = None
        query_lower = user_query.lower()
        global _drug_dictionary
        if _drug_dictionary:
            for keyword in _drug_dictionary:
                if keyword in query_lower:
                    found_drug = keyword
                    break
        if found_drug:
            logger.info(f"⚠️ Reranker returned 0 chunks, but found drug '{found_drug}' in query — forcing context (stream)")
            fallback_chunks = [doc for doc in documents if found_drug in doc.metadata.get('drug_name', '').lower()]
            if not fallback_chunks:
                fallback_chunks = documents[:1]
            reranked_chunks = fallback_chunks
        else:
            logger.info("⚠️ Reranker returned 0 chunks — out-of-domain (stream)")
            yield get_out_of_domain_response()
            return

    # Format context
    context = format_documents(reranked_chunks)

    # Stream answer
    full_answer = ""
    for chunk in generate_answer_stream(
        question     = user_query,
        context      = context,
        chat_history = chat_history,
    ):
        full_answer += chunk
        yield chunk

    # Cache after streaming completes
    sources = list(set([
        f"{doc.metadata.get('drug_name', 'Unknown')} "
        f"(page {doc.metadata.get('page', '?')})"
        for doc in reranked_chunks
    ]))

    # CHANGE: Yield sources as a special JSON marker at end of stream.
    #
    # WHAT YOU WROTE:
    #   sources were computed but only cached — never sent to frontend.
    #   Frontend had sources = [] hardcoded.
    #
    # WHY IT WAS WRONG:
    #   User never saw which drug/page the answer came from.
    #
    # WHAT WAS CHANGED:
    #   After all answer text is streamed, yield a special separator
    #   followed by JSON with sources. Frontend splits on this marker
    #   to extract sources and show them in a dropdown.
    if sources:
        yield f"\n\n__SOURCES__{json.dumps(sources)}"

    cache_response(user_query, {
        "answer"          : full_answer,
        "rewritten_query" : user_query,
        "sub_queries"     : [user_query],
        "chunks_used"     : len(reranked_chunks),
        "sources"         : sources,
    })
