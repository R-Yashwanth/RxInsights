# 💊 Pharma Drug Intelligence Assistant

> A production-grade Advanced RAG system for pharmaceutical drug information, built with LangChain, Groq, HuggingFace, and Chroma.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![LangChain](https://img.shields.io/badge/LangChain-1.x-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135-teal)
![Streamlit](https://img.shields.io/badge/Streamlit-1.55-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Overview

The **Pharma Drug Intelligence Assistant** is an AI-powered Q&A system that allows medical professionals, researchers, and students to query detailed pharmaceutical information from FDA-approved drug labels (DailyMed PDFs).

Ask natural language questions about:
- Drug indications and usage
- Side effects and adverse reactions
- Dosage and administration
- Contraindications and warnings
- Drug interactions
- Clinical trial outcomes

---

## 🏗️ Architecture
User Query
↓
Guardrails (Small talk / Out-of-domain)
↓
Response Cache (instant for repeated queries)
↓
Rule Based Query Router (zero LLM call)
↓
Hybrid Search — BM25 + Semantic (with metadata filtering)
↓
TinyBERT Cross-Encoder Reranking
↓
Groq LLaMA 3.3 70B → Streaming Answer

---

## ✨ Features

### Advanced RAG Pipeline
- **Semantic Chunking** — splits PDFs by meaning shift not character count
- **Hybrid Search** — BM25 keyword + semantic vector search combined
- **Cross-Encoder Reranking** — TinyBERT scores each chunk for precision
- **Metadata Filtering** — searches only relevant drug chunks
- **Streaming Responses** — ChatGPT-like word by word output

### Production Optimizations
- **Model Preloading** — all models loaded once at startup
- **Response Caching** — repeated queries return instantly
- **Rule Based Router** — zero LLM call for query analysis
- **Dynamic Drug Dictionary** — built from Chroma metadata, no hardcoding
- **Guardrails** — small talk detection, out-of-domain rejection

### Smart UI
- Dynamic FAQs generated from ingested drug data
- Chat history for follow-up questions
- Source citations with drug name and page number
- Knowledge base status indicator

---

## 🗂️ Project Structure

pharma-rag/
│
├── backend/
│   ├── main.py                     ← FastAPI entry point
│   ├── requirements.txt            ← dependencies
│   ├── drug_config.json            ← comparison keywords config
│   │
│   ├── data/
│   │   └── pdfs/                   ← drug PDF files (DailyMed)
│   │
│   ├── ingestion/
│   │   ├── pdf_loader.py           ← PyMuPDF loading + cleaning
│   │   ├── chunker.py              ← SemanticChunker
│   │   └── embedder.py             ← HuggingFace + Chroma storage
│   │
│   ├── retrieval/
│   │   ├── hybrid_search.py        ← BM25 + Semantic + metadata filter
│   │   ├── reranker.py             ← TinyBERT cross-encoder
│   │   └── compressor.py           ← context compression (optional)
│   │
│   ├── pipeline/
│   │   └── rag_pipeline.py         ← full pipeline orchestration
│   │
│   └── utils/
│       ├── config.py               ← environment config
│       ├── logger.py               ← production logging
│       └── helpers.py              ← shared utilities
│
├── frontend/
│   └── app.py                      ← Streamlit UI
│
├── vectorstore/
│   └── chroma_db/                  ← persisted vector store
│
├── .env                            ← API keys (never commit)
├── .gitignore
└── README.md

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Groq API key (free at console.groq.com)

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/pharma-rag.git
cd pharma-rag
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 4. Configure Environment
Create `backend/.env`:
```bash
# LLM
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# Embeddings
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Chroma
CHROMA_PERSIST_DIR=./vectorstore/chroma_db
COLLECTION_NAME=pharma_rag

# Chunking
BREAKPOINT_THRESHOLD_TYPE=percentile
BREAKPOINT_THRESHOLD_AMOUNT=85

# Retrieval
TOP_K=10
SCORE_THRESHOLD=0.70
FINAL_TOP_K=3
BM25_WEIGHT=0.4
SEMANTIC_WEIGHT=0.6

# Reranker
RERANKER_MODEL=cross-encoder/ms-marco-TinyBERT-L-2-v2

# API
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=True

# Data
PDF_DIR=./data
```

### 5. Add Drug PDFs
Download drug label PDFs from [DailyMed](https://dailymed.nlm.nih.gov) and place in `backend/data/`:

backend/data/
├── jardiance.pdf
├── ozempic.pdf
├── farxiga.pdf
└── ... (any drug PDFs)

### 6. Run Ingestion
```bash
cd backend
python -c "from pipeline.rag_pipeline import run_ingestion; run_ingestion()"
```

### 7. Start Backend
```bash
cd backend
python main.py
```
API runs at `http://localhost:8000`
Docs at `http://localhost:8000/docs`

### 8. Start Frontend
Open new terminal:
```bash
cd pharma-rag
streamlit run frontend/app.py
```
UI runs at `http://localhost:8501`

---

## 📊 Tech Stack

| Component | Technology | Why |
|---|---|---|
| **LLM** | Groq + LLaMA 3.3 70B | Free, fast, 128K context |
| **Embeddings** | HuggingFace all-MiniLM-L6-v2 | Free, local, production grade |
| **Vector Store** | Chroma | Local, persistent, no server |
| **Reranker** | TinyBERT cross-encoder | 5x faster than MiniLM-L6 |
| **PDF Parsing** | PyMuPDF | Best for complex drug label layouts |
| **Chunking** | LangChain SemanticChunker | Splits by meaning not character count |
| **BM25** | LangChain BM25Retriever | Exact medical term matching |
| **Backend** | FastAPI | Production grade, async, auto docs |
| **Frontend** | Streamlit | Fast UI, free cloud deployment |

---

## 🔬 RAG Techniques Used

### Pre-Retrieval
| Technique | Implementation |
|---|---|
| Semantic Chunking | LangChain SemanticChunker with percentile breakpoints |
| Drug Metadata Extraction | PyMuPDF content parsing for brand + generic names |

### Retrieval
| Technique | Implementation |
|---|---|
| Hybrid Search | LangChain EnsembleRetriever (BM25 40% + Semantic 60%) |
| Metadata Filtering | Chroma filter by drug_name before search |
| Rule Based Routing | String matching — zero LLM call |

### Post-Retrieval
| Technique | Implementation |
|---|---|
| Cross-Encoder Reranking | TinyBERT ms-marco model |
| Score Thresholding | Drop chunks below 0.70 relevance |
| Context Formatting | Source-tagged chunks for LLM |

---

## ⚡ Performance

| Metric | Value |
|---|---|
| First query (cold) | ~15 sec (model loading) |
| Subsequent queries | 5-8 sec |
| Cached queries | < 0.1 sec |
| Ingestion (10 PDFs) | ~20 min (one time) |

---

## 🛡️ Guardrails

| Guardrail | Behavior |
|---|---|
| Small talk | Instant friendly response, no RAG |
| Out-of-domain | Polite decline, no RAG |
| Context grounding | LLM restricted to provided context only |
| Hallucination prevention | Strict prompt boundaries |

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/status` | Vectorstore readiness |
| GET | `/drugs` | Available drugs list |
| POST | `/ingest` | Run ingestion pipeline |
| POST | `/query` | Run RAG query |
| POST | `/query/stream` | Streaming RAG query |

---

## 📈 Why This Architecture

### Design Decisions

**Removed Query Rewriting:**
LLaMA 3.3 70B natively understands medical terminology. Query rewriting added 3-5 seconds per query without meaningful quality improvement.

**Removed Compressor:**
SemanticChunker produces focused chunks. TinyBERT reranking filters irrelevant content effectively. Compression added 9-15 seconds (3 LLM calls) with marginal benefit.

**TinyBERT over MiniLM-L6:**
2 transformer layers vs 6 layers. 5x faster on CPU with minimal accuracy loss for short medical passages. Saves 15 seconds per query.

**Dynamic Drug Dictionary:**
Brand and generic names extracted from PDF content during ingestion and stored in Chroma metadata. Zero hardcoding — add new drugs by simply adding PDFs and re-ingesting.s