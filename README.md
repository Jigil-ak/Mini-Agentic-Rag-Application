# 🤖 Mini Agentic RAG Application

A production-ready **Retrieval-Augmented Generation** application with intelligent query routing, LLM fallback resilience, tool use (calculator + web search), and structured tracing.

## ✨ Key Features

- **Intelligent Routing** — Automatically decides between knowledge base (RAG) and tools (web search / calculator) based on retrieval similarity
- **Dual LLM with Fallback** — Primary: Google Gemini 2.0 Flash, Fallback: Groq Llama 3.1 8B. Seamless failover if primary is unavailable
- **Multi-Source Ingestion** — Upload PDFs or scrape URLs into the vector knowledge base
- **Safe Calculator** — AST-based math evaluator (no `eval()`) for arithmetic queries
- **Web Search** — DuckDuckGo-powered search for queries outside the knowledge base
- **Vector Search** — Milvus Lite with cosine similarity, no Docker required
- **Structured Tracing** — Every query produces a JSONL trace with routing decisions, latency, and model info
- **Modern UI** — Streamlit frontend with real-time trace visualisation

## 🏗️ Architecture

```mermaid
flowchart TD
    A["User Query"] --> B["FastAPI Backend"]
    B --> C["RouterAgent"]
    C --> D["Embed Query<br/>(all-MiniLM-L6-v2)"]
    D --> E["Milvus Search<br/>(Cosine Similarity)"]
    E --> F{similarity >= 0.6?}

    F -->|Yes| G["RAG Agent"]
    G --> H["Retrieve Top-K Chunks"]
    H --> I["Build Context Prompt"]

    F -->|No| J["Tool Agent"]
    J --> K{Calculator or Search?}
    K -->|Math Pattern| L["Calculator<br/>(ast evaluator)"]
    K -->|General| M["Web Search<br/>(DuckDuckGo)"]
    L --> N["Build Tool Prompt"]
    M --> N

    I --> O["Gemini 2.0 Flash"]
    N --> O
    O -->|Fails| P["Groq Llama 3.1"]
    O -->|Success| Q["TraceManager"]
    P --> Q

    Q --> R["JSONL Log"]
    Q --> S["API Response"]
    S --> T["Streamlit UI"]
```

## 📋 Prerequisites

- **Python** 3.11 or higher
- **pip** (latest version recommended)
- **API Keys:**
  - [Google Gemini API Key](https://aistudio.google.com/app/apikey) (free tier available)
  - [Groq API Key](https://console.groq.com/keys) (free tier available)

## 🚀 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Jigil-ak/Mini-Agentic-Rag-Application.git
cd Mini-Agentic-Rag-Application
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
GEMINI_API_KEY=your_actual_gemini_api_key
GROQ_API_KEY=your_actual_groq_api_key
```

### 5. Start the backend

```bash
uvicorn backend.main:app --reload
```

The API will be available at `http://localhost:8000`

### 6. Start the frontend (in a new terminal)

```bash
streamlit run frontend/app.py
```

The UI will open at `http://localhost:8501`

## ⚙️ Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | ✅ Yes | — |
| `GROQ_API_KEY` | Groq API key for fallback LLM | ✅ Yes | — |
| `SIMILARITY_THRESHOLD` | RAG vs Tool routing threshold (0.0–1.0) | No | `0.6` |
| `CHUNK_SIZE` | Text chunk size in characters | No | `500` |
| `CHUNK_OVERLAP` | Overlap between consecutive chunks | No | `50` |
| `TOP_K` | Number of chunks to retrieve | No | `3` |
| `EMBEDDING_MODEL` | Sentence transformer model name | No | `all-MiniLM-L6-v2` |
| `PRIMARY_MODEL` | Primary LLM model | No | `gemini-2.0-flash` |
| `FALLBACK_MODEL` | Fallback LLM model | No | `llama-3.1-8b-instant` |
| `MILVUS_DB_PATH` | Path to Milvus Lite database file | No | `./data/milvus_demo.db` |
| `COLLECTION_NAME` | Milvus collection name | No | `rag_documents` |
| `LOG_FILE` | Path to JSONL trace log file | No | `./logs/agent_logs.jsonl` |
| `FASTAPI_PORT` | FastAPI server port | No | `8000` |

## 📡 API Reference

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "ok",
  "milvus": "connected",
  "documents_indexed": 42
}
```

### Load a PDF

```bash
curl -X POST http://localhost:8000/load \
  -F "file=@document.pdf"
```

Response:
```json
{
  "status": "success",
  "chunks_loaded": 15,
  "source": "document.pdf"
}
```

### Load a URL

```bash
curl -X POST http://localhost:8000/load \
  -F "url=https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
```

Response:
```json
{
  "status": "success",
  "chunks_loaded": 23,
  "source": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
}
```

### Ask a Question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is retrieval-augmented generation?"}'
```

Response:
```json
{
  "answer": "Retrieval-augmented generation (RAG) is...",
  "trace": {
    "trace_id": "a1b2c3d4-...",
    "query": "What is retrieval-augmented generation?",
    "path_taken": "rag",
    "similarity_score": 0.82,
    "response_time_ms": 1250.5,
    "...": "..."
  }
}
```

### Get Recent Traces

```bash
curl http://localhost:8000/traces
```

## 🔄 Agent Workflow

1. **User submits a query** via the Streamlit UI or API
2. **TraceManager** starts a new trace record with a unique ID
3. **RouterAgent** embeds the query using `all-MiniLM-L6-v2`
4. **Milvus search** finds the most similar document chunk (cosine similarity)
5. **Routing decision:**
   - Similarity ≥ 0.6 → **RAG path**: retrieve top-K chunks, build context prompt
   - Similarity < 0.6 → **Tool path**: detect calculator vs web search intent
6. **LLM generation** with fallback: try Gemini first, fall back to Groq on failure
7. **TraceManager** records the full trace (path, model, latency, scores) to JSONL
8. **Response** is returned with the answer and complete trace metadata

## 🔀 Fallback Logic

```mermaid
flowchart TD
    A["Prompt"] --> B["Gemini 2.0 Flash"]
    B -->|Success + Non-Empty| C["Return Response<br/>fallback_triggered: false"]
    B -->|Exception| D["Log Warning"]
    B -->|Empty/None/Whitespace| D
    D --> E["Groq Llama 3.1 8B"]
    E -->|Success + Non-Empty| F["Return Response<br/>fallback_triggered: true"]
    E -->|Exception| G["Raise RuntimeError<br/>'Both LLMs failed'"]
    E -->|Empty| G
```

## 📊 Trace Log Format

Each query appends one JSON line to `logs/agent_logs.jsonl`:

```json
{
  "trace_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "query": "What is machine learning?",
  "timestamp": "2024-01-15T10:30:00.123456+00:00",
  "retrieval_hit": true,
  "similarity_score": 0.847,
  "path_taken": "rag",
  "routing_reason": "Found relevant context (similarity: 0.847 >= threshold: 0.6)",
  "tool_used": null,
  "tool_output_preview": null,
  "primary_model": "gemini-2.0-flash",
  "fallback_triggered": false,
  "chunks_used": 3,
  "response_time_ms": 1423.56,
  "error": null
}
```

| Field | Type | Description |
|-------|------|-------------|
| `trace_id` | string | Unique UUID for this query |
| `query` | string | The user's original question |
| `timestamp` | string | ISO 8601 timestamp (UTC) |
| `retrieval_hit` | bool | Whether the RAG path was taken |
| `similarity_score` | float | Normalised cosine similarity (0.0–1.0) |
| `path_taken` | string | `"rag"` or `"tool"` |
| `routing_reason` | string | Human-readable routing explanation |
| `tool_used` | string\|null | `"calculator"`, `"web_search"`, or null |
| `tool_output_preview` | string\|null | First 200 chars of tool output |
| `primary_model` | string | LLM model that generated the response |
| `fallback_triggered` | bool | Whether the fallback LLM was used |
| `chunks_used` | int\|null | Number of RAG chunks used (null for tool path) |
| `response_time_ms` | float | Total query processing time in milliseconds |
| `error` | string\|null | Error message if the query failed |

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Expected output:
```
tests/test_calculator.py::TestBasicArithmetic::test_addition PASSED
tests/test_calculator.py::TestBasicArithmetic::test_subtraction PASSED
tests/test_calculator.py::TestSecurityPrevention::test_import_blocked PASSED
tests/test_router.py::TestRAGRouting::test_high_similarity_routes_to_rag PASSED
tests/test_router.py::TestToolRouting::test_low_similarity_routes_to_tool PASSED
tests/test_fallback.py::TestPrimarySuccess::test_primary_succeeds_no_fallback PASSED
tests/test_fallback.py::TestFallbackTriggered::test_primary_exception_triggers_fallback PASSED
tests/test_url_loader.py::TestSuccessfulExtraction::test_valid_html_returns_text PASSED
...
```

## 📁 Project Structure

```
project/
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py          # FastAPI endpoints
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── router.py          # Query routing (RAG vs Tool)
│   │   ├── rag_agent.py       # RAG pipeline
│   │   ├── tool_agent.py      # Tool selection + execution
│   │   └── fallback.py        # LLM fallback orchestration
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pdf_loader.py      # PDF text extraction
│   │   ├── url_loader.py      # URL scraping
│   │   ├── chunking.py        # Text splitting
│   │   └── embedding.py       # Sentence transformer embeddings
│   ├── vectordb/
│   │   ├── __init__.py
│   │   └── milvus_client.py   # Milvus Lite vector store
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── primary_llm.py     # Gemini 2.0 Flash
│   │   └── fallback_llm.py    # Groq Llama 3.1 8B
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── calculator.py      # Safe math evaluator
│   │   └── web_search.py      # DuckDuckGo search
│   ├── tracing/
│   │   ├── __init__.py
│   │   └── tracer.py          # JSONL trace logger
│   ├── __init__.py
│   ├── config.py              # Central configuration
│   └── main.py                # FastAPI app entry point
├── frontend/
│   └── app.py                 # Streamlit UI
├── tests/
│   ├── __init__.py
│   ├── test_calculator.py
│   ├── test_router.py
│   ├── test_fallback.py
│   └── test_url_loader.py
├── data/                      # Milvus Lite database (auto-created)
├── logs/                      # Trace logs (auto-created)
├── requirements.txt
├── .env.example
└── README.md
```

## 📄 License

This project is for educational and demonstration purposes.
