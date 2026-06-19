# AI Knowledge Assistant

A production-grade Retrieval-Augmented Generation (RAG) platform that transforms your documents into an interactive knowledge base. Built for high-performance semantic search, context-aware conversations, and automated quality evaluation via RAGAS.

---

## Key Features

- **Conversational Memory** — Remembers previous questions and automatically reformulates follow-up queries using conversation history.
- **Advanced RAG Pipeline** — State-of-the-art chunking, BGE embeddings, FAISS vector search, and CrossEncoder reranking for precision retrieval.
- **Incremental Indexing** — Uploading a new document only embeds that document's chunks, not the entire corpus.
- **FAISS Persistence** — Vector index is saved to disk after every change; restarts load from cache in ~1s instead of re-embedding from scratch.
- **RAGAS Evaluation** — Automated quality scoring (Faithfulness, Answer Relevancy, Context Precision, Context Recall) runs as a non-blocking background task after every query and via an offline batch evaluation script.
- **Cloud Storage** — Syncs with AWS S3 at startup; uploads and deletes propagate to S3 automatically.
- **Structured Logging** — All services emit JSON-structured logs with timestamps and correlation IDs, compatible with any log aggregator.
- **Production Security** — Optional API-key authentication, restricted CORS origins, PDF magic-byte validation, and file-size limits.
- **Health Probes** — `/health` and `/ready` endpoints for load-balancer integration.
- **Modern UI** — React + Vite dashboard with Chat and Document Management tabs, session management, and file-scoped querying.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | [Groq](https://groq.com/) — Llama 3.1 8B Instant (configurable) |
| Embeddings | `BAAI/bge-base-en-v1.5` via Sentence Transformers (local) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local) |
| Vector DB | FAISS `IndexFlatL2` (persisted to disk) |
| RAG Evaluation | RAGAS 0.2.x with Groq as judge LLM |
| Backend | Python 3.14, FastAPI, Uvicorn |
| Frontend | React 18, Vite, Lucide React |
| Storage | AWS S3, Boto3 |

---

## Architecture

```
User Question
     │
     ▼
[memory.py]  ──── get chat history ────►  [llm_service.py]
                                           reformulate_query()
                                                │
                                                ▼
                                        [retrieval.py]
                                     FAISS semantic search
                                      + keyword fallback
                                                │
                                                ▼
                                        [reranker.py]
                                     CrossEncoder scoring
                                                │
                                                ▼
                                        [llm_service.py]
                                       generate_rag_answer()
                                          (Groq / Llama)
                                                │
                          ┌─────────────────────┴──────────────────────┐
                          ▼                                             ▼
                  Response to user                     [ragas_evaluator.py]
                                                  Background: Faithfulness +
                                                  Answer Relevancy scored and
                                                  logged to logs/ragas_eval_log.jsonl
```

---

## Setup

### 1. Environment variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Required variables:

```env
GROQ_API_KEY=gsk_...
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
S3_BUCKET_NAME=your-bucket-name
```

Optional variables with defaults:

```env
LLM_MODEL=llama-3.1-8b-instant   # Groq model name
LLM_MAX_TOKENS=800                 # Max tokens in generated answers
VECTOR_DB_PATH=./vector_store      # Where FAISS index is saved

API_KEY=                           # Set to enable X-API-Key auth on all endpoints
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000
UPLOAD_MAX_MB=50

AWS_REGION=us-east-1
S3_PREFIX=documents/
```

### 2. Backend

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

API docs: `http://localhost:8000/docs`

### 3. Frontend

```bash
cd ui
npm install
npm run dev
```

UI: `http://localhost:5173`

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe — always 200 |
| `GET` | `/ready` | Readiness probe — 200 once RAG is initialised |
| `GET` | `/ask` | Main RAG query (params: `question`, `session_id`, `files`) |
| `GET` | `/ask-anything` | Direct LLM query, no retrieval |
| `GET` | `/documents` | List all indexed documents |
| `POST` | `/upload` | Upload a PDF (validates size + magic bytes) |
| `DELETE` | `/documents/{filename}` | Delete a document |
| `POST` | `/evaluate` | Trigger RAGAS batch evaluation (background) |
| `GET` | `/eval-summary` | Average RAGAS scores from the evaluation log |

If `API_KEY` is set in `.env`, all endpoints except `/health`, `/ready`, and `/ping` require the header:
```
X-API-Key: <your-api-key>
```

---

## RAGAS Evaluation

RAGAS automatically measures RAG pipeline quality without needing human graders.

### Metrics

| Metric | What it measures | Ground truth needed |
|---|---|---|
| **Faithfulness** | Are all claims in the answer supported by retrieved context? Detects hallucination. | No |
| **Answer Relevancy** | Does the answer address what was actually asked? | No |
| **Context Precision** | Are the retrieved chunks useful? (retrieval signal-to-noise) | Yes |
| **Context Recall** | Did retrieval find all the information needed to answer? | Yes |

### Online evaluation (automatic, per-request)

After every `/ask` response is sent to the user, FastAPI runs Faithfulness and Answer Relevancy as a background task — the user sees no added latency. Scores are appended to `logs/ragas_eval_log.jsonl`.

### Offline batch evaluation

```bash
# Run against the curated test set in data/eval_testset.json
python scripts/run_evaluation.py

# Fail CI if scores fall below thresholds
python scripts/run_evaluation.py --min-faithfulness 0.7 --min-relevancy 0.75
```

### Customise the test set

Edit `data/eval_testset.json`. Each entry:

```json
{
  "question": "What is the monthly fee in the service agreement?",
  "ground_truth": "The monthly fee is INR 150,000 per resource introduced...",
  "reference_contexts": [
    "The Client shall pay the Contractor INR 150,000 per Resource introduced..."
  ]
}
```

The file ships with 20 questions pre-written from the project's sample documents.

---

## Project Structure

```
ai-knowledge-assistant/
├── app/
│   ├── main.py                    # FastAPI app — all endpoints, middleware
│   ├── config.py                  # Env var loading and validation
│   └── services/
│       ├── logger.py              # JSON-structured logging
│       ├── rag_pipeline.py        # FAISS init, persistence, incremental index
│       ├── ragas_evaluator.py     # RAGAS metrics — online + offline evaluation
│       ├── llm_service.py         # Groq LLM — reformulation + generation
│       ├── memory.py              # Thread-safe in-memory chat sessions
│       ├── retrieval.py           # FAISS search + keyword fallback
│       ├── reranker.py            # CrossEncoder reranking
│       ├── embedding.py           # Sentence Transformer embeddings
│       ├── chunking.py            # LangChain text splitter
│       ├── ingestion.py           # pdfplumber PDF text extraction
│       ├── document_loader.py     # Batch load all PDFs from folder
│       └── s3_storage.py          # AWS S3 upload / delete / sync
├── ui/                            # React + Vite frontend
│   └── src/
│       ├── App.jsx
│       └── components/
│           ├── ChatTab.jsx
│           └── DocumentsTab.jsx
├── data/
│   └── eval_testset.json          # 20 curated Q&A pairs for RAGAS offline eval
├── scripts/
│   └── run_evaluation.py          # CLI batch evaluation script
├── logs/                          # RAGAS evaluation log output (gitignored)
├── vector_store/                  # Persisted FAISS index + chunk metadata (gitignored)
├── documents/                     # Local PDF cache synced from S3 (gitignored)
├── .env.example                   # Template for required environment variables
└── requirements.txt
```

---

## Author

**Neerali Acharya**
