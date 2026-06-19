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
| RAG Evaluation | RAGAS 0.2.14, LangChain-Groq, LangChain-HuggingFace |
| Backend | Python 3.14, FastAPI, Uvicorn |
| Frontend | React 18, Vite, Lucide React |
| Storage | AWS S3, Boto3 |
| Testing | pytest 9, anyio, 98 tests (unit / integration / API) |

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
                                      top-10 candidates
                                                │
                                                ▼
                                        [reranker.py]
                                  CrossEncoder scoring → top-5
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

RAGAS automatically measures RAG pipeline quality across four metrics without needing human graders. Each metric targets a different failure mode in the pipeline.

### Metrics

| Metric | What it measures | Targets | Ground truth needed |
|---|---|---|---|
| **Faithfulness** | Are all claims in the answer supported by the retrieved context? Detects hallucination. | LLM generation | No |
| **Answer Relevancy** | Does the answer fully address what was asked? Penalises incomplete or off-topic answers. | LLM generation | No |
| **Context Precision** | Of the chunks retrieved, how many were actually useful? (signal-to-noise ratio) | Retrieval | Yes |
| **Context Recall** | Did retrieval surface all the information needed to answer correctly? | Retrieval | Yes |

### Current benchmark scores (88 samples)

| Metric | Score |
|---|---|
| Faithfulness | **0.8790** |
| Answer Relevancy | **0.7585** |
| Context Precision | **0.8121** |
| Context Recall | **0.7882** |

### How scores improved through iteration

The initial pipeline scored Answer Relevancy 0.68 and Context Recall 0.71. Two changes closed the gap:

**1. Removed keyword pre-filter before reranking (Context Recall: 0.71 → 0.79)**

The original `retrieval.py` fetched 10 candidates from FAISS but then filtered down to 3 using keyword overlap before passing to the Cross-Encoder reranker. This discarded semantically relevant chunks that used synonyms or paraphrasing — exactly the cases where embedding search excels and keyword matching fails. The fix: pass all 10 FAISS results to the reranker and let it score them properly.

```python
# Before — keyword filter discards paraphrased-but-relevant chunks
if keyword_matches:
    return keyword_matches[:3]
return retrieved_chunks[:3]

# After — reranker sees the full FAISS candidate pool
return retrieved_chunks
```

**2. Removed 1–2 sentence answer constraint (Answer Relevancy: 0.68 → 0.76)**

The prompt previously enforced `Return a complete and meaningful 1–2 sentence answer.` RAGAS measures Answer Relevancy by generating synthetic questions from the answer and comparing them to the original question. A 2-sentence answer to a multi-part question produces narrow synthetic questions that diverge from the original — low score by design. Removing the length cap and requiring the LLM to address every aspect of the question resolved this.

**3. Increased reranker output from top-3 to top-5**

More verified-relevant chunks in the LLM prompt means more of the ground truth information is present, which directly raises Context Recall without adding noise (the Cross-Encoder scores each candidate).

### How RAGAS is wired in

```
Metrics that need ground truth (Context Precision, Context Recall)
└── Offline only → scripts/run_evaluation.py → data/eval_testset.json

Metrics that don't need ground truth (Faithfulness, Answer Relevancy)
└── Online: background task after every /ask request → logs/ragas_eval_log.jsonl
└── Offline: same script, same test set
```

### Online evaluation (automatic, per-request)

After every `/ask` response is returned to the user, FastAPI schedules Faithfulness and Answer Relevancy as a `BackgroundTask` — zero added latency to the user. Scores are appended to `logs/ragas_eval_log.jsonl`.

### Offline batch evaluation

```bash
# Run against the curated test set in data/eval_testset.json
python scripts/run_evaluation.py

# Fail CI if scores fall below thresholds
python scripts/run_evaluation.py --min-faithfulness 0.7 --min-relevancy 0.75
```

### Python 3.14 compatibility note

RAGAS imports `nest_asyncio` at module level, which replaces `asyncio.Task` with a pure-Python `_PyTask`. Python 3.12+ moved `asyncio.current_task()` to a C built-in that only tracks C-level task instances — making it return `None` inside all RAGAS tasks and causing `asyncio.timeout()` to raise `RuntimeError: Timeout should be used inside a task`. Every metric silently returned NaN.

The fix (applied in `ragas_evaluator.py` immediately after RAGAS imports) patches `asyncio.current_task` to read from `asyncio.tasks._current_tasks`, the Python dict that `_PyTask` correctly maintains:

```python
if not getattr(asyncio, '_current_task_patched', False):
    def _fixed_current_task(loop=None):
        if loop is None:
            try: loop = asyncio.get_running_loop()
            except RuntimeError: return None
        return asyncio.tasks._current_tasks.get(loop)
    asyncio.current_task = _fixed_current_task
    asyncio.tasks.current_task = _fixed_current_task
    asyncio._current_task_patched = True
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

The file ships with 20 questions pre-written from the project's sample documents. Add your own questions from your actual documents to get scores that reflect your real use case.

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
│   └── eval_testset.json          # 20 curated Q&A pairs for RAGAS offline eval (extend with your own)
├── scripts/
│   ├── run_evaluation.py          # CLI batch evaluation script
│   └── verify.sh                  # Full system verification (unit + integration + API tests)
├── logs/                          # RAGAS evaluation log output (gitignored)
├── vector_store/                  # Persisted FAISS index + chunk metadata (gitignored)
├── documents/                     # Local PDF cache synced from S3 (gitignored)
├── .env.example                   # Template for required environment variables
└── requirements.txt
```

---

## Author

**Neerali Acharya**
