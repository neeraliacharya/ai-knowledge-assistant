"""
FastAPI application entry point.

Production changes vs. original
---------------------------------
1. CORS restricted to ALLOWED_ORIGINS from config (was allow_origins=["*"])
2. API-key auth middleware (opt-in via API_KEY env var)
3. Request logging middleware — every request gets a correlation ID + latency
4. /health and /ready endpoints for load-balancer health checks
5. /ask: input sanitisation, guard for empty index, RAGAS background evaluation
6. /upload: file-size cap + PDF magic-byte validation, incremental indexing
7. /delete: calls remove_document_from_index() instead of full initialize_rag()
8. /evaluate: trigger offline RAGAS batch evaluation
9. Duplicate app = FastAPI() removed (was created twice in original)
"""
import os
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import app.services.rag_pipeline as rag_pipeline
import app.services.ragas_evaluator as ragas_eval
import app.services.s3_storage as _s3
from app.config import (
    ALLOWED_ORIGINS,
    API_KEY,
    UPLOAD_MAX_MB,
)
from app.services.chunking import chunk_text
from app.services.embedding import generate_embeddings, embedding_model
from app.services.ingestion import extract_text_from_pdf
from app.services.llm_service import ask_llm, generate_rag_answer, reformulate_query
from app.services.logger import get_logger
from app.services.memory import get_history, add_message
from app.services.reranker import rerank_chunks
from app.services.retrieval import retrieve_chunks
from app.services.s3_storage import (
    list_s3_documents,
    sync_s3_to_local,
    upload_to_s3,
)
from app.services.vector_store import create_vector_store

log = get_logger(__name__)

DOCUMENTS_FOLDER = "documents/"
UPLOAD_MAX_BYTES = UPLOAD_MAX_MB * 1024 * 1024


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting AI Knowledge Assistant")
    sync_s3_to_local(DOCUMENTS_FOLDER)
    rag_pipeline.initialize_rag()
    log.info("Startup complete", extra={"chunks": len(rag_pipeline.chunks_store)})
    yield
    log.info("Shutting down")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(title="AI Knowledge Assistant", lifespan=lifespan)

# ── CORS ───────────────────────────────────────────────────────────────────────
# Previously was allow_origins=["*"] which allows any website to call the API.
# Now restricted to the origins listed in ALLOWED_ORIGINS (env var).
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Auth middleware ────────────────────────────────────────────────────────────
# If API_KEY is set in the environment, every request must include the header:
#   X-API-Key: <value>
# Health and readiness probes are exempted so load balancers can reach them
# without credentials.
UNPROTECTED_PATHS = {"/", "/health", "/ready", "/ping"}

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if API_KEY and request.url.path not in UNPROTECTED_PATHS:
        if request.headers.get("X-API-Key") != API_KEY:
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid X-API-Key header"},
            )
    return await call_next(request)


# ── Request logging middleware ─────────────────────────────────────────────────
# Assigns a short correlation ID to every request and logs method, path,
# status code, and wall-clock latency. The request_id appears in every
# downstream log line, making it easy to trace a single user request through
# all service calls.
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    log.info(
        "http_request",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "latency_ms": latency_ms,
        },
    )
    return response


# ── Health / readiness ─────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Always returns 200. Used by load balancers to know the process is alive."""
    return {"status": "ok", "ts": datetime.now(timezone.utc).isoformat()}


@app.get("/ready")
def ready():
    """
    Returns 200 only once the RAG pipeline is initialised.
    Load balancers should wait for /ready before sending traffic.
    """
    if rag_pipeline.vector_index is None and not rag_pipeline.chunks_store:
        raise HTTPException(status_code=503, detail="RAG pipeline not yet initialised")
    return {
        "status": "ready",
        "chunks": len(rag_pipeline.chunks_store),
    }


# ── Debug endpoints ────────────────────────────────────────────────────────────

@app.get("/")
def home():
    return {"message": "AI Knowledge Assistant API Running"}


@app.get("/ping")
def ping():
    return {"status": "alive"}


@app.get("/test-llm")
def test_llm():
    response = ask_llm("Explain what a vector database is in just 2 lines")
    return {"response": response}


@app.get("/ask-anything")
def ask_anything(question: str):
    response = ask_llm(question)
    return {"answer": response}


# ── Main RAG query endpoint ────────────────────────────────────────────────────

@app.get("/ask")
def ask(
    question: str,
    session_id: str = Query("default"),
    files: list[str] = Query(None),
    background_tasks: BackgroundTasks = None,
):
    # ── Input sanitisation ──────────────────────────────────────────────────
    question = question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if len(question) > 2000:
        question = question[:2000]

    # ── Guard: no documents indexed yet ────────────────────────────────────
    if rag_pipeline.vector_index is None:
        return {
            "question": question,
            "search_query": question,
            "answer": "No documents have been uploaded yet. Please upload a PDF first.",
            "sources": [],
        }

    request_id = str(uuid.uuid4())[:8]

    history = get_history(session_id)
    search_query = reformulate_query(question, history)

    retrieved_chunks = retrieve_chunks(
        search_query,
        embedding_model,
        rag_pipeline.vector_index,
        rag_pipeline.chunks_store,
        top_k=10,
        filters=files,
    )

    reranked_chunks = rerank_chunks(search_query, retrieved_chunks)

    # Group reranked chunks by source document
    doc_chunks: dict[str, list[str]] = {}
    for chunk in reranked_chunks:
        source = chunk["metadata"]["source"]
        doc_chunks.setdefault(source, []).append(chunk["text"])

    if not doc_chunks:
        answer = generate_rag_answer(question, "", history)
        sources: list[str] = []
    else:
        top_doc = list(doc_chunks.keys())[0]
        context = "\n\n".join(doc_chunks[top_doc][:4])
        answer = generate_rag_answer(question, context, history)
        sources = [top_doc]

    add_message(session_id, "user", question)
    add_message(session_id, "assistant", answer)

    # ── RAGAS online evaluation (non-blocking background task) ──────────────
    #
    # HOW THE BACKGROUND TASK WORKS:
    # FastAPI's BackgroundTasks runs evaluate_and_log() AFTER the response has
    # been sent to the user — the user never waits for it.
    # We pass the raw chunk texts (not the concatenated context string) because
    # RAGAS needs individual passages to check coverage and precision per chunk.
    # No ground_truth is available for live traffic, so only Faithfulness and
    # AnswerRelevancy are computed.
    if background_tasks and ragas_eval.is_available():
        background_tasks.add_task(
            ragas_eval.evaluate_and_log,
            question=question,
            answer=answer,
            contexts=[c["text"] for c in reranked_chunks],
            ground_truth=None,
            request_id=request_id,
        )

    return {
        "question": question,
        "search_query": search_query,
        "answer": answer,
        "sources": sources,
    }


# ── Document management ────────────────────────────────────────────────────────

@app.get("/documents")
def list_documents():
    return list_s3_documents()


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF, validate it, store locally + S3, then add it to the index
    incrementally (only the new document's chunks are embedded — not all docs).
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Read entire file into memory first so we can validate before writing
    content = await file.read()

    # ── File size check ─────────────────────────────────────────────────────
    if len(content) > UPLOAD_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {UPLOAD_MAX_MB} MB upload limit",
        )

    # ── PDF magic-byte validation ───────────────────────────────────────────
    # A real PDF always starts with the bytes %PDF- regardless of extension.
    # This prevents uploading a renamed .exe or .html as a .pdf.
    if not content.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400, detail="File content is not a valid PDF"
        )

    try:
        os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)
        local_path = os.path.join(DOCUMENTS_FOLDER, file.filename)

        with open(local_path, "wb") as f_out:
            f_out.write(content)

        success = upload_to_s3(local_path, file.filename)
        if not success:
            raise HTTPException(status_code=500, detail="S3 upload failed")

        # ── Incremental indexing ──────────────────────────────────────────────
        # Instead of re-embedding ALL documents (old behaviour), we extract text
        # only from this new document and append its chunks to the existing index.
        doc_text = extract_text_from_pdf(local_path)
        rag_pipeline.add_document_to_index(doc_text, file.filename)

        return {"message": f"{file.filename} uploaded and indexed successfully"}

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Upload failed: {e}", extra={"filename": file.filename})
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    """
    Delete a PDF from S3 + local storage and remove its chunks from the index.
    The index is rebuilt from the remaining documents only.
    """
    try:
        s3_success = _s3.delete_from_s3(filename)
        if not s3_success:
            raise HTTPException(status_code=404, detail="File not found in S3")

        local_path = os.path.join(DOCUMENTS_FOLDER, filename)
        if os.path.isfile(local_path):
            os.remove(local_path)

        # Remove this document's chunks and rebuild from remaining docs
        rag_pipeline.remove_document_from_index(filename)

        return {"message": f"{filename} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Delete failed: {e}", extra={"filename": filename})
        raise HTTPException(status_code=500, detail=str(e))


# ── RAGAS batch evaluation ─────────────────────────────────────────────────────

@app.post("/evaluate")
def run_evaluation(background_tasks: BackgroundTasks):
    """
    Trigger an offline RAGAS batch evaluation in the background.

    Reads data/eval_testset.json, runs the full RAG pipeline on each question,
    evaluates with RAGAS (all four metrics including ContextPrecision and
    ContextRecall), and writes per-sample scores to logs/ragas_eval_log.jsonl.

    You can also run this offline without the server:
        python scripts/run_evaluation.py
    """
    if not ragas_eval.is_available():
        raise HTTPException(
            status_code=503,
            detail=(
                "RAGAS is not installed. "
                "Run: pip install ragas langchain-groq langchain-huggingface"
            ),
        )
    if not os.path.exists("data/eval_testset.json"):
        raise HTTPException(
            status_code=404,
            detail="No test set found at data/eval_testset.json",
        )

    background_tasks.add_task(_run_batch_eval)
    return {
        "message": "Batch evaluation started",
        "output": "logs/ragas_eval_log.jsonl",
    }


def _run_batch_eval():
    """Called as a background task by POST /evaluate."""
    import json

    with open("data/eval_testset.json") as f:
        testset = json.load(f)

    for item in testset:
        question = item["question"]
        ground_truth = item.get("ground_truth")

        if rag_pipeline.vector_index is None:
            log.warning("Skipping eval — no index loaded")
            return

        search_query = reformulate_query(question, [])
        retrieved = retrieve_chunks(
            search_query, embedding_model,
            rag_pipeline.vector_index, rag_pipeline.chunks_store,
            top_k=10,
        )
        reranked = rerank_chunks(search_query, retrieved)
        context = "\n\n".join(c["text"] for c in reranked[:4])
        answer = generate_rag_answer(question, context, [])

        ragas_eval.evaluate_and_log(
            question=question,
            answer=answer,
            contexts=[c["text"] for c in reranked],
            ground_truth=ground_truth,
        )

    log.info("Batch evaluation complete", extra={"samples": len(testset)})


@app.get("/eval-summary")
def eval_summary():
    """Return averaged RAGAS scores across all logged evaluations."""
    summary = ragas_eval.get_scores_summary()
    return summary
