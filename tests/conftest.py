"""
Shared fixtures for the entire test suite.

Scope strategy
--------------
session  — expensive one-time setup: ML models, FAISS index, TestClient startup
module   — per-test-file setup
function — per-test setup (default); used for anything that mutates state
"""
import os
import sys
import pytest

# Make project root importable regardless of how pytest is invoked
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Small synthetic corpus used by integration tests ──────────────────────────
# These are tiny fixed texts so integration tests don't depend on the real PDFs
# being present, while still exercising the real embedding + retrieval logic.
TEST_CHUNKS = [
    {
        "text": "Nimbus TechKnox charges INR 150,000 per resource per month under the service agreement.",
        "metadata": {"source": "agreement.pdf"},
    },
    {
        "text": "Payment is due within fifteen days of invoice receipt by electronic transfer.",
        "metadata": {"source": "agreement.pdf"},
    },
    {
        "text": "The contract can be terminated with one month written notice by either party.",
        "metadata": {"source": "agreement.pdf"},
    },
    {
        "text": "The go-to-market strategy has three phases spread over twelve months.",
        "metadata": {"source": "gtm.pdf"},
    },
    {
        "text": "Phase 1 activates the existing network through LinkedIn outreach and referrals.",
        "metadata": {"source": "gtm.pdf"},
    },
    {
        "text": "Neerali Acharya is a Senior Software Engineer at Oracle India with 6.5 years of experience.",
        "metadata": {"source": "cv.pdf"},
    },
]


@pytest.fixture(scope="session")
def test_chunks():
    return list(TEST_CHUNKS)


# ── ML models — loaded once per session ───────────────────────────────────────

@pytest.fixture(scope="session")
def embed_model():
    """The real BGE embedding model. Loads once, reused across all tests."""
    from app.services.embedding import embedding_model
    return embedding_model


@pytest.fixture(scope="session")
def test_index(embed_model):
    """
    A real FAISS index built from TEST_CHUNKS.
    Session-scoped so the embedding + index build happens only once.
    """
    from app.services.embedding import generate_embeddings
    from app.services.vector_store import create_vector_store
    texts = [c["text"] for c in TEST_CHUNKS]
    embeddings = generate_embeddings(texts)
    return create_vector_store(embeddings)


# ── Real PDF bytes — for upload endpoint tests ─────────────────────────────────

@pytest.fixture(scope="session")
def pdf_bytes():
    """
    Raw bytes of a real small PDF from the documents/ folder.
    Used to POST to /upload in API tests.
    Skips the test if no PDF is found.
    """
    for fname in ["Python-Basics_compressed.pdf", "nimbustk-service-agreement.pdf"]:
        path = os.path.join("documents", fname)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return f.read(), fname
    pytest.skip("No PDF found in documents/ — upload tests skipped")


# ── FastAPI TestClient — boots the real app with S3 mocked ───────────────────

@pytest.fixture(scope="session")
def api_client():
    """
    TestClient that starts the full FastAPI app (including lifespan/startup)
    with all S3 operations patched to no-ops.

    WHY mock S3?
    Tests must not hit real AWS. We mock at the function level in s3_storage
    so the rest of the app (RAG pipeline, auth, validation) runs for real.

    WHY session scope?
    The lifespan boots the embedding model and CrossEncoder reranker. Loading
    them twice per test run would add ~30s. Session scope means they load once.

    The RAG pipeline initialises from the real documents/ folder (or disk cache)
    so /ask tests use real retrieved chunks, not mocked data.
    """
    from unittest.mock import patch, MagicMock
    from fastapi.testclient import TestClient

    s3_mocks = [
        patch("app.services.s3_storage.sync_s3_to_local", return_value=None),
        patch("app.services.s3_storage.upload_to_s3",     return_value=True),
        patch("app.services.s3_storage.delete_from_s3",   return_value=True),
        patch("app.services.s3_storage.list_s3_documents",
              return_value=[{"name": "NimbusTechKnox_GTM_Strategy.pdf", "size": 99999}]),
        # Disable RAGAS during API tests. The background task makes real Groq
        # LLM calls (~90s each) and nest_asyncio patches the anyio event loop
        # class, corrupting task-context tracking for subsequent requests.
        # RAGAS correctness is verified separately via scripts/run_evaluation.py.
        patch("app.services.ragas_evaluator.is_available", return_value=False),
    ]
    for m in s3_mocks:
        m.start()

    from app.main import app
    with TestClient(app) as client:
        yield client

    for m in s3_mocks:
        m.stop()


# ── LLM mock helpers ───────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm(monkeypatch):
    """
    Patch the LLM functions imported into app.main so no Groq API calls are
    made during endpoint tests. The patches return deterministic fake values.

    WHY patch app.main.* and not app.services.llm_service.*?
    app/main.py does `from app.services.llm_service import reformulate_query`.
    That creates a local name in app.main. Patching app.services.llm_service
    won't affect app.main's already-bound reference. So we must patch the name
    where it's used, not where it's defined.
    """
    monkeypatch.setattr("app.main.reformulate_query",
                        lambda question, history: question)
    monkeypatch.setattr("app.main.generate_rag_answer",
                        lambda q, ctx, hist=None: f"Mocked answer for: {q[:40]}")
    monkeypatch.setattr("app.main.ask_llm",
                        lambda prompt: "Mocked direct LLM response")
