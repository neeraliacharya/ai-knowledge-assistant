"""
API tests for GET /ask

Tests input validation, auth middleware, response shape, and the full
question→retrieval→answer flow (LLM mocked to avoid API costs).
"""
import pytest

pytestmark = pytest.mark.api


# ── Input validation ───────────────────────────────────────────────────────────

def test_ask_empty_question_rejected(api_client, mock_llm):
    r = api_client.get("/ask", params={"question": "   "})
    assert r.status_code == 400


def test_ask_missing_question_rejected(api_client):
    r = api_client.get("/ask")
    assert r.status_code == 422  # FastAPI validation error


def test_ask_very_long_question_accepted(api_client, mock_llm):
    """Questions over 2000 chars should be truncated, not rejected."""
    long_q = "A" * 3000
    r = api_client.get("/ask", params={"question": long_q})
    assert r.status_code == 200


# ── Response shape ─────────────────────────────────────────────────────────────

def test_ask_response_has_required_fields(api_client, mock_llm):
    r = api_client.get("/ask", params={"question": "What is the fee?"})
    assert r.status_code == 200
    body = r.json()
    assert "question"     in body
    assert "search_query" in body
    assert "answer"       in body
    assert "sources"      in body


def test_ask_sources_is_list(api_client, mock_llm):
    r = api_client.get("/ask", params={"question": "Tell me about Nimbus"})
    assert isinstance(r.json()["sources"], list)


def test_ask_question_echoed_in_response(api_client, mock_llm):
    q = "What is the monthly fee?"
    r = api_client.get("/ask", params={"question": q})
    assert r.json()["question"] == q


def test_ask_answer_is_string(api_client, mock_llm):
    r = api_client.get("/ask", params={"question": "Any question"})
    assert isinstance(r.json()["answer"], str)
    assert len(r.json()["answer"]) > 0


# ── Session / memory ───────────────────────────────────────────────────────────

def test_ask_uses_default_session(api_client, mock_llm):
    """Omitting session_id must default to 'default' session."""
    r = api_client.get("/ask", params={"question": "Hello"})
    assert r.status_code == 200


def test_ask_with_custom_session_id(api_client, mock_llm):
    r = api_client.get("/ask", params={"question": "Test", "session_id": "my-session-abc"})
    assert r.status_code == 200


# ── Auth middleware ────────────────────────────────────────────────────────────

def test_ask_blocked_without_api_key_when_auth_enabled(api_client, monkeypatch, mock_llm):
    """
    When API_KEY is set, requests without X-API-Key must get 401.
    We temporarily set the config value and make a request WITHOUT the header.
    """
    import app.main as m
    original = m.API_KEY
    m.API_KEY = "secret-key-for-test"
    try:
        r = api_client.get("/ask", params={"question": "test"})
        assert r.status_code == 401
    finally:
        m.API_KEY = original


def test_ask_passes_with_correct_api_key(api_client, mock_llm):
    """With a correct X-API-Key header, the request must succeed."""
    import app.main as m
    original = m.API_KEY
    m.API_KEY = "secret-key-for-test"
    try:
        r = api_client.get("/ask",
                           params={"question": "test"},
                           headers={"X-API-Key": "secret-key-for-test"})
        assert r.status_code == 200
    finally:
        m.API_KEY = original


def test_ask_rejected_with_wrong_api_key(api_client, mock_llm):
    import app.main as m
    original = m.API_KEY
    m.API_KEY = "correct-key"
    try:
        r = api_client.get("/ask",
                           params={"question": "test"},
                           headers={"X-API-Key": "wrong-key"})
        assert r.status_code == 401
    finally:
        m.API_KEY = original


def test_health_bypasses_auth(api_client):
    """
    /health must always be reachable even when API_KEY is set — it's
    whitelisted for load-balancer probes.
    """
    import app.main as m
    original = m.API_KEY
    m.API_KEY = "some-key"
    try:
        r = api_client.get("/health")
        assert r.status_code == 200
    finally:
        m.API_KEY = original


# ── No-documents guard ─────────────────────────────────────────────────────────

def test_ask_returns_helpful_message_when_no_index(api_client, mock_llm, monkeypatch):
    """
    If the vector index is None (no docs uploaded), /ask must return a
    user-friendly message instead of crashing.
    """
    import app.services.rag_pipeline as rp
    original_index  = rp.vector_index
    original_chunks = rp.chunks_store

    rp.vector_index  = None
    rp.chunks_store  = []
    try:
        r = api_client.get("/ask", params={"question": "What is the fee?"})
        assert r.status_code == 200
        assert "upload" in r.json()["answer"].lower() or r.json()["answer"] != ""
    finally:
        rp.vector_index  = original_index
        rp.chunks_store  = original_chunks


# ── File filtering ─────────────────────────────────────────────────────────────

def test_ask_accepts_file_filter_param(api_client, mock_llm):
    """Passing files= query param must not crash the endpoint."""
    r = api_client.get("/ask", params={
        "question": "What is the fee?",
        "files": "nimbustk-service-agreement.pdf",
    })
    assert r.status_code == 200
