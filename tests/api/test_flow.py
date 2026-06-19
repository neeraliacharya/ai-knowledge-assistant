"""
End-to-end flow simulation tests.

These tests simulate a real user session from start to finish:
  1. App starts — RAG pipeline initialises from documents/ (or disk cache)
  2. User asks a question — answer is retrieved and generated (LLM mocked)
  3. User asks a follow-up — memory carries over the conversation context
  4. User uploads a new document — it gets indexed without full rebuild
  5. User asks a question scoped to the new document — correct source returned
  6. User deletes the document — it is removed from the index
  7. Same question now returns no source from the deleted document

For live LLM tests (real Groq calls), use:
    pytest tests/api/test_flow.py -m live
"""
import io
import os
import pytest
import uuid

pytestmark = pytest.mark.api


# ── Scenario 1: Basic question → answer → follow-up ──────────────────────────

def test_first_question_returns_structured_response(api_client, mock_llm):
    """A fresh question must return all four required fields."""
    session = f"flow-test-{uuid.uuid4()}"
    r = api_client.get("/ask", params={
        "question": "What is the monthly fee in the service agreement?",
        "session_id": session,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["question"] != ""
    assert body["answer"]   != ""
    assert isinstance(body["sources"], list)


def test_follow_up_question_uses_same_session(api_client, mock_llm):
    """
    Two questions in the same session must both succeed. The second
    call reuses the session_id so the backend can access conversation history.
    """
    session = f"flow-session-{uuid.uuid4()}"

    r1 = api_client.get("/ask", params={
        "question": "What does Nimbus TechKnox do?",
        "session_id": session,
    })
    assert r1.status_code == 200

    r2 = api_client.get("/ask", params={
        "question": "How many phases does their GTM strategy have?",
        "session_id": session,
    })
    assert r2.status_code == 200
    # Both requests must have answers
    assert r1.json()["answer"] != ""
    assert r2.json()["answer"] != ""


def test_independent_sessions_do_not_interfere(api_client, mock_llm):
    """
    Two simultaneous sessions must not share conversation history.
    """
    sid_a = f"session-a-{uuid.uuid4()}"
    sid_b = f"session-b-{uuid.uuid4()}"

    api_client.get("/ask", params={"question": "Question for session A", "session_id": sid_a})
    api_client.get("/ask", params={"question": "Question for session B", "session_id": sid_b})

    # Both sessions are still live and independent — no 500 errors
    r_a = api_client.get("/ask", params={"question": "Follow-up A", "session_id": sid_a})
    r_b = api_client.get("/ask", params={"question": "Follow-up B", "session_id": sid_b})
    assert r_a.status_code == 200
    assert r_b.status_code == 200


# ── Scenario 2: Upload → query → delete ───────────────────────────────────────

def test_upload_indexes_and_query_returns_source(api_client, pdf_bytes, tmp_path):
    """
    Full cycle:
      1. Upload a PDF
      2. Ask a question scoped to that PDF
      3. Verify the response comes back (not a crash)
      4. Delete the PDF
      5. Verify delete succeeds
    """
    import app.main as m
    import app.services.rag_pipeline as rp
    from unittest.mock import patch

    pdf_content, original_fname = pdf_bytes
    test_fname = f"_flowtest_{uuid.uuid4().hex[:8]}.pdf"

    temp_docs = str(tmp_path / "docs")
    os.makedirs(temp_docs)

    original_folder = m.DOCUMENTS_FOLDER
    m.DOCUMENTS_FOLDER = temp_docs + "/"

    # Track what was indexed before upload
    chunks_before = len(rp.chunks_store)

    try:
        # ── Step 1: Upload ──────────────────────────────────────────────────
        r_upload = api_client.post(
            "/upload",
            files={"file": (test_fname, io.BytesIO(pdf_content), "application/pdf")},
        )
        assert r_upload.status_code == 200, f"Upload failed: {r_upload.json()}"

        # ── Step 2: Index should have grown ───────────────────────────────
        chunks_after = len(rp.chunks_store)
        assert chunks_after > chunks_before, (
            f"Expected more chunks after upload: before={chunks_before}, after={chunks_after}"
        )

        # ── Step 3: Query scoped to the uploaded file ──────────────────────
        # LLM is NOT mocked here so we use a simple patched reformulate to
        # avoid a real API call while still exercising retrieval.
        with patch("app.main.reformulate_query", lambda q, h: q), \
             patch("app.main.generate_rag_answer", lambda q, ctx, hist=None: f"Answer: {ctx[:80]}"):
            r_ask = api_client.get("/ask", params={
                "question": "Summarize the document",
                "files": test_fname,
            })
        assert r_ask.status_code == 200

        # ── Step 4: Delete ─────────────────────────────────────────────────
        with patch("app.services.s3_storage.delete_from_s3", return_value=True):
            r_delete = api_client.delete(f"/documents/{test_fname}")
        assert r_delete.status_code == 200

        # ── Step 5: Chunks from that file should be gone ───────────────────
        remaining_sources = {c["metadata"]["source"] for c in rp.chunks_store}
        assert test_fname not in remaining_sources, (
            f"Chunks for {test_fname} still present after deletion"
        )

    finally:
        m.DOCUMENTS_FOLDER = original_folder
        # Ensure cleanup even if test fails
        for leftover in [os.path.join("documents", test_fname),
                         os.path.join(temp_docs, test_fname)]:
            if os.path.exists(leftover):
                os.remove(leftover)


# ── Scenario 3: Eval summary endpoint ─────────────────────────────────────────

def test_eval_summary_returns_dict(api_client):
    """/eval-summary must always return a dict (may be empty if no evals run yet)."""
    r = api_client.get("/eval-summary")
    assert r.status_code == 200
    assert isinstance(r.json(), dict)


def test_evaluate_endpoint_needs_testset(api_client):
    """
    POST /evaluate must return 404 when data/eval_testset.json is missing,
    or 200/503 depending on RAGAS availability and testset presence.
    """
    r = api_client.post("/evaluate")
    # 200 = started, 404 = no testset, 503 = ragas not available
    assert r.status_code in (200, 404, 503)


# ── Scenario 4: Input edge cases ──────────────────────────────────────────────

@pytest.mark.parametrize("question", [
    "hi",
    "hello",
    "what is this app?",
    "summarize everything",
])
def test_common_greetings_and_commands_dont_crash(api_client, mock_llm, question):
    """Short greetings and broad commands must all return 200."""
    r = api_client.get("/ask", params={"question": question})
    assert r.status_code == 200


# ── Scenario 5: Live LLM integration (opt-in) ─────────────────────────────────

@pytest.mark.live
def test_live_ask_returns_relevant_answer(api_client):
    """
    Real end-to-end test with actual Groq LLM call.
    Run with: pytest tests/api/test_flow.py -m live

    Verifies that the answer to a factual question about the indexed documents
    contains content that is grounded in the actual document text.
    """
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set — skipping live LLM test")

    import app.services.rag_pipeline as rp
    if rp.vector_index is None:
        pytest.skip("No documents indexed — skipping live retrieval test")

    session = f"live-test-{uuid.uuid4()}"
    r = api_client.get("/ask", params={
        "question": "What is the monthly fee in the service agreement?",
        "session_id": session,
    })
    assert r.status_code == 200
    body = r.json()
    answer = body["answer"].lower()
    # The answer should mention the fee amount (150,000) or the word "fee"
    assert any(keyword in answer for keyword in ["150", "fee", "inr", "payment", "month"]), (
        f"Expected a relevant answer about fees, got: {body['answer']}"
    )


@pytest.mark.live
def test_live_memory_carries_context(api_client):
    """
    Two real LLM calls in the same session. The second question uses a
    pronoun that only makes sense with the first question's context.
    Verifies that query reformulation works with real history.
    """
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set")

    session = f"live-memory-{uuid.uuid4()}"

    # First question establishes context
    api_client.get("/ask", params={
        "question": "Tell me about Neerali Acharya",
        "session_id": session,
    })

    # Second question uses pronoun — reformulate_query must expand "she"
    r2 = api_client.get("/ask", params={
        "question": "What company does she work at?",
        "session_id": session,
    })
    assert r2.status_code == 200
    body = r2.json()
    # search_query should no longer contain a bare pronoun
    search_query = body["search_query"].lower()
    assert "neerali" in search_query or "oracle" in search_query or "she" not in search_query, (
        f"Expected pronoun resolved in search_query, got: {body['search_query']}"
    )
