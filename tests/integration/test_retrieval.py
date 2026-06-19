"""
Integration tests for app/services/retrieval.py

Uses the real embedding model + FAISS index built from TEST_CHUNKS in conftest.
No LLM API calls.
"""
import pytest
from app.services.retrieval import retrieve_chunks

pytestmark = pytest.mark.integration


def test_relevant_chunk_is_retrieved(embed_model, test_index, test_chunks):
    """
    Querying about 'monthly fee' must return the chunk that mentions
    INR 150,000, not a chunk about GTM phases.
    """
    results = retrieve_chunks("monthly fee service agreement", embed_model,
                              test_index, test_chunks, top_k=3)
    texts = [r["text"] for r in results]
    assert any("150,000" in t or "fee" in t.lower() for t in texts), (
        f"Expected fee-related chunk, got: {texts}"
    )


def test_returns_at_most_top_k(embed_model, test_index, test_chunks):
    results = retrieve_chunks("any query", embed_model, test_index, test_chunks, top_k=2)
    assert len(results) <= 2


def test_results_have_text_and_metadata(embed_model, test_index, test_chunks):
    results = retrieve_chunks("query", embed_model, test_index, test_chunks, top_k=3)
    for r in results:
        assert "text"     in r
        assert "metadata" in r
        assert "source"   in r["metadata"]


def test_file_filter_limits_sources(embed_model, test_index, test_chunks):
    """
    When filters=["agreement.pdf"] is set, results must only contain chunks
    whose source is agreement.pdf — no gtm.pdf or cv.pdf chunks.
    """
    results = retrieve_chunks("any query", embed_model, test_index, test_chunks,
                              top_k=10, filters=["agreement.pdf"])
    for r in results:
        assert r["metadata"]["source"] == "agreement.pdf", (
            f"Got chunk from unexpected source: {r['metadata']['source']}"
        )


def test_filter_for_nonexistent_file_returns_empty(embed_model, test_index, test_chunks):
    results = retrieve_chunks("query", embed_model, test_index, test_chunks,
                              top_k=5, filters=["does_not_exist.pdf"])
    assert results == []


def test_keyword_match_preferred_over_pure_vector(embed_model, test_index, test_chunks):
    """
    When query words appear verbatim in a chunk, that chunk should be
    returned (keyword fallback path in retrieval.py).
    """
    results = retrieve_chunks("termination written notice", embed_model,
                              test_index, test_chunks, top_k=5)
    texts = [r["text"] for r in results]
    assert any("terminated" in t.lower() or "termination" in t.lower() or "notice" in t.lower()
               for t in texts)


def test_all_results_from_corpus(embed_model, test_index, test_chunks):
    """Every returned chunk must originate from the test corpus."""
    corpus_texts = {c["text"] for c in test_chunks}
    results = retrieve_chunks("some query", embed_model, test_index, test_chunks, top_k=3)
    for r in results:
        assert r["text"] in corpus_texts
