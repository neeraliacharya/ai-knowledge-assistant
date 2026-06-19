"""
Integration tests for app/services/reranker.py

Uses the real CrossEncoder model. No LLM API calls.
"""
import pytest
from app.services.reranker import rerank_chunks

pytestmark = pytest.mark.integration

CHUNKS = [
    {"text": "The sky is blue and the sun is bright today.",
     "metadata": {"source": "weather.pdf"}},
    {"text": "The monthly fee is INR 150,000 per resource.",
     "metadata": {"source": "agreement.pdf"}},
    {"text": "Payment must be made within 15 days of invoice.",
     "metadata": {"source": "agreement.pdf"}},
    {"text": "Phase 1 covers the first three months of the strategy.",
     "metadata": {"source": "gtm.pdf"}},
]


def test_most_relevant_chunk_ranked_first():
    """
    For a query about payment fee, the chunk mentioning INR 150,000 should
    be ranked higher than an unrelated chunk about the sky.
    """
    query = "What is the monthly payment fee?"
    reranked = rerank_chunks(query, CHUNKS, top_k=4)
    top_text = reranked[0]["text"]
    assert "150,000" in top_text or "fee" in top_text.lower() or "payment" in top_text.lower(), (
        f"Expected fee/payment chunk at top, got: {top_text}"
    )


def test_sky_chunk_ranked_below_payment_chunks():
    """Irrelevant chunks about weather should rank below financial chunks for a payment query."""
    query = "monthly fee payment terms"
    reranked = rerank_chunks(query, CHUNKS, top_k=4)
    texts = [r["text"] for r in reranked]
    sky_index     = next((i for i, t in enumerate(texts) if "sky" in t.lower()), None)
    payment_index = next((i for i, t in enumerate(texts) if "fee" in t.lower() or "payment" in t.lower()), None)
    if sky_index is not None and payment_index is not None:
        assert payment_index < sky_index, (
            "Payment chunk should rank above sky chunk for a payment query"
        )


def test_returns_at_most_top_k():
    reranked = rerank_chunks("any query", CHUNKS, top_k=2)
    assert len(reranked) <= 2


def test_single_chunk_returned_unchanged():
    single = [CHUNKS[0]]
    reranked = rerank_chunks("any query", single, top_k=3)
    assert len(reranked) == 1
    assert reranked[0]["text"] == CHUNKS[0]["text"]


def test_output_preserves_chunk_structure():
    """Each returned chunk must still have 'text' and 'metadata' keys."""
    reranked = rerank_chunks("query", CHUNKS, top_k=3)
    for chunk in reranked:
        assert "text"     in chunk
        assert "metadata" in chunk
        assert "source"   in chunk["metadata"]


def test_empty_chunks_returns_empty():
    reranked = rerank_chunks("query", [], top_k=3)
    assert reranked == []
