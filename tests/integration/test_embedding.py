"""
Integration tests for app/services/embedding.py

Uses the real BAAI/bge-base-en-v1.5 model. Model weights are loaded once
per test session via the session-scoped embed_model fixture in conftest.py.
No LLM API calls — embeddings are computed locally.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.integration


def test_embeddings_correct_shape(embed_model):
    """generate_embeddings must return shape (n_texts, 768) for BGE-base."""
    from app.services.embedding import generate_embeddings
    texts = ["first sentence", "second sentence", "third sentence"]
    embeddings = generate_embeddings(texts)
    assert embeddings.shape == (3, 768)


def test_single_text_works(embed_model):
    from app.services.embedding import generate_embeddings
    embeddings = generate_embeddings(["only one sentence"])
    assert embeddings.shape == (1, 768)


def test_embeddings_are_float32(embed_model):
    from app.services.embedding import generate_embeddings
    embeddings = generate_embeddings(["test sentence"])
    assert embeddings.dtype in (np.float32, np.float64)


def test_embeddings_are_deterministic(embed_model):
    """Same text must produce identical embedding vectors on every call."""
    from app.services.embedding import generate_embeddings
    text = ["deterministic test sentence"]
    e1 = generate_embeddings(text)
    e2 = generate_embeddings(text)
    np.testing.assert_array_almost_equal(e1, e2, decimal=5)


def test_different_texts_produce_different_embeddings(embed_model):
    """Semantically unrelated texts must not have identical embeddings."""
    from app.services.embedding import generate_embeddings
    e = generate_embeddings(["monthly payment fee", "kubernetes deployment pipeline"])
    # Cosine similarity between two unrelated texts should not be 1.0
    sim = np.dot(e[0], e[1]) / (np.linalg.norm(e[0]) * np.linalg.norm(e[1]))
    assert sim < 0.99, f"Unrelated texts too similar: cosine={sim:.4f}"


def test_similar_texts_have_higher_similarity(embed_model):
    """
    A semantically similar pair must score higher than an unrelated pair.
    This validates the model is actually doing semantic search, not random.
    """
    from app.services.embedding import generate_embeddings
    embeddings = generate_embeddings([
        "monthly service fee INR 150000",
        "the payment amount per month",
        "kubernetes cluster deployment",
    ])
    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    similar_score   = cosine(embeddings[0], embeddings[1])
    unrelated_score = cosine(embeddings[0], embeddings[2])
    assert similar_score > unrelated_score, (
        f"Expected similar_score ({similar_score:.4f}) > "
        f"unrelated_score ({unrelated_score:.4f})"
    )


def test_batch_equals_individual(embed_model):
    """Embedding a batch must produce the same vector as embedding each text alone."""
    from app.services.embedding import generate_embeddings
    texts = ["sentence one", "sentence two"]
    batch = generate_embeddings(texts)
    single_0 = generate_embeddings([texts[0]])
    single_1 = generate_embeddings([texts[1]])
    np.testing.assert_array_almost_equal(batch[0], single_0[0], decimal=4)
    np.testing.assert_array_almost_equal(batch[1], single_1[0], decimal=4)
