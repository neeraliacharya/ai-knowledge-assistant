"""
Integration tests for app/services/rag_pipeline.py

Tests: FAISS persistence (save/load), incremental add, document removal,
disk-match detection. Uses a temp directory so the real vector_store is
never touched.
"""
import os
import pickle
import pytest
import numpy as np

pytestmark = pytest.mark.integration


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    """
    A fresh rag_pipeline module state using a temp VECTOR_DB_PATH so tests
    are fully isolated from the real vector_store/ on disk.

    monkeypatch.setenv changes the env var before config is re-read.
    We also reset the module globals (chunks_store, vector_index) so each
    test starts from a clean slate.
    """
    vs_dir = str(tmp_path / "vs")
    os.makedirs(vs_dir)
    monkeypatch.setenv("VECTOR_DB_PATH", vs_dir)

    # Re-import so module-level constants pick up the patched env
    import importlib
    import app.config
    importlib.reload(app.config)

    import app.services.rag_pipeline as rp
    importlib.reload(rp)  # reset module globals

    return rp, vs_dir


def _seed_pipeline(rp, chunks):
    """Directly set module globals to a known state without hitting the filesystem."""
    from app.services.embedding import generate_embeddings
    from app.services.vector_store import create_vector_store
    texts = [c["text"] for c in chunks]
    embeddings = generate_embeddings(texts)
    rp.chunks_store = list(chunks)
    rp.vector_index = create_vector_store(embeddings)


SAMPLE_CHUNKS = [
    {"text": "The fee is INR 150,000 per resource per month.",
     "metadata": {"source": "doc_a.pdf"}},
    {"text": "Payment terms are fifteen days from invoice date.",
     "metadata": {"source": "doc_a.pdf"}},
    {"text": "Phase 1 is network activation over the first three months.",
     "metadata": {"source": "doc_b.pdf"}},
]


def test_save_and_load_round_trip(pipeline):
    """After _save_to_disk, _load_from_disk must restore identical state."""
    rp, vs_dir = pipeline
    _seed_pipeline(rp, SAMPLE_CHUNKS)

    rp._save_to_disk()

    # Reset module globals
    rp.chunks_store = []
    rp.vector_index = None

    success = rp._load_from_disk()
    assert success, "Expected _load_from_disk to return True"
    assert len(rp.chunks_store) == 3
    texts = {c["text"] for c in rp.chunks_store}
    assert "The fee is INR 150,000 per resource per month." in texts


def test_faiss_file_created_on_save(pipeline):
    rp, vs_dir = pipeline
    _seed_pipeline(rp, SAMPLE_CHUNKS)
    rp._save_to_disk()
    assert os.path.exists(os.path.join(vs_dir, "faiss.index"))
    assert os.path.exists(os.path.join(vs_dir, "chunks_store.pkl"))


def test_load_returns_false_when_no_files(pipeline):
    rp, _ = pipeline
    result = rp._load_from_disk()
    assert result is False


def test_add_document_to_index_increases_chunk_count(pipeline):
    """
    add_document_to_index must append new chunks without removing existing ones.
    """
    rp, _ = pipeline
    _seed_pipeline(rp, SAMPLE_CHUNKS)
    original_count = len(rp.chunks_store)

    rp.add_document_to_index(
        "A new document with some unique content about deployments.",
        "new_doc.pdf"
    )

    assert len(rp.chunks_store) > original_count


def test_add_document_sets_correct_source(pipeline):
    rp, _ = pipeline
    _seed_pipeline(rp, SAMPLE_CHUNKS)
    rp.add_document_to_index("New doc text about monitoring.", "monitor.pdf")
    sources = {c["metadata"]["source"] for c in rp.chunks_store}
    assert "monitor.pdf" in sources


def test_remove_document_deletes_only_target_chunks(pipeline):
    """
    remove_document_from_index("doc_a.pdf") must remove all doc_a chunks
    while leaving doc_b chunks intact.
    """
    rp, _ = pipeline
    _seed_pipeline(rp, SAMPLE_CHUNKS)

    rp.remove_document_from_index("doc_a.pdf")

    sources = {c["metadata"]["source"] for c in rp.chunks_store}
    assert "doc_a.pdf" not in sources, "doc_a.pdf chunks should be removed"
    assert "doc_b.pdf" in sources,     "doc_b.pdf chunks must remain"


def test_remove_document_reduces_count(pipeline):
    rp, _ = pipeline
    _seed_pipeline(rp, SAMPLE_CHUNKS)
    doc_a_count = sum(1 for c in rp.chunks_store if c["metadata"]["source"] == "doc_a.pdf")
    before = len(rp.chunks_store)

    rp.remove_document_from_index("doc_a.pdf")

    assert len(rp.chunks_store) == before - doc_a_count


def test_remove_last_document_clears_index(pipeline):
    """Removing the only document must set vector_index to None."""
    rp, _ = pipeline
    _seed_pipeline(rp, [SAMPLE_CHUNKS[0]])  # only one doc

    rp.remove_document_from_index("doc_a.pdf")

    assert rp.chunks_store == []
    assert rp.vector_index is None


def test_add_document_when_index_is_none(pipeline):
    """add_document_to_index on an empty pipeline must create a new index."""
    rp, _ = pipeline
    assert rp.vector_index is None

    rp.add_document_to_index("Creating index from scratch.", "first_doc.pdf")

    assert rp.vector_index is not None
    assert len(rp.chunks_store) >= 1
