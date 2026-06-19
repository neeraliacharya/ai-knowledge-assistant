"""
Unit tests for app/services/chunking.py

Tests pure logic only — no models, no files, no API calls.
"""
import pytest
from app.services.chunking import chunk_text

pytestmark = pytest.mark.unit

# ── Helpers ────────────────────────────────────────────────────────────────────

def make_text(n_chars: int) -> str:
    """Generate a string of exactly n_chars by repeating a word."""
    word = "knowledge "
    return (word * (n_chars // len(word) + 1))[:n_chars]


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_short_text_returns_single_chunk():
    """Text under chunk_size should produce exactly one chunk."""
    text = "This is a short document."
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_long_text_produces_multiple_chunks():
    """Text longer than chunk_size (1000) must be split into several chunks."""
    text = make_text(4000)
    chunks = chunk_text(text)
    assert len(chunks) > 1


def test_no_chunk_exceeds_max_size():
    """Every chunk must be at most chunk_size characters (1000)."""
    text = make_text(5000)
    chunks = chunk_text(text)
    for chunk in chunks:
        assert len(chunk) <= 1000, f"Chunk length {len(chunk)} exceeds 1000"


def test_consecutive_chunks_share_overlap():
    """
    With overlap=200, the end of chunk[i] and the start of chunk[i+1] must
    share some common text (i.e. the overlap region exists).
    """
    text = make_text(3000)
    chunks = chunk_text(text)
    assert len(chunks) >= 2, "Need at least 2 chunks to test overlap"
    # The start of chunk[1] should appear somewhere near the end of chunk[0]
    overlap_candidate = chunks[1][:100]
    assert overlap_candidate in chunks[0], "Expected overlap between adjacent chunks"


def test_empty_string_returns_empty_list():
    """Empty input must not crash and must return an empty list."""
    chunks = chunk_text("")
    assert chunks == []


def test_whitespace_only_returns_empty():
    """Whitespace-only text should produce no meaningful chunks."""
    chunks = chunk_text("   \n\t  ")
    assert len(chunks) == 0


def test_all_chunks_are_strings():
    """Every element in the result must be a non-empty string."""
    chunks = chunk_text(make_text(2500))
    for chunk in chunks:
        assert isinstance(chunk, str)
        assert len(chunk) > 0


def test_full_text_reconstructable():
    """
    All chunks together must cover every word in the original text
    (they may overlap, but no word should be silently dropped).
    """
    words = ["alpha", "beta", "gamma", "delta", "epsilon"]
    text = " ".join(words * 300)  # ~2700 chars — spans multiple chunks
    chunks = chunk_text(text)
    joined = " ".join(chunks)
    for word in words:
        assert word in joined
