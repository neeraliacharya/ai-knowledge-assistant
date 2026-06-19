"""
Unit tests for app/services/memory.py

Tests in-memory session store: CRUD, max-size enforcement, isolation between
sessions, and thread safety under concurrent writes.
"""
import threading
import pytest
from app.services.memory import get_history, add_message, clear_session

pytestmark = pytest.mark.unit


# ── Helpers ────────────────────────────────────────────────────────────────────

def unique_session():
    """Return a globally unique session ID to prevent test interference."""
    import uuid
    return f"test-{uuid.uuid4()}"


# ── Basic CRUD ─────────────────────────────────────────────────────────────────

def test_new_session_is_empty():
    sid = unique_session()
    assert get_history(sid) == []


def test_add_and_retrieve_user_message():
    sid = unique_session()
    add_message(sid, "user", "Hello there")
    history = get_history(sid)
    assert len(history) == 1
    assert history[0] == {"role": "user", "content": "Hello there"}


def test_add_multiple_messages_preserves_order():
    sid = unique_session()
    add_message(sid, "user",      "What is the fee?")
    add_message(sid, "assistant", "The fee is INR 150,000.")
    add_message(sid, "user",      "And the payment terms?")
    history = get_history(sid)
    assert len(history) == 3
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    assert history[2]["content"] == "And the payment terms?"


def test_clear_session_empties_history():
    sid = unique_session()
    add_message(sid, "user", "question")
    add_message(sid, "assistant", "answer")
    clear_session(sid)
    assert get_history(sid) == []


def test_clear_nonexistent_session_does_not_raise():
    """Clearing a session that was never created must not raise."""
    clear_session("nonexistent-session-xyz")


# ── Max-size enforcement ───────────────────────────────────────────────────────

def test_max_10_messages_enforced():
    """
    Adding more than 10 messages must keep only the last 10.
    This prevents unbounded memory growth in long conversations.
    """
    sid = unique_session()
    for i in range(15):
        role = "user" if i % 2 == 0 else "assistant"
        add_message(sid, role, f"message {i}")
    history = get_history(sid)
    assert len(history) == 10


def test_oldest_messages_dropped_when_overflow():
    """When the buffer overflows, the OLDEST messages are dropped."""
    sid = unique_session()
    for i in range(12):
        add_message(sid, "user", f"message {i}")
    history = get_history(sid)
    contents = [m["content"] for m in history]
    assert "message 0"  not in contents, "Oldest message should have been dropped"
    assert "message 11" in contents,     "Newest message must still be present"


# ── Session isolation ──────────────────────────────────────────────────────────

def test_sessions_are_isolated():
    """Messages added to session A must not appear in session B."""
    sid_a = unique_session()
    sid_b = unique_session()
    add_message(sid_a, "user", "message for A only")
    add_message(sid_b, "user", "message for B only")
    assert all(m["content"] != "message for A only" for m in get_history(sid_b))
    assert all(m["content"] != "message for B only" for m in get_history(sid_a))


def test_get_history_returns_copy_not_reference():
    """
    Mutating the returned list must not affect the stored history.
    This guards against accidental external modification of internal state.
    """
    sid = unique_session()
    add_message(sid, "user", "original")
    history = get_history(sid)
    history.append({"role": "user", "content": "injected"})
    assert len(get_history(sid)) == 1, "External mutation should not affect stored history"


# ── Thread safety ──────────────────────────────────────────────────────────────

def test_concurrent_writes_do_not_corrupt_state():
    """
    50 threads simultaneously adding messages to different sessions must not
    raise exceptions or corrupt each session's message count.
    """
    session_ids = [unique_session() for _ in range(50)]
    errors = []

    def write_messages(sid):
        try:
            for i in range(5):
                add_message(sid, "user", f"concurrent message {i}")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=write_messages, args=(sid,)) for sid in session_ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Thread errors: {errors}"

    for sid in session_ids:
        hist = get_history(sid)
        assert len(hist) == 5, f"Session {sid} has {len(hist)} messages, expected 5"
