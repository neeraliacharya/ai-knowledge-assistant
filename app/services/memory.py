"""
In-memory chat session store with thread safety.

Production change: a threading.Lock() protects all reads and writes so that
concurrent requests for different session IDs don't corrupt each other's history.
get_history() returns a shallow copy of the list so callers can't accidentally
mutate the stored history from outside this module.
"""
import threading

_chat_sessions: dict[str, list[dict]] = {}
_lock = threading.Lock()


def get_history(session_id: str) -> list[dict]:
    with _lock:
        if session_id not in _chat_sessions:
            _chat_sessions[session_id] = []
        return list(_chat_sessions[session_id])  # copy, not reference


def add_message(session_id: str, role: str, content: str) -> None:
    with _lock:
        if session_id not in _chat_sessions:
            _chat_sessions[session_id] = []
        _chat_sessions[session_id].append({"role": role, "content": content})
        # Keep the last 10 messages (5 pairs) to limit context window usage
        if len(_chat_sessions[session_id]) > 10:
            _chat_sessions[session_id] = _chat_sessions[session_id][-10:]


def clear_session(session_id: str) -> None:
    with _lock:
        _chat_sessions.pop(session_id, None)
