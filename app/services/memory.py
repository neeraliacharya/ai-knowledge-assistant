# In-memory session store for chat memory
chat_sessions = {}

def get_history(session_id: str):
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    return chat_sessions[session_id]

def add_message(session_id: str, role: str, content: str):
    history = get_history(session_id)
    history.append({"role": role, "content": content})
    # Keep only the last 10 messages (5 user, 5 assistant) to prevent context bloat
    if len(history) > 10:
        chat_sessions[session_id] = history[-10:]

def clear_session(session_id: str):
    if session_id in chat_sessions:
        chat_sessions[session_id] = []
