"""
LLM service — Groq integration.

Production changes
-------------------
- LLM_MODEL and LLM_MAX_TOKENS are now read from config (env vars) instead of
  being hardcoded. This lets you switch models or raise the token limit without
  a code change.
- All print() replaced with structured logging.
"""
from groq import Groq

from app.config import GROQ_API_KEY, LLM_MODEL, LLM_MAX_TOKENS
from app.services.logger import get_logger

log = get_logger(__name__)
client = Groq(api_key=GROQ_API_KEY)


def ask_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def reformulate_query(question: str, history: list) -> str:
    if not history:
        return question

    history_text = "\n".join(
        f"{msg['role']}: {msg['content']}" for msg in history[-4:]
    )

    prompt = f"""You are a query rewriting assistant.
Rewrite the latest 'Follow-up Question' into a standalone search query using context from the 'Conversation History'.
Resolve any pronouns (he, she, it, they, this, that) to their specific entities from the history.
If the question is already fully self-contained or is just a greeting/simple command, return it exactly as is.
Do not answer the question. Do not provide any conversational filler. Return ONLY the final standalone query string.

Conversation History:
{history_text}

Follow-up Question:
{question}
"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=60,
    )

    new_query = response.choices[0].message.content.strip().strip('"')
    log.info("Query reformulated", extra={"original": question, "reformulated": new_query})
    return new_query


def generate_rag_answer(question: str, context: str, history: list | None = None) -> str:
    prompt = f"""
    You are an AI assistant helping users find and understand information from their uploaded documents.

    If the user's input is a simple greeting (like 'hi', 'hello', 'hey'), respond politely as an AI Knowledge Assistant and ask how you can help them with their documents. DO NOT use the context to answer greetings.

    For all other queries, use ONLY the information in the context below. Answer every part of the question fully and directly — do not truncate or omit details that appear in the context.

    RULES:
    1. Address every aspect of the question. If the question has multiple parts, answer each part.
    2. If the user uses a specific pronoun (e.g. 'she', 'her', 'he', 'his'), use that same pronoun in your response.
    3. If the context does not state a pronoun, use the person's name or gender-neutral language. DO NOT assume gender from names or job titles.
    4. If the query is about a person, include their role, experience, and any other relevant details present in the context.
    5. If the user asks a specific factual question and the answer is genuinely not in the context, say EXACTLY:
    "I cannot find the answer to this in the uploaded documents. Please provide a more specific prompt so I can find the right information for you."
    6. If the user asks for a summary or overview, provide a thorough summary covering all key points in the context.

    Context:
    {context}

    Question / Command:
    {question}

    Answer:
    """

    messages: list[dict] = []
    if history:
        messages.extend(history[-6:])
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        timeout=20,
        temperature=0,
        max_tokens=LLM_MAX_TOKENS,
    )

    text = response.choices[0].message.content
    if "<think>" in text:
        text = text.split("</think>")[-1]

    return text.strip()
