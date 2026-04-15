from groq import Groq
from app.config import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)

def ask_llm(prompt: str):

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

def generate_llm_thinkinh(question, context_chunks):

    context = "\n\n".join(context_chunks)

    prompt = f"""
You are an AI assistant answering questions based on company documents.

Use ONLY the provided context to answer.

Context:
{context}

Question:
{question}

If the answer is not in the context, say:
"I could not find this information in the provided documents."
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

def reformulate_query(question: str, history: list):
    if not history:
        return question
        
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-4:]])
    
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
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=60
    )
    
    new_query = response.choices[0].message.content.strip()
    if new_query.startswith('"') and new_query.endswith('"'):
        new_query = new_query[1:-1]
        
    return new_query

def generate_rag_answer(question, context, history=None):

    # Build context from retrieved chunks
    #context = "\n\n".join([c["text"] for c in retrieved_chunks])
    
    #context = "\n\n".join([c["text"] for c in retrieved_chunks[:3]])

    prompt = f"""
    You are an AI assistant helping a user extract information from or summarize their uploaded documents.

    If the user's input is a simple greeting (like 'hi', 'hello', 'hey'), respond politely as an AI Knowledge Assistant and ask how you can help them with their documents. DO NOT use the context to answer greetings.

    For queries or commands (e.g. "summarize"), use the information in the context below to fulfill the request.
    Return a complete and meaningful 1–2 sentence answer.
    If the query is about a person, include their role and experience. Do not just return a name.

    IMPORTANT:
    1. If the user uses a specific pronoun (e.g. 'she', 'her', 'he', 'his'), MUST use that same pronoun in your response.
    2. If the user does not use a pronoun and the context doesn't explicitly state one, use the person's name or gender-neutral language. DO NOT assume gender based on job roles or names.
    3. If the user asks a specific factual question and the context does not contain the answer, say EXACTLY:
    "I cannot find the answer to this in the uploaded documents. Please provide a more specific prompt so I can find the right information for you."
    
    HOWEVER, if the user asks for a summary or overview, provide the summary based on the given context without saying the answer cannot be found.

    Context:
    {context}

    Question / Command:
    {question}

    Answer:
    """

    # Call LLM
    messages = []
    if history:
        messages.extend(history[-6:])  # Keep max 6 messages for context window
        
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        timeout=20,
        temperature=0,
        max_tokens=300
    )

    response_text = response.choices[0].message.content

    # Remove reasoning tokens if model emits them
    if "<think>" in response_text:
        response_text = response_text.split("</think>")[-1]

    return response_text