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

def generate_rag_answer(question, retrieved_chunks):

    # Build context from retrieved chunks
    #context = "\n\n".join([c["text"] for c in retrieved_chunks])
    context = "\n\n".join([c["text"] for c in retrieved_chunks[:3]])

    prompt = f"""
You are a helpful assistant answering questions using company documents.

Use ONLY the provided context to answer.

Do NOT include reasoning steps.
Do NOT output <think> tags.

If the answer is not present in the context, reply exactly:
"I could not find this information in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

    # Call LLM
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        timeout=20,
        temperature=0,
        max_tokens=300
    )

    response_text = response.choices[0].message.content

    # Remove reasoning tokens if model emits them
    if "<think>" in response_text:
        response_text = response_text.split("</think>")[-1]

    return response_text.strip()