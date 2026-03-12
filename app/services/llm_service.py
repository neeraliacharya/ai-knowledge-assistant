from groq import Groq
from app.config import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)

def ask_llm(prompt: str):

    response = client.chat.completions.create(
        model="qwen/qwen3-32b",
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
        model="qwen/qwen3-32b",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

def generate_rag_answer(question, context_chunks):

    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a helpful assistant answering questions using company documents.

Answer ONLY using the provided context.

Do NOT explain your reasoning.
Do NOT include thinking steps.

Return ONLY the final answer in one short paragraph.

If the answer is not present in the context, respond exactly with:
"I could not find this information in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="qwen/qwen3-32b",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()