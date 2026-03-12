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