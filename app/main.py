from fastapi import FastAPI
from app.services.llm_service import ask_llm


app = FastAPI(title="AI Knowledge Assistant")


@app.get("/")
def home():
    return {"message": "AI Knowledge Assistant API Running"}


@app.get("/test-llm")
def test_llm():
    response = ask_llm("Explain what a vector database is in just 2 line")
    return {"response": response}


@app.get("/ask-anything")
def ask(question: str):
    response = ask_llm(question)
    return {"answer": response}
