from fastapi import FastAPI

app = FastAPI(title="AI Knowledge Assistant")

@app.get("/")
def home():
    return {"message": "AI Knowledge Assistant API Running"}