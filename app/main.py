from fastapi import FastAPI
from app.services.chunking import chunk_text
from app.services.embedding import generate_embeddings, embedding_model
from app.services.file_watcher import start_watcher
from app.services.ingestion import extract_text_from_pdf
from app.services.llm_service import ask_llm, generate_rag_answer
from app.services.rag_pipeline import initialize_rag, vector_index, chunks_store
from app.services.reranker import rerank_chunks
from app.services.retrieval import retrieve_chunks
from app.services.vector_store import create_vector_store
from contextlib import asynccontextmanager
import app.services.rag_pipeline as rag_pipeline

app = FastAPI(title="AI Knowledge Assistant")

observer = None

@asynccontextmanager
async def lifespan(app: FastAPI):

    global observer

    rag_pipeline.initialize_rag()

    observer = start_watcher()

    yield

    if observer:
        observer.stop()
        observer.join()


app = FastAPI(
    title="AI Knowledge Assistant",
    lifespan=lifespan
)

@asynccontextmanager
async def lifespan(app: FastAPI):

    rag_pipeline.initialize_rag()

    global observer
    observer = start_watcher()

    yield

    observer.stop()
    observer.join()

document_path = "documents/NimbusTechKnox_GTM_Strategy.pdf"
# use /docs for all rest end point list


# normal rest
@app.get("/")
def home():
    return {"message": "AI Knowledge Assistant API Running"}


# gives result of the string passed in below method from LLM model
@app.get("/test-llm")
def test_llm():
    response = ask_llm("Explain what a vector database is in just 2 line")
    return {"response": response}


# pass ?question=[your_prompt] at the end of URL to get response for passed prompt
@app.get("/ask-anything")
def ask(question: str):
    response = ask_llm(question)
    return {"answer": response}


# to test vector chunk fetching
@app.get("/test-chunks")
def test_chunks():

    text = extract_text_from_pdf(document_path)

    chunks = chunk_text(text)

    return {"total_chunks": len(chunks), "first_chunk": chunks[0]}


# to test vector embeddings
@app.get("/test-embeddings")
def test_embeddings():

    text = extract_text_from_pdf(document_path)

    chunks = chunk_text(text)

    embeddings = generate_embeddings(chunks)

    index = create_vector_store(embeddings)

    return {"total_chunks": len(chunks), "embedding_dimension": len(embeddings[0])}


# search based on prompt in the given pdf through vector embeddings
@app.get("/search")
def search(question: str):

    text = extract_text_from_pdf(document_path)

    chunks = chunk_text(text)

    embeddings = generate_embeddings(chunks)

    index = create_vector_store(embeddings)

    results = retrieve_chunks(question, embedding_model, index, chunks)

    return {"question": question, "retrieved_chunks": results}


@app.get("/ping")
def ping():
    print("Ping called")
    return {"status": "alive"}

@app.get("/ask")
def ask(question: str):

    retrieved_chunks = retrieve_chunks(
        question,
        embedding_model,
        rag_pipeline.vector_index,
        rag_pipeline.chunks_store,
        top_k=10
    )

    reranked_chunks = rerank_chunks(question, retrieved_chunks, top_k=3)

    answer = generate_rag_answer(question, reranked_chunks)

    sources = list(set([c["metadata"]["source"] for c in reranked_chunks]))[:2]

    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }

@app.get("/debug-durgesh")
def debug_durgesh():

    import app.services.rag_pipeline as rag_pipeline

    matches = []

    for chunk in rag_pipeline.chunks_store:
        if "durgesh" in chunk["text"].lower():
            matches.append(chunk["text"])

    return {
        "matches_found": len(matches),
        "examples": matches[:3]
    }