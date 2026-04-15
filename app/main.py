import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from app.services.chunking import chunk_text
from app.services.embedding import generate_embeddings, embedding_model
from app.services.ingestion import extract_text_from_pdf
from app.services.llm_service import ask_llm, generate_rag_answer
from app.services.rag_pipeline import initialize_rag, vector_index, chunks_store
from app.services.reranker import rerank_chunks
from app.services.retrieval import retrieve_chunks
from app.services.vector_store import create_vector_store
from app.services.s3_storage import (
    upload_to_s3,
    delete_from_s3,
    list_s3_documents,
    sync_s3_to_local,
)
from contextlib import asynccontextmanager
import app.services.rag_pipeline as rag_pipeline

app = FastAPI(title="AI Knowledge Assistant")

document_path = "documents/NimbusTechKnox_GTM_Strategy.pdf"
DOCUMENTS_FOLDER  = "documents/"
# use /docs for all rest end point list


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Sync documents from S3 to local folder before RAG init
    sync_s3_to_local(DOCUMENTS_FOLDER)

    rag_pipeline.initialize_rag()

    yield


app = FastAPI(
    title="AI Knowledge Assistant",
    lifespan=lifespan
)



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

from app.services.memory import get_history, add_message
from app.services.llm_service import reformulate_query

@app.get("/ask")
def ask(question: str, session_id: str = Query("default"), files: list[str] = Query(None)):

    history = get_history(session_id)
    search_query = reformulate_query(question, history)

    retrieved_chunks = retrieve_chunks(
        search_query,
        embedding_model,
        rag_pipeline.vector_index,
        rag_pipeline.chunks_store,
        top_k=10,
        filters=files
    )

    reranked_chunks = rerank_chunks(search_query, retrieved_chunks)

    # group chunks by document
    doc_chunks = {}

    for chunk in reranked_chunks:
        source = chunk["metadata"]["source"]

        if source not in doc_chunks:
            doc_chunks[source] = []

        doc_chunks[source].append(chunk["text"])

    if not doc_chunks:
        answer = generate_rag_answer(question, "", history)
        sources = []
    else:
        # pick top document
        top_doc = list(doc_chunks.keys())[0]

        # take multiple chunks from same document
        context = "\n\n".join(doc_chunks[top_doc][:4])  # increase to 4–5

        answer = generate_rag_answer(question, context, history)
        sources = [top_doc]

    # Save to memory
    add_message(session_id, "user", question)
    add_message(session_id, "assistant", answer)

    return {
        "question": question,
        "search_query": search_query,
        "answer": answer,
        "sources": sources
    }

@app.get("/documents")
def list_documents():
    """List all PDF documents stored in S3."""
    return list_s3_documents()

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF to local storage and S3, then re-initialise the RAG pipeline."""

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    try:
        os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)
        local_path = os.path.join(DOCUMENTS_FOLDER, file.filename)

        # Save locally first (needed for RAG ingestion)
        with open(local_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Upload to S3
        success = upload_to_s3(local_path, file.filename)
        if not success:
            raise HTTPException(status_code=500, detail="S3 upload failed")

        # Re-ingest so the new document is searchable immediately
        rag_pipeline.initialize_rag()

        return {"message": f"{file.filename} uploaded successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{filename}")
def delete_document(filename: str):
    """Delete a PDF from both S3 and local storage, then re-initialise the RAG pipeline."""

    try:
        # Delete from S3
        s3_success = delete_from_s3(filename)
        if not s3_success:
            raise HTTPException(status_code=404, detail="File not found in S3")

        # Remove local copy if it exists
        local_path = os.path.join(DOCUMENTS_FOLDER, filename)
        if os.path.isfile(local_path):
            os.remove(local_path)

        # Re-ingest so deleted doc is removed from vector store
        rag_pipeline.initialize_rag()

        return {"message": f"{filename} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))