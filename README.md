# AI Knowledge Assistant

Enterprise-style Retrieval-Augmented Generation (RAG) system that allows users to ask questions about documents.

The system ingests PDFs, converts them into embeddings, stores them in a vector database, and retrieves relevant context to generate accurate answers.

---

## Features

- Document ingestion (PDF)
- Text chunking and embedding
- Semantic search using vector similarity
- Context-aware question answering
- FastAPI backend API

---

## Tech Stack

- Python
- FastAPI
- Sentence Transformers
- FAISS
- LangChain
- Uvicorn

---

## Architecture

User Question
↓
FastAPI API
↓
Embedding Model
↓
Vector Database (FAISS)
↓
Retrieve Relevant Context
↓
LLM Generates Answer

---

## Project Structure

ai-knowledge-assistant
│
├── app
│ ├── api
│ ├── services
│ ├── utils
│ ├── prompts
│ ├── config.py
│ └── main.py
│
├── requirements.txt
├── README.md
└── .gitignore


---

## Installation

Clone repository:


git clone https://github.com/neeraliacharya/ai-knowledge-assistant.git

cd ai-knowledge-assistant


Create virtual environment:


python -m venv venv
source venv/bin/activate


Install dependencies:


pip install -r requirements.txt


---

## Run the API


uvicorn app.main:app --reload


Server runs at:


http://127.0.0.1:8000


API documentation:


http://127.0.0.1:8000/docs


---

## Future Improvements

- Streaming responses
- Support for multiple document formats
- Persistent vector database
- UI interface

---

## Author

Neerali Acharya