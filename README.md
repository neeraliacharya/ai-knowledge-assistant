# AI Knowledge Assistant

A professional-grade Retrieval-Augmented Generation (RAG) platform that transforms your documents into an interactive knowledge base. Built for high-performance semantic search and context-aware conversations.

---

## 🚀 Key Features

- **🧠 Conversational Memory**: Remembers previous questions and context. Automatically reformulates follow-up queries for accurate information retrieval.
- **📄 Advanced RAG Pipeline**: Uses state-of-the-art chunking, embedding, and **Reranking** to ensure the most relevant information is retrieved.
- **☁️ Cloud Storage Integration**: Seamlessly syncs with **AWS S3** for persistent document storage and automatic local synchronization.
- **🔍 Precision Search**: Includes semantic search capabilities with source tracking—cite exactly which document provides the answer.
- **⚡ Real-time Indexing**: Automatically re-indexes the vector store whenever you upload or delete a document.
- **💻 Modern UI**: A sleek, responsive dashboard built with React and Vite featuring dedicated Chat and Document Management tabs.

---

## 🛠 Tech Stack

- **Large Language Model**: [Groq](https://groq.com/) (Llama-3.1-8b-instant)
- **Backend**: Python 3.13, FastAPI, FAISS (Vector DB), Sentence Transformers
- **Frontend**: React 18, Vite, Lucide React
- **Infrastructure**: AWS S3, Boto3

---

## 🏗 Architecture

1. **Ingestion**: Documents are uploaded $\rightarrow$ Saved to S3 $\rightarrow$ Synced locally.
2. **Indexing**: Documents are chunked $\rightarrow$ Embedded $\rightarrow$ Stored in FAISS vector index.
3. **Retrieval**: User query reformulated (Memory) $\rightarrow$ Vector search $\rightarrow$ Reranking.
4. **Generation**: Top context + Conversation history $\rightarrow$ Groq LLM $\rightarrow$ Answer.

---

## ⚙️ Setup Instructions

### 1. Prerequisite Environments
Create a `.env` file in the root directory:
```env
ANTHROPIC_API_KEY=your_key
GEMINI_API_KEY=your_key
GROQ_API_KEY=your_key
HF_TOKEN=your_key

# AWS Config
AWS_ACCESS_KEY_ID=your_id
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-east-1
S3_BUCKET_NAME=your_bucket
```

### 2. Backend Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn app.main:app --reload --port 8000
```
*API docs available at: `http://localhost:8000/docs`*

### 3. Frontend Setup
```bash
cd ui
npm install
npm run dev
```
*UI available at: `http://localhost:5173`*

---

## 📁 Project Structure

```text
├── app/
│   ├── services/      # RAG logic, S3, Memory, LLM integration
│   ├── config.py      # Environment configuration
│   └── main.py        # FastAPI endpoints
├── ui/                # React/Vite Frontend
├── venv/              # Python virtual environment
├── .env               # Secrets (Ignored)
└── README.md          # This file
```

---

## 👤 Author

**Neerali Acharya**
