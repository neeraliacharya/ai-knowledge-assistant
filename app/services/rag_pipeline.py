from app.services.document_loader import load_documents
from app.services.chunking import chunk_text
from app.services.embedding import generate_embeddings
from app.services.vector_store import create_vector_store

chunks_store = []
vector_index = None


def initialize_rag():

    global chunks_store
    global vector_index

    documents = load_documents()

    all_chunks = []
    metadata = []

    for doc in documents:

        chunks = chunk_text(doc["text"])

        for chunk in chunks:

            all_chunks.append(chunk)

            metadata.append({
                "source": doc["source"]
            })

    chunks_store = [
        {"text": chunk, "metadata": meta}
        for chunk, meta in zip(all_chunks, metadata)
    ]

    if not all_chunks:
        print("No documents found. Vector DB cleared.")
        vector_index = None
        return

    embeddings = generate_embeddings(all_chunks)

    vector_index = create_vector_store(embeddings)

    print(f"RAG initialized with {len(all_chunks)} chunks")