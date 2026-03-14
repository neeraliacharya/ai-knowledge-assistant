import numpy as np


def retrieve_chunks(query, embedding_model, index, chunks_store, top_k=10):

    # Step 1: embed query
    query_embedding = embedding_model.encode([query])

    # Step 2: vector search
    distances, indices = index.search(np.array(query_embedding), top_k)

    retrieved_chunks = []

    for i in indices[0]:
        retrieved_chunks.append(chunks_store[i])

    # Step 3: keyword fallback filter
    query_words = query.lower().split()

    keyword_matches = []

    for chunk in retrieved_chunks:

        text = chunk["text"].lower()

        if any(word in text for word in query_words):
            keyword_matches.append(chunk)

    # Step 4: prefer keyword matches if they exist
    if keyword_matches:
        return keyword_matches[:3]

    # fallback to vector results
    return retrieved_chunks[:3]