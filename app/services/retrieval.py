import numpy as np


def retrieve_chunks(query, embedding_model, index, chunks_store, top_k=10, filters=None):

    # Step 1: embed query
    query_embedding = embedding_model.encode([query])

    # If filtering, we need to fetch a larger pool from FAISS to ensure 
    # we get enough matches from the specifically requested documents.
    fetch_k = top_k * 5 if filters else top_k

    # Step 2: vector search
    distances, indices = index.search(np.array(query_embedding), fetch_k)

    retrieved_chunks = []

    for i in indices[0]:
        chunk = chunks_store[i]

        # Apply strict file filtering
        if filters:
            source = chunk["metadata"]["source"]
            if source not in filters:
                continue

        retrieved_chunks.append(chunk)

        # Stop early if we have enough chunks after filtering
        if len(retrieved_chunks) >= top_k:
            break

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