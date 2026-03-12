import numpy as np


def retrieve_chunks(query, embedding_model, index, chunks, top_k=3):

    query_embedding = embedding_model.encode([query])

    distances, indices = index.search(np.array(query_embedding), top_k)

    results = []

    for i in indices[0]:
        results.append(chunks[i])

    return results