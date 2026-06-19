from sentence_transformers import CrossEncoder

reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_chunks(query, chunks, top_k=5):

    pairs = [(query, chunk["text"]) for chunk in chunks]

    scores = reranker_model.predict(pairs)

    ranked = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [item[0] for item in ranked[:top_k]]