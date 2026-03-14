from sentence_transformers import SentenceTransformer

# load embedding model once
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")


def generate_embeddings(chunks):

    embeddings = embedding_model.encode(chunks)

    return embeddings