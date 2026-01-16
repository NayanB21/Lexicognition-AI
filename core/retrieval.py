# core/retrieval.py
from core.embeddings import embedding_model

def retrieve_chunks(query, index, semantic_chunks, top_k=3):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return [semantic_chunks[i] for i in indices[0]]
