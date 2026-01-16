import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def build_faiss_index(chunks):
    """
    Takes a list of semantic text chunks and builds a FAISS index.

    Returns:
    - faiss_index: FAISS index containing embeddings
    - embeddings: numpy array of embeddings
    """

    # Step 1: Generate embeddings
    embeddings = embedding_model.encode(
        chunks,
        show_progress_bar=False
    )

    embeddings = np.array(embeddings).astype("float32")

    # Step 2: Create FAISS index
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)

    # Step 3: Add embeddings to index
    faiss_index.add(embeddings)

    return faiss_index, embeddings