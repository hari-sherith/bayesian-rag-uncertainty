from pathlib import Path
from typing import Tuple

import faiss
import numpy as np


class VectorStore:
    """FAISS-based vector store using inner product on L2-normalized vectors (cosine similarity)."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        embeddings = np.array(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        faiss.normalize_L2(embeddings)
        return embeddings

    def add_embeddings(self, embeddings: np.ndarray) -> None:
        """Normalize and add embeddings to the index."""
        normed = self._normalize(embeddings.copy())
        self.index.add(normed)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Search the index and return (similarities, indices)."""
        query = self._normalize(query_embedding.copy())
        similarities, indices = self.index.search(query, top_k)
        return similarities[0], indices[0]

    def save(self, path: str) -> None:
        """Save the FAISS index to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    def load(self, path: str) -> None:
        """Load a FAISS index from disk."""
        self.index = faiss.read_index(str(path))
        self.dimension = self.index.d
