import hashlib
import os
from pathlib import Path
from typing import List

import numpy as np
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer


class EmbeddingManager:
    """Manages document and query embeddings with disk caching."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: str = "data/embeddings",
    ):
        self.model = SentenceTransformer(model_name)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _content_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed_documents(self, documents: List[Document]) -> np.ndarray:
        """Embed a list of documents, using cached embeddings when available."""
        embeddings = []
        texts_to_embed = []
        indices_to_embed = []

        for i, doc in enumerate(documents):
            h = self._content_hash(doc.page_content)
            cache_path = self.cache_dir / f"{h}.npy"
            if cache_path.exists():
                embeddings.append(np.load(cache_path))
            else:
                embeddings.append(None)
                texts_to_embed.append(doc.page_content)
                indices_to_embed.append(i)

        if texts_to_embed:
            new_embeddings = self.model.encode(texts_to_embed, show_progress_bar=False)
            for idx, emb in zip(indices_to_embed, new_embeddings):
                emb = np.array(emb, dtype=np.float32)
                embeddings[idx] = emb
                h = self._content_hash(documents[idx].page_content)
                np.save(self.cache_dir / f"{h}.npy", emb)

        return np.vstack(embeddings).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string (no caching)."""
        embedding = self.model.encode([query], show_progress_bar=False)
        return np.array(embedding[0], dtype=np.float32)
