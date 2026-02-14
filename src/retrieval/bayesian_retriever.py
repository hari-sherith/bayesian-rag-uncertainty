from typing import Dict, List

import numpy as np
from scipy.special import betaln, digamma
from scipy.stats import beta as beta_dist

from .document_loader import DocumentLoader
from .embeddings import EmbeddingManager
from .vector_store import VectorStore


class BayesianRetriever:
    """Retriever that maintains a Beta posterior over document relevance.

    Instead of raw similarity scores, computes calibrated uncertainty metrics
    via conjugate Beta-Bernoulli updates.
    """

    def __init__(
        self,
        document_loader: DocumentLoader,
        embedding_manager: EmbeddingManager,
        vector_store: VectorStore,
        prior_alpha: float = 2.0,
        prior_beta: float = 2.0,
    ):
        self.document_loader = document_loader
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.documents = []

    def index_documents(self, data_dir: str) -> None:
        """Load, chunk, embed, and index documents from a directory."""
        raw_docs = self.document_loader.load_documents(data_dir)
        self.documents = self.document_loader.chunk_documents(raw_docs)
        embeddings = self.embedding_manager.embed_documents(self.documents)
        self.vector_store.add_embeddings(embeddings)

    @staticmethod
    def _similarity_to_observation(cosine_sim: float) -> float:
        """Map cosine similarity from [-1, 1] to [0, 1]."""
        return (cosine_sim + 1.0) / 2.0

    def _compute_posterior(self, cosine_sim: float) -> Dict[str, float]:
        """Compute Beta posterior parameters and uncertainty metrics."""
        s = self._similarity_to_observation(cosine_sim)

        # Posterior update: Beta(alpha + s, beta + (1 - s))
        a = self.prior_alpha + s
        b = self.prior_beta + (1.0 - s)

        # Posterior mean and variance
        mean = a / (a + b)
        variance = (a * b) / ((a + b) ** 2 * (a + b + 1))

        # Entropy of the Beta distribution
        entropy = (
            betaln(a, b)
            - (a - 1) * digamma(a)
            - (b - 1) * digamma(b)
            + (a + b - 2) * digamma(a + b)
        )

        # 95% credible interval
        ci_lower = float(beta_dist.ppf(0.025, a, b))
        ci_upper = float(beta_dist.ppf(0.975, a, b))

        return {
            "similarity": float(cosine_sim),
            "posterior_mean": float(mean),
            "posterior_std": float(np.sqrt(variance)),
            "entropy": float(entropy),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    def get_relevant_docs(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve documents with Bayesian uncertainty estimates."""
        query_embedding = self.embedding_manager.embed_query(query)
        similarities, indices = self.vector_store.search(query_embedding, top_k)

        results = []
        for sim, idx in zip(similarities, indices):
            if idx == -1:
                continue
            posterior = self._compute_posterior(float(sim))
            posterior["document"] = self.documents[idx]
            results.append(posterior)

        return results
