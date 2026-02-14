import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.special import betaln, digamma
from scipy.stats import beta as beta_dist

from src.retrieval.bayesian_retriever import BayesianRetriever
from src.retrieval.document_loader import DocumentLoader
from src.retrieval.embeddings import EmbeddingManager
from src.retrieval.vector_store import VectorStore


# ── Beta math tests ──────────────────────────────────────────────────────────


class TestBetaMath:
    """Test the Bayesian posterior update formulas."""

    def test_similarity_to_observation_bounds(self):
        assert BayesianRetriever._similarity_to_observation(-1.0) == pytest.approx(0.0)
        assert BayesianRetriever._similarity_to_observation(0.0) == pytest.approx(0.5)
        assert BayesianRetriever._similarity_to_observation(1.0) == pytest.approx(1.0)

    def test_posterior_mean_in_unit_interval(self):
        retriever = _make_bare_retriever()
        for sim in [-0.5, 0.0, 0.3, 0.7, 1.0]:
            result = retriever._compute_posterior(sim)
            assert 0.0 < result["posterior_mean"] < 1.0

    def test_higher_similarity_gives_higher_posterior(self):
        retriever = _make_bare_retriever()
        low = retriever._compute_posterior(0.2)
        high = retriever._compute_posterior(0.9)
        assert high["posterior_mean"] > low["posterior_mean"]

    def test_entropy_is_finite(self):
        """Differential entropy of Beta can be negative; verify it's finite and matches scipy."""
        retriever = _make_bare_retriever()
        for sim in [-0.5, 0.0, 0.5, 1.0]:
            result = retriever._compute_posterior(sim)
            assert np.isfinite(result["entropy"])
            # Verify against scipy
            s = (sim + 1.0) / 2.0
            a, b = 2.0 + s, 2.0 + (1.0 - s)
            assert result["entropy"] == pytest.approx(beta_dist.entropy(a, b), abs=1e-10)

    def test_credible_interval_contains_mean(self):
        retriever = _make_bare_retriever()
        for sim in [-0.5, 0.0, 0.5, 1.0]:
            r = retriever._compute_posterior(sim)
            assert r["ci_lower"] <= r["posterior_mean"] <= r["ci_upper"]

    def test_credible_interval_bounds(self):
        retriever = _make_bare_retriever()
        r = retriever._compute_posterior(0.5)
        assert 0.0 <= r["ci_lower"]
        assert r["ci_upper"] <= 1.0

    def test_posterior_std_is_positive(self):
        retriever = _make_bare_retriever()
        r = retriever._compute_posterior(0.5)
        assert r["posterior_std"] > 0

    def test_posterior_formulas_match_scipy(self):
        """Verify our manual formulas against scipy Beta distribution."""
        retriever = _make_bare_retriever()
        sim = 0.6
        s = (sim + 1.0) / 2.0
        a = 2.0 + s
        b = 2.0 + (1.0 - s)

        result = retriever._compute_posterior(sim)

        assert result["posterior_mean"] == pytest.approx(beta_dist.mean(a, b), abs=1e-10)
        assert result["posterior_std"] == pytest.approx(beta_dist.std(a, b), abs=1e-10)
        assert result["ci_lower"] == pytest.approx(beta_dist.ppf(0.025, a, b), abs=1e-10)
        assert result["ci_upper"] == pytest.approx(beta_dist.ppf(0.975, a, b), abs=1e-10)


# ── Document loader tests ────────────────────────────────────────────────────


class TestDocumentLoader:
    def test_load_documents(self, tmp_path):
        (tmp_path / "a.txt").write_text("Hello world")
        (tmp_path / "b.txt").write_text("Foo bar")
        (tmp_path / "c.csv").write_text("not,a,text,file")

        loader = DocumentLoader()
        docs = loader.load_documents(str(tmp_path))

        assert len(docs) == 2
        assert docs[0].metadata["source"].endswith("a.txt")
        assert docs[1].page_content == "Foo bar"

    def test_chunk_documents_metadata(self, tmp_path):
        (tmp_path / "doc.txt").write_text("word " * 300)

        loader = DocumentLoader(chunk_size=100, chunk_overlap=10)
        docs = loader.load_documents(str(tmp_path))
        chunks = loader.chunk_documents(docs)

        assert len(chunks) > 1
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_id"] == i
            assert "source" in chunk.metadata

    def test_empty_directory(self, tmp_path):
        loader = DocumentLoader()
        docs = loader.load_documents(str(tmp_path))
        assert docs == []


# ── Embedding tests ──────────────────────────────────────────────────────────


class TestEmbeddingManager:
    def test_embed_query_shape(self):
        mgr = EmbeddingManager()
        emb = mgr.embed_query("test query")
        assert emb.shape == (384,)
        assert emb.dtype == np.float32

    def test_embed_documents_shape(self, tmp_path):
        from langchain_core.documents import Document

        docs = [
            Document(page_content="Hello world"),
            Document(page_content="Bayesian inference"),
        ]
        mgr = EmbeddingManager(cache_dir=str(tmp_path / "cache"))
        embs = mgr.embed_documents(docs)
        assert embs.shape == (2, 384)

    def test_cache_roundtrip(self, tmp_path):
        from langchain_core.documents import Document

        docs = [Document(page_content="cached content")]
        cache_dir = str(tmp_path / "cache")
        mgr = EmbeddingManager(cache_dir=cache_dir)

        embs1 = mgr.embed_documents(docs)
        # Second call should use cache
        embs2 = mgr.embed_documents(docs)
        np.testing.assert_array_equal(embs1, embs2)

        # Verify cache file exists
        cache_files = list(Path(cache_dir).glob("*.npy"))
        assert len(cache_files) == 1


# ── Vector store tests ────────────────────────────────────────────────────────


class TestVectorStore:
    def test_add_and_search(self):
        store = VectorStore(dimension=4)
        embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=np.float32)
        store.add_embeddings(embeddings)

        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        sims, indices = store.search(query, top_k=2)

        assert indices[0] == 0  # most similar
        assert sims[0] == pytest.approx(1.0, abs=1e-5)

    def test_save_and_load(self, tmp_path):
        store = VectorStore(dimension=4)
        embeddings = np.random.randn(5, 4).astype(np.float32)
        store.add_embeddings(embeddings)

        path = str(tmp_path / "test.index")
        store.save(path)

        store2 = VectorStore()
        store2.load(path)
        assert store2.index.ntotal == 5

    def test_empty_index(self):
        store = VectorStore(dimension=4)
        assert store.index.ntotal == 0


# ── End-to-end test ──────────────────────────────────────────────────────────


class TestEndToEnd:
    def test_full_pipeline(self, tmp_path):
        # Create sample documents
        (tmp_path / "doc1.txt").write_text(
            "Bayesian statistics uses prior distributions and likelihood functions "
            "to compute posterior distributions over parameters of interest."
        )
        (tmp_path / "doc2.txt").write_text(
            "The Python programming language is widely used for web development, "
            "data science, and automation tasks."
        )

        cache_dir = str(tmp_path / "cache")
        loader = DocumentLoader()
        mgr = EmbeddingManager(cache_dir=cache_dir)
        store = VectorStore()
        retriever = BayesianRetriever(loader, mgr, store)

        retriever.index_documents(str(tmp_path))
        results = retriever.get_relevant_docs("What is Bayesian inference?", top_k=2)

        assert len(results) == 2
        for r in results:
            assert "document" in r
            assert 0 < r["posterior_mean"] < 1
            assert np.isfinite(r["entropy"])
            assert r["ci_lower"] <= r["posterior_mean"] <= r["ci_upper"]
            assert r["posterior_std"] > 0

        # The Bayesian doc should rank higher
        assert "Bayesian" in results[0]["document"].page_content


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_bare_retriever(**kwargs) -> BayesianRetriever:
    """Create a BayesianRetriever with dummy dependencies for math-only tests."""
    return BayesianRetriever(
        document_loader=DocumentLoader(),
        embedding_manager=None,
        vector_store=None,
        **kwargs,
    )
