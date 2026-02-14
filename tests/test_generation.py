import torch
import torch.nn as nn
import numpy as np
import pytest

from src.generation.mc_dropout import MCDropoutModel
from src.generation.uncertainty import UncertaintyEstimator
from src.generation.generator import UncertainGenerator


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def mc_model():
    """Load model once for the whole test module -- it's slow."""
    return MCDropoutModel(model_name="distilgpt2")


@pytest.fixture
def estimator():
    return UncertaintyEstimator()


# ── MC Dropout model tests ───────────────────────────────────────────────────


class TestMCDropout:
    def test_model_loads(self, mc_model):
        assert mc_model.model is not None
        assert mc_model.tokenizer is not None

    def test_has_dropout_layers(self, mc_model):
        dropout_layers = [m for m in mc_model.model.modules() if isinstance(m, nn.Dropout)]
        assert len(dropout_layers) > 0, "distilgpt2 should have dropout layers"

    def test_enable_dropout_sets_train(self, mc_model):
        mc_model._enable_dropout()
        for m in mc_model.model.modules():
            if isinstance(m, nn.Dropout):
                assert m.training, "Dropout should be in train mode"
        mc_model._disable_dropout()

    def test_disable_dropout_sets_eval(self, mc_model):
        mc_model._disable_dropout()
        for m in mc_model.model.modules():
            if isinstance(m, nn.Dropout):
                assert not m.training

    def test_forward_pass_shape(self, mc_model):
        tokens = mc_model.tokenizer("hello world", return_tensors="pt")["input_ids"]
        n = 3
        logits = mc_model.forward_pass(tokens, n_samples=n)
        seq_len = tokens.shape[1]
        vocab_size = mc_model.model.config.vocab_size
        assert logits.shape == (n, seq_len, vocab_size)

    def test_samples_differ_with_dropout(self, mc_model):
        """With dropout on, different forward passes should give different logits."""
        tokens = mc_model.tokenizer("The cat sat on", return_tensors="pt")["input_ids"]
        logits = mc_model.forward_pass(tokens, n_samples=5)
        # check that not all samples are identical
        diffs = (logits[0] - logits[1]).abs().sum().item()
        # could be 0 if dropout rate is 0, but distilgpt2 default is 0.1
        # so this should almost certainly pass
        assert diffs > 0, "Expected different logits from different dropout masks"

    def test_generate_returns_texts(self, mc_model):
        result = mc_model.generate("Once upon a", n_samples=2, max_new_tokens=10)
        assert len(result["texts"]) == 2
        assert all(isinstance(t, str) for t in result["texts"])


# ── Uncertainty math tests ───────────────────────────────────────────────────


class TestUncertaintyMath:
    def test_predictive_entropy_nonnegative(self, estimator):
        # random logits
        logits = torch.randn(5, 4, 100)
        h = estimator.predictive_entropy(logits)
        assert (h >= 0).all()

    def test_expected_entropy_nonnegative(self, estimator):
        logits = torch.randn(5, 4, 100)
        h = estimator.expected_entropy(logits)
        assert (h >= 0).all()

    def test_mi_nonnegative(self, estimator):
        logits = torch.randn(8, 3, 50)
        mi = estimator.mutual_information(logits)
        assert (mi >= 0).all()

    def test_mi_leq_predictive_entropy(self, estimator):
        logits = torch.randn(8, 3, 50)
        h_pred = estimator.predictive_entropy(logits)
        mi = estimator.mutual_information(logits)
        # MI should be <= H_pred (it's H_pred - H_expected, and H_expected >= 0)
        assert (mi <= h_pred + 1e-5).all()

    def test_sequence_entropy_is_mean_of_tokens(self, estimator):
        logits = torch.randn(5, 6, 100)
        token_h = estimator.predictive_entropy(logits)
        seq_h = estimator.sequence_entropy(logits)
        assert abs(seq_h - token_h.mean().item()) < 1e-5

    def test_uniform_distribution_high_entropy(self, estimator):
        """Uniform logits should give near-maximum entropy."""
        vocab = 50
        # all zeros -> uniform softmax
        logits = torch.zeros(3, 2, vocab)
        h = estimator.predictive_entropy(logits)
        max_entropy = np.log(vocab)
        assert (h > max_entropy * 0.99).all()

    def test_peaked_distribution_low_entropy(self, estimator):
        """Very peaked logits should give low entropy."""
        vocab = 50
        logits = torch.full((3, 2, vocab), -100.0)
        logits[:, :, 0] = 100.0  # one token dominates
        h = estimator.predictive_entropy(logits)
        assert (h < 0.01).all()

    def test_identical_samples_zero_mi(self, estimator):
        """If all MC samples give the same logits, MI should be ~0."""
        base = torch.randn(1, 4, 30)
        logits = base.expand(10, -1, -1)  # repeat, no variance
        mi = estimator.mutual_information(logits)
        assert mi.max().item() < 1e-5

    def test_compute_all_keys(self, estimator):
        logits = torch.randn(3, 2, 20)
        samples = ["hello world", "goodbye moon", "foo bar"]
        result = estimator.compute_all(logits, samples)
        expected_keys = {
            "sequence_entropy", "mean_mi", "predictive_entropy",
            "expected_entropy", "mutual_information",
            "unique_ratio", "pairwise_diversity",
        }
        assert set(result.keys()) == expected_keys


# ── Diversity tests ──────────────────────────────────────────────────────────


class TestDiversity:
    def test_identical_samples_low_unique_ratio(self, estimator):
        samples = ["same text"] * 10
        assert estimator.unique_ratio(samples) == pytest.approx(0.1)

    def test_distinct_samples_high_unique_ratio(self, estimator):
        samples = [f"text number {i}" for i in range(10)]
        assert estimator.unique_ratio(samples) == pytest.approx(1.0)

    def test_identical_pairwise_zero(self, estimator):
        samples = ["the cat sat"] * 5
        assert estimator.pairwise_diversity(samples) == pytest.approx(0.0)

    def test_different_pairwise_positive(self, estimator):
        samples = ["the cat sat on the mat", "a dog ran in the park", "fish swim in water"]
        div = estimator.pairwise_diversity(samples)
        assert div > 0.0

    def test_empty_samples(self, estimator):
        assert estimator.unique_ratio([]) == 0.0
        assert estimator.pairwise_diversity([]) == 0.0


# ── Prompt formatting tests ─────────────────────────────────────────────────


class TestPromptFormatting:
    def test_basic_prompt_structure(self):
        gen = UncertainGenerator.__new__(UncertainGenerator)
        gen.estimator = UncertaintyEstimator()
        docs = [
            {"document": "Paris is the capital of France."},
            {"document": "Berlin is the capital of Germany."},
        ]
        prompt = gen._build_prompt("What is the capital of France?", docs)
        assert "Context:" in prompt
        assert "Question: What is the capital of France?" in prompt
        assert "Answer:" in prompt
        assert "[1]" in prompt
        assert "[2]" in prompt

    def test_truncation(self):
        gen = UncertainGenerator.__new__(UncertainGenerator)
        gen.estimator = UncertaintyEstimator()
        long_text = "x" * 1000
        docs = [{"document": long_text}]
        prompt = gen._build_prompt("test?", docs, max_doc_chars=100)
        # should be truncated with ...
        assert "..." in prompt
        # the doc portion shouldn't contain all 1000 chars
        assert len(prompt) < 1000

    def test_fallback_to_text_key(self):
        gen = UncertainGenerator.__new__(UncertainGenerator)
        gen.estimator = UncertaintyEstimator()
        docs = [{"text": "some content"}]
        prompt = gen._build_prompt("q?", docs)
        assert "some content" in prompt


# ── Retrieval summary tests ─────────────────────────────────────────────────


class TestRetrievalSummary:
    def test_summarize_with_stats(self):
        gen = UncertainGenerator.__new__(UncertainGenerator)
        docs = [
            {"posterior_std": 0.1, "entropy": 0.5},
            {"posterior_std": 0.3, "entropy": 0.7},
        ]
        summary = gen._summarize_retrieval(docs)
        assert summary["mean_posterior_std"] == pytest.approx(0.2)
        assert summary["mean_entropy"] == pytest.approx(0.6)

    def test_summarize_empty(self):
        gen = UncertainGenerator.__new__(UncertainGenerator)
        summary = gen._summarize_retrieval([])
        assert summary["mean_posterior_std"] is None
        assert summary["mean_entropy"] is None


# ── Integration test (slow, needs model download) ───────────────────────────


@pytest.mark.slow
class TestIntegration:
    def test_end_to_end(self, mc_model, estimator):
        """Full pipeline: generate with uncertainty from fake retrieval docs."""
        gen = UncertainGenerator(
            mc_model=mc_model, estimator=estimator,
            default_samples=3, max_new_tokens=20,
        )
        fake_docs = [
            {
                "document": "The Eiffel Tower is in Paris, France.",
                "posterior_std": 0.12,
                "entropy": 0.45,
            },
            {
                "document": "Paris has a population of about 2 million.",
                "posterior_std": 0.08,
                "entropy": 0.32,
            },
        ]
        result = gen.generate("Where is the Eiffel Tower?", fake_docs, n_samples=3)

        # check structure
        assert "samples" in result
        assert len(result["samples"]) == 3
        assert "uncertainty" in result
        assert "retrieval_uncertainty" in result

        unc = result["uncertainty"]
        assert "sequence_entropy" in unc
        assert "mean_mi" in unc
        assert "unique_ratio" in unc
        assert unc["sequence_entropy"] >= 0
        assert unc["mean_mi"] >= 0

        ret_unc = result["retrieval_uncertainty"]
        assert ret_unc["mean_posterior_std"] is not None
        assert ret_unc["mean_entropy"] is not None
