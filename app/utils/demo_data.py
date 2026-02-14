import numpy as np


SAMPLE_QUERIES = [
    "What is Bayesian inference and how does it update beliefs?",
    "How does dropout work as a regularization technique?",
    "Explain how vector space models are used in information retrieval.",
    "What is the difference between supervised and unsupervised learning?",
    "How do transformers use self-attention mechanisms?",
    "What are conjugate priors and why are they useful?",
]


def get_sample_queries():
    """Return the list of sample queries."""
    return SAMPLE_QUERIES


def generate_synthetic_calibration_data(n_samples=200):
    """Create deliberately overconfident synthetic data for the calibration tab.

    Returns dict with retrieval and generation calibration inputs.
    """
    rng = np.random.RandomState(42)

    # --- Retrieval calibration data ---
    # Overconfident: high confidence but only ~50% actually relevant
    retrieval_confidences = rng.beta(5, 2, size=n_samples)  # skewed high
    # Ground truth: relevant with probability proportional to sqrt(confidence)
    # so the model is systematically overconfident
    true_relevance_prob = np.sqrt(retrieval_confidences) * 0.6
    relevance_labels = rng.binomial(1, true_relevance_prob).astype(float)

    # --- Generation calibration data ---
    n_classes = 50  # small vocab slice for demo
    # Create overconfident logits: one class gets a large spike
    logits = rng.randn(n_samples, n_classes).astype(float)
    # Make one logit per sample artificially high -> overconfident
    dominant_class = rng.randint(0, n_classes, size=n_samples)
    for i in range(n_samples):
        logits[i, dominant_class[i]] += 3.0  # push confidence up

    # True labels: the dominant class is correct only ~60% of the time
    labels = dominant_class.copy()
    flip_mask = rng.rand(n_samples) > 0.6
    labels[flip_mask] = rng.randint(0, n_classes, size=flip_mask.sum())

    return {
        "retrieval_confidences": retrieval_confidences,
        "relevance_labels": relevance_labels,
        "generation_logits": logits,
        "generation_labels": labels,
    }
