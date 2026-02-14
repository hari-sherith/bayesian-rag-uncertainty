import torch
import torch.nn.functional as F


class UncertaintyEstimator:
    """Computes uncertainty metrics from MC Dropout logits.

    All the core math lives here: predictive entropy, expected entropy,
    mutual information, plus some simple text-diversity stuff.
    """

    def __init__(self, eps=1e-10):
        self.eps = eps  # for numerical stability in log

    def predictive_entropy(self, logits):
        """Total uncertainty via entropy of the mean predictive distribution.

        Args:
            logits: (N, seq_len, vocab_size) from N forward passes
        Returns:
            (seq_len,) tensor of per-token predictive entropy
        """
        # average the softmax across samples -> p_bar
        probs = F.softmax(logits, dim=-1)
        p_bar = probs.mean(dim=0)  # (seq_len, vocab)

        # H = -sum(p * log(p))
        log_p = torch.log(p_bar + self.eps)
        h = -(p_bar * log_p).sum(dim=-1)
        return h

    def expected_entropy(self, logits):
        """Aleatoric uncertainty: average entropy of individual samples.

        This is the uncertainty you'd have even with infinite data --
        irreducible noise from the model's own softmax distributions.
        """
        probs = F.softmax(logits, dim=-1)  # (N, seq_len, vocab)
        log_p = torch.log(probs + self.eps)

        # per-sample entropy, then average
        h_per_sample = -(probs * log_p).sum(dim=-1)  # (N, seq_len)
        return h_per_sample.mean(dim=0)  # (seq_len,)

    def mutual_information(self, logits):
        """Epistemic uncertainty: MI = H_predictive - H_expected.

        Should always be >= 0 by Jensen's inequality. We clamp just in case
        of floating point weirdness.
        """
        h_pred = self.predictive_entropy(logits)
        h_exp = self.expected_entropy(logits)
        mi = h_pred - h_exp
        return torch.clamp(mi, min=0.0)

    def sequence_entropy(self, logits):
        """Mean of token-level predictive entropies. Single number summary."""
        return self.predictive_entropy(logits).mean().item()

    def sequence_mi(self, logits):
        """Mean mutual information across tokens."""
        return self.mutual_information(logits).mean().item()

    # -- text diversity metrics --

    def unique_ratio(self, samples):
        """Fraction of unique strings in the sample set."""
        if not samples:
            return 0.0
        n = len(samples)
        n_unique = len(set(samples))
        # subtract 1 from both to get ratio of *distinct* beyond the first
        # actually just keep it simple
        return n_unique / n

    def pairwise_diversity(self, samples):
        """Average pairwise lexical difference (0 = all identical, 1 = all different).

        Uses simple token-level jaccard distance. Not fancy but does the job.
        """
        if len(samples) < 2:
            return 0.0

        dists = []
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                s1 = set(samples[i].split())
                s2 = set(samples[j].split())
                union = s1 | s2
                if not union:
                    dists.append(0.0)
                    continue
                jaccard = len(s1 & s2) / len(union)
                dists.append(1.0 - jaccard)  # distance, not similarity

        return sum(dists) / len(dists)

    def compute_all(self, logits, samples):
        """One-stop shop: compute everything and return a dict.

        Args:
            logits: (N, seq_len, vocab_size) tensor
            samples: list of N generated text strings
        """
        return {
            "sequence_entropy": self.sequence_entropy(logits),
            "mean_mi": self.sequence_mi(logits),
            "predictive_entropy": self.predictive_entropy(logits).tolist(),
            "expected_entropy": self.expected_entropy(logits).tolist(),
            "mutual_information": self.mutual_information(logits).tolist(),
            "unique_ratio": self.unique_ratio(samples),
            "pairwise_diversity": self.pairwise_diversity(samples),
        }
