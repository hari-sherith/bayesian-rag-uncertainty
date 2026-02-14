import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax, log_softmax


class TemperatureScaling:
    """Post-hoc temperature scaling for neural net calibration.

    The idea: divide logits by a scalar T before softmax.
    T > 1 softens the distribution (less overconfident),
    T < 1 sharpens it. We find T that minimizes NLL on a val set.
    """

    def __init__(self):
        self.temperature = 1.0
        self._fitted = False

    def fit(self, logits: np.ndarray, labels: np.ndarray, bounds=(0.1, 10.0)):
        """Find optimal temperature on a validation set.

        Args:
            logits: (N, C) array of raw logits, N samples, C classes
            labels: (N,) integer class labels
        """
        logits = np.asarray(logits, dtype=float)
        labels = np.asarray(labels, dtype=int)

        def nll(T):
            # scale logits and compute log-softmax
            scaled = log_softmax(logits / T, axis=1)
            # pick the log-prob of the true class for each sample
            per_sample = scaled[np.arange(len(labels)), labels]
            return -per_sample.mean()

        result = minimize_scalar(nll, bounds=bounds, method="bounded")
        self.temperature = float(result.x)
        self._fitted = True
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling: softmax(logits / T)."""
        logits = np.asarray(logits, dtype=float)
        return softmax(logits / self.temperature, axis=-1)

    def __repr__(self):
        status = f"T={self.temperature:.4f}" if self._fitted else "unfitted"
        return f"TemperatureScaling({status})"


class PlattScaling:
    """Platt scaling — fit a logistic regression on top of scores.

    Simpler than temperature scaling, works on 1D confidence scores
    rather than full logit vectors. Good for binary calibration.
    """

    def __init__(self):
        self.a = 1.0
        self.b = 0.0
        self._fitted = False

    def _sigmoid(self, x):
        # clamp to avoid overflow
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def fit(self, scores: np.ndarray, labels: np.ndarray):
        """Fit a * score + b to minimize log loss.

        Args:
            scores: (N,) raw model scores or confidences
            labels: (N,) binary labels (0 or 1)
        """
        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels, dtype=float)

        def neg_log_likelihood(params):
            a, b = params
            p = self._sigmoid(a * scores + b)
            p = np.clip(p, 1e-10, 1 - 1e-10)
            ll = labels * np.log(p) + (1 - labels) * np.log(1 - p)
            return -ll.mean()

        from scipy.optimize import minimize
        result = minimize(
            neg_log_likelihood,
            x0=[1.0, 0.0],
            method="Nelder-Mead",
        )
        self.a, self.b = result.x
        self._fitted = True
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Apply Platt scaling to get calibrated probabilities."""
        scores = np.asarray(scores, dtype=float)
        return self._sigmoid(self.a * scores + self.b)

    def __repr__(self):
        status = f"a={self.a:.4f}, b={self.b:.4f}" if self._fitted else "unfitted"
        return f"PlattScaling({status})"
