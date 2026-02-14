import numpy as np
from typing import Dict, List, Tuple


def expected_calibration_error(
    predictions: np.ndarray,
    confidences: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error — the go-to calibration metric.

    Bins predictions by confidence, computes per-bin |accuracy - confidence|,
    returns the weighted average. Lower is better, 0 = perfectly calibrated.
    """
    predictions = np.asarray(predictions)
    confidences = np.asarray(confidences)
    bin_edges = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    n_total = len(predictions)
    if n_total == 0:
        return 0.0

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        # last bin is inclusive on both sides
        if hi == bin_edges[-1]:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        n_bin = mask.sum()
        if n_bin == 0:
            continue

        avg_conf = confidences[mask].mean()
        avg_acc = predictions[mask].mean()
        ece += (n_bin / n_total) * abs(avg_acc - avg_conf)

    return float(ece)


def maximum_calibration_error(
    predictions: np.ndarray,
    confidences: np.ndarray,
    n_bins: int = 10,
) -> float:
    """MCE = worst-case bin calibration gap. Catches local miscalibration."""
    predictions = np.asarray(predictions)
    confidences = np.asarray(confidences)
    bin_edges = np.linspace(0, 1, n_bins + 1)

    mce = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        if hi == bin_edges[-1]:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        if mask.sum() == 0:
            continue

        avg_conf = confidences[mask].mean()
        avg_acc = predictions[mask].mean()
        mce = max(mce, abs(avg_acc - avg_conf))

    return float(mce)


def brier_score(probabilities: np.ndarray, targets: np.ndarray) -> float:
    """Brier score: mean squared error between predicted probs and outcomes.

    BS = (1/N) * sum((p_i - y_i)^2)
    Range [0, 1], lower is better. Decomposes into calibration + resolution + uncertainty
    but we just return the raw score here.
    """
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(targets, dtype=float)
    return float(np.mean((p - y) ** 2))


def reliability_diagram(
    predictions: np.ndarray,
    confidences: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, List]:
    """Compute bin-level data for plotting a reliability diagram.

    Returns dict with lists: bin_midpoints, bin_accuracies, bin_confidences,
    bin_counts. Empty bins get NaN for accuracy/confidence.
    """
    predictions = np.asarray(predictions)
    confidences = np.asarray(confidences)
    bin_edges = np.linspace(0, 1, n_bins + 1)

    midpoints = []
    accuracies = []
    avg_confs = []
    counts = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mid = (lo + hi) / 2
        midpoints.append(float(mid))

        if hi == bin_edges[-1]:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        n_bin = mask.sum()
        counts.append(int(n_bin))

        if n_bin == 0:
            accuracies.append(float("nan"))
            avg_confs.append(float("nan"))
        else:
            accuracies.append(float(predictions[mask].mean()))
            avg_confs.append(float(confidences[mask].mean()))

    return {
        "bin_midpoints": midpoints,
        "bin_accuracies": accuracies,
        "bin_confidences": avg_confs,
        "bin_counts": counts,
        "bin_edges": bin_edges.tolist(),
    }
