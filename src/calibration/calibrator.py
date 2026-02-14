import numpy as np
from typing import Dict, List, Optional

from .metrics import expected_calibration_error, maximum_calibration_error, brier_score, reliability_diagram
from .temperature_scaling import TemperatureScaling, PlattScaling


class RAGCalibrator:
    """Calibration wrapper for the full RAG pipeline.

    Collects confidence scores from retrieval and generation,
    compares against ground truth, and applies post-hoc calibration.
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.retrieval_scaler = PlattScaling()
        self.generation_scaler = TemperatureScaling()
        self._retrieval_report = None
        self._generation_report = None

    def calibrate_retrieval(
        self,
        confidences: np.ndarray,
        relevance_labels: np.ndarray,
    ) -> Dict:
        """Calibrate retrieval confidence scores against ground truth relevance.

        Args:
            confidences: (N,) posterior means from BayesianRetriever
            relevance_labels: (N,) binary — 1 if doc was actually relevant
        """
        confidences = np.asarray(confidences, dtype=float)
        relevance_labels = np.asarray(relevance_labels, dtype=float)

        # metrics before calibration
        preds_binary = (confidences >= 0.5).astype(float)
        ece_before = expected_calibration_error(relevance_labels, confidences, self.n_bins)
        mce_before = maximum_calibration_error(relevance_labels, confidences, self.n_bins)
        bs_before = brier_score(confidences, relevance_labels)
        diagram_before = reliability_diagram(relevance_labels, confidences, self.n_bins)

        # fit platt scaling (binary task)
        self.retrieval_scaler.fit(confidences, relevance_labels)
        calibrated = self.retrieval_scaler.transform(confidences)

        # metrics after
        ece_after = expected_calibration_error(relevance_labels, calibrated, self.n_bins)
        mce_after = maximum_calibration_error(relevance_labels, calibrated, self.n_bins)
        bs_after = brier_score(calibrated, relevance_labels)
        diagram_after = reliability_diagram(relevance_labels, calibrated, self.n_bins)

        self._retrieval_report = {
            "before": {
                "ece": ece_before, "mce": mce_before, "brier": bs_before,
                "diagram": diagram_before,
            },
            "after": {
                "ece": ece_after, "mce": mce_after, "brier": bs_after,
                "diagram": diagram_after,
            },
            "scaler": repr(self.retrieval_scaler),
        }
        return self._retrieval_report

    def calibrate_generation(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
    ) -> Dict:
        """Calibrate generation model using temperature scaling.

        Args:
            logits: (N, C) raw logits from the language model
            labels: (N,) integer labels — index of the correct next token
        """
        logits = np.asarray(logits, dtype=float)
        labels = np.asarray(labels, dtype=int)

        # before calibration: take argmax confidence
        from scipy.special import softmax
        probs_before = softmax(logits, axis=1)
        conf_before = probs_before.max(axis=1)
        correct_before = (probs_before.argmax(axis=1) == labels).astype(float)

        ece_before = expected_calibration_error(correct_before, conf_before, self.n_bins)
        mce_before = maximum_calibration_error(correct_before, conf_before, self.n_bins)
        diagram_before = reliability_diagram(correct_before, conf_before, self.n_bins)

        # fit temperature scaling
        self.generation_scaler.fit(logits, labels)
        probs_after = self.generation_scaler.transform(logits)
        conf_after = probs_after.max(axis=1)
        correct_after = (probs_after.argmax(axis=1) == labels).astype(float)

        ece_after = expected_calibration_error(correct_after, conf_after, self.n_bins)
        mce_after = maximum_calibration_error(correct_after, conf_after, self.n_bins)
        diagram_after = reliability_diagram(correct_after, conf_after, self.n_bins)

        self._generation_report = {
            "before": {
                "ece": ece_before, "mce": mce_before,
                "diagram": diagram_before,
            },
            "after": {
                "ece": ece_after, "mce": mce_after,
                "diagram": diagram_after,
            },
            "temperature": self.generation_scaler.temperature,
            "scaler": repr(self.generation_scaler),
        }
        return self._generation_report

    def get_calibration_report(self) -> Dict:
        """Combined report for both retrieval and generation calibration."""
        return {
            "retrieval": self._retrieval_report,
            "generation": self._generation_report,
        }
