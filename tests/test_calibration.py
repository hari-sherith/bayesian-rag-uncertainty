import numpy as np
import pytest
from scipy.special import softmax

from src.calibration.metrics import (
    expected_calibration_error,
    maximum_calibration_error,
    brier_score,
    reliability_diagram,
)
from src.calibration.temperature_scaling import TemperatureScaling, PlattScaling
from src.calibration.calibrator import RAGCalibrator


# ── ECE tests ────────────────────────────────────────────────────────────────


class TestECE:
    def test_perfectly_calibrated(self):
        """If confidence == accuracy in every bin, ECE should be ~0."""
        np.random.seed(42)
        n = 1000
        confs = np.random.uniform(0, 1, n)
        # generate outcomes that match confidence (bernoulli with p=conf)
        preds = (np.random.uniform(0, 1, n) < confs).astype(float)
        ece = expected_calibration_error(preds, confs, n_bins=10)
        # won't be exactly 0 due to finite samples, but should be small
        assert ece < 0.05

    def test_overconfident_high_ece(self):
        """Always confident but often wrong -> high ECE."""
        preds = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=float)
        confs = np.array([0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95])
        ece = expected_calibration_error(preds, confs, n_bins=10)
        # accuracy = 0.5 but confidence = 0.95 -> gap ~ 0.45
        assert ece > 0.3

    def test_ece_bounds(self):
        """ECE should be in [0, 1]."""
        np.random.seed(0)
        for _ in range(20):
            n = 100
            preds = np.random.randint(0, 2, n).astype(float)
            confs = np.random.uniform(0, 1, n)
            ece = expected_calibration_error(preds, confs)
            assert 0 <= ece <= 1

    def test_empty_inputs(self):
        ece = expected_calibration_error(np.array([]), np.array([]))
        assert ece == 0.0


# ── MCE tests ────────────────────────────────────────────────────────────────


class TestMCE:
    def test_mce_geq_ece(self):
        """MCE (max gap) should always be >= ECE (weighted avg gap)."""
        np.random.seed(7)
        preds = np.random.randint(0, 2, 200).astype(float)
        confs = np.random.uniform(0, 1, 200)
        ece = expected_calibration_error(preds, confs)
        mce = maximum_calibration_error(preds, confs)
        assert mce >= ece - 1e-10

    def test_perfectly_calibrated_low_mce(self):
        np.random.seed(42)
        n = 2000
        confs = np.random.uniform(0, 1, n)
        preds = (np.random.uniform(0, 1, n) < confs).astype(float)
        mce = maximum_calibration_error(preds, confs, n_bins=10)
        assert mce < 0.1


# ── Brier score tests ────────────────────────────────────────────────────────


class TestBrierScore:
    def test_perfect_predictions(self):
        bs = brier_score(np.array([1.0, 0.0, 1.0]), np.array([1.0, 0.0, 1.0]))
        assert bs == pytest.approx(0.0)

    def test_worst_predictions(self):
        bs = brier_score(np.array([0.0, 1.0]), np.array([1.0, 0.0]))
        assert bs == pytest.approx(1.0)

    def test_brier_range(self):
        np.random.seed(3)
        p = np.random.uniform(0, 1, 100)
        y = np.random.randint(0, 2, 100).astype(float)
        bs = brier_score(p, y)
        assert 0 <= bs <= 1

    def test_uncertain_predictions(self):
        """Predicting 0.5 for everything -> BS = 0.25."""
        bs = brier_score(np.full(100, 0.5), np.random.randint(0, 2, 100).astype(float))
        assert abs(bs - 0.25) < 0.05  # approx due to randomness


# ── Reliability diagram tests ────────────────────────────────────────────────


class TestReliabilityDiagram:
    def test_correct_number_of_bins(self):
        preds = np.random.randint(0, 2, 50).astype(float)
        confs = np.random.uniform(0, 1, 50)
        diag = reliability_diagram(preds, confs, n_bins=5)
        assert len(diag["bin_midpoints"]) == 5
        assert len(diag["bin_accuracies"]) == 5
        assert len(diag["bin_counts"]) == 5

    def test_counts_sum_to_n(self):
        n = 100
        preds = np.random.randint(0, 2, n).astype(float)
        confs = np.random.uniform(0, 1, n)
        diag = reliability_diagram(preds, confs)
        assert sum(diag["bin_counts"]) == n

    def test_midpoints_are_centered(self):
        diag = reliability_diagram(np.array([1.0]), np.array([0.5]), n_bins=10)
        # first bin midpoint should be 0.05, second 0.15, etc
        assert diag["bin_midpoints"][0] == pytest.approx(0.05)
        assert diag["bin_midpoints"][4] == pytest.approx(0.45)

    def test_empty_bins_have_nan(self):
        # all confidences in one bin
        preds = np.ones(10)
        confs = np.full(10, 0.95)
        diag = reliability_diagram(preds, confs, n_bins=10)
        # most bins should be empty -> NaN
        nan_count = sum(1 for a in diag["bin_accuracies"] if np.isnan(a))
        assert nan_count >= 8


# ── Temperature scaling tests ────────────────────────────────────────────────


class TestTemperatureScaling:
    def _make_overconfident_data(self, n=500, n_classes=5):
        """Create logits that are too sharp (overconfident).

        We make the model right ~70% of the time but with very high confidence,
        so temperature scaling needs to soften (T > 1).
        """
        np.random.seed(42)
        true_labels = np.random.randint(0, n_classes, n)
        # start with small noise
        logits = np.random.randn(n, n_classes) * 0.3
        # spike the correct class with a huge margin -> near-100% softmax confidence
        for i in range(n):
            logits[i, true_labels[i]] += 10.0
        # but make 30% of labels wrong, so accuracy ~ 70% while confidence ~ 100%
        labels = true_labels.copy()
        flip_idx = np.random.choice(n, size=int(n * 0.3), replace=False)
        for i in flip_idx:
            wrong = (labels[i] + 1) % n_classes
            labels[i] = wrong
        return logits, labels

    def test_fit_finds_temperature(self):
        logits, labels = self._make_overconfident_data()
        ts = TemperatureScaling()
        ts.fit(logits, labels)
        # overconfident -> T should be > 1 to soften
        assert ts.temperature > 1.0
        assert ts._fitted

    def test_transform_sums_to_one(self):
        logits = np.random.randn(10, 5)
        ts = TemperatureScaling()
        ts.temperature = 2.0
        probs = ts.transform(logits)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_higher_temp_softer_distribution(self):
        logits = np.array([[3.0, 1.0, 0.5]])
        ts_low = TemperatureScaling()
        ts_low.temperature = 0.5
        ts_high = TemperatureScaling()
        ts_high.temperature = 5.0

        p_low = ts_low.transform(logits)
        p_high = ts_high.transform(logits)
        # higher T -> max prob should be lower (softer)
        assert p_high.max() < p_low.max()

    def test_temperature_scaling_reduces_ece(self):
        """The whole point: after fitting, ECE should go down."""
        logits, labels = self._make_overconfident_data()

        probs_before = softmax(logits, axis=1)
        conf_before = probs_before.max(axis=1)
        correct = (probs_before.argmax(axis=1) == labels).astype(float)
        ece_before = expected_calibration_error(correct, conf_before)

        ts = TemperatureScaling()
        ts.fit(logits, labels)
        probs_after = ts.transform(logits)
        conf_after = probs_after.max(axis=1)
        correct_after = (probs_after.argmax(axis=1) == labels).astype(float)
        ece_after = expected_calibration_error(correct_after, conf_after)

        assert ece_after < ece_before

    def test_repr(self):
        ts = TemperatureScaling()
        assert "unfitted" in repr(ts)
        ts.temperature = 1.5
        ts._fitted = True
        assert "1.5000" in repr(ts)


# ── Platt scaling tests ──────────────────────────────────────────────────────


class TestPlattScaling:
    def test_fit_and_transform(self):
        np.random.seed(0)
        scores = np.concatenate([
            np.random.normal(0.3, 0.1, 50),
            np.random.normal(0.7, 0.1, 50),
        ])
        labels = np.array([0] * 50 + [1] * 50, dtype=float)
        ps = PlattScaling()
        ps.fit(scores, labels)
        calibrated = ps.transform(scores)
        # output should be valid probabilities
        assert (calibrated >= 0).all() and (calibrated <= 1).all()
        assert ps._fitted

    def test_transform_output_range(self):
        ps = PlattScaling()
        ps.a = 2.0
        ps.b = -1.0
        out = ps.transform(np.array([0.0, 0.5, 1.0]))
        assert (out >= 0).all() and (out <= 1).all()


# ── RAGCalibrator tests ──────────────────────────────────────────────────────


class TestRAGCalibrator:
    def test_calibrate_retrieval(self):
        np.random.seed(42)
        cal = RAGCalibrator(n_bins=10)
        confs = np.random.uniform(0.4, 1.0, 200)
        labels = (np.random.uniform(0, 1, 200) < 0.6).astype(float)
        report = cal.calibrate_retrieval(confs, labels)

        assert "before" in report and "after" in report
        assert "ece" in report["before"]
        assert "diagram" in report["after"]

    def test_calibrate_generation(self):
        np.random.seed(42)
        cal = RAGCalibrator()
        n, c = 300, 10
        true_labels = np.random.randint(0, c, n)
        logits = np.random.randn(n, c) * 0.3
        for i in range(n):
            logits[i, true_labels[i]] += 10.0
        # flip 30% of labels so model is overconfident
        labels = true_labels.copy()
        flip_idx = np.random.choice(n, size=int(n * 0.3), replace=False)
        for i in flip_idx:
            labels[i] = (labels[i] + 1) % c

        report = cal.calibrate_generation(logits, labels)
        assert "temperature" in report
        assert report["temperature"] > 1.0  # should soften overconfident logits

    def test_full_report(self):
        np.random.seed(42)
        cal = RAGCalibrator()

        # retrieval
        cal.calibrate_retrieval(
            np.random.uniform(0.3, 0.9, 100),
            np.random.randint(0, 2, 100).astype(float),
        )
        # generation
        logits = np.random.randn(100, 5)
        labels = np.random.randint(0, 5, 100)
        cal.calibrate_generation(logits, labels)

        report = cal.get_calibration_report()
        assert report["retrieval"] is not None
        assert report["generation"] is not None
