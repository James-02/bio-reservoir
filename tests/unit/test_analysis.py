"""Unit tests for utils.analysis — Metrics, evaluate_performance, aggregation."""

import pytest
import numpy as np

from utils.analysis import (
    Metrics,
    evaluate_performance,
    compute_mean_dicts,
    compute_class_dicts,
    compute_mean_metrics,
    count_labels,
    measure_dataset_deviation,
    _is_class_key,
)


# ---------------------------------------------------------------------------
# _is_class_key
# ---------------------------------------------------------------------------

class TestIsClassKey:
    def test_digit_strings(self):
        assert _is_class_key("0")
        assert _is_class_key("42")

    def test_non_digit_strings(self):
        assert not _is_class_key("accuracy")
        assert not _is_class_key("macro avg")
        assert not _is_class_key("weighted avg")

    def test_non_string(self):
        assert not _is_class_key(0)
        assert not _is_class_key(None)


# ---------------------------------------------------------------------------
# Metrics dataclass
# ---------------------------------------------------------------------------

class TestMetrics:
    @pytest.fixture
    def sample_metrics(self):
        return Metrics(
            f1=0.9,
            accuracy=0.85,
            precision=0.88,
            recall=0.87,
            mcc=0.75,
            kappa=0.74,
            runtime=1.5,
            confusion_matrix=np.array([[10, 2], [1, 12]]),
            class_metrics={"0": {"precision": 0.83, "recall": 0.91}, "1": {"precision": 0.92, "recall": 0.86}},
        )

    def test_to_dict_roundtrip(self, sample_metrics):
        d = sample_metrics.to_dict()
        restored = Metrics.from_dict(d)
        assert restored.f1 == pytest.approx(0.9)
        assert restored.kappa == pytest.approx(0.74)

    def test_from_dict_ignores_unknown_keys(self):
        d = dict(f1=0.9, accuracy=0.8, precision=0.8, recall=0.8,
                 mcc=0.7, kappa=0.7, runtime=1.0,
                 confusion_matrix=np.eye(2), class_metrics={},
                 UNKNOWN_KEY="should be ignored")
        m = Metrics.from_dict(d)
        assert m.f1 == pytest.approx(0.9)

    def test_from_dict_defaults_mcc_kappa(self):
        """Legacy artefacts without mcc/kappa get NaN defaults."""
        d = dict(f1=0.9, accuracy=0.8, precision=0.8, recall=0.8,
                 runtime=1.0, confusion_matrix=np.eye(2), class_metrics={})
        m = Metrics.from_dict(d)
        assert np.isnan(m.mcc)
        assert np.isnan(m.kappa)

    def test_dict_style_access(self, sample_metrics):
        assert sample_metrics["f1"] == 0.9
        assert sample_metrics.get("f1", 0.0) == 0.9
        assert sample_metrics.get("nonexistent", -1) == -1
        assert "accuracy" in sample_metrics
        assert "bogus" not in sample_metrics

    def test_keys_values_items(self, sample_metrics):
        keys = list(sample_metrics.keys())
        assert "f1" in keys
        assert "confusion_matrix" in keys
        items = list(sample_metrics.items())
        assert len(items) == len(keys)

    def test_getitem_raises_key_error(self, sample_metrics):
        with pytest.raises(KeyError):
            _ = sample_metrics["nonexistent"]


# ---------------------------------------------------------------------------
# evaluate_performance
# ---------------------------------------------------------------------------

class TestEvaluatePerformance:
    def test_perfect_prediction(self):
        Y_true = np.eye(3)[[0, 1, 2, 0, 1, 2]]  # (6, 3) one-hot
        Y_pred = Y_true.copy()
        m = evaluate_performance(Y_true, Y_pred, time=2.0)
        assert m.f1 == pytest.approx(1.0)
        assert m.accuracy == pytest.approx(1.0)
        assert m.mcc == pytest.approx(1.0)
        assert m.kappa == pytest.approx(1.0)
        assert m.runtime == pytest.approx(2.0)

    def test_random_prediction_lower_scores(self):
        rng = np.random.default_rng(0)
        Y_true = np.eye(3)[rng.integers(0, 3, size=100)]
        Y_pred = np.eye(3)[rng.integers(0, 3, size=100)]
        m = evaluate_performance(Y_true, Y_pred)
        assert m.f1 < 0.9  # random should be bad
        assert m.confusion_matrix.shape == (3, 3)

    def test_class_metrics_contains_per_class(self):
        Y_true = np.eye(2)[[0, 0, 1, 1]]
        Y_pred = np.eye(2)[[0, 1, 1, 1]]
        m = evaluate_performance(Y_true, Y_pred)
        assert "0" in m.class_metrics
        assert "1" in m.class_metrics


# ---------------------------------------------------------------------------
# compute_mean_dicts / compute_class_dicts
# ---------------------------------------------------------------------------

class TestComputeMeanDicts:
    def test_simple_average(self):
        d1 = {"0": {"p": 0.8, "r": 0.7}, "1": {"p": 0.9, "r": 0.6}}
        d2 = {"0": {"p": 0.6, "r": 0.9}, "1": {"p": 0.7, "r": 0.8}}
        result = compute_mean_dicts([d1, d2])
        assert result["0"]["p"] == pytest.approx(0.7)
        assert result["1"]["r"] == pytest.approx(0.7)

    def test_ignores_non_class_keys(self):
        d = {"0": {"p": 1.0}, "macro avg": {"p": 0.9}, "weighted avg": {"p": 0.95}}
        result = compute_mean_dicts([d])
        assert "macro avg" not in result
        assert "0" in result

    def test_empty_list(self):
        assert compute_mean_dicts([]) == {}

    def test_no_class_keys(self):
        d = {"accuracy": 0.9, "macro avg": {"p": 0.8}}
        assert compute_mean_dicts([d]) == {}


class TestComputeClassDicts:
    def test_mean_and_std(self):
        d1 = {"0": {"p": 0.8, "r": 0.6}, "1": {"p": 0.9, "r": 0.7}}
        d2 = {"0": {"p": 0.6, "r": 0.8}, "1": {"p": 0.7, "r": 0.9}}
        result = compute_class_dicts([d1, d2])
        assert result["0"]["p_mean"] == pytest.approx(0.7)
        assert result["0"]["p_std"] == pytest.approx(0.1)
        assert result["1"]["r_mean"] == pytest.approx(0.8)

    def test_empty_input(self):
        assert compute_class_dicts([]) == {}


# ---------------------------------------------------------------------------
# compute_mean_metrics
# ---------------------------------------------------------------------------

class TestComputeMeanMetrics:
    def test_averages_scalars(self):
        m1 = Metrics(f1=0.8, accuracy=0.7, precision=0.75, recall=0.65,
                     mcc=0.6, kappa=0.5, runtime=1.0,
                     confusion_matrix=np.array([[5, 1], [2, 4]]),
                     class_metrics={"0": {"p": 0.8}, "1": {"p": 0.7}})
        m2 = Metrics(f1=0.9, accuracy=0.8, precision=0.85, recall=0.75,
                     mcc=0.7, kappa=0.6, runtime=2.0,
                     confusion_matrix=np.array([[6, 0], [1, 5]]),
                     class_metrics={"0": {"p": 0.9}, "1": {"p": 0.8}})
        avg = compute_mean_metrics([m1, m2])
        assert avg.f1 == pytest.approx(0.85)
        assert avg.runtime == pytest.approx(1.5)

    def test_confusion_matrix_summed(self):
        cm1 = np.array([[5, 1], [2, 4]])
        cm2 = np.array([[6, 0], [1, 5]])
        m1 = Metrics(f1=0.8, accuracy=0.7, precision=0.75, recall=0.65,
                     mcc=0.6, kappa=0.5, runtime=1.0,
                     confusion_matrix=cm1, class_metrics={})
        m2 = Metrics(f1=0.9, accuracy=0.8, precision=0.85, recall=0.75,
                     mcc=0.7, kappa=0.6, runtime=2.0,
                     confusion_matrix=cm2, class_metrics={})
        avg = compute_mean_metrics([m1, m2])
        np.testing.assert_array_equal(avg.confusion_matrix, cm1 + cm2)


# ---------------------------------------------------------------------------
# count_labels
# ---------------------------------------------------------------------------

class TestCountLabels:
    def test_one_hot(self):
        Y = [np.array([[1, 0, 0]]), np.array([[0, 1, 0]]), np.array([[0, 1, 0]])]
        counts = count_labels(Y)
        assert counts[0] == 1
        assert counts[1] == 2

    def test_scalar_labels(self):
        Y = [np.array([0]), np.array([2]), np.array([2]), np.array([1])]
        counts = count_labels(Y)
        assert counts[0] == 1
        assert counts[2] == 2


# ---------------------------------------------------------------------------
# measure_dataset_deviation
# ---------------------------------------------------------------------------

class TestMeasureDatasetDeviation:
    def test_known_data(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        mean, std = measure_dataset_deviation(X)
        np.testing.assert_allclose(mean, [2.0, 3.0])
        np.testing.assert_allclose(std, [1.0, 1.0])
