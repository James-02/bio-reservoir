"""Unit tests for optimization.baseline (classifiers on the raw beats)."""

import numpy as np
import optuna
import pytest

from optimization import baseline

KNN = {"classifier": "KNN", "knn_n_neighbors": 3, "knn_weights": "uniform", "knn_p": 2.0}


def _pool(n_per_class=20, steps=16, classes=3, seed=0):
    """Noisy sines of a different frequency per class: X (N, T, 1), one-hot Y (N, 1, C)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, steps)
    X = np.concatenate([np.sin((c + 1) * np.pi * t) + 0.6 * rng.normal(size=(n_per_class, steps))
                        for c in range(classes)])[..., None]
    Y = np.eye(classes)[np.repeat(np.arange(classes), n_per_class)][:, None, :]
    return X, Y


@pytest.fixture
def pool(monkeypatch):
    """A 60-beat three-class pool in the dataset cache (as 40 training and 20 test beats)."""
    X, Y = _pool()
    monkeypatch.setattr(baseline, "_DATA_CACHE", {
        (60, False, True, None, "sequence_zscore", 1): (X[:40], Y[:40], X[40:], Y[40:])})
    return X, Y


def _kwargs(**changes):
    kwargs = {"classifiers": ["KNN"], "instances": 60, "binary": False,
              "balance_classes": True, "max_per_class": None, "scaler_type": "sequence_zscore",
              "noise_rate": 0.0, "noise_ratio": 0.0, "folds": 3, "seed": 1,
              "objective_average": "macro"}
    kwargs.update(changes)
    return kwargs


def test_macro_and_weighted_averages_are_stored(pool):
    """The macro and weighted averages (metric_macro_*, metric_weighted_*) were not
    stored, because the helper that averaged the folds kept the class rows only."""
    trial = optuna.trial.FixedTrial(KNN)
    baseline.objective(trial, **_kwargs())
    attrs = trial.user_attrs
    assert attrs["metric_macro_f1"] == pytest.approx(attrs["metric_f1_macro"])
    assert attrs["metric_weighted_f1"] == pytest.approx(attrs["metric_f1_weighted"])
    assert attrs["metric_macro_precision"] == pytest.approx(attrs["metric_precision_macro"])
    assert attrs["metric_weighted_recall"] == pytest.approx(attrs["metric_recall_weighted"])
    assert attrs["metric_macro_support"] == pytest.approx(20.0)
