"""Integration tests for the cross-validation pipeline with the biological reservoir.

These run a full (small) cross-validation end to end; they are slower than unit tests
and kept in ``integration/`` so they can be run selectively with::

    pytest tests/integration/
"""

import importlib

import numpy as np
import pytest
from reservoirpy.nodes import Ridge

from bioreservoir import BioReservoir

# training/__init__.py re-exports a function named cross_validate, which hides the module.
cv = importlib.import_module("training.cross_validate")
SECTIONS = ("train_reservoir", "fit_readout", "val_reservoir", "predict")


@pytest.fixture(autouse=True)
def _keep_global_state(monkeypatch):
    """run_fold configures logging and thread limits for worker processes."""
    monkeypatch.setattr(cv, "setup_logging", lambda **kwargs: None)
    monkeypatch.setattr(cv, "_limit_native_threads", lambda: None)


def _run(**kwargs):
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 1.0, 12)
    X = np.concatenate([t + 0.3 * rng.normal(size=(10, 12)),
                        1.0 - t + 0.3 * rng.normal(size=(10, 12))])[..., None]
    Y = np.eye(2)[np.repeat([0, 1], 10)][:, None, :]
    reservoir = BioReservoir(units=3, warmup=0, rk4_substeps=16, seed=1, input_dim=1,
                             output_variables="He", parallel=False)
    return cv.cross_validate(reservoir, Ridge(ridge=1e-6), X, Y, folds=2, seed=3,
                             primary_average="macro", **kwargs)


def test_cross_validate_two_folds():
    """cross_validate() completes on a tiny dataset with two folds."""
    metrics = _run()
    assert {"f1", "accuracy", "precision", "recall", "mcc", "kappa", "confusion_matrix",
            "class_metrics", "folds"} <= set(metrics)
    assert [fold["fold_index"] for fold in metrics["folds"]] == [0, 1]
    assert np.asarray(metrics["confusion_matrix"]).sum() == 20
    assert metrics["f1"] == pytest.approx(np.mean([fold["f1"] for fold in metrics["folds"]]))
    assert "profiling" not in metrics


def test_cross_validate_profile_flag():
    """cross_validate(profile=True) times each stage of every fold."""
    metrics = _run(profile=True)
    assert set(SECTIONS) <= set(metrics["profiling"])
    assert all(metrics["profiling"][name]["total_s"] > 0 for name in SECTIONS)
    assert len(metrics["fold_runtimes"]["per_fold"]) == 2
