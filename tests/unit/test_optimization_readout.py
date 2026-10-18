"""Unit tests for optimization.readout."""

import numpy as np
import optuna
import pytest

from optimization import readout
from utils.analysis import evaluate_performance
from utils.results import FoldArtifact

RIDGE = {"classifier": "Ridge", "ridge_alpha": 1.0, "ridge_fit_intercept": True,
         "ridge_tol": 1e-3, "ridge_solver": "auto"}


@pytest.fixture
def saved_states(tmp_path, monkeypatch):
    """Two folds of saved states for an imbalanced two-class problem, in a temporary
    results folder."""
    monkeypatch.setenv("BIORESERVOIR_RESULTS_DIR", str(tmp_path))
    rng = np.random.default_rng(0)

    def part(n):
        labels = np.repeat([0, 1, 1], n)
        states = rng.normal(size=(len(labels), 4)) + 1.5 * labels[:, None]
        return states, np.eye(2)[labels][:, None, :]

    for k in range(2):
        train_states, Y_train = part(12)
        test_states, Y_test = part(6)
        FoldArtifact(trial_name="synthetic", fold_index=k, Y_train=Y_train, Y_test=Y_test,
                     Y_pred=Y_test, metrics=evaluate_performance(Y_test, Y_test, 0.0),
                     train_states=train_states, test_states=test_states,
                     ).save(save_states=True)
    return "synthetic"


def test_readout_studies_optimise_macro_f1_by_default(saved_states):
    """Without --primary_average the readout objective used to optimise weighted F1; the
    reported metric (and every archived readout study) is macro F1."""
    trial = optuna.trial.FixedTrial(RIDGE)
    value = readout.objective(trial, classifiers=["Ridge"], trial_name=saved_states)
    assert trial.user_attrs["metric_primary_average"] == "macro"
    assert value == pytest.approx(trial.user_attrs["metric_f1_macro"])
    assert value != pytest.approx(trial.user_attrs["metric_f1_weighted"])
