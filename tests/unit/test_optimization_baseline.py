"""Unit tests for optimization.baseline (classifiers on the raw beats): the fold
protocol, the objective against a direct scikit-learn computation, and the commands."""

import argparse
import re
import warnings

import numpy as np
import optuna
import pytest
from optuna.distributions import CategoricalDistribution, IntDistribution
from optuna.trial import create_trial
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from optimization import base, baseline

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


class TestFoldHelpers:
    def test_beats_are_flattened(self):
        X = np.arange(12.0).reshape(2, 6, 1)
        assert baseline._flatten_X(X).shape == (2, 6)
        assert baseline._flatten_X(X[..., 0]).shape == (2, 6)
        with pytest.raises(ValueError, match="Unexpected"):
            baseline._flatten_X(np.zeros((2, 6, 2)))

    def test_folds_are_standardised_and_only_training_beats_augmented(self):
        X, Y = _pool()
        train, test = np.arange(0, 60, 2), np.arange(1, 60, 2)
        X_tr, Y_tr, X_va, Y_va = baseline._prepare_fold(X, Y, train, test, 0, noise_rate=0.0,
                                                        noise_ratio=0.0, seed=1)
        assert (X_tr.shape, X_va.shape, len(Y_tr), len(Y_va)) == ((30, 16), (30, 16), 30, 30)
        np.testing.assert_allclose(X_tr.mean(axis=1), 0.0, atol=1e-12)
        np.testing.assert_allclose(X_va.std(axis=1), 1.0)
        X_aug, Y_aug, X_va2, _ = baseline._prepare_fold(X, Y, train, test, 0, noise_rate=0.2,
                                                        noise_ratio=0.5, seed=1)
        assert (len(X_aug), len(Y_aug)) == (45, 45)  # 5 noisy copies per class of 10
        np.testing.assert_array_equal(X_va2, X_va)


def test_the_dataset_is_loaded_once_per_setting(monkeypatch):
    X, Y = _pool()
    calls = []
    monkeypatch.setattr(baseline, "_DATA_CACHE", {})
    monkeypatch.setattr(baseline, "load_ecg_data",
                        lambda **kwargs: calls.append(kwargs) or (X, Y, X, Y))
    for seed in (1, 1, 2):
        baseline._get_dataset(instances=60, binary=False, seed=seed)
    assert [call["seed"] for call in calls] == [1, 2]
    assert (calls[0]["rows"], calls[0]["noise_rate"], calls[0]["preserve_split"]) == (
        60, 0.0, False)


class TestObjective:
    def test_the_score_is_a_stratified_cross_validation_of_the_classifier(self, pool):
        trial = optuna.trial.FixedTrial(KNN)
        value = baseline.objective(trial, **_kwargs())
        X, Y = pool
        labels = np.argmax(Y[:, 0, :], axis=1)
        beats = X[..., 0]
        beats = (beats - beats.mean(axis=1, keepdims=True)) / beats.std(axis=1, keepdims=True)
        expected = []
        for train, test in StratifiedKFold(3, shuffle=True, random_state=1).split(beats, labels):
            model = KNeighborsClassifier(n_neighbors=3, weights="uniform", p=2.0)
            model.fit(beats[train], labels[train])
            expected.append(f1_score(labels[test], model.predict(beats[test]), average="macro"))
        attrs = trial.user_attrs
        assert value == pytest.approx(np.mean(expected), abs=1e-12)
        assert [attrs[f"fold_{k}_f1"] for k in range(3)] == pytest.approx(expected, abs=1e-12)
        assert attrs["metric_f1_std"] == pytest.approx(np.std(expected, ddof=1))
        assert (attrs["dataset_cv_pool_size"], attrs["feature_dim"], attrs["classifier"]) == (
            60, 16, "KNN")
        assert np.asarray(attrs["metric_confusion_matrix"]).sum() == 60
        assert {"metric_class_2_recall_std", "profile_fit_mean_s", "profile_predict_total_s",
                "trial_end_utc"} <= set(attrs)

    def test_larger_pools_are_subsampled_to_the_requested_size(self, pool):
        cache = baseline._DATA_CACHE
        cache[(45, False, True, None, "sequence_zscore", 1)] = cache[
            (60, False, True, None, "sequence_zscore", 1)]
        trial = optuna.trial.FixedTrial(KNN)
        baseline.objective(trial, **_kwargs(instances=45))
        assert trial.user_attrs["dataset_cv_pool_size"] == 45

    def test_trial_times_are_utc_from_a_timezone_aware_clock(self, pool):
        trial = optuna.trial.FixedTrial(KNN)
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*utcnow", category=DeprecationWarning)
            baseline.objective(trial, **_kwargs())
        for key in ("trial_start_utc", "trial_end_utc"):
            assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", trial.user_attrs[key])


class TestCommands:
    @pytest.fixture
    def workspace(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BIORESERVOIR_CONFIG", str(tmp_path / "config.env"))
        monkeypatch.setenv("BIORESERVOIR_RESULTS_DIR", str(tmp_path))
        monkeypatch.setattr(base, "setup_logging", lambda **kwargs: None)
        return tmp_path

    @staticmethod
    def _research_args(**changes):
        args = {"study_name": "B9-custom", "study": None, "storage": None, "seed": 1,
                "classifiers": ["KNN"], "job_id": 0, "processes": 1, "trials": 2,
                "instances": None, "binary": False, "max_per_class": None, "noise_rate": None,
                "noise_ratio": None, "folds": None, "primary_average": None}
        args.update(changes)
        return argparse.Namespace(**args)

    @staticmethod
    def _evaluate_args(**changes):
        args = {"study_name": "B9-test", "study": None, "storage": None, "show": "params,env",
                "classifiers": None, "top_n": 3, "plots": "none", "classifier_plots": None,
                "max_params": 4}
        args.update(changes)
        return argparse.Namespace(**args)

    @pytest.fixture
    def recorded(self, monkeypatch):
        calls = []
        monkeypatch.setattr(baseline, "research", lambda study, trials, objective, **kwargs:
                            calls.append((trials, kwargs)))
        return calls

    def test_command_line_values_win_over_the_study_entry(self, workspace, recorded):
        baseline.cmd_research(self._research_args(
            study_name="B2-unbalanced", instances=900, max_per_class=50, noise_rate=0.1,
            noise_ratio=0.2, folds=3, primary_average="weighted", trials=7, binary=True))
        ((trials, kwargs),) = recorded
        assert trials == 7
        assert (kwargs["instances"], kwargs["max_per_class"], kwargs["binary"],
                kwargs["folds"]) == (900, 50, True, 3)
        assert (kwargs["noise_rate"], kwargs["noise_ratio"], kwargs["objective_average"]) == (
            0.1, 0.2, "weighted")

    def test_unregistered_studies_use_the_defaults(self, workspace, recorded):
        baseline.cmd_research(self._research_args(trials=None))
        ((trials, kwargs),) = recorded
        assert trials == 500
        assert (kwargs["instances"], kwargs["binary"], kwargs["max_per_class"],
                kwargs["folds"], kwargs["objective_average"]) == (4015, False, None, 5, "macro")
        assert (kwargs["noise_rate"], kwargs["noise_ratio"]) == (0.0, 0.0)

    def test_research_runs_the_trials(self, workspace, pool):
        baseline.cmd_research(self._research_args(instances=60, folds=3))
        study = optuna.load_study(study_name="B9-custom",
                                  storage=base.get_default_storage("B9-custom"))
        assert [t.state for t in study.trials] == [optuna.trial.TrialState.COMPLETE] * 2
        assert {t.user_attrs["dataset_cv_pool_size"] for t in study.trials} == {60}

    @pytest.fixture
    def study_db(self, workspace):
        study = optuna.create_study(study_name="B9-test", direction="maximize",
                                    storage=base.get_default_storage("B9-test"))
        classifier = CategoricalDistribution(["KNN", "RF"])
        for value, params, name in [(0.8, {"classifier": "KNN", "knn_n_neighbors": 3},
                                     "knn_n_neighbors"),
                                    (0.9, {"classifier": "RF", "rf_max_depth": 10},
                                     "rf_max_depth")]:
            study.add_trial(create_trial(
                params=params, value=value, user_attrs={"env_python_version": "3.12"},
                distributions={"classifier": classifier, name: IntDistribution(1, 50)}))
        return study

    def test_evaluate_prints_the_tables_and_requests_the_plots(self, study_db, monkeypatch,
                                                               capsys):
        plotted = []
        monkeypatch.setattr(baseline, "plot_results",
                            lambda *args, **kwargs: plotted.append(kwargs["filename"]))
        baseline.cmd_evaluate(self._evaluate_args(
            classifiers=["knn"], plots="slice,importance,edf",
            classifier_plots="slice,importance"))
        out = capsys.readouterr().out
        assert "Environment \u2014 trial 1" in out and "ALL classifiers \u2014 B9-test" in out
        assert "classifier=KNN" in out and "classifier=RF" not in out
        assert "=== Summary table (F1 over top 3 and all trials) ===" in out
        assert plotted == ["B9-test-slice.png", "B9-test-importance.png", "B9-test-edf.png",
                           "B9-test-KNN-slice.png", "B9-test-KNN-importance.png",
                           "B9-test-RF-slice.png", "B9-test-RF-importance.png"]

    def test_evaluate_without_plots_or_a_name(self, study_db, monkeypatch, capsys):
        plotted = []
        monkeypatch.setattr(baseline, "plot_results",
                            lambda *args, **kwargs: plotted.append(kwargs["filename"]))
        baseline.cmd_evaluate(self._evaluate_args())
        assert plotted == [] and "B9-test" in capsys.readouterr().out
        with pytest.raises(SystemExit, match="study_name"):
            baseline.cmd_evaluate(self._evaluate_args(study_name=None))
