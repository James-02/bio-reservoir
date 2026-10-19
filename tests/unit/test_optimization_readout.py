"""Unit tests for optimization.readout: the classifier search space, the readout objective
on saved states, and the research and evaluate commands."""

import argparse

import numpy as np
import optuna
import pandas as pd
import pytest
from optuna.distributions import CategoricalDistribution
from optuna.trial import create_trial
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score

from optimization import base, readout
from utils.analysis import evaluate_performance
from utils.results import FoldArtifact

RIDGE = {"classifier": "Ridge", "ridge_alpha": 1.0, "ridge_fit_intercept": True,
         "ridge_tol": 1e-3, "ridge_solver": "auto"}
CLASSIFIERS = ["Ridge", "Bayes", "LR", "Perceptron", "SVM", "MLP", "KNN", "DT", "RF", "GB"]


@pytest.fixture(autouse=True)
def _keep_logging(monkeypatch):
    monkeypatch.setattr(base, "setup_logging", lambda **kwargs: None)


@pytest.fixture
def saved_states(tmp_path, monkeypatch):
    """Two folds of saved states for an imbalanced two-class problem, in a temporary
    results folder."""
    monkeypatch.setenv("BIORESERVOIR_CONFIG", str(tmp_path / "config.env"))
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


def _labels(Y):
    return np.argmax(Y.reshape(len(Y), -1), axis=1)


def test_readout_studies_optimise_macro_f1_by_default(saved_states):
    """Without --primary_average the readout objective used to optimise weighted F1; the
    reported metric (and every archived readout study) is macro F1."""
    trial = optuna.trial.FixedTrial(RIDGE)
    value = readout.objective(trial, classifiers=["Ridge"], trial_name=saved_states)
    assert trial.user_attrs["metric_primary_average"] == "macro"
    assert value == pytest.approx(trial.user_attrs["metric_f1_macro"])
    assert value != pytest.approx(trial.user_attrs["metric_f1_weighted"])


class TestChooseClassifier:
    @pytest.mark.parametrize("name", CLASSIFIERS)
    def test_every_family_builds_a_classifier_that_fits(self, name):
        trial = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=1)).ask()
        cls, hypers = readout.choose_classifier(trial, name, seed=3)
        rng = np.random.default_rng(0)
        y = np.repeat([0, 1, 2], 10)
        X = rng.normal(size=(30, 4)) + y[:, None]
        assert cls(**hypers).fit(X, y).predict(X).shape == (30,)
        if name not in ("Ridge", "Bayes", "KNN"):
            assert hypers["random_state"] == 3

    def test_a_linear_svm_has_no_gamma_or_degree(self):
        _, hypers = readout.choose_classifier(optuna.trial.FixedTrial(
            {"svc_kernel": "linear", "svc_C": 1.0, "svc_tol": 1e-3}), "SVM")
        assert hypers["gamma"] == "scale" and "degree" not in hypers

    def test_a_polynomial_svm_has_a_degree(self):
        _, hypers = readout.choose_classifier(optuna.trial.FixedTrial(
            {"svc_kernel": "poly", "svc_C": 1.0, "svc_gamma": "auto", "svc_tol": 1e-3,
             "svc_degree": 3}), "SVM")
        assert (hypers["gamma"], hypers["degree"]) == ("auto", 3)

    @pytest.mark.parametrize("binary, losses", [(False, ("log_loss",)),
                                                (True, ("log_loss", "exponential"))])
    def test_the_exponential_loss_is_for_binary_tasks_only(self, binary, losses):
        trial = optuna.create_study().ask()
        readout.choose_classifier(trial, "GB", binary=binary)
        assert trial.distributions["gb_loss"].choices == losses

    def test_unknown_families_are_rejected(self):
        with pytest.raises(ValueError, match="Unknown classifier"):
            readout.choose_classifier(optuna.trial.FixedTrial({}), "XGB")


class TestObjective:
    def test_a_ridge_readout_scores_as_scikit_learn_does(self, saved_states):
        trial = optuna.trial.FixedTrial(RIDGE)
        value = readout.objective(trial, classifiers=["Ridge"], trial_name=saved_states,
                                  study_name="P4-test", job_id=3, seed=5)
        expected = []
        for fold in FoldArtifact.load_trial(saved_states):
            model = RidgeClassifier(alpha=1.0, fit_intercept=True, tol=1e-3, solver="auto",
                                    max_iter=2000)
            model.fit(fold.train_states, _labels(fold.Y_train))
            expected.append(f1_score(_labels(fold.Y_test), model.predict(fold.test_states),
                                     average="macro"))
        attrs = trial.user_attrs
        assert value == pytest.approx(np.mean(expected), abs=1e-12)
        assert [attrs["fold_0_f1"], attrs["fold_1_f1"]] == pytest.approx(expected, abs=1e-12)
        assert attrs["metric_f1_std"] == pytest.approx(np.std(expected, ddof=1))
        assert (attrs["metric_n_folds"], attrs["readout_classifier"]) == (2, "Ridge")
        assert (attrs["study_name"], attrs["job_id"], attrs["trial_name"]) == (
            "P4-test", 3, saved_states)
        assert np.asarray(attrs["metric_confusion_matrix"]).sum() == 36
        assert {"metric_class_0_f1_mean", "metric_class_1_f1_std", "metric_macro_f1",
                "metric_weighted_recall", "profile_fit_readout_mean_s",
                "profile_predict_total_s", "trial_start_utc", "trial_end_utc"} <= set(attrs)
        assert any(key.startswith("env_") for key in attrs)

    def test_folds_without_states_are_refused(self, saved_states):
        Y = np.eye(2)[[0, 1]][:, None, :]
        FoldArtifact(trial_name="stateless", fold_index=0, Y_train=Y, Y_test=Y, Y_pred=Y,
                     metrics=evaluate_performance(Y, Y, 0.0)).save()
        with pytest.raises(SystemExit, match="no saved states"):
            readout.objective(optuna.trial.FixedTrial(RIDGE), classifiers=["Ridge"],
                              trial_name="stateless")


class TestSummaries:
    def test_group_statistics(self):
        stats = readout._compute_group_stats(pd.DataFrame({"value": [0.9, 0.8, np.nan, 0.7]}),
                                             top_n=2)
        assert stats["n_trials"] == 3
        assert (stats["top_mean"], stats["top_min"], stats["top_max"]) == pytest.approx(
            (0.85, 0.8, 0.9))
        assert stats["all_std"] == pytest.approx(np.std([0.9, 0.8, 0.7]))
        empty = readout._compute_group_stats(pd.DataFrame({"value": [np.nan]}), top_n=2)
        assert empty["n_trials"] == 0 and np.isnan(empty["top_mean"])

    def test_the_summary_table_has_a_row_per_classifier(self, capsys):
        frame = pd.DataFrame({"value": [0.9, 0.8, 0.7],
                              "params_classifier": ["KNN", "RF", "KNN"]})
        readout._print_classifier_summary_table(frame, top_n=1)
        lines = capsys.readouterr().out.splitlines()
        assert lines[1] == "=== Summary table (F1 over top 1 and all trials) ==="
        assert [line.split()[:2] for line in lines[3:]] == [["ALL", "3"], ["KNN", "2"],
                                                            ["RF", "1"]]
        readout._print_classifier_summary_table(pd.DataFrame({"number": [1]}), top_n=1)
        assert capsys.readouterr().out == ""

    def test_parameter_names_stand_in_when_importances_fail(self, monkeypatch):
        def fail(study):
            raise RuntimeError("no importances")

        monkeypatch.setattr("optuna.importance.get_param_importances", fail)
        study = optuna.create_study()
        study.add_trial(create_trial(
            params={"b": "x", "a": "y", "fixed_c": 1},
            distributions={"b": CategoricalDistribution(["x"]),
                           "a": CategoricalDistribution(["y"]),
                           "fixed_c": CategoricalDistribution([1])},
            value=0.5))
        assert readout._important_params(study) == ["a", "b"]
        assert readout._important_params(study, max_n=1) == ["a"]


class TestCommands:
    @pytest.fixture
    def study_db(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BIORESERVOIR_CONFIG", str(tmp_path / "config.env"))
        monkeypatch.setenv("BIORESERVOIR_RESULTS_DIR", str(tmp_path))
        storage = base.get_default_storage("P4-test")
        study = optuna.create_study(study_name="P4-test", storage=storage,
                                    direction="maximize")
        classifier = CategoricalDistribution(["Ridge", "KNN"])
        for value, params, distributions in [
            (0.8, {"classifier": "Ridge", "ridge_alpha": 1.0},
             {"ridge_alpha": optuna.distributions.FloatDistribution(1e-5, 1e2, log=True)}),
            (0.7, {"classifier": "KNN", "knn_n_neighbors": 3},
             {"knn_n_neighbors": optuna.distributions.IntDistribution(1, 25)}),
            (0.9, {"classifier": "Ridge", "ridge_alpha": 0.1},
             {"ridge_alpha": optuna.distributions.FloatDistribution(1e-5, 1e2, log=True)}),
        ]:
            study.add_trial(create_trial(
                params=params, distributions={"classifier": classifier, **distributions},
                value=value, user_attrs={"env_python_version": "3.12"}))
        return "P4-test"

    @staticmethod
    def _evaluate_args(**changes):
        args = {"study_name": "P4-test", "study": None, "storage": None, "archive": None,
                "show": "params,metrics,env", "classifiers": None, "top_n": 2,
                "plots": "none", "classifier_plots": None, "max_params": 4}
        args.update(changes)
        return argparse.Namespace(**args)

    def test_research_runs_trials_on_saved_states(self, saved_states, tmp_path):
        readout.cmd_research(argparse.Namespace(
            study_name="P4-new", study=None, storage=None, seed=1, classifiers=["Ridge"],
            job_id=2, trial_name=saved_states, primary_average="weighted", trials=2,
            processes=1))
        study = optuna.load_study(study_name="P4-new",
                                  storage=f"sqlite:///{tmp_path / 'optuna' / 'optuna-P4-new.db'}")
        assert [t.state for t in study.trials] == [optuna.trial.TrialState.COMPLETE] * 2
        assert {(t.user_attrs["trial_name"], t.user_attrs["metric_primary_average"],
                 t.user_attrs["job_id"]) for t in study.trials} == {(saved_states, "weighted", 2)}

    def test_evaluate_prints_the_tables(self, study_db, capsys, monkeypatch):
        plotted = []
        monkeypatch.setattr(readout, "plot_results",
                            lambda *args, **kwargs: plotted.append(kwargs["filename"]))
        readout.cmd_evaluate(self._evaluate_args(classifiers=["ridge"]))
        out = capsys.readouterr().out
        assert "Environment \u2014 trial 2" in out
        assert "ALL classifiers \u2014 P4-test" in out
        assert "classifier=Ridge" in out and "classifier=KNN" not in out
        assert "=== Summary table (F1 over top 2 and all trials) ===" in out
        assert plotted == []

    def test_evaluate_requests_the_plots(self, study_db, capsys, monkeypatch):
        plotted = []
        monkeypatch.setattr(readout, "plot_results",
                            lambda *args, **kwargs: plotted.append(kwargs["filename"]))
        readout.cmd_evaluate(self._evaluate_args(plots="slice,importance,edf",
                                                 classifier_plots="slice,importance"))
        assert plotted == ["P4-test-slice.png", "P4-test-importance.png", "P4-test-edf.png",
                           "P4-test-KNN-slice.png", "P4-test-KNN-importance.png",
                           "P4-test-Ridge-slice.png", "P4-test-Ridge-importance.png"]

    def test_evaluate_needs_a_study_name(self):
        with pytest.raises(SystemExit, match="study_name"):
            readout.cmd_evaluate(self._evaluate_args(study_name=None))
