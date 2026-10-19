"""Unit tests for optimization.training: study parameters, the reservoir objective on a
small echo state network, and the research, evaluate and rerun-best commands."""

import argparse
import importlib
import logging

import numpy as np
import optuna
import pytest
from optuna.distributions import FloatDistribution
from optuna.trial import create_trial

from optimization import base
from optimization import training as training_mod
from optimization.studies import BASELINE, STUDY_REGISTRY

cv_module = importlib.import_module("training.cross_validate")

ESN = {"reservoir_type": "esn", "units": 12, "esn_spectral_radius": 0.9,
       "esn_activation": "tanh", "input_scaling": 1.0, "input_connectivity": 1.0,
       "input_bias": False, "bias_scaling": 0.0, "noise_in": 0.0, "noise_rc": 0.0,
       "rc_connectivity": 0.5, "readout_aggregation": "final", "noise_rate": 0.0,
       "noise_ratio": 0.0}
STUDY = {"description": "unit test", "folds": 2, "instances": 40, "fixed": ESN,
         "optimize": {"esn_leak_rate": {"type": "float", "low": 0.1, "high": 1.0}}}


def _pool(n_per_class=20, steps=16, seed=0):
    """Rising and falling noisy ramps: X (N, T, 1), one-hot Y (N, 1, 2)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, steps)
    X = np.concatenate([t + 0.3 * rng.normal(size=(n_per_class, steps)),
                        1.0 - t + 0.3 * rng.normal(size=(n_per_class, steps))])[..., None]
    return X, np.eye(2)[np.repeat([0, 1], n_per_class)][:, None, :]


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Results under tmp_path, a 40-beat synthetic dataset, and no global side effects."""
    monkeypatch.setenv("BIORESERVOIR_CONFIG", str(tmp_path / "config.env"))
    monkeypatch.setenv("BIORESERVOIR_RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(base, "setup_logging", lambda **kwargs: None)
    monkeypatch.setattr(cv_module, "setup_logging", lambda **kwargs: None)
    monkeypatch.setattr(cv_module, "_limit_native_threads", lambda: None)
    X, Y = _pool()
    loads = []
    monkeypatch.setattr(training_mod, "_DATA_CACHE", {})
    monkeypatch.setattr(training_mod, "load_ecg_data", lambda **kwargs: loads.append(kwargs)
                        or (X[::2], Y[::2], X[1::2], Y[1::2]))
    return tmp_path, loads


def _objective(study_config=STUDY, params=None, **kwargs):
    trial = optuna.trial.FixedTrial(params or {"esn_leak_rate": 0.5})
    kwargs = {"study_name": "P9-test", "study_config": study_config, "seed": 1, **kwargs}
    return trial, training_mod.objective(trial, **kwargs)


class TestStudyParameters:
    def test_suggestions_follow_their_specs(self):
        trial = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0)).ask()
        specs = {
            "rate": {"type": "float", "low": 1e-3, "high": 1.0, "log": True},
            "size": {"type": "int", "low": 10, "high": 50, "step": 10},
            "kind": {"type": "categorical", "choices": ["a", "b"]},
        }
        values = {name: training_mod._suggest(trial, name, spec, {})
                  for name, spec in specs.items()}
        assert 1e-3 <= values["rate"] <= 1.0 and values["size"] in (10, 20, 30, 40, 50)
        assert values["kind"] in ("a", "b")
        assert trial.distributions["rate"].log and trial.distributions["size"].step == 10

    def test_conditional_parameters_are_fixed_when_their_condition_fails(self):
        trial = optuna.create_study().ask()
        spec = {"type": "int", "low": 2, "high": 9, "conditional_on": "mode",
                "condition_value": "last_N"}
        assert training_mod._suggest(trial, "window", spec, {"mode": "final",
                                                             "window": 1}) == 1
        assert "fixed_window" in trial.params and "window" not in trial.params
        assert 2 <= training_mod._suggest(trial, "window", spec, {"mode": "last_N"}) <= 9

    def test_unknown_spec_types_are_rejected(self):
        with pytest.raises(ValueError, match="Unknown suggest type"):
            training_mod._suggest(optuna.create_study().ask(), "x", {"type": "set"}, {})

    def test_parameters_merge_baseline_fixed_and_suggested_values(self):
        trial = optuna.trial.FixedTrial({"esn_leak_rate": 0.3})
        params = training_mod._build_params(trial, STUDY)
        assert params["esn_leak_rate"] == 0.3 and params["units"] == 12
        assert params["scaler_type"] == BASELINE["scaler_type"]

    def test_suggested_values_win_over_fixed_placeholders(self):
        trial = create_trial(params={"window": 5, "fixed_window": 1, "fixed_mode": "final"},
                             distributions={"window": FloatDistribution(1, 10),
                                            "fixed_window": FloatDistribution(1, 10),
                                            "fixed_mode": optuna.distributions
                                            .CategoricalDistribution(["final"])},
                             value=0.5)
        params = training_mod._resolve_params_from_trial(trial, {"fixed": {"units": 7}})
        assert (params["window"], params["mode"], params["units"]) == (5, "final", 7)

    def test_the_topology_helper_is_retired(self):
        with pytest.raises(RuntimeError, match="build_reservoir"):
            training_mod._build_topology({}, seed=1)

    def test_parameter_names_stand_in_when_importances_fail(self, monkeypatch):
        def fail(study):
            raise RuntimeError("no importances")

        monkeypatch.setattr("optuna.importance.get_param_importances", fail)
        study = optuna.create_study()
        study.add_trial(create_trial(params={"b": 1.0, "a": 2.0, "fixed_c": 1.0},
                                     distributions={k: FloatDistribution(0, 5)
                                                    for k in ("a", "b", "fixed_c")},
                                     value=0.5))
        assert training_mod._important_params(study) == ["a", "b"]


def test_the_dataset_is_loaded_once_per_setting(workspace):
    _, loads = workspace
    for noise_rate in (0.0, 0.0, 0.1):
        training_mod._get_dataset(instances=40, binary=False, noise_rate=noise_rate)
    assert [load["noise_rate"] for load in loads] == [0.0, 0.1]


class TestObjective:
    def test_cross_validation_trial(self, workspace):
        tmp_path, loads = workspace
        trial, value = _objective()
        attrs = trial.user_attrs
        assert 0.0 <= value <= 1.0 and value == attrs["metric_f1"]
        assert loads[0]["preserve_split"] is False and loads[0]["noise_rate"] == 0.0
        assert (attrs["dataset_cv_pool_size"], attrs["metric_n_folds"]) == (40, 2)
        assert (attrs["reservoir_type"], attrs["res_units"]) == ("esn", 12)
        assert (attrs["readout_aggregation"], attrs["readout_window"]) == ("final", 1)
        assert attrs["metric_f1_std"] == pytest.approx(
            np.std([attrs["fold_0_f1"], attrs["fold_1_f1"]], ddof=1))
        assert attrs["metric_macro_f1"] == pytest.approx(attrs["metric_f1_macro"])
        assert {"profile_train_reservoir_total_s", "fold_runtime_mean_s",
                "fold_runtime_fold_1_s", "metric_class_1_f1_mean", "env_python_version",
                "trial_end_utc"} <= set(attrs)
        assert sorted(p.name for p in (tmp_path / "runs" / "P9-test-0").iterdir()) == [
            "fold-0.npz", "fold-1.npz"]

    def test_larger_pools_are_subsampled(self, workspace):
        trial, _ = _objective(instances=30)
        assert trial.user_attrs["dataset_cv_pool_size"] == 30

    def test_a_one_step_last_n_window_falls_back_to_the_mean(self, workspace):
        study = dict(STUDY, fixed=dict(ESN, readout_aggregation="last_N", readout_window=1))
        trial, _ = _objective(study)
        assert (trial.user_attrs["readout_aggregation"],
                trial.user_attrs["readout_window"]) == ("signal_mean", 1)

    def test_the_original_split_needs_one_fold(self, workspace):
        with pytest.raises(ValueError, match="folds=1"):
            _objective(preserve_split=True)

    def test_the_original_split_is_recorded_as_one_fold(self, workspace):
        """A trial on the original split used to record 0 folds and no fold metrics."""
        _, loads = workspace
        trial, value = _objective(dict(STUDY, folds=1), preserve_split=True)
        attrs = trial.user_attrs
        assert loads[-1]["preserve_split"] is True
        assert (attrs["metric_n_folds"], attrs["dataset_cv_pool_size"]) == (1, 40)
        assert attrs["fold_0_f1"] == value and attrs["metric_f1_std"] == 0.0
        assert attrs["metric_macro_f1"] == pytest.approx(attrs["metric_f1_macro"])
        assert "metric_class_1_f1_mean" in attrs


class TestCommands:
    @staticmethod
    def _research_args(**changes):
        args = {"study": "P2-distance", "study_name": None, "storage": None, "trials": None,
                "folds": None, "instances": None, "binary": False, "sampler": None,
                "save_threshold": 0.5, "save_states": True, "fold_workers": 2,
                "profile": False, "seed": 3, "preserve_split": False,
                "primary_average": None, "log_level": "INFO", "log_file": None,
                "verbosity": 0, "job_id": 4, "processes": 8}
        args.update(changes)
        return argparse.Namespace(**args)

    def test_research_takes_the_study_entry_unless_overridden(self, workspace, monkeypatch):
        calls = []
        monkeypatch.setattr(training_mod, "research", lambda study, trials, objective, **kw:
                            calls.append((study, trials, kw)))
        training_mod.cmd_research(self._research_args())
        training_mod.cmd_research(self._research_args(
            study_name="P9-copy", trials=3, folds=5, instances=600, binary=True,
            sampler="random"))
        (study, trials, kwargs), (copy, copy_trials, copy_kwargs) = calls
        entry = STUDY_REGISTRY["P2-distance"]
        assert (trials, kwargs["folds"], kwargs["instances"]) == (
            entry["n_trials"], entry["folds"], entry["instances"])
        assert kwargs["study_config"] is entry and kwargs["processes"] == 8
        assert (kwargs["log_level"], kwargs["job_id"], kwargs["seed"]) == (logging.INFO, 4, 3)
        assert (copy_trials, copy_kwargs["folds"], copy_kwargs["instances"],
                copy_kwargs["binary"]) == (3, 5, 600, True)
        assert copy.study_name == "P9-copy"
        assert isinstance(copy.sampler, optuna.samplers.RandomSampler)

    def test_studies_are_listed(self, capsys):
        training_mod.cmd_list_studies(None)
        out = capsys.readouterr().out
        assert all(name in out for name in STUDY_REGISTRY)

    @pytest.fixture
    def study_db(self, workspace):
        study = optuna.create_study(study_name="P9-test", direction="maximize",
                                    storage=base.get_default_storage("P9-test"))
        for value in (0.6, 0.8, 0.7):
            study.add_trial(create_trial(
                params={"esn_leak_rate": value},
                distributions={"esn_leak_rate": FloatDistribution(0.1, 1.0)},
                value=value, user_attrs=dict(ESN_ATTRS)))
        return study

    @staticmethod
    def _evaluate_args(**changes):
        args = {"study_name": "P9-test", "study": None, "storage": None, "show": "params",
                "top_n": 2, "plots": "history,edf,rank", "max_params": 4}
        args.update(changes)
        return argparse.Namespace(**args)

    def test_evaluate_prints_the_report_and_requests_the_plots(self, study_db, monkeypatch,
                                                               capsys):
        plotted, confusion = [], []
        monkeypatch.setattr(training_mod, "plot_results",
                            lambda study, fn, params=None, filename=None:
                            plotted.append((filename, params)))
        monkeypatch.setattr(training_mod, "_plot_confusion_matrix",
                            lambda study, name, config, filename=None:
                            confusion.append((name, config, filename)))
        training_mod.cmd_evaluate(self._evaluate_args())
        assert "study \u2014 P9-test" in capsys.readouterr().out
        assert plotted == [("P9-test-rank.png", None), ("P9-test-edf.png", None),
                           ("P9-test-history.png", None)]
        training_mod.cmd_evaluate(self._evaluate_args(plots="slice,confusion"))
        assert plotted[-1] == ("P9-test-slice.png", ["esn_leak_rate"])
        assert confusion == [("P9-test", {}, "P9-test-confusion.png")]

    def test_evaluate_needs_a_study_name(self):
        with pytest.raises(SystemExit, match="study_name"):
            training_mod.cmd_evaluate(self._evaluate_args(study_name=None))

    def test_the_confusion_matrix_is_summed_over_the_folds(self, study_db, monkeypatch):
        drawn = []
        monkeypatch.setattr("utils.visualisation.plot_confusion_matrix",
                            lambda cm, filename=None, show=True: drawn.append((cm, filename)))
        config = dict(STUDY, fixed=dict(ESN, esn_leak_rate=0.5))
        training_mod._plot_confusion_matrix(study_db, "P9-test", config, filename="cm.png")
        ((cm, filename),) = drawn
        assert filename == "cm.png" and np.asarray(cm).sum() == 40

    def test_no_confusion_matrix_without_completed_trials(self, workspace, monkeypatch, caplog):
        monkeypatch.setattr("utils.visualisation.plot_confusion_matrix",
                            lambda *args, **kwargs: pytest.fail("nothing to draw"))
        with caplog.at_level(logging.WARNING, logger="optimization.training"):
            training_mod._plot_confusion_matrix(optuna.create_study(), "P9-empty", STUDY)
        assert any("No completed trials" in message for message in caplog.messages)


# What an ESN study trial records (as build_reservoir and the objective store it).
ESN_ATTRS = {"reservoir_type": "esn", "res_units": 12, "res_input_scaling": 1.0,
             "res_noise_in": 0.0, "res_noise_rc": 0.0, "res_input_bias": False,
             "res_bias_scaling": 0.0, "res_input_connectivity": 1.0,
             "topo_rc_connectivity": 0.5, "esn_spectral_radius": 0.9,
             "esn_activation": "tanh", "dataset_noise_rate": 0.0, "dataset_noise_ratio": 0.0,
             "readout_aggregation": "final", "readout_window": 1}


class TestRerunBestOnTheOriginalSplit:
    @pytest.fixture
    def best_trial(self, workspace):
        study = optuna.create_study(study_name="P3-binary", direction="maximize",
                                    storage=base.get_default_storage("P3-binary"))
        study.add_trial(create_trial(params={}, distributions={}, value=0.9,
                                     user_attrs=dict(ESN_ATTRS, res_units=8)))
        return study

    @staticmethod
    def _args(**changes):
        args = {"study_name": "P3-binary", "study": None, "storage": None, "seed": 2,
                "instances": None, "binary": False, "folds": 1, "trial_name": None,
                "primary_average": None, "noise_rate": None, "noise_ratio": None,
                "preserve_split": True, "archive": None, "processes": 1, "profile": False}
        args.update(changes)
        return argparse.Namespace(**args)

    def test_the_best_trial_is_rerun_with_its_states_saved(self, best_trial, monkeypatch):
        calls = []
        monkeypatch.setattr(training_mod, "classify", lambda *args, **kwargs:
                            calls.append((args, kwargs)) or {"f1": 0.9})
        training_mod.cmd_rerun_best(self._args())
        ((args, kwargs),) = calls
        reservoir, readout, X_train, Y_train, X_test, Y_test = args
        assert reservoir.hypers["units"] == 8
        assert (len(X_train), len(X_test)) == (20, 20)
        assert (kwargs["trial_name"], kwargs["save_states"]) == ("P3-binary-best", True)
        assert (kwargs["readout_aggregation"], kwargs["primary_average"]) == ("final", "macro")

    def test_the_original_split_needs_one_fold(self, best_trial):
        with pytest.raises(SystemExit, match="--folds 1"):
            training_mod.cmd_rerun_best(self._args(folds=5))

    def test_a_study_without_completed_trials_is_reported(self, workspace):
        optuna.create_study(study_name="P3-binary",
                            storage=base.get_default_storage("P3-binary"))
        with pytest.raises(SystemExit, match="No completed trials"):
            training_mod.cmd_rerun_best(self._args())

    def test_a_study_name_is_required(self):
        with pytest.raises(SystemExit, match="study_name"):
            training_mod.cmd_rerun_best(self._args(study_name=None))
