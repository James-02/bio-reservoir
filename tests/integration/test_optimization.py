"""Integration tests for the Optuna studies: a reservoir trial end to end, pruned and
failed trials, and the no-reservoir baseline studies against the archive.

Run with::

    pytest tests/integration/test_optimization.py
"""

import importlib

import numpy as np
import optuna
import pytest
from optuna.trial import TrialState

from optimization import base, baseline
from optimization import training as training_mod
from optimization.studies import STUDY_REGISTRY
from utils import config
from utils.preprocessing import load_ecg_data

cv_module = importlib.import_module("training.cross_validate")


@pytest.fixture
def quiet(monkeypatch):
    """The study runner and the folds configure logging and threads for workers."""
    monkeypatch.setattr(base, "setup_logging", lambda **kwargs: None)
    monkeypatch.setattr(cv_module, "setup_logging", lambda **kwargs: None)
    monkeypatch.setattr(cv_module, "_limit_native_threads", lambda: None)


def test_single_optuna_trial_populates_user_attrs(tmp_path, monkeypatch, quiet):
    """One reservoir trial (a 3-oscillator BioReservoir, 40 synthetic beats, 2 folds)
    runs through the study runner and records its parameters, metrics and timings."""
    monkeypatch.setenv("BIORESERVOIR_RESULTS_DIR", str(tmp_path))
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 1.0, 12)
    X = np.concatenate([t + 0.3 * rng.normal(size=(20, 12)),
                        1.0 - t + 0.3 * rng.normal(size=(20, 12))])[..., None]
    Y = np.eye(2)[np.repeat([0, 1], 20)][:, None, :]
    monkeypatch.setattr(training_mod, "_DATA_CACHE", {})
    monkeypatch.setattr(training_mod, "load_ecg_data",
                        lambda **kwargs: (X[::2], Y[::2], X[1::2], Y[1::2]))
    study_config = {
        "folds": 2, "description": "integration test",
        "fixed": {"units": 3, "warmup": 0, "rk4_substeps": 16, "output_variables": "He",
                  "input_scaling": 1.0, "rc_scaling": 0.1, "dde_scaling": 0.01,
                  "interaction_diameters": 3.0, "fade_alpha": 0.05, "sparsity": 1.0,
                  "signed": False, "p_excite": 1.0, "readout_aggregation": "final",
                  "noise_rate": 0.0, "noise_ratio": 0.0, "noise_in": 0.0, "noise_rc": 0.0},
        "optimize": {"cell_coupling": {"type": "float", "low": 1.0, "high": 10.0}},
    }
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.RandomSampler(seed=0))
    base.research(study, 1, training_mod.objective, processes=1, study_name="P9-trial",
                  study_config=study_config, seed=1)
    (trial,) = study.trials
    assert trial.state == TrialState.COMPLETE and 0.0 <= trial.value <= 1.0
    attrs = trial.user_attrs
    assert (attrs["res_units"], attrs["topo_type"], attrs["reservoir_type"]) == (
        3, "distance", "bioreservoir")
    assert attrs["res_cell_coupling"] == trial.params["cell_coupling"]
    assert (attrs["metric_n_folds"], attrs["dataset_cv_pool_size"]) == (2, 40)
    assert attrs["metric_f1"] == trial.value
    for key in ("profile_train_reservoir_total_s", "profile_fit_readout_mean_s",
                "fold_runtime_mean_s", "metric_macro_f1", "metric_class_0_f1_mean",
                "env_git_commit", "trial_start_utc", "trial_end_utc"):
        assert key in attrs, key


def test_pruned_and_failed_trials_are_recorded_not_raised(quiet):
    """A pruner stops unpromising trials and failing trials are marked FAIL; neither
    escapes the study runner."""
    def objective(trial, **kwargs):
        x = trial.suggest_float("x", 0.0, 1.0)
        for step in range(4):
            trial.report(x * (step + 1), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if x > 0.9:
            raise RuntimeError("diverged")
        return x

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.RandomSampler(seed=0),
        pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=2))
    base.research(study, 30, objective, processes=1)
    states = [t.state for t in study.trials]
    assert len(states) == 30
    assert {TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL} <= set(states)


@pytest.mark.parametrize("study_name", ["B1-balanced", "B2-unbalanced", "B3-binary"])
def test_baseline_best_trials_reproduce_the_archive(study_name, monkeypatch):
    """The best trial of each archived no-reservoir study, rerun from its registry entry
    (no augmentation) on its v0.1 pool (v0.1 cap rule and, for binary, the v0.1 binary
    file), gives the archived macro F1 in every fold."""
    path = config.optuna_db(study_name, "v0.1")
    if not path.is_file():
        pytest.skip("archive not available")
    entry = STUDY_REGISTRY[study_name]
    fixed = entry["fixed"]
    data_file = config.data_dir() / ("ecg_archive/binary_ecg_data.npz" if entry["binary"]
                                     else "ecg/ecg_data.npz")
    if not data_file.is_file():
        pytest.skip(f"{data_file.name} not in the data folder")
    best = optuna.load_study(study_name=study_name, storage=f"sqlite:///{path}").best_trial
    kwargs = {"classifiers": [best.params["classifier"]], "instances": entry["instances"],
              "binary": entry["binary"], "folds": entry["folds"], "seed": config.SEED,
              **{key: fixed[key] for key in ("balance_classes", "max_per_class",
                                             "scaler_type", "noise_rate", "noise_ratio",
                                             "objective_average")}}
    pool = load_ecg_data(
        rows=entry["instances"], binary=entry["binary"],
        balance_classes=fixed["balance_classes"], max_per_class=fixed["max_per_class"],
        noise_rate=0.0, noise_ratio=0.0, scaler_type=fixed["scaler_type"], seed=config.SEED,
        preserve_split=False, cap_includes_test=False, data_dir=str(data_file.parent),
        **{"binary_save_file" if entry["binary"] else "save_file": data_file.name})
    key = (entry["instances"], entry["binary"], fixed["balance_classes"],
           fixed["max_per_class"], fixed["scaler_type"], config.SEED)
    monkeypatch.setattr(baseline, "_DATA_CACHE", {key: pool})
    trial = optuna.trial.FixedTrial(best.params)
    value = baseline.objective(trial, **kwargs)
    folds = range(entry["folds"])
    assert value == pytest.approx(best.value, abs=1e-12)
    assert [trial.user_attrs[f"fold_{k}_f1"] for k in folds] == pytest.approx(
        [best.user_attrs[f"fold_{k}_f1"] for k in folds], abs=1e-12)
    assert trial.user_attrs["dataset_cv_pool_size"] == best.user_attrs["dataset_cv_pool_size"]
