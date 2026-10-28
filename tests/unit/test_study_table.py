"""Unit tests for the study table (training.study.trials_table, seed_table, study-table)."""

import importlib
from datetime import datetime, timedelta

import numpy as np
import optuna
import pandas as pd
import pytest
import yaml

from training import run as run_module
from training import study as study_module
from utils import provenance

cv_module = importlib.import_module("training.cross_validate")

RESERVOIR = {
    "units": 3, "warmup": 0, "rk4_substeps": 16,
    "input_scaling": 1.0, "input_connectivity": 1.0, "input_bias": False, "bias_scaling": 0.0,
    "rc_scaling": 0.1, "dde_scaling": 0.01, "cell_coupling": 1.0, "noise_in": 0.0,
    "noise_rc": 0.0, "output_variables": "He", "topology_type": "distance",
    "interaction_diameters": 3.0, "fade_alpha": 0.05, "sparsity": 1.0, "signed": False,
    "p_excite": 1.0,
}
SCALING = "reservoir.params.input_scaling"


def _pool(n_per_class=12, steps=8, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, steps)
    X = np.concatenate([t + 0.3 * rng.normal(size=(n_per_class, steps)),
                        1.0 - t + 0.3 * rng.normal(size=(n_per_class, steps))])[..., None]
    return X, np.eye(2)[np.repeat([0, 1], n_per_class)][:, None, :]


def _config(name="unit-table", search=None, seed=3, metric="f1"):
    return {
        "name": name, "seed": 3, "reservoir": {"seed": seed, "params": dict(RESERVOIR)},
        "readout": {"classifier": "Ridge", "params": {"alpha": 1e-3}},
        "task": {"rows": 24}, "cv": {"folds": 3},
        "study": {"trials": 4, "metric": metric, "sampler": {"type": "grid", "seed": 1},
                  "search": search or {SCALING: {"type": "float", "choices": [0.5, 2.0]}}},
    }


def _fake_study(cfg, trials):
    """Store ``trials`` (params, fold F1 or None for a failed trial) in the study of ``cfg``,
    with the attributes the study runner records; trial k takes 30 + k seconds."""
    study = optuna.create_study(study_name=study_module.name(cfg),
                                storage=study_module.storage_url(cfg))
    start = datetime(2026, 10, 24, 12, 0, 0)
    for number, (params, folds) in enumerate(trials):
        finished = folds is not None
        attrs = {}
        if finished:
            attrs = {"fold_f1": folds,
                     "summary": {"f1": float(np.mean(folds)),
                                 "f1_sd": float(np.std(folds, ddof=1))},
                     "simulation": {"simulate_s": 20.0 + number},
                     "code": {"commit": "0123456789abcdef", "citable": True}}
        study.add_trial(optuna.trial.FrozenTrial(
            number=number, trial_id=number, value=float(np.mean(folds)) if finished else None,
            state=(optuna.trial.TrialState.COMPLETE if finished
                   else optuna.trial.TrialState.FAIL),
            datetime_start=start, datetime_complete=start + timedelta(seconds=30 + number),
            params=params,
            distributions={p: optuna.distributions.FloatDistribution(0.0, 10.0)
                           for p in params},
            user_attrs=attrs, system_attrs={}, intermediate_values={}))
    return study


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("BIORESERVOIR_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(provenance, "REGISTRY_FILE", tmp_path / "registry.jsonl")
    monkeypatch.setattr(provenance, "git_info",
                        lambda: {"commit": "abc", "describe": "abc", "dirty": False})
    monkeypatch.setattr(run_module, "load_task", lambda task: (*_pool(), []))
    monkeypatch.setattr(cv_module, "setup_logging", lambda **kwargs: None)
    return tmp_path


def _seed_grid():
    return _config(search={SCALING: {"type": "float", "choices": [0.5, 2.0]},
                           "reservoir.seed": {"type": "categorical", "choices": [3, 4]}})


def test_each_trial_row_holds_its_values_fold_scores_seed_and_times():
    cfg = _seed_grid()
    study = study_module.run_study(cfg)
    table = study_module.trials_table(cfg)
    assert list(table.columns) == ["trial", "state", SCALING, "reservoir.seed", "f1", "f1_sd",
                                   "seconds", "simulate_s", "commit"]
    assert len(table) == 4
    for trial, row in zip(study.trials, table.itertuples(index=False), strict=True):
        resolved = yaml.safe_load(trial.user_attrs["config"])
        assert row.trial == trial.number and row.state == "COMPLETE"
        assert row[2] == trial.params[SCALING]
        assert row[3] == resolved["reservoir"]["seed"] == trial.params["reservoir.seed"]
        assert row.f1 == pytest.approx(np.mean(trial.user_attrs["fold_f1"]))
        assert row.f1_sd == pytest.approx(np.std(trial.user_attrs["fold_f1"], ddof=1))
        assert row.seconds == pytest.approx(
            (trial.datetime_complete - trial.datetime_start).total_seconds())
        assert 0 < row.simulate_s <= row.seconds
        assert row.commit == "abc"


def test_a_raw_beat_trial_lists_its_run_seed():
    cfg = {"name": "unit-raw-table", "seed": 3, "reservoir": {"type": "none"},
           "readout": {"classifier": "Ridge", "params": {"alpha": 1e-3}},
           "task": {"rows": 24, "seed": 3}, "cv": {"folds": 3, "seed": 3},
           "study": {"trials": 2, "sampler": {"type": "grid", "seed": 1},
                     "search": {"seed": {"type": "categorical", "choices": [5, 6]}}}}
    study = study_module.run_study(cfg)
    table = study_module.trials_table(cfg)
    assert sorted(table["reservoir.seed"]) == [5, 6]
    for trial, seed in zip(study.trials, table["reservoir.seed"], strict=True):
        assert seed == trial.params["seed"]


def test_the_grouped_view_averages_each_point_over_its_reservoir_seeds():
    cfg = _seed_grid()
    study_module.run_study(cfg)
    trials = study_module.trials_table(cfg)
    table = study_module.seed_table(trials)
    assert list(table.columns) == [SCALING, "n", "f1", "f1_seed_sd", "f1_fold_sd", "seeds",
                                   "seconds"]
    assert table[SCALING].tolist() == [0.5, 2.0]
    for row in table.itertuples(index=False):
        point = trials[trials[SCALING] == row[0]]
        assert row.n == 2 and row.seeds == "3 4"
        assert row.f1 == pytest.approx(point["f1"].mean())
        assert row.f1_seed_sd == pytest.approx(point["f1"].std(ddof=1))
        assert row.f1_fold_sd == pytest.approx(point["f1_sd"].mean())
        assert row.seconds == pytest.approx(point["seconds"].mean())


def test_studies_that_differ_only_in_their_seeds_read_as_one_grouped_table():
    cfgs = [_config(name=f"unit-table-{seed}", seed=seed) for seed in (3, 4)]
    for k, cfg in enumerate(cfgs):
        _fake_study(cfg, [({SCALING: 0.5}, [0.6 + k / 10, 0.7 + k / 10]),
                          ({SCALING: 2.0}, [0.8, 0.9 + k / 10])])
    trials = study_module.study_table(cfgs)
    assert trials["study"].tolist() == ["unit-table-3"] * 2 + ["unit-table-4"] * 2
    assert trials["reservoir.seed"].tolist() == [3, 3, 4, 4]
    table = study_module.study_table(cfgs, group=True)
    assert table[SCALING].tolist() == [0.5, 2.0]
    assert table["n"].tolist() == [2, 2] and table["seeds"].tolist() == ["3 4", "3 4"]
    assert table["f1"].tolist() == pytest.approx([0.7, 0.875])
    assert table["f1_seed_sd"].tolist() == pytest.approx([np.std([0.65, 0.75], ddof=1),
                                                          np.std([0.85, 0.9], ddof=1)])


def test_unfinished_trials_are_listed_but_not_averaged():
    cfg = _config()
    _fake_study(cfg, [({SCALING: 0.5}, [0.6, 0.7]), ({SCALING: 0.5}, None),
                      ({SCALING: 2.0}, None)])
    trials = study_module.trials_table(cfg)
    assert trials["state"].tolist() == ["COMPLETE", "FAIL", "FAIL"]
    assert trials["f1"].isna().tolist() == [False, True, True]
    assert trials["seconds"].tolist() == [30.0, 31.0, 32.0]
    table = study_module.seed_table(trials)
    assert table[SCALING].tolist() == [0.5] and table["n"].tolist() == [1]
    assert np.isnan(table["f1_seed_sd"].iloc[0])


def test_a_study_without_a_database_is_reported_and_not_created():
    cfg = _config()
    with pytest.raises(FileNotFoundError, match="unit-table"):
        study_module.trials_table(cfg)
    assert not study_module.settings.optuna_db("unit-table").exists()


def test_studies_scored_on_different_metrics_are_not_combined():
    cfgs = [_config(name="unit-a"), _config(name="unit-b", metric="accuracy")]
    with pytest.raises(ValueError, match="different metrics"):
        study_module.study_table(cfgs)


def test_the_command_line_prints_the_table_and_writes_it_as_csv(tmp_path, capsys):
    cfg = _config()
    _fake_study(cfg, [({SCALING: 0.5}, [0.61234, 0.7]), ({SCALING: 2.0}, [0.8, 0.9])])
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(cfg))
    run_module.main(["study-table", str(path), "--csv", str(tmp_path / "trials.csv")])
    out = capsys.readouterr().out
    assert "input_scaling" in out and SCALING not in out
    assert "0.6562" in out and "0123456" in out and "0123456789" not in out
    written = pd.read_csv(tmp_path / "trials.csv")
    assert list(written.columns)[:4] == ["trial", "state", SCALING, "reservoir.seed"]
    assert written["commit"].tolist() == ["0123456789abcdef"] * 2
    run_module.main(["study-table", str(path), "--group", "--csv", str(tmp_path / "grid.csv")])
    assert "f1_seed_sd" in capsys.readouterr().out
    assert pd.read_csv(tmp_path / "grid.csv")["n"].tolist() == [1, 1]
