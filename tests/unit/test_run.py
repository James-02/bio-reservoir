"""Unit tests for the run framework (training.run and training.compare)."""

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import optuna
import pytest
import yaml

from training import compare as compare_module
from training import run as run_module
from utils import config, provenance

cv_module = importlib.import_module("training.cross_validate")
BENCHMARKS = Path(__file__).resolve().parents[2] / "experiments" / "benchmarks"
LOAD_TASK = run_module.load_task

RESERVOIR = {
    "units": 3, "warmup": 0, "rk4_substeps": 16, "rk4_substep_aggregation": 1,
    "input_scaling": 1.0, "input_connectivity": 1.0, "input_bias": False, "bias_scaling": 0.0,
    "rc_scaling": 0.1, "dde_scaling": 0.01, "cell_coupling": 1.0, "noise_in": 0.0,
    "noise_rc": 0.0, "output_variables": "He", "topology_type": "distance",
    "interaction_diameters": 3.0, "fade_alpha": 0.05, "sparsity": 1.0, "signed": False,
    "p_excite": 1.0,
}


def _config(**sections):
    cfg = {"name": "unit", "seed": 3, "reservoir": {"params": dict(RESERVOIR)},
           "readout": {"classifier": "Ridge", "params": {"alpha": 1e-3}},
           "task": {"rows": 24}, "cv": {"folds": 3}}
    cfg.update(sections)
    return cfg


def _pool(n_per_class=12, steps=8, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, steps)
    X = np.concatenate([t + 0.3 * rng.normal(size=(n_per_class, steps)),
                        1.0 - t + 0.3 * rng.normal(size=(n_per_class, steps))])[..., None]
    return X, np.eye(2)[np.repeat([0, 1], n_per_class)][:, None, :]


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Results, archive and registry under tmp_path; a synthetic two-class pool."""
    monkeypatch.setenv("BIORESERVOIR_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("BIORESERVOIR_ARCHIVE_DIR", str(tmp_path / "archive"))
    monkeypatch.setattr(provenance, "REGISTRY_FILE", tmp_path / "registry.jsonl")
    monkeypatch.setattr(run_module, "load_task", lambda task: (*_pool(), []))
    monkeypatch.setattr(cv_module, "setup_logging", lambda **kwargs: None)
    monkeypatch.setattr(cv_module, "_limit_native_threads", lambda: None)
    return tmp_path


def _execute(cfg, out, **kwargs):
    return run_module.execute(cfg, out=out, allow_dirty=True, **kwargs)


def _predictions(run_dir):
    return [np.load(p, allow_pickle=True)["Y_pred"] for p in sorted(run_dir.glob("fold-*.npz"))]


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

class TestResolve:
    def test_defaults_and_seeds_are_filled_in(self):
        cfg = _config()
        cfg["reservoir"]["params"]["noise_rate"] = 0.3  # not a reservoir parameter
        resolved = run_module.resolve(cfg)
        assert resolved["task"] == {"labels": "five-class", "rows": 24, "balance_classes": True,
                                    "max_per_class": None, "scaler": "sequence_zscore",
                                    "seed": 3}
        assert resolved["cv"]["seed"] == resolved["reservoir"]["seed"] == 3
        assert resolved["features"] == {"aggregation": "final", "window": 1}
        assert resolved["outputs"] == {"save_states": False}
        assert "noise_rate" not in resolved["reservoir"]["params"]
        assert resolved["reservoir"]["type"] == "bioreservoir"

    def test_a_resolved_config_resolves_to_itself(self):
        resolved = run_module.resolve(_config())
        assert run_module.resolve(resolved) == resolved
        assert yaml.safe_load(run_module.dump_config(resolved)) == resolved

    @pytest.mark.parametrize("change, message", [
        ({"name": ""}, "name"),
        ({"readout": {}}, "readout"),
        ({"readout": {"classifier": "Forest"}}, "Unknown readout"),
        ({"task": {"labels": "three-class"}}, "labels"),
        ({"reservoir": {"params": {"warmup": 0}}}, "units"),
    ])
    def test_invalid_configs_are_rejected(self, change, message):
        with pytest.raises(ValueError, match=message):
            run_module.resolve(_config(**change))

    def test_study_trials_supply_the_parameters(self, isolated):
        optuna_dir = isolated / "results" / "optuna"
        optuna_dir.mkdir(parents=True)
        reservoir_study = optuna.create_study(
            study_name="P2-distance", storage=f"sqlite:///{optuna_dir / 'optuna-P2-distance.db'}")
        reservoir_study.add_trial(optuna.trial.create_trial(
            params={"units": 7, "readout_aggregation": "last_N", "readout_window": 5},
            distributions={"units": optuna.distributions.IntDistribution(1, 2000),
                           "readout_aggregation": optuna.distributions.CategoricalDistribution(
                               ["final", "last_N"]),
                           "readout_window": optuna.distributions.IntDistribution(1, 200)},
            value=0.9))
        readout_study = optuna.create_study(
            study_name="P4-readout-balanced",
            storage=f"sqlite:///{optuna_dir / 'optuna-P4-readout-balanced.db'}")
        readout_study.add_trial(optuna.trial.create_trial(
            params={"classifier": "KNN", "knn_n_neighbors": 3, "knn_weights": "distance",
                    "knn_p": 2.0},
            distributions={
                "classifier": optuna.distributions.CategoricalDistribution(["KNN", "RF"]),
                "knn_n_neighbors": optuna.distributions.IntDistribution(1, 25),
                "knn_weights": optuna.distributions.CategoricalDistribution(
                    ["uniform", "distance"]),
                "knn_p": optuna.distributions.FloatDistribution(1.0, 4.0)},
            value=0.8))

        resolved = run_module.resolve({
            "name": "from-studies", "reservoir": {"study": "P2-distance"},
            "readout": {"study": "P4-readout-balanced"}, "cv": {"noise_rate": 0.0}})
        assert resolved["reservoir"]["params"]["units"] == 7
        assert resolved["reservoir"]["params"]["topology_type"] == "distance"
        assert resolved["reservoir"]["source"] == {"study": "P2-distance", "trial": 0,
                                                   "archive": None}
        assert resolved["features"] == {"aggregation": "last_N", "window": 5}
        assert resolved["task"]["rows"] == 4015
        assert resolved["cv"]["noise_rate"] == 0.0 and resolved["cv"]["noise_ratio"] > 0
        assert resolved["readout"]["classifier"] == "KNN"
        assert resolved["readout"]["params"] == {"n_neighbors": 3, "weights": "distance",
                                                 "p": 2.0}

    def test_missing_study_databases_are_reported(self):
        with pytest.raises(FileNotFoundError, match="P2-distance"):
            run_module.resolve({"name": "x", "reservoir": {"study": "P2-distance"},
                                "readout": {"classifier": "KNN"}})


@pytest.mark.parametrize("path", sorted(BENCHMARKS.rglob("*.yaml")), ids=lambda p: p.stem)
def test_benchmark_configs_are_explicit(path):
    cfg = run_module.load_config(path)
    assert run_module.resolve(cfg) == cfg
    assert cfg["reference"].startswith("archive:")


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

class TestExecute:
    def test_a_run_folder_has_every_output(self, tmp_path):
        run_dir = _execute(_config(), tmp_path / "run")
        assert {p.name for p in run_dir.iterdir()} == {
            "config.yaml", "metrics.json", "manifest.json", "run.log", "figures",
            "fold-0.npz", "fold-1.npz", "fold-2.npz"}
        assert yaml.safe_load((run_dir / "config.yaml").read_text()) == \
            run_module.resolve(_config())
        assert {p.name for p in (run_dir / "figures").iterdir()} == {
            "confusion_matrix.png", "fold_metrics.png"}
        assert "Pool: 24 beats" in (run_dir / "run.log").read_text()

    def test_metrics_report_folds_mean_and_sample_sd(self, tmp_path):
        metrics = json.loads((_execute(_config(), tmp_path / "run") / "metrics.json")
                             .read_text())
        f1 = np.array([fold["f1"] for fold in metrics["folds"]])
        assert metrics["n_folds"] == 3 and [f["fold"] for f in metrics["folds"]] == [0, 1, 2]
        assert metrics["mean"]["f1"] == pytest.approx(f1.mean())
        assert metrics["sd"]["f1"] == pytest.approx(f1.std(ddof=1))
        assert np.sum(metrics["pooled"]["confusion_matrix"]) == 24
        assert metrics["summary"]["f1"] == metrics["mean"]["f1"]

    def test_manifest_and_registry_record_the_run(self, tmp_path, isolated):
        run_dir = _execute(_config(), tmp_path / "run")
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["spec"] == run_module.resolve(_config())
        assert len(manifest["data_pool_sha256"]) == 64
        assert {o["path"].split("/")[-1] for o in manifest["outputs"]} >= {
            "metrics.json", "config.yaml", "fold-0.npz", "run.log"}
        (entry,) = provenance.load_registry(isolated / "registry.jsonl").values()
        assert entry["run_id"] == manifest["run_id"]
        assert entry["status"] == ("ok" if manifest["citable"] else "exploratory")
        assert entry["metrics"]["f1"] == pytest.approx(
            json.loads((run_dir / "metrics.json").read_text())["mean"]["f1"])

    def test_runs_are_reproducible(self, tmp_path):
        a = _execute(_config(), tmp_path / "a")
        b = _execute(_config(), tmp_path / "b")
        for pa, pb in zip(_predictions(a), _predictions(b), strict=True):
            np.testing.assert_array_equal(pa, pb)
        manifests = [json.loads((d / "manifest.json").read_text()) for d in (a, b)]
        assert manifests[0]["data_pool_sha256"] == manifests[1]["data_pool_sha256"]
        assert manifests[0]["spec_hash"] == manifests[1]["spec_hash"]

    @pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
    def test_refitting_on_saved_states_reproduces_the_run(self, tmp_path):
        first = _execute(_config(outputs={"save_states": True}), tmp_path / "first")
        refit = _execute({"name": "refit", "reservoir": {"states_from": str(first)},
                          "readout": _config()["readout"]}, tmp_path / "refit")
        for pa, pb in zip(_predictions(first), _predictions(refit), strict=True):
            np.testing.assert_array_equal(pa, pb)
        parallel = _execute({"name": "refit", "reservoir": {"states_from": str(first)},
                             "readout": _config()["readout"]}, tmp_path / "parallel",
                            processes=2)
        for pa, pb in zip(_predictions(first), _predictions(parallel), strict=True):
            np.testing.assert_array_equal(pa, pb)

    def test_refitting_needs_saved_states(self, tmp_path):
        first = _execute(_config(), tmp_path / "first")
        with pytest.raises(SystemExit, match="no saved states"):
            _execute({"name": "refit", "reservoir": {"states_from": str(first)},
                      "readout": _config()["readout"]}, tmp_path / "refit")

    def test_an_existing_run_folder_is_not_overwritten(self, tmp_path):
        _execute(_config(), tmp_path / "run")
        with pytest.raises(SystemExit, match="already exists"):
            _execute(_config(), tmp_path / "run")

    def test_uncommitted_code_needs_permission(self, tmp_path, monkeypatch, isolated):
        monkeypatch.setattr(provenance, "git_info",
                            lambda: {"commit": "abc", "describe": "abc", "dirty": True})
        with pytest.raises(SystemExit, match="Uncommitted"):
            run_module.execute(_config(), out=tmp_path / "refused")
        run_dir = _execute(_config(), tmp_path / "allowed")
        assert json.loads((run_dir / "manifest.json").read_text())["citable"] is False
        (entry,) = provenance.load_registry(isolated / "registry.jsonl").values()
        assert entry["status"] == "exploratory"

    def test_default_folder_is_named_after_the_config(self, isolated):
        run_dir = run_module.execute(_config(), allow_dirty=True)
        assert run_dir.parent == isolated / "results" / "runs"
        assert run_dir.name.startswith("unit-")


def test_load_task_subsamples_to_the_requested_rows(monkeypatch):
    X, Y = _pool(n_per_class=30)
    monkeypatch.setattr("utils.preprocessing.load_ecg_data",
                        lambda **kwargs: (X[:40], Y[:40], X[40:], Y[40:]))
    task = {"labels": "five-class", "rows": 25, "balance_classes": True, "max_per_class": None,
            "scaler": "sequence_zscore", "seed": 1}
    first, labels, _ = LOAD_TASK(task)
    again, _, _ = LOAD_TASK(task)
    expected = np.sort(np.random.default_rng(1).permutation(60)[:25])
    np.testing.assert_array_equal(first, X[expected])
    np.testing.assert_array_equal(labels, Y[expected])
    np.testing.assert_array_equal(first, again)


# ---------------------------------------------------------------------------
# Comparing
# ---------------------------------------------------------------------------

class TestCompare:
    def test_identical_runs_agree_completely(self, tmp_path):
        a = _execute(_config(outputs={"save_states": True}), tmp_path / "a")
        b = _execute(_config(outputs={"save_states": True}), tmp_path / "b")
        result = compare_module.compare(a, b, out=tmp_path / "cmp", n_boot=200)
        assert result["paired"] and result["agreement"] == 1.0
        assert result["pooled_delta_f1_macro"]["delta"] == 0.0
        assert result["pooled_delta_f1_macro"]["ci95"] == [0.0, 0.0]
        assert all(r["delta_f1_macro"] == 0.0 for r in result["folds"])
        assert result["paired_by"] == "beat position"
        assert {p.name for p in (tmp_path / "cmp").iterdir()} == {
            "compare.json", "fold_f1.png", "confusion_matrices.png", "state_features.png"}
        assert "agreement 1.0000" in compare_module.format_comparison(result)

    def test_different_folds_are_not_paired(self, tmp_path):
        a = _execute(_config(), tmp_path / "a")
        b = _execute(_config(cv={"folds": 3, "seed": 99}), tmp_path / "b")
        result = compare_module.compare(a, b, n_boot=50)
        assert not result["paired"] and "pooled_delta_f1_macro" not in result
        assert "not paired" in compare_module.format_comparison(result)

    def test_runs_are_found_by_folder_registry_id_name_or_archive(self, tmp_path, isolated):
        run_dir = _execute(_config(), isolated / "results" / "runs" / "named")
        (run_id,) = provenance.load_registry(isolated / "registry.jsonl")
        archived = isolated / "archive" / "v0.1" / "runs" / "frozen"
        archived.mkdir(parents=True)
        (archived / "fold-0.npz").write_bytes((run_dir / "fold-0.npz").read_bytes())
        assert compare_module.run_path(str(run_dir)) == run_dir
        assert compare_module.run_path("named") == run_dir
        assert compare_module.run_path(run_id) == run_dir
        assert compare_module.run_path("archive:v0.1/frozen") == archived
        with pytest.raises(FileNotFoundError):
            compare_module.run_path("missing")

    def test_bootstrap_interval_contains_the_observed_difference(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 3, 300)
        pred_a = np.where(rng.random(300) < 0.7, y, rng.integers(0, 3, 300))
        pred_b = np.where(rng.random(300) < 0.9, y, rng.integers(0, 3, 300))
        delta, low, high = compare_module.bootstrap_delta(y, pred_a, pred_b, n_boot=300)
        assert low < delta < high and delta > 0
        assert compare_module.bootstrap_delta(y, pred_a, pred_a, n_boot=50) == (0.0, 0.0, 0.0)


class TestOptimizeCommands:
    """optimize rerun-best and apply-best build a run config and call the runner."""

    @pytest.fixture
    def calls(self, monkeypatch):
        calls = []
        monkeypatch.setattr(run_module, "execute",
                            lambda cfg, **kwargs: calls.append((cfg, kwargs)) or Path("run"))
        return calls

    def test_rerun_best(self, calls):
        from optimization.training import cmd_rerun_best

        cmd_rerun_best(argparse.Namespace(
            study_name=None, study="P2-distance", trial_name=None, folds=5, instances=None,
            binary=True, seed=7, processes=3, profile=False, preserve_split=False,
            primary_average=None, readout_study="P4-readout-binary", readout_trial=None,
            noise_rate=0.0, noise_ratio=None, archive="v0.1", storage=None))
        ((cfg, kwargs),) = calls
        assert cfg["name"] == "P2-distance-best" and cfg["seed"] == 7
        assert cfg["reservoir"] == {"study": "P2-distance", "trial": "best", "archive": "v0.1",
                                    "storage": None}
        assert cfg["readout"] == {"study": "P4-readout-binary", "trial": "best",
                                  "archive": "v0.1"}
        assert cfg["task"] == {"labels": "binary"}
        assert cfg["cv"] == {"folds": 5, "profile": False, "noise_rate": 0.0}
        assert cfg["outputs"] == {"save_states": True}
        assert kwargs == {"out": config.runs_dir() / "P2-distance-best", "processes": 3,
                          "allow_dirty": True}

    def test_rerun_best_defaults_to_the_knn_readout(self, calls):
        from optimization.training import _KNN_HYPERS, cmd_rerun_best

        cmd_rerun_best(argparse.Namespace(
            study_name="P2-distance", trial_name="mine", folds=None, instances=600,
            binary=False, seed=1, processes=1, profile=True, preserve_split=False,
            primary_average="weighted", readout_study=None, readout_trial=None,
            noise_rate=None, noise_ratio=None, archive=None, storage=None))
        ((cfg, kwargs),) = calls
        assert cfg["readout"] == {"classifier": "KNN", "params": _KNN_HYPERS}
        assert cfg["task"] == {"rows": 600}
        assert cfg["cv"] == {"folds": 1, "profile": True, "primary_average": "weighted"}
        assert kwargs["out"] == config.runs_dir() / "mine"

    def test_apply_best(self, calls):
        from optimization.readout import cmd_apply_best

        cmd_apply_best(argparse.Namespace(
            study_name="P4-readout-balanced", trial_name="archive:v0.1/P2-distance-best",
            readout_trial=12, processes=5, seed=2, archive="v0.1", storage=None))
        ((cfg, kwargs),) = calls
        assert cfg["name"] == "P4-readout-balanced-best"
        assert cfg["reservoir"] == {"states_from": "archive:v0.1/P2-distance-best"}
        assert cfg["readout"] == {"study": "P4-readout-balanced", "archive": "v0.1",
                                  "trial": 12, "storage": None}
        assert kwargs == {"out": config.runs_dir() / "P4-readout-balanced-best",
                          "processes": 5, "allow_dirty": True}


def test_command_line(tmp_path, capsys):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(_config()))
    run_module.main(["resolve", str(path)])
    assert yaml.safe_load(capsys.readouterr().out) == run_module.resolve(_config())
    run_module.main(["run", str(path), "--out", str(tmp_path / "a"), "--allow-dirty"])
    run_module.main(["run", str(path), "--out", str(tmp_path / "b"), "--allow-dirty"])
    capsys.readouterr()
    run_module.main(["compare", str(tmp_path / "a"), str(tmp_path / "b"), "--bootstrap", "50"])
    assert "prediction agreement 1.0000" in capsys.readouterr().out
