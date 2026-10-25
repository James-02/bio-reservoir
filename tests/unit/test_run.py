"""Unit tests for the run framework (training.run and training.compare)."""

import argparse
import hashlib
import importlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import optuna
import pytest
import yaml
from sklearn.metrics import f1_score

from training import compare as compare_module
from training import run as run_module
from utils import config, provenance

cv_module = importlib.import_module("training.cross_validate")
BENCHMARKS = Path(__file__).resolve().parents[2] / "experiments" / "benchmarks"
LOAD_TASK = run_module.load_task

RESERVOIR = {
    "units": 3, "warmup": 0, "rk4_substeps": 16,
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
    def test_defaults_and_seeds_are_filled_in(self, caplog):
        cfg = _config()
        cfg["reservoir"]["params"]["noise_rate"] = 0.3  # not a reservoir parameter
        with caplog.at_level(logging.WARNING, logger="training.run"):
            resolved = run_module.resolve(cfg)
        assert resolved["task"] == {"labels": "five-class", "rows": 24, "balance_classes": True,
                                    "max_per_class": None, "cap_includes_test": True,
                                    "scaler": "sequence_zscore", "data_file": None, "seed": 3}
        assert resolved["cv"]["seed"] == resolved["reservoir"]["seed"] == 3
        assert resolved["features"] == {"aggregation": "final", "window": 1, "variables": None,
                                        "measurement_noise": 0.0, "calibration_error": 0.0}
        assert resolved["outputs"] == {"save_states": False}
        assert "noise_rate" not in resolved["reservoir"]["params"]
        assert any("Ignoring reservoir parameters" in r.getMessage()
                   and "noise_rate" in r.getMessage() for r in caplog.records)
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
        assert (resolved["features"]["aggregation"], resolved["features"]["window"]) == (
            "last_N", 5)
        assert resolved["task"]["rows"] == 4015
        assert resolved["cv"]["noise_rate"] == 0.0 and resolved["cv"]["noise_ratio"] > 0
        assert resolved["cv"]["simulate"] == "per_fold"
        assert resolved["readout"]["classifier"] == "KNN"
        assert resolved["readout"]["params"] == {"n_neighbors": 3, "weights": "distance",
                                                 "p": 2.0}

    def test_missing_study_databases_are_reported(self):
        with pytest.raises(FileNotFoundError, match="P2-distance"):
            run_module.resolve({"name": "x", "reservoir": {"study": "P2-distance"},
                                "readout": {"classifier": "KNN"}})

    def test_recorded_trial_values_win_over_the_current_baseline(self, isolated):
        """Phase 1 trials ran with the BASELINE of their time, which later phases
        changed; resolving them from BASELINE gave other reservoirs."""
        storage = f"sqlite:///{isolated / 'optuna-P1-scaling.db'}"
        study = optuna.create_study(study_name="P1-scaling", storage=storage)
        float_range = optuna.distributions.FloatDistribution(1e-5, 1.0, log=True)
        study.add_trial(optuna.trial.create_trial(
            params={"input_scaling": 0.08, "rc_scaling": 0.019, "dde_scaling": 0.0037},
            distributions={"input_scaling": float_range, "rc_scaling": float_range,
                           "dde_scaling": float_range},
            user_attrs={"res_warmup": 40, "res_rk4_substeps": 44, "res_cell_coupling": 16.28,
                        "res_output_variables": "all", "topo_signed": False,
                        "res_heterogeneity_seed": None, "dataset_noise_rate": 0.0825,
                        "dataset_noise_ratio": 0.2348, "dataset_max_per_class": "None",
                        "dataset_instances": 600, "dataset_binary": True,
                        "readout_aggregation": "final", "readout_window": 1},
            value=0.88))

        resolved = run_module.resolve({
            "name": "phase-1", "reservoir": {"study": "P1-scaling", "storage": storage},
            "readout": {"classifier": "KNN"}})
        params = resolved["reservoir"]["params"]
        assert (params["warmup"], params["rk4_substeps"], params["cell_coupling"]) == (
            40, 44, 16.28)
        assert params["output_variables"] == "all" and params["signed"] is False
        assert params["input_scaling"] == 0.08
        assert (resolved["features"]["aggregation"], resolved["features"]["window"]) == (
            "final", 1)
        assert (resolved["cv"]["noise_rate"], resolved["cv"]["noise_ratio"]) == (0.0825, 0.2348)
        assert resolved["task"]["labels"] == "binary" and resolved["task"]["rows"] == 600
        assert resolved["task"]["max_per_class"] is None


@pytest.mark.parametrize("path", sorted(BENCHMARKS.rglob("*.yaml")), ids=lambda p: p.stem)
def test_benchmark_configs_are_explicit(path):
    cfg = run_module.load_config(path)
    assert run_module.resolve(cfg) == cfg
    assert cfg["reference"].startswith("archive:")


def test_a_config_lists_only_what_differs_from_its_base(tmp_path):
    (tmp_path / "designs").mkdir()
    (tmp_path / "designs" / "root.yaml").write_text(
        "seed: 3\nreservoir: {params: {units: 4, warmup: 5}}\n")
    (tmp_path / "designs" / "design.yaml").write_text(
        "base: root.yaml\nreadout: {classifier: Ridge, params: {alpha: 1.0}, scale: true}\n")
    (tmp_path / "study.yaml").write_text(
        "base: designs/design.yaml\nname: s\nreservoir: {params: {units: 9}}\n"
        "readout: {classifier: RF, params: {n_estimators: 10}}\n")
    assert run_module.load_config(tmp_path / "study.yaml") == {
        "seed": 3, "name": "s", "reservoir": {"params": {"units": 9, "warmup": 5}},
        "readout": {"classifier": "RF", "params": {"alpha": 1.0, "n_estimators": 10},
                    "scale": True}}


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

    def test_a_run_without_the_plotting_packages_skips_its_figures(self, tmp_path,
                                                                     monkeypatch):
        monkeypatch.setitem(sys.modules, "matplotlib", None)
        run_dir = _execute(_config(), tmp_path / "run")
        assert (run_dir / "manifest.json").is_file()
        assert not list(run_dir.glob("figures/*"))
        assert "No figures" in (run_dir / "run.log").read_text()

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

    @pytest.mark.parametrize("section", [
        {"task": {"scramble": True}}, {"task": {"rows": 12}}, {"cv": {"folds": 2}}])
    def test_a_refit_cannot_change_the_beats_or_folds_of_its_source(self, section):
        with pytest.raises(ValueError, match="A refit reads the beats, labels and folds"):
            run_module.resolve({"name": "refit", "reservoir": {"states_from": "run"},
                                "readout": {"classifier": "KNN"}, **section})
        assert run_module.resolve({"name": "refit", "reservoir": {"states_from": "run"},
                                   "readout": {"classifier": "KNN"},
                                   "cv": {"primary_average": "weighted"}})

    @pytest.mark.parametrize("seed", [{}, {"seed": 5}])
    def test_the_resolved_config_of_a_refit_resolves_again(self, seed):
        resolved = run_module.resolve({"name": "refit", "reservoir": {"states_from": "run"},
                                       "readout": {"classifier": "KNN"}, **seed})
        assert run_module.resolve(resolved) == resolved

    def test_a_refit_runs_again_from_the_config_in_its_folder(self, tmp_path):
        first = _execute(_config(outputs={"save_states": True}), tmp_path / "first")
        refit = _execute({"name": "refit", "reservoir": {"states_from": str(first)},
                          "readout": _config()["readout"]}, tmp_path / "refit")
        again = _execute(run_module.load_config(refit / "config.yaml"), tmp_path / "again")
        for pa, pb in zip(_predictions(refit), _predictions(again), strict=True):
            np.testing.assert_array_equal(pa, pb)

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


class TestSimulation:
    def test_beats_are_simulated_once_except_in_the_v0_1_preset(self):
        assert run_module.resolve(_config())["cv"]["simulate"] == "once"
        assert run_module.resolve(_config(preset="v0.1"))["cv"]["simulate"] == "per_fold"

    def test_without_noise_both_paths_give_the_same_predictions(self, tmp_path):
        once = _execute(_config(), tmp_path / "once")
        per_fold = _execute(_config(cv={"folds": 3, "simulate": "per_fold"}),
                            tmp_path / "per_fold")
        for pa, pb in zip(_predictions(once), _predictions(per_fold), strict=True):
            np.testing.assert_array_equal(pa, pb)
        metrics = json.loads((once / "metrics.json").read_text())
        assert metrics["simulation"]["beats"] == 24
        assert "simulation" not in json.loads((per_fold / "metrics.json").read_text())

    def test_a_run_warms_its_reservoir_up_once(self, tmp_path, monkeypatch):
        from bioreservoir.node import BioReservoir

        warm_ups = []
        warm_up = BioReservoir._warmup_nodes
        monkeypatch.setattr(BioReservoir, "_warmup_nodes",
                            lambda node: warm_ups.append(node) or warm_up(node))
        _execute(_config(), tmp_path / "run")
        assert len(warm_ups) == 1

    @pytest.mark.parametrize("change, message", [
        ({"cv": {"simulate": "twice"}}, "cv.simulate"),
        ({"cv": {"noise_rate": 0.1, "noise_ratio": 0.2}}, "augmentation"),
        ({"task": {"rows": 24, "scaler": "standard"}}, "fitted on a training fold"),
    ])
    def test_settings_that_need_the_per_fold_path_are_rejected(self, change, message):
        with pytest.raises(ValueError, match=message):
            run_module.resolve(_config(**change))
        per_fold = {**change, "cv": {**change.get("cv", {}), "simulate": "per_fold"}}
        if change.get("cv", {}).get("simulate") != "twice":
            run_module.resolve(_config(**per_fold))


def _reporter_config(**features):
    cfg = _config(features=features, outputs={"save_states": True})
    cfg["reservoir"]["params"]["output_variables"] = "IHe"
    return cfg


def _states(run_dir, kind):
    return [np.load(p, allow_pickle=True)[f"{kind}_states"][:, 0]
            for p in sorted(run_dir.glob("fold-*.npz"))]


class TestObservation:
    def test_the_reporter_alone_is_observed(self, tmp_path):
        both = _execute(_reporter_config(), tmp_path / "both")
        reporter = _execute(_reporter_config(variables=["I"]), tmp_path / "reporter")
        for whole, part in zip(_states(both, "test"), _states(reporter, "test"), strict=True):
            np.testing.assert_array_equal(whole[:, 0::2], part)

    def test_images_are_snapshots_ending_with_the_final_state(self, tmp_path):
        two = _execute(_reporter_config(variables=["I"], aggregation="images:2"),
                       tmp_path / "two")
        final = _execute(_reporter_config(variables=["I"]), tmp_path / "final")
        for images, last in zip(_states(two, "test"), _states(final, "test"), strict=True):
            assert images.shape[1] == 6
            np.testing.assert_array_equal(images[:, 3:], last)

    def test_images_at_chosen_times_are_the_images_k_at_the_same_times(self, tmp_path):
        """With 8-step beats, images:2 takes the states after 4 and 8 steps; the calibration
        gains of the test rows count both images."""
        features = dict(variables=["I"], measurement_noise=0.05, calibration_error=0.1)
        even = _execute(_reporter_config(aggregation="images:2", **features), tmp_path / "even")
        chosen = _execute(_reporter_config(aggregation="images_at:4,8", **features),
                          tmp_path / "chosen")
        for kind in ("train", "test"):
            for a, b in zip(_states(even, kind), _states(chosen, kind), strict=True):
                assert a.shape[1] == 6
                np.testing.assert_array_equal(a, b)
        with pytest.raises(ValueError, match="8 steps long"):
            _execute(_reporter_config(aggregation="images_at:4,9"), tmp_path / "long")

    def test_measurement_noise_is_multiplicative_and_repeatable(self, tmp_path):
        clean = _states(_execute(_reporter_config(), tmp_path / "clean"), "train")
        noisy = [_states(_execute(_reporter_config(measurement_noise=0.05), tmp_path / name),
                         "train") for name in ("a", "b")]
        for c, a, b in zip(clean, *noisy, strict=True):
            np.testing.assert_array_equal(a, b)
            ratio = a / c
            assert 0.03 < np.std(ratio) < 0.07 and abs(np.mean(ratio) - 1) < 0.02

    def test_calibration_error_changes_only_the_test_beats(self, tmp_path):
        clean = _execute(_reporter_config(), tmp_path / "clean")
        biased = _execute(_reporter_config(calibration_error=0.1), tmp_path / "biased")
        for c, b in zip(_states(clean, "train"), _states(biased, "train"), strict=True):
            np.testing.assert_array_equal(c, b)
        for c, b in zip(_states(clean, "test"), _states(biased, "test"), strict=True):
            gains = b / c
            np.testing.assert_allclose(gains, np.broadcast_to(gains[0], gains.shape))
            assert not np.allclose(gains[0], 1.0)

    @pytest.mark.parametrize("change, message", [
        ({"features": {"variables": ["I"]}, "cv": {"simulate": "per_fold"}},
         "cv.simulate: once"),
        ({"features": {"measurement_noise": 0.01}, "cv": {"simulate": "per_fold"}},
         "cv.simulate: once"),
        ({"features": {"measurement_noise": -0.01}}, "0 or more"),
        ({"features": {"calibration_error": -1}}, "0 or more"),
        ({"features": {"variables": "I"}}, "variables"),
        ({"features": {"variables": ["LuxI"]}}, "variables"),
        ({"features": {"variables": ["I", "I"]}}, "each once"),
        ({"features": {"aggregation": "images:0"}}, "images:K"),
        ({"features": {"aggregation": "images:4", "window": 5}}, "window"),
        ({"features": {"aggregation": "images_at:8,4"}}, "strictly increasing"),
        ({"features": {"aggregation": "images_at:0,4"}}, "1 or more"),
        ({"features": {"aggregation": "images_at:4,8", "window": 5}}, "window"),
        ({"features": {"variables": ["I"]},
          "reservoir": {"params": {"units": 3, "reservoir_type": "esn"}}}, "bioreservoir"),
        ({"features": {"calibration_error": 0.1}, "reservoir": {"states_from": "run"}},
         "refit"),
    ])
    def test_invalid_observations_are_rejected(self, change, message):
        with pytest.raises(ValueError, match=message):
            run_module.resolve(_config(**change))

    def test_variables_the_reservoir_does_not_output_are_reported(self, tmp_path):
        with pytest.raises(ValueError, match="output variables"):
            _execute(_config(features={"variables": ["I"]}), tmp_path / "run")


def test_load_task_subsamples_to_the_requested_rows(monkeypatch):
    X, Y = _pool(n_per_class=30)
    monkeypatch.setattr("utils.preprocessing.load_ecg_data",
                        lambda **kwargs: (X[:40], Y[:40], X[40:], Y[40:]))
    task = {"labels": "five-class", "rows": 25, "balance_classes": True, "max_per_class": None,
            "cap_includes_test": True, "scaler": "sequence_zscore", "seed": 1}
    first, labels, _ = LOAD_TASK(task)
    again, _, _ = LOAD_TASK(task)
    expected = np.sort(np.random.default_rng(1).permutation(60)[:25])
    np.testing.assert_array_equal(first, X[expected])
    np.testing.assert_array_equal(labels, Y[expected])
    np.testing.assert_array_equal(first, again)


def test_load_task_reads_the_named_data_file(tmp_path, monkeypatch):
    """task.data_file replaces the prepared dataset; task.data_sha256 must match it."""
    monkeypatch.setenv("BIORESERVOIR_DATA_DIR", str(tmp_path / "data"))
    pool_file = tmp_path / "data" / "ecg_archive" / "binary_ecg_data.npz"
    pool_file.parent.mkdir(parents=True)
    pool_file.write_bytes(b"v0.1 pool")
    X, Y = _pool(n_per_class=10)
    calls = []
    monkeypatch.setattr("utils.preprocessing.load_ecg_data",
                        lambda **kwargs: calls.append(kwargs) or (X[:15], Y[:15], X[15:], Y[15:]))
    task = {"labels": "binary", "rows": 20, "balance_classes": True, "max_per_class": None,
            "cap_includes_test": False, "scaler": "sequence_zscore", "seed": 1,
            "data_file": "ecg_archive/binary_ecg_data.npz"}
    _, _, sources = LOAD_TASK(task)
    assert sources == [pool_file]
    assert calls[0]["data_dir"] == str(pool_file.parent)
    assert calls[0]["binary_save_file"] == "binary_ecg_data.npz"
    LOAD_TASK(dict(task, data_sha256=provenance.sha256_file(pool_file)))
    with pytest.raises(ValueError, match="data_sha256"):
        LOAD_TASK(dict(task, data_sha256="0" * 64))
    with pytest.raises(FileNotFoundError, match="data_file"):
        LOAD_TASK(dict(task, data_file="ecg_archive/missing.npz"))
    LOAD_TASK(dict(task, labels="five-class", data_file="ecg_archive/binary_ecg_data.npz"))
    assert calls[-1]["save_file"] == "binary_ecg_data.npz"


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

    def test_a_line_joins_each_fold_s_two_scores_only_when_paired(self, tmp_path, monkeypatch,
                                                                    pyplot):
        import utils.visualisation as vis

        a, b = _execute(_config(), tmp_path / "a"), _execute(_config(), tmp_path / "b")
        c = _execute(_config(cv={"folds": 3, "seed": 99}), tmp_path / "c")
        drawn = {}
        monkeypatch.setattr(vis, "save_figure",
                            lambda fig, name, **kwargs: drawn.update({name: fig.axes[0]}) or [])
        for other, paired in ((b, True), (c, False)):
            result = compare_module.compare(a, other, out=tmp_path / "cmp", n_boot=50)
            ax = drawn["fold_f1"]
            joined = [line for line in ax.lines if line.get_color() == "0.75"]
            assert len(joined) == (len(result["folds"]) if paired else 0)
            assert [t.get_text() for t in ax.get_xticklabels()] == [result["a"], result["b"]]
            assert ax.containers[0].lines[0].get_ydata()[0] == pytest.approx(
                result["summary_a"]["mean"]["f1_macro"])

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

    def test_paired_runs_get_a_non_inferiority_bound_and_the_gap_closed(self, tmp_path):
        # kNN over all 16 training beats predicts one class for every beat.
        weak = _execute(_config(readout={"classifier": "KNN", "params": {"n_neighbors": 16}}),
                        tmp_path / "weak")
        strong = _execute(_config(), tmp_path / "strong")
        result = compare_module.compare(weak, strong, n_boot=200, margin=-0.02,
                                        ceiling=strong)
        pooled = result["pooled_delta_f1_macro"]
        assert pooled["ci95"][0] <= pooled["lower_95_one_sided"] <= pooled["delta"]
        assert pooled["non_inferior"] == (pooled["lower_95_one_sided"] > -0.02)
        gap = result["gap_closed"]
        assert pooled["delta"] > 0
        assert gap["gap"] == 1.0 and gap["ci95"] == [1.0, 1.0]
        text = compare_module.format_comparison(result)
        assert "margin -0.020" in text and "gap closed toward" in text
        other = _execute(_config(cv={"folds": 3, "seed": 99}), tmp_path / "other")
        with pytest.raises(ValueError, match="does not share the folds"):
            compare_module.compare(weak, strong, n_boot=10, ceiling=other)
        with pytest.raises(ValueError, match="on the same folds and beats"):
            compare_module.compare(weak, other, n_boot=10, margin=-0.02)

    def test_the_gap_closed_is_undefined_without_a_gap(self):
        np.testing.assert_allclose(compare_module.gap_closed(0.7, [0.8, 0.7], [0.9, 0.9]),
                                   [0.5, 0.0])
        assert np.isnan(compare_module.gap_closed(0.7, 0.8, 0.7))

    def test_scores_average_over_the_classes_of_the_test_labels(self):
        y = np.array([0, 0, 1, 1])
        pred = np.array([0, 2, 1, 1])
        assert compare_module._scores(y, pred)["f1_macro"] == pytest.approx(
            f1_score(y, pred, labels=[0, 1], average="macro"))
        low, high = compare_module.bootstrap_ci(y, y, n_boot=50)
        assert low == high == 1.0


# ---------------------------------------------------------------------------
# Held-out evaluation
# ---------------------------------------------------------------------------

def _three_classes(n_per_class=10, steps=8, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, steps)
    shapes = (t, 1.0 - t, np.sin(np.pi * t))
    X = np.concatenate([s + 0.3 * rng.normal(size=(n_per_class, steps)) for s in shapes])
    return X[..., None], np.eye(3)[np.repeat([0, 1, 2], n_per_class)][:, None, :]


class TestHeldOutEvaluation:
    """``test: {split}``: the readout is fitted on the whole pool and scored on the split."""

    @pytest.fixture
    def split(self, monkeypatch):
        from utils import heldout

        X, Y = _three_classes()
        X_split, Y_split = _three_classes(n_per_class=6, seed=1)
        keep = Y_split[:, 0, 2] == 0  # the split has no beats of class 2
        X_split, Y_split = X_split[keep], Y_split[keep]
        ids = np.arange(100, 100 + len(X_split))
        calls = []
        monkeypatch.setattr(run_module, "load_task", lambda task: (X, Y, []))
        monkeypatch.setattr(heldout, "load_split", lambda name, scaler_type: (
            calls.append((name, scaler_type)) or (X_split, Y_split, ids)))
        return X, X_split, Y_split, ids, calls

    def test_the_split_is_scored_after_fitting_on_the_whole_pool(self, tmp_path, split,
                                                                isolated):
        X, _, Y_split, ids, calls = split
        run_dir = _execute(_config(test={"split": "validation-1"}), tmp_path / "run")
        assert calls == [("validation-1", "sequence_zscore")]
        assert [p.name for p in run_dir.glob("fold-*.npz")] == ["fold-0.npz"]
        with np.load(run_dir / "fold-0.npz", allow_pickle=True) as fold:
            assert len(fold["Y_train"]) == len(X)
            np.testing.assert_array_equal(fold["Y_test"], Y_split)
            np.testing.assert_array_equal(fold["test_index"], ids)
            y_true, y_pred = (run_module.class_indices(fold[k]) for k in ("Y_test", "Y_pred"))
        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert metrics["n_folds"] == 1
        assert metrics["test"]["split"] == "validation-1"
        assert metrics["test"]["n_beats"] == len(ids) and metrics["test"]["classes"] == [0, 1]
        assert metrics["mean"]["f1"] == pytest.approx(
            f1_score(y_true, y_pred, labels=[0, 1], average="macro"))
        low, high = metrics["test"]["f1_macro_ci95"]
        assert 0.0 <= low <= high <= 1.0
        assert metrics["summary"]["f1_macro_ci95_low"] == low
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["beat_ids_sha256"] == hashlib.sha256(ids.astype(np.int64).tobytes()
                                                             ).hexdigest()
        assert manifest["spec"]["test"] == {"split": "validation-1"}
        (entry,) = provenance.load_registry(isolated / "registry.jsonl").values()
        assert entry["evaluation"] == "validation-1"

    def test_two_runs_on_a_split_are_paired_by_beat(self, tmp_path, split):
        a = _execute(_config(test={"split": "validation-1"}), tmp_path / "a")
        b = _execute(_config(test={"split": "validation-1"}), tmp_path / "b")
        result = compare_module.compare(a, b, n_boot=50)
        assert result["paired"] and result["paired_by"] == "beat position"
        assert result["agreement"] == 1.0

    def test_the_test_split_needs_final_test(self, tmp_path, split):
        with pytest.raises(SystemExit, match="final test set"):
            _execute(_config(test={"split": "test"}), tmp_path / "refused")
        assert not (tmp_path / "refused").exists()
        run_dir = _execute(_config(test={"split": "test"}), tmp_path / "run", final_test=True)
        assert json.loads((run_dir / "metrics.json").read_text())["test"]["split"] == "test"

    def test_split_beats_in_the_pool_are_refused(self, tmp_path, split, monkeypatch):
        from utils import heldout

        X, X_split, Y_split, ids, _ = split
        Y = np.eye(3)[np.zeros(2, dtype=int)][:, None, :]
        monkeypatch.setattr(heldout, "load_split", lambda name, scaler_type: (
            np.concatenate([X_split, X[:2]]), np.concatenate([Y_split, Y]),
            np.arange(len(ids) + 2)))
        with pytest.raises(ValueError, match="2 beats of the held-out split"):
            _execute(_config(test={"split": "validation-2"}), tmp_path / "run")

    @pytest.mark.parametrize("change, message", [
        ({"test": {"split": "draw-3"}}, "Unknown held-out split"),
        ({"test": {"split": "test"}, "task": {"labels": "binary"}}, "five-class"),
    ])
    def test_invalid_splits_are_rejected(self, change, message):
        with pytest.raises(ValueError, match=message):
            run_module.resolve(_config(**change))

    def test_a_split_needs_a_simulated_reservoir(self):
        with pytest.raises(ValueError, match="states_from"):
            run_module.resolve({"name": "x", "reservoir": {"states_from": "run"},
                                "readout": {"classifier": "KNN"},
                                "test": {"split": "validation-1"}})

    def test_cross_validated_configs_have_no_test_section(self):
        assert "test" not in run_module.resolve(_config())


def _two_scales(n=60, seed=0):
    """Two classes separated on both features; feature 0 is 10^4 times larger than feature 1."""
    rng = np.random.default_rng(seed)
    labels = np.repeat([0, 1], n // 2)
    centres = np.array([[0.0, 0.0], [1.0, 1.0]])
    X = centres[labels] + 0.6 * rng.normal(size=(n, 2))
    X[:, 0] *= 1e4
    return [x.reshape(1, 2) for x in X], np.eye(2)[labels][:, None, :]


class TestReadoutScaling:
    """The v0.1 readouts were fitted on unscaled features."""

    def test_readouts_are_scaled_by_default_but_not_in_the_v0_1_preset(self):
        assert run_module.resolve(_config())["readout"]["scale"] is True
        v0_1 = run_module.resolve(_config(preset="v0.1"))
        assert v0_1["readout"]["scale"] is False and v0_1["preset"] == "v0.1"

    def test_the_scaler_is_fitted_on_the_training_fold_only(self):
        from training.pipeline import fit_readout

        states, Y = _two_scales()
        readout = run_module.make_readout({"classifier": "KNN", "params": {"n_neighbors": 3},
                                           "scale": True})
        fit_readout(readout, states[:40], Y[:40])
        scaler = readout._fitted_sklearn.named_steps["scale"]
        np.testing.assert_allclose(scaler.mean_, np.vstack(states[:40]).mean(axis=0))

    def test_scaled_predictions_do_not_depend_on_feature_units(self):
        from training.pipeline import fit_readout, predict

        states, Y = _two_scales()
        rescaled = [s * np.array([[1e-4, 1.0]]) for s in states]
        outputs = []
        for data, scale in ((states, True), (rescaled, True), (states, False),
                            (rescaled, False)):
            readout = run_module.make_readout({"classifier": "KNN",
                                               "params": {"n_neighbors": 3}, "scale": scale})
            fit_readout(readout, data[:40], Y[:40])
            outputs.append(np.vstack(predict(readout, data[40:])))
        np.testing.assert_array_equal(outputs[0], outputs[1])
        assert not np.array_equal(outputs[2], outputs[3])


class TestReadoutSelection:
    """The v0.1 readout hyperparameters were chosen on the reported folds."""

    def test_hyperparameters_are_chosen_inside_the_training_fold(self):
        from sklearn.model_selection import GridSearchCV, StratifiedKFold
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        from training.pipeline import fit_readout

        states, Y = _two_scales()
        cfg = run_module.resolve(_config(readout={
            "classifier": "KNN", "params": {}, "search": {"n_neighbors": [1, 5, 15]}}))
        assert cfg["readout"]["inner_folds"] == 3 and cfg["readout"]["seed"] == 3
        readout = run_module.make_readout(cfg["readout"])
        fit_readout(readout, states[:40], Y[:40])
        reference = GridSearchCV(
            Pipeline([("scale", StandardScaler()), ("classifier", KNeighborsClassifier())]),
            {"classifier__n_neighbors": [1, 5, 15]}, scoring="f1_macro",
            cv=StratifiedKFold(3, shuffle=True, random_state=3),
        ).fit(np.vstack(states[:40]), np.argmax(Y[:40, 0], axis=1))
        assert readout._fitted_sklearn.best_params_ == reference.best_params_

    def test_ridge_cv_chooses_alpha_by_leave_one_out_on_the_training_fold(self):
        from sklearn.linear_model import RidgeClassifierCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        from training.pipeline import fit_readout, predict, selected_readout_params

        states, Y = _two_scales()
        cfg = run_module.resolve(_config(readout={"classifier": "RidgeCV"}))
        assert cfg["readout"]["params"] == {"alphas": list(run_module.RIDGE_ALPHAS)}
        readout = run_module.make_readout(cfg["readout"])
        fit_readout(readout, states[:40], Y[:40])
        reference = make_pipeline(StandardScaler(),
                                  RidgeClassifierCV(alphas=run_module.RIDGE_ALPHAS))
        reference.fit(np.vstack(states[:40]), np.argmax(Y[:40, 0], axis=1))
        assert selected_readout_params(readout) == {"alpha": reference[-1].alpha_}
        np.testing.assert_array_equal(
            np.argmax(np.vstack(predict(readout, states[40:])), axis=1),
            reference.predict(np.vstack(states[40:])))

    @pytest.mark.parametrize("readout, chosen", [
        ({"classifier": "Ridge", "params": {}, "search": {"alpha": [0.01, 1.0, 100.0]}},
         "classifier__alpha"),
        ({"classifier": "RidgeCV", "params": {"alphas": [0.01, 1.0, 100.0]}}, "alpha"),
    ])
    def test_runs_record_the_hyperparameters_chosen_in_each_fold(self, tmp_path, readout,
                                                                 chosen):
        metrics = json.loads((_execute(_config(readout=readout), tmp_path / "run")
                              / "metrics.json").read_text())
        for fold in metrics["folds"]:
            assert set(fold["selected"]) == {chosen}

    def test_the_default_ridge_penalties_are_the_exact_half_powers_of_ten(self):
        from decimal import Decimal, getcontext

        getcontext().prec = 50
        assert run_module.RIDGE_ALPHAS == tuple(
            float(Decimal(10) ** (Decimal(k) / 2)) for k in range(-8, 13))

    def test_settings_chosen_at_the_edge_of_their_grid_are_named(self):
        readout = {"classifier": "SVM", "params": {},
                   "search": {"C": [1.0, 10.0, 100.0], "kernel": ["rbf", "linear"]}}
        folds = [{"fold": 0, "selected": {"classifier__C": 10.0, "classifier__kernel": "rbf"}},
                 {"fold": 1, "selected": {"classifier__C": 100.0,
                                          "classifier__kernel": "linear"}}]
        assert run_module.grid_edges(readout, folds) == ["fold 1: classifier__C = 100.0"]
        ridge = {"classifier": "RidgeCV", "params": {"alphas": [0.01, 1.0, 100.0]}}
        assert run_module.grid_edges(ridge, [{"fold": 2, "selected": {"alpha": 0.01}}]) == [
            "fold 2: alpha = 0.01"]

    def test_a_run_warns_when_a_fold_chooses_the_edge_of_a_grid(self, tmp_path):
        cfg = _config(readout={"classifier": "RidgeCV", "params": {"alphas": [0.01, 100.0]}})
        run_log = (_execute(cfg, tmp_path / "run") / "run.log").read_text()
        assert "Readout settings at the edge of their grid (widen it): fold 0: alpha" in run_log

    def test_a_readout_search_runs_on_the_run_processes_with_the_same_result(
            self, tmp_path, monkeypatch):
        cfg = _config(reservoir={"type": "none"}, readout={
            "classifier": "KNN", "params": {}, "search": {"n_neighbors": [1, 5, 15]}})
        jobs = []
        make_readout = run_module.make_readout
        monkeypatch.setattr(run_module, "make_readout", lambda readout, n_jobs=1: (
            jobs.append(n_jobs) or make_readout(readout, n_jobs)))
        one = _execute(cfg, tmp_path / "one")
        two = _execute(cfg, tmp_path / "two", processes=2)
        assert jobs == [1, 2]
        for a, b in zip(_predictions(one), _predictions(two), strict=True):
            np.testing.assert_array_equal(a, b)


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
