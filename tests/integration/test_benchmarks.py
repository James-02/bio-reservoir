"""The v0.1 headline benchmark configs (experiments/benchmarks/v0.1) against the archive.

Integration checks (need the archive and the dataset): each config builds the archived
reservoir and readout and loads the archived beat pool, and refitting the balanced
readout on the archived states reproduces the Phase 4 study's value. The full reruns are
benchmarks (``pytest -m benchmark`` on a capable host): each config runs through the
runner and is compared fold by fold with its archived run.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from optimization.reservoirs import build_reservoir
from training import compare, run
from utils import config

CONFIGS = sorted((Path(__file__).resolve().parents[2] / "experiments" / "benchmarks" / "v0.1")
                 .glob("*.yaml"))
NODE_KEYS = {
    "bioreservoir": ("units", "warmup", "input_scaling", "rc_scaling", "dde_scaling",
                     "cell_coupling", "noise_in", "noise_rc", "input_connectivity",
                     "bias_scaling", "output_variables"),
    "esn": ("units", "lr", "sr", "input_scaling", "noise_in", "noise_rc", "rc_connectivity",
            "input_connectivity", "bias_scaling"),
}
# Archived runs whose folds the loader cannot rebuild, compared by mean F1 instead: the
# loader's binary-full pool has the archived size and per-fold class counts, but the
# archived fold labels come in an order no pool it draws reproduces.
UNPAIRED = {"binary-full"}


def _reference_fold(cfg):
    try:
        path = compare.run_path(cfg["reference"])
    except FileNotFoundError:
        pytest.skip(f"archived run {cfg['reference']} not available")
    with np.load(path / "fold-0.npz", allow_pickle=True) as data:
        return {key: data[key] for key in ("Y_train", "Y_test")} | {
            "model_hypers": data["model_hypers"].item()}


def _require_dataset(cfg):
    name = "binary_ecg_data.npz" if cfg["task"]["labels"] == "binary" else "ecg_data.npz"
    if not (config.data_dir() / "ecg" / name).is_file():
        pytest.skip("dataset not prepared (python -m utils.preprocessing)")


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_config_builds_the_archived_models(path):
    cfg = run.load_config(path)
    archived = _reference_fold(cfg)["model_hypers"]
    params = dict(cfg["reservoir"]["params"], reservoir_type=cfg["reservoir"]["type"])
    node, _ = build_reservoir(params, seed=cfg["reservoir"]["seed"])
    node.initialize(np.zeros((1, 1)))  # the archived hypers were recorded after a run
    for key in NODE_KEYS[cfg["reservoir"]["type"]]:
        assert node.hypers[key] == archived[key], key
    archived_readout = archived["model_hypers"]
    for key, value in cfg["readout"]["params"].items():
        assert archived_readout[key] == value, key
    assert run.classifier_class(cfg["readout"]["classifier"]) is archived["model"]


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_config_loads_the_archived_pool(path):
    cfg = run.load_config(path)
    archived = _reference_fold(cfg)
    _require_dataset(cfg)
    _, Y, _ = run.load_task(cfg["task"])
    archived_labels = np.concatenate([archived["Y_train"], archived["Y_test"]])
    assert len(Y) == len(archived_labels)
    np.testing.assert_array_equal(
        np.bincount(np.argmax(Y.reshape(len(Y), -1), axis=1)),
        np.bincount(np.argmax(archived_labels.reshape(len(Y), -1), axis=1)))


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_refitting_the_balanced_readout_reproduces_the_phase4_value(tmp_path):
    import optuna

    if not config.optuna_db("P4-readout-balanced", "v0.1").is_file():
        pytest.skip("archive not available")
    try:
        compare.run_path("archive:v0.1/P2-distance-best")
    except FileNotFoundError:
        pytest.skip("archived states not available")
    study = optuna.load_study(
        study_name="P4-readout-balanced",
        storage=f"sqlite:///{config.optuna_db('P4-readout-balanced', 'v0.1')}")
    run_dir = run.execute(
        {"name": "refit-P4-readout-balanced", "preset": "v0.1",
         "reservoir": {"states_from": "archive:v0.1/P2-distance-best"},
         "readout": {"study": "P4-readout-balanced", "trial": 833, "archive": "v0.1"}},
        out=tmp_path / "refit", processes=5, allow_dirty=True,
        registry=tmp_path / "registry.jsonl")
    metrics = json.loads((run_dir / "metrics.json").read_text())
    (trial,) = [t for t in study.trials if t.number == 833]
    assert metrics["mean"]["f1"] == pytest.approx(trial.value, abs=1e-12)


@pytest.mark.benchmark
@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_headline_rerun_matches_the_archive(path, tmp_path):
    cfg = run.load_config(path)
    _reference_fold(cfg)
    _require_dataset(cfg)
    processes = int(os.environ.get("BIORESERVOIR_BENCHMARK_PROCESSES", "5"))
    run_dir = run.execute(cfg, out=tmp_path / cfg["name"], processes=processes,
                          allow_dirty=True, registry=tmp_path / "registry.jsonl")
    result = compare.compare(cfg["reference"], run_dir, out=tmp_path / "compare")
    print(compare.format_comparison(result))
    if path.stem in UNPAIRED:
        means = [result[side]["mean"]["f1_macro"] for side in ("summary_a", "summary_b")]
        assert abs(means[1] - means[0]) <= 0.005
        return
    assert result["paired"], "the rerun's folds differ from the archived run's"
    assert result["agreement"] >= 0.99
    assert abs(result["pooled_delta_f1_macro"]["delta"]) <= 0.005
