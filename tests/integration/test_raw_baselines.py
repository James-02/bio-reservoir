"""The v0.1 raw-beat baselines rerun as raw-beat runs (reservoir: {type: none}).

The archived baseline studies fitted their classifiers on the raw beat with their own code
(optimization/baseline.py). A run with a trial's classifier on the raw beat gives the same
fold scores. Skipped without the v0.1 archive.
"""

import json

import pytest

from training import run
from utils import config

TASK = {"rows": 4015, "balance_classes": True, "max_per_class": None,
        "scaler": "sequence_zscore", "seed": 6337}
BINARY_POOL = {"data_file": "ecg_archive/binary_ecg_data.npz",
               "data_sha256": "d0a083a8002b6567b02ae8fd18d9328f4a22cbabacc4669dca0ff4cf4b3782ed"}


@pytest.mark.parametrize("study, trial, task", [
    ("B1-balanced", 253, {**TASK, "labels": "five-class"}),
    ("B3-binary", 356, {**TASK, "labels": "binary", **BINARY_POOL}),
])
def test_a_raw_beat_run_reproduces_the_archived_baseline(study, trial, task, tmp_path):
    import optuna

    database = config.optuna_db(study, "v0.1")
    if not database.is_file():
        pytest.skip("archive not available")
    data = config.data_dir() / task.get("data_file", "ecg/ecg_data.npz")
    if not data.is_file():
        pytest.skip(f"{data.name} not in the data folder")
    # The v0.1 baselines fitted their classifiers on unscaled features.
    cfg = {"name": f"{study}-raw", "seed": 6337, "task": task, "reservoir": {"type": "none"},
           "readout": {"study": study, "trial": trial, "archive": "v0.1", "scale": False},
           "cv": {"folds": 5, "seed": 6337}}
    archived = optuna.load_study(study_name=study, storage=f"sqlite:///{database}")
    (frozen,) = [t for t in archived.trials if t.number == trial]
    run_dir = run.execute(cfg, out=tmp_path / "rerun", processes=1, allow_dirty=True,
                          registry=tmp_path / "registry.jsonl")
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert [f["f1"] for f in metrics["folds"]] == pytest.approx(
        [frozen.user_attrs[f"fold_{k}_f1"] for k in range(5)], abs=1e-12)
    assert metrics["mean"]["f1"] == pytest.approx(frozen.value, abs=1e-12)
