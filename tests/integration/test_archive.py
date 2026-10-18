"""Integrity checks for the frozen v0.1 archive (skipped when archive/v0.1 is absent)."""

import csv
import hashlib
import json
import os
import sqlite3

import numpy as np
import optuna
import pytest

from optimization.studies import STUDY_REGISTRY
from utils import config

ARCHIVE = config.results_root(archive="v0.1")

pytestmark = pytest.mark.skipif(not ARCHIVE.is_dir(), reason="archive/v0.1 not present")

# study -> (trials, complete trials, best trial, best macro F1); SI Table S5.
STUDIES = {
    "P1-scaler": (21, 21, 2, 0.7601452409788916),
    "P1-scaling": (288, 288, 156, 0.8957258910404304),
    "P1-integration": (128, 128, 61, 0.8157754301168445),
    "P1-topology-distance": (256, 256, 169, 0.8363974881507239),
    "P1-topology-random": (256, 256, 147, 0.7862000685318802),
    "P1-noise-data": (96, 96, 49, 0.8032070520911844),
    "P1-noise-reservoir": (144, 144, 111, 0.7945191825935767),
    "P1-input": (192, 192, 118, 0.8650401874000209),
    "P1-readout": (192, 191, 181, 0.7831075849135057),
    "P1-units": (25, 25, 7, 0.8953305770503306),
    "P1-coupling": (128, 128, 68, 0.8961690814812822),
    "P2-distance": (768, 768, 416, 0.9046992931610176),
    "P2-random": (1044, 800, 1017, 0.9068414407391866),
    "P3-augmentation": (49, 49, 37, 0.8843841350374861),
    "P3-imbalance": (10, 5, 1, 0.9034640970796113),
    "P4-readout-balanced": (950, 946, 833, 0.8952663735055323),
    "P4-readout-unbalanced": (939, 929, 604, 0.9160648991200807),
    "P4-readout-binary": (974, 971, 651, 0.9328112065567975),
    "ESN-balanced": (240, 240, 239, 0.8790407910522854),
    "ESN-readout": (1008, 1008, 364, 0.8957813795939952),
    "B1-balanced": (480, 480, 253, 0.8863582622041463),
    "B2-unbalanced": (480, 480, 225, 0.9153699025708306),
    "B3-binary": (480, 480, 356, 0.9354693897256402),
}
HOST_KEYS = {
    "env_hostname", "env_ip_address", "env_cwd", "env_script_path", "env_output_dir",
    "env_python_executable",
}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def test_checksums_match():
    lines = (ARCHIVE / "SHA256SUMS").read_text().splitlines()
    listed = {line.split("  ", 1)[1][2:]: line.split("  ", 1)[0] for line in lines}
    present = {p.relative_to(ARCHIVE).as_posix() for p in ARCHIVE.rglob("*") if p.is_file()}
    assert present - {"SHA256SUMS"} == set(listed)
    for rel, digest in listed.items():
        assert _sha256(ARCHIVE / rel) == digest, rel


def test_archive_is_read_only():
    for path in [ARCHIVE, *ARCHIVE.rglob("*")]:
        assert not os.access(path, os.W_OK) or os.geteuid() == 0, path


def test_expected_studies_only():
    names = {p.name[len("optuna-"):-len(".db")] for p in (ARCHIVE / "optuna").glob("optuna-*.db")}
    assert names == set(STUDIES)


@pytest.mark.parametrize("study_name", sorted(STUDIES))
def test_study_contents(study_name):
    n_trials, n_complete, best_trial, best_value = STUDIES[study_name]
    path = ARCHIVE / "optuna" / f"optuna-{study_name}.db"
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        keys = {k for (k,) in con.execute("SELECT DISTINCT key FROM trial_user_attributes")}
        assert not keys & HOST_KEYS
        labelled = con.execute(
            "SELECT count(DISTINCT trial_id) FROM trial_user_attributes WHERE key='env_machine_label'"
        ).fetchone()[0]
        assert labelled == n_trials
        stored = {json.loads(v) for (v,) in con.execute(
            "SELECT DISTINCT value_json FROM trial_user_attributes WHERE key='study_name'")}
        assert stored <= {study_name}
    finally:
        con.close()

    study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{path}")
    trials = study.get_trials(deepcopy=False)
    complete = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert (len(trials), len(complete)) == (n_trials, n_complete)
    assert study.best_value == best_value
    assert best_trial in {t.number for t in complete if t.value == best_value}


def test_manifest_lists_every_run_and_study():
    with open(ARCHIVE / "MANIFEST.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    listed = {row["path"] for row in rows}
    runs = {f"runs/{p.name}" for p in (ARCHIVE / "runs").iterdir()}
    dbs = {f"optuna/{p.name}" for p in (ARCHIVE / "optuna").iterdir()}
    assert runs | dbs <= listed


def test_run_names_match_fold_files():
    for run in sorted((ARCHIVE / "runs").iterdir()):
        folds = sorted(run.glob("fold-*.npz"))
        assert folds, run
        with np.load(folds[0], allow_pickle=True) as z:
            assert z["trial_name"].item().decode() == run.name


@pytest.mark.parametrize("study_name", ["B1-balanced", "B2-unbalanced", "B3-binary"])
def test_baseline_registry_entries_match_the_archived_trials(study_name):
    """The registry runs the no-reservoir baselines as the archived trials ran."""
    entry = STUDY_REGISTRY[study_name]
    fixed = entry["fixed"]
    expected = {
        "dataset_instances": entry["instances"], "dataset_binary": entry["binary"],
        "dataset_balance_classes": fixed["balance_classes"],
        "dataset_max_per_class": str(fixed["max_per_class"]),
        "dataset_scaler": fixed["scaler_type"], "dataset_noise_rate": fixed["noise_rate"],
        "dataset_noise_ratio": fixed["noise_ratio"],
        "metric_primary_average": fixed["objective_average"], "metric_n_folds": entry["folds"],
    }
    path = ARCHIVE / "optuna" / f"optuna-{study_name}.db"
    study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{path}")
    for trial in study.get_trials(deepcopy=False,
                                  states=[optuna.trial.TrialState.COMPLETE]):
        assert {key: trial.user_attrs[key] for key in expected} == expected, trial.number


@pytest.mark.parametrize("study_name", sorted(
    name for name in STUDIES if STUDY_REGISTRY.get(name, {}).get("type") != "baseline"
    and name in STUDY_REGISTRY))
def test_best_trials_resolve_to_the_values_they_ran_with(study_name):
    """The run framework rebuilds every archived reservoir study's best trial."""
    from optimization.training import RECORDED
    from training import run

    resolved = run.resolve({"name": "x", "reservoir": {"study": study_name, "archive": "v0.1"},
                            "readout": {"classifier": "KNN"}})
    path = ARCHIVE / "optuna" / f"optuna-{study_name}.db"
    attrs = optuna.load_study(study_name=study_name,
                              storage=f"sqlite:///{path}").best_trial.user_attrs
    params = resolved["reservoir"]["params"]
    checked = [key for attr, key in RECORDED.items()
               if attrs.get(attr) is not None and key in params]
    assert checked
    for attr, key in RECORDED.items():
        if attrs.get(attr) is not None and key in params:
            assert params[key] == attrs[attr], key
    assert resolved["features"]["aggregation"] == attrs.get("readout_aggregation", "final")
    assert resolved["cv"]["noise_rate"] == attrs["dataset_noise_rate"]
    assert resolved["cv"]["noise_ratio"] == attrs["dataset_noise_ratio"]
