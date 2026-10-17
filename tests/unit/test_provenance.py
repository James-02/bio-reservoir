"""Unit tests for run manifests and the run registry (revision plan, section 7)."""

import functools
import getpass
import json
import re
import socket
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import optimization.environment as env_mod
from utils import config, provenance

CLEAN = {"commit": "a" * 40, "describe": "v0.1-10-gaaaaaaa", "dirty": False}
START = datetime(2026, 10, 21, 9, 0, 0, tzinfo=timezone.utc)
END = START + timedelta(seconds=90)
SPEC = {"preset": "P2-distance", "units": 20, "folds": 2}


@pytest.fixture
def clean_tree(monkeypatch):
    monkeypatch.setattr(provenance, "git_info", lambda: dict(CLEAN))
    monkeypatch.setattr(env_mod, "_ENV_CACHE", None)


def _manifest(tmp_path, **kwargs):
    data = tmp_path / "data.npz"
    data.write_bytes(b"beats")
    out = tmp_path / "metrics.json"
    out.write_text("{}")
    return provenance.build_manifest("X2-balanced", SPEC, seeds=[6337], started=START, ended=END,
                                     data_files=[data], beat_ids=np.arange(5), outputs=[out],
                                     **kwargs)


def test_run_id_format():
    run_id = provenance.make_run_id("X2-balanced", SPEC, "a" * 40, START)
    assert re.fullmatch(r"20261021T090000Z-X2-balanced-aaaaaaa-[0-9a-f]{8}", run_id)


def test_spec_hash_ignores_key_order():
    assert provenance.spec_hash({"a": 1, "b": 2}) == provenance.spec_hash({"b": 2, "a": 1})


def test_manifest_records_commit_hashes_and_timing(clean_tree, tmp_path):
    manifest = _manifest(tmp_path)
    assert manifest["git"]["commit"] == "a" * 40 and manifest["citable"] is True
    assert manifest["spec_hash"] == provenance.spec_hash(SPEC)
    assert len(manifest["data"][0]["sha256"]) == 64
    assert len(manifest["outputs"][0]["sha256"]) == 64
    assert len(manifest["beat_ids_sha256"]) == 64
    assert manifest["timing"]["wall_s"] == 90.0
    assert manifest["environment"]["cpu_model"]


def test_manifest_has_no_host_names_users_or_absolute_paths(clean_tree, tmp_path):
    text = json.dumps(_manifest(tmp_path))
    assert not re.search(r'"(/|~|[A-Za-z]:\\\\)', text)
    for word in {socket.gethostname(), getpass.getuser()}:
        assert len(word) < 3 or word not in text


def test_dirty_tree_is_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(provenance, "git_info", lambda: dict(CLEAN, dirty=True))
    with pytest.raises(SystemExit, match="allow-dirty"):
        _manifest(tmp_path)
    assert _manifest(tmp_path, allow_dirty=True)["citable"] is False


def test_registry_is_append_only_and_folds_status(clean_tree, tmp_path):
    registry = tmp_path / "registry.jsonl"
    first = _manifest(tmp_path, run_id="run-a")
    second = _manifest(tmp_path, run_id="run-b")
    provenance.register(first, tmp_path / "a", metrics={"f1": 0.80}, registry=registry)
    provenance.register(second, tmp_path / "b", metrics={"f1": 0.85}, registry=registry)
    provenance.mark("run-a", "superseded", by="run-b", registry=registry)
    runs = provenance.load_registry(registry)
    assert len(registry.read_text().splitlines()) == 3
    assert runs["run-a"]["status"] == "superseded" and runs["run-a"]["superseded_by"] == "run-b"
    assert runs["run-b"]["status"] == "ok"
    diff = provenance.diff_runs(runs["run-a"], runs["run-b"])
    assert diff["metrics"]["f1"] == pytest.approx(0.05)


def test_unknown_status_is_rejected(clean_tree, tmp_path):
    with pytest.raises(ValueError):
        provenance.register(_manifest(tmp_path), tmp_path, status="done",
                            registry=tmp_path / "r.jsonl")


def test_git_info_reads_the_checkout():
    if not (config.PROJECT_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    info = provenance.git_info()
    assert re.fullmatch(r"[0-9a-f]{40}", info["commit"])
    assert isinstance(info["dirty"], bool)


def test_git_info_falls_back_to_the_deployment_settings(tmp_path, monkeypatch):
    def no_git(*args, **kwargs):
        raise OSError("git is not installed")

    monkeypatch.setattr(provenance.subprocess, "check_output", no_git)
    monkeypatch.setenv("BIORESERVOIR_CONFIG", str(tmp_path / "absent.env"))
    monkeypatch.setenv("BIORESERVOIR_GIT_COMMIT", "b" * 40)
    monkeypatch.setenv("BIORESERVOIR_GIT_DIRTY", "false")
    assert provenance.git_info() == {"commit": "b" * 40, "describe": "N/A", "dirty": False}
    monkeypatch.delenv("BIORESERVOIR_GIT_DIRTY")
    assert provenance.git_info()["dirty"] is True


def test_write_manifest_round_trips(clean_tree, tmp_path):
    manifest = _manifest(tmp_path)
    path = provenance.write_manifest(tmp_path / "run", manifest)
    assert path == tmp_path / "run" / "manifest.json"
    assert json.loads(path.read_text()) == json.loads(json.dumps(manifest, default=str))


def test_missing_registry_is_empty_and_unknown_runs_are_refused(tmp_path):
    registry = tmp_path / "registry.jsonl"
    assert provenance.load_registry(registry) == {}
    with pytest.raises(SystemExit, match="unknown run"):
        provenance.mark("run-x", "superseded", registry=registry)
    with pytest.raises(ValueError):
        provenance.mark("run-x", "done", registry=registry)


def test_diff_lists_changed_manifest_fields(clean_tree, tmp_path):
    a = json.loads(json.dumps(_manifest(tmp_path, run_id="run-a"), default=str))
    b = json.loads(json.dumps(a))
    b["run_id"], b["spec"]["units"], b["timing"]["wall_s"] = "run-b", 40, 1.0
    diff = provenance.diff_runs({"metrics": {"f1": 0.8}}, {"metrics": {"f1": 0.9}}, (a, b))
    assert diff["metrics"]["f1"] == pytest.approx(0.1)
    assert diff["manifest"] == {"spec.units": [20, 40]}


@pytest.fixture
def registry(clean_tree, tmp_path, monkeypatch):
    """Two registered runs (the second not citable) with manifests in a temporary project."""
    path = tmp_path / "registry.jsonl"
    load = provenance.load_registry
    monkeypatch.setattr(provenance.config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(provenance, "load_registry", lambda registry=path: load(registry))
    monkeypatch.setattr(provenance, "mark", functools.partial(provenance.mark, registry=path))
    for run_id, f1, citable in (("run-a", 0.80, True), ("run-b", 0.85, False)):
        manifest = dict(_manifest(tmp_path, run_id=run_id), citable=citable)
        run_dir = tmp_path / "runs" / run_id
        provenance.write_manifest(run_dir, manifest)
        provenance.register(manifest, run_dir, metrics={"f1": f1}, registry=path)
    return path


def test_cli_lists_runs(registry, capsys):
    provenance.main(["list"])
    assert capsys.readouterr().out.splitlines() == [
        'run-a  ok  {"f1": 0.8}', 'run-b  ok  {"f1": 0.85}  (not citable)']
    provenance.main(["list", "--exp", "X3-other"])
    provenance.main(["list", "--status", "superseded"])
    assert capsys.readouterr().out == ""


def test_cli_marks_shows_and_diffs_runs(registry, capsys):
    provenance.main(["mark", "run-a", "superseded", "--by", "run-b", "--note", "rerun"])
    provenance.main(["show", "run-a"])
    shown = json.loads(capsys.readouterr().out)
    assert shown["registry"]["status"] == "superseded"
    assert shown["registry"]["superseded_by"] == "run-b"
    assert shown["manifest"]["run_id"] == "run-a"
    provenance.main(["diff", "run-a", "run-b"])
    diff = json.loads(capsys.readouterr().out)
    assert diff["metrics"]["f1"] == pytest.approx(0.05)
    assert diff["manifest"] == {"citable": [True, False]}


def test_cli_rejects_unknown_runs(registry):
    with pytest.raises(SystemExit, match="unknown run"):
        provenance.main(["diff", "run-a", "run-x"])
