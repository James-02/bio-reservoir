"""Unit tests for Optuna storage resolution (working results/optuna/ vs frozen archive/)."""

import sys

import pytest

from optimization import optimize
from optimization.base import get_default_storage


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("BIORESERVOIR_CONFIG", str(tmp_path / "config.env"))
    monkeypatch.setenv("BIORESERVOIR_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("BIORESERVOIR_ARCHIVE_DIR", str(tmp_path / "archive"))
    monkeypatch.setenv("BIORESERVOIR_LOGS_DIR", str(tmp_path / "logs"))
    return tmp_path


def test_working_storage_is_in_results(workspace):
    db = workspace / "results" / "optuna" / "optuna-P2-distance.db"
    assert get_default_storage("P2-distance") == f"sqlite:///{db}?timeout=60"
    assert db.parent.is_dir()


def test_missing_working_study_is_an_error_when_required(workspace):
    with pytest.raises(SystemExit, match="Study database not found"):
        get_default_storage("P2-distance", must_exist=True)


def test_archived_storage_points_to_archive(workspace):
    db = workspace / "archive" / "v0.1" / "optuna" / "optuna-P2-distance.db"
    db.parent.mkdir(parents=True)
    db.touch()
    assert get_default_storage("P2-distance", archive="v0.1") == f"sqlite:///{db}"
    assert not (workspace / "results").exists()


def test_missing_archived_study_is_an_error(workspace):
    with pytest.raises(SystemExit, match="Archived study database not found"):
        get_default_storage("P2-distance", archive="v0.1")


def test_research_refuses_archive(workspace, monkeypatch):
    argv = ["optimize", "research", "--type", "training", "--study", "P2-distance",
            "--archive", "v0.1"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="read-only"):
        optimize.main()
    assert not (workspace / "logs").exists()
