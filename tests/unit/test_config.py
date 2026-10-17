"""Unit tests for utils.config: setting precedence and the standard workspace paths."""

import re

import pytest

from utils import config

KEYS = ("BIORESERVOIR_RESULTS_DIR", "BIORESERVOIR_LOGS_DIR", "BIORESERVOIR_ARCHIVE_DIR",
        "BIORESERVOIR_TEST_SETTING")


@pytest.fixture
def config_env(tmp_path, monkeypatch):
    """A temporary, initially absent config.env, with no settings in the environment."""
    path = tmp_path / "config.env"
    monkeypatch.setenv("BIORESERVOIR_CONFIG", str(path))
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)
    return path


def test_default_used_when_setting_absent(config_env):
    assert config.get("BIORESERVOIR_TEST_SETTING", "default") == "default"


def test_config_file_overrides_default(config_env):
    config_env.write_text("# comment\n\nBIORESERVOIR_TEST_SETTING = 'from-file'\n")
    assert config.get("BIORESERVOIR_TEST_SETTING", "default") == "from-file"


def test_environment_overrides_config_file(config_env, monkeypatch):
    config_env.write_text("BIORESERVOIR_TEST_SETTING=from-file\n")
    monkeypatch.setenv("BIORESERVOIR_TEST_SETTING", "from-env")
    assert config.get("BIORESERVOIR_TEST_SETTING", "default") == "from-env"


def test_empty_value_falls_back_to_default(config_env):
    config_env.write_text("BIORESERVOIR_TEST_SETTING=\n")
    assert config.get("BIORESERVOIR_TEST_SETTING", "default") == "default"


def test_default_layout(config_env):
    results = config.PROJECT_ROOT / "results"
    assert config.optuna_db("P2-distance") == results / "optuna" / "optuna-P2-distance.db"
    assert config.runs_dir() == results / "runs"
    assert config.metrics_dir() == results / "metrics"
    assert config.figures_dir("optimization") == results / "figures" / "optimization"
    assert config.logs_dir() == config.PROJECT_ROOT / "logs"


def test_archive_layout(config_env):
    archived = config.PROJECT_ROOT / "archive" / "v0.1"
    assert config.optuna_db("P2-distance", archive="v0.1") == (
        archived / "optuna" / "optuna-P2-distance.db"
    )
    assert config.runs_dir(archive="v0.1") == archived / "runs"
    assert config.metrics_dir(archive="v0.1") == archived / "metrics"


def test_relative_directory_resolves_against_project_root(config_env):
    config_env.write_text("BIORESERVOIR_RESULTS_DIR=scratch/results\n")
    assert config.runs_dir() == config.PROJECT_ROOT / "scratch" / "results" / "runs"


def test_absolute_directory_is_used_as_given(config_env, tmp_path, monkeypatch):
    monkeypatch.setenv("BIORESERVOIR_LOGS_DIR", str(tmp_path / "logs"))
    assert config.logs_dir() == tmp_path / "logs"


def test_log_file_name(config_env, tmp_path, monkeypatch):
    monkeypatch.setenv("BIORESERVOIR_LOGS_DIR", str(tmp_path))
    path = config.log_file("research P2-distance/test")
    assert path.parent == tmp_path
    assert re.fullmatch(r"research-P2-distance-test-\d{8}T\d{6}Z\.log", path.name)
    assert not tmp_path.joinpath(path.name).exists()
