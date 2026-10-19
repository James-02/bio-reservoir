"""Project settings and standard paths, read by every module and script.

Precedence for each setting: environment variable, then the gitignored ``config.env``
(see ``config.env.example``), then the default.  Relative paths resolve against the
repository root, so commands behave the same from any working directory::

    results/                  BIORESERVOIR_RESULTS_DIR   working outputs
        optuna/               Optuna study databases: optuna-<study>.db
        runs/<run>/           fold artefacts: fold-<k>.npz
        metrics/              derived metric tables and statistics
        figures/<topic>/      figures
    logs/                     BIORESERVOIR_LOGS_DIR      log files only
    data/                     BIORESERVOIR_DATA_DIR      datasets; ECG files in data/ecg/
    archive/<version>/        BIORESERVOIR_ARCHIVE_DIR   frozen, read-only results trees
"""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Seed of every reported run (data splits, topology, noise); CLIs take --seed to override.
SEED = 6337


def config_file() -> Path:
    """Return the settings file (``BIORESERVOIR_CONFIG`` or ``<repo>/config.env``)."""
    return Path(os.environ.get("BIORESERVOIR_CONFIG", PROJECT_ROOT / "config.env"))


def file_settings() -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines of the settings file (missing file: no settings)."""
    path = config_file()
    values: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip("'\"")
    return values


def get(name: str, default: str | None = None) -> str | None:
    """Return a setting from the environment, then ``config.env``, then *default*."""
    value = os.environ.get(name)
    if value is None:
        value = file_settings().get(name)
    return default if value in (None, "") else value


def _directory(name: str, default: str) -> Path:
    path = Path(get(name, default))
    return path if path.is_absolute() else PROJECT_ROOT / path


def archive_dir() -> Path:
    return _directory("BIORESERVOIR_ARCHIVE_DIR", "archive")


def data_dir() -> Path:
    return _directory("BIORESERVOIR_DATA_DIR", "data")


def results_root(archive: str | None = None) -> Path:
    """Return the working results root, or the frozen ``archive/<archive>`` tree."""
    if archive:
        return archive_dir() / archive
    return _directory("BIORESERVOIR_RESULTS_DIR", "results")


def optuna_db(study_name: str, archive: str | None = None) -> Path:
    return results_root(archive) / "optuna" / f"optuna-{study_name}.db"


def runs_dir(archive: str | None = None) -> Path:
    return results_root(archive) / "runs"


def metrics_dir(archive: str | None = None) -> Path:
    return results_root(archive) / "metrics"


def figures_dir(topic: str = "") -> Path:
    return results_root() / "figures" / topic


def logs_dir() -> Path:
    return _directory("BIORESERVOIR_LOGS_DIR", "logs")


def log_file(name: str) -> Path:
    """Return a new log file path ``logs/<name>-<UTC timestamp>.log``."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "run"
    return logs_dir() / f"{safe}-{stamp}.log"
