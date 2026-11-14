"""The paper's figure, table and number scripts (scripts/paper/): each starts and prints its
usage, and the SI table of the starting state is built from two small oscillation CSVs."""

import csv
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "scripts" / "paper"
SCRIPTS = sorted(p.stem for p in PAPER.glob("*.py"))


def test_every_paper_script_is_listed():
    assert SCRIPTS == ["article_tables", "convergence", "flat_share", "oscillation_map",
                       "paper_figures", "paper_panels", "search_landscape", "si_tables"]


@pytest.mark.parametrize("name", SCRIPTS)
def test_paper_script_prints_its_usage(name):
    env = {"PYTHONPATH": f"{ROOT}:{PAPER}", "MPLBACKEND": "Agg", "PATH": "/usr/bin:/bin"}
    done = subprocess.run([sys.executable, str(PAPER / f"{name}.py"), "--help"], cwd=ROOT,
                          env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert "usage:" in done.stdout


def _oscillation_csv(path, rows, start_ahl=""):
    fields = ["design", "flow", "density", "seed", "start_ahl", "units", "oscillating",
              "period", "ratio"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for flow, density, share, period in rows:
            for seed in (1, 2):
                writer.writerow({"design": "p576", "flow": flow, "density": density,
                                 "seed": seed, "start_ahl": start_ahl, "units": 576,
                                 "oscillating": share, "period": period, "ratio": 2.0})


def test_start_table_compares_the_two_starts(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(PAPER))
    from si_tables import start_table

    low, ahl = tmp_path / "low.csv", tmp_path / "ahl.csv"
    _oscillation_csv(low, [(0.3, 0.8, 0.99, 56.0), (1.2, 0.8, 0.0, "nan")])
    _oscillation_csv(ahl, [(0.3, 0.8, 0.99, 56.0), (1.2, 0.8, 1.0, 45.0)], start_ahl=0.1)
    table = start_table(low, ahl)
    assert r"\label{tab:si-start}" in table
    assert "AHL at 0.1" in table
    assert "0.3 & 99 / 99" in table
    assert "1.2 & 0 / 100" in table
    assert "1.2 & 45" in table
