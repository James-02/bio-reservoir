"""Unit tests for the optimisation command line (python -m optimization.optimize)."""

import logging
import sys

import pytest

from optimization import baseline, optimize, readout, training


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    for name, folder in (("RESULTS", "results"), ("ARCHIVE", "archive"), ("LOGS", "logs")):
        monkeypatch.setenv(f"BIORESERVOIR_{name}_DIR", str(tmp_path / folder))
    monkeypatch.setenv("BIORESERVOIR_CONFIG", str(tmp_path / "config.env"))
    logging_calls = []
    monkeypatch.setattr(optimize, "setup_logging",
                        lambda **kwargs: logging_calls.append(kwargs))
    return tmp_path, logging_calls


def _main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["optimize", *argv])
    optimize.main()


def _record(monkeypatch, module, command):
    calls = []
    monkeypatch.setattr(module, command, lambda args: calls.append(args))
    return calls


@pytest.mark.parametrize("argv, module, command", [
    (["research", "--type", "training", "--study", "P2-distance"], training, "cmd_research"),
    (["research", "--type", "readout", "--study_name", "P4-x", "--trial_name", "run"],
     readout, "cmd_research"),
    (["research", "--type", "baseline", "--study_name", "B1-balanced"], baseline,
     "cmd_research"),
    (["evaluate", "--type", "training", "--study", "P2-distance"], training, "cmd_evaluate"),
    (["evaluate", "--type", "readout", "--study_name", "P4-x"], readout, "cmd_evaluate"),
    (["evaluate", "--type", "baseline", "--study_name", "B1-balanced"], baseline,
     "cmd_evaluate"),
    (["rerun-best", "--study", "P2-distance", "--folds", "5"], training, "cmd_rerun_best"),
    (["apply-best", "--study_name", "P4-x", "--trial_name", "run"], readout,
     "cmd_apply_best"),
])
def test_commands_reach_their_handlers(workspace, monkeypatch, argv, module, command):
    calls = _record(monkeypatch, module, command)
    _main(monkeypatch, *argv)
    (args,) = calls
    assert args.command == argv[0]


def test_runs_log_to_a_file_named_after_the_study(workspace, monkeypatch):
    tmp_path, logging_calls = workspace
    calls = _record(monkeypatch, training, "cmd_research")
    _main(monkeypatch, "research", "--type", "training", "--study", "P2-distance",
          "--log-level", "WARNING")
    (args,) = calls
    assert args.log_file.startswith(str(tmp_path / "logs" / "research-P2-distance-"))
    assert logging_calls == [{"level": logging.WARNING, "log_file": args.log_file}]


def test_evaluating_writes_no_log_file(workspace, monkeypatch):
    calls = _record(monkeypatch, training, "cmd_evaluate")
    _main(monkeypatch, "evaluate", "--type", "training", "--study", "P2-distance")
    assert calls[0].log_file is None


def test_archived_studies_are_read_from_the_archive(workspace, monkeypatch):
    tmp_path, _ = workspace
    db = tmp_path / "archive" / "v0.1" / "optuna" / "optuna-P2-distance.db"
    db.parent.mkdir(parents=True)
    db.touch()
    calls = _record(monkeypatch, training, "cmd_evaluate")
    _main(monkeypatch, "evaluate", "--type", "training", "--study", "P2-distance",
          "--archive", "v0.1")
    assert calls[0].storage == f"sqlite:///{db}"


def test_studies_are_listed(workspace, monkeypatch, capsys):
    _main(monkeypatch, "list-studies")
    assert "P2-distance" in capsys.readouterr().out


def test_a_research_type_is_required(workspace, monkeypatch):
    with pytest.raises(SystemExit):
        _main(monkeypatch, "research", "--study", "P2-distance")
