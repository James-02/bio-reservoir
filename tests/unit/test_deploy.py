"""Unit tests for the deployment helpers: host details only come from config, dry runs
execute nothing, and remote runs are labelled with the host label and git SHA."""

import json
import logging
import shutil
import subprocess

import pytest

from optimization import collect, deploy
from utils import config


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("BIORESERVOIR_CONFIG", str(tmp_path / "config.env"))
    monkeypatch.setenv("BIORESERVOIR_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("BIORESERVOIR_LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("BIORESERVOIR_DEPLOY_HOSTS", "host1=node1.example.org, host2=192.0.2.10")
    monkeypatch.setenv("BIORESERVOIR_DEPLOY_USER", "runner")
    monkeypatch.setenv("BIORESERVOIR_DEPLOY_WORKSPACE", "/srv/bioreservoir/")
    for key in ("BIORESERVOIR_DEPLOY_PYTHON", "BIORESERVOIR_DEPLOY_PIP_PROXY",
                "BIORESERVOIR_DEPLOY_SERVICE", "BIORESERVOIR_DEPLOY_WORKSPACES"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(deploy, "git_state", lambda: ("0" * 40, "false"))
    monkeypatch.setattr(deploy.subprocess, "run", _no_execution)
    monkeypatch.setattr(shutil, "which", lambda name: f"/opt/uv/bin/{name}")
    return deploy.load_settings()


def _no_execution(*args, **kwargs):
    raise AssertionError("a dry run executed a command")


@pytest.mark.parametrize("glibc, problem", [
    ("2.12", "needs glibc 2.17"), ("", "needs glibc 2.17"), ("2.17", "cannot draw figures"),
    ("2.28", None), ("2.34", None)])
def test_hosts_need_glibc_2_17_and_2_28_for_figures(glibc, problem):
    problems = deploy.check_requirements({"glibc": glibc, "configured_python": "3.9.21"})
    assert problems == [] if problem is None else problem in problems[0]


def test_settings_come_from_config(settings):
    assert settings.hosts == {"host1": "node1.example.org", "host2": "192.0.2.10"}
    assert settings.workspace == "/srv/bioreservoir"
    assert deploy.Deployer(settings).target("host2") == "runner@192.0.2.10"


def test_a_host_can_have_its_own_workspace(settings, monkeypatch):
    monkeypatch.setenv("BIORESERVOIR_DEPLOY_WORKSPACES", "host2=/root/bioreservoir/")
    deployer = deploy.Deployer(deploy.load_settings(), dry_run=True)
    deployer.sync("host1")
    deployer.sync("host2")
    copies = [cmd[-1] for cmd in deployer.commands if cmd[0] == "rsync"]
    assert "runner@node1.example.org:/srv/bioreservoir/uv.lock" in copies
    assert "runner@192.0.2.10:/root/bioreservoir/uv.lock" in copies
    collect.collect_host(deployer, "host2", studies=["P2-distance"])
    assert deployer.commands[-1][-2] == (
        "runner@192.0.2.10:/root/bioreservoir/results/optuna/optuna-P2-distance.db")
    for value, message in (("host9=/root/bioreservoir", "not a configured host"),
                           ("host2=/root", "WORKSPACES")):
        monkeypatch.setenv("BIORESERVOIR_DEPLOY_WORKSPACES", value)
        with pytest.raises(SystemExit, match=message):
            deploy.load_settings()


def test_missing_settings_are_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("BIORESERVOIR_CONFIG", str(tmp_path / "config.env"))
    monkeypatch.delenv("BIORESERVOIR_DEPLOY_HOSTS", raising=False)
    with pytest.raises(SystemExit, match="BIORESERVOIR_DEPLOY_HOSTS"):
        deploy.load_settings()


@pytest.mark.parametrize("value", ["host1", "=node1", "host1=", "host 1=node1"])
def test_malformed_hosts_are_rejected(value):
    with pytest.raises(SystemExit):
        deploy.parse_hosts(value)


@pytest.mark.parametrize("workspace", ["/srv", "relative/path", "/srv/../etc"])
def test_unsafe_workspace_is_rejected(settings, monkeypatch, workspace):
    monkeypatch.setenv("BIORESERVOIR_DEPLOY_WORKSPACE", workspace)
    with pytest.raises(SystemExit, match="WORKSPACE"):
        deploy.load_settings()


def test_unknown_host_is_rejected(settings):
    with pytest.raises(SystemExit, match="Unknown host"):
        deploy.Deployer(settings).target("host9")


def test_unknown_study_is_rejected():
    with pytest.raises(SystemExit, match="Unknown study"):
        deploy.study_jobs(["not-a-study"], None, [])


def test_launch_labels_runs_with_host_label_and_commit(settings):
    deployer = deploy.Deployer(settings, dry_run=True)
    deployer.launch("host1", deploy.study_jobs(["P2-distance"], None, ["--trials", "5"]))
    *_, target, script = deployer.commands[-1]
    assert target == "runner@node1.example.org"
    assert "BIORESERVOIR_MACHINE_LABEL=host1" in script
    assert f"BIORESERVOIR_GIT_COMMIT={'0' * 40}" in script
    assert '--study P2-distance --processes "$NPROC" --trials 5' in script
    assert "node1.example.org" not in script


def test_push_only_accepts_results_or_archive(settings):
    deployer = deploy.Deployer(settings, dry_run=True)
    with pytest.raises(SystemExit, match="--push"):
        deployer.push("host1", ["config.env"])


def test_collect_reads_results_optuna(settings):
    deployer = deploy.Deployer(settings, dry_run=True)
    collect.collect_host(deployer, "host2", studies=["P2-distance"])
    source, destination = deployer.commands[-1][-2:]
    assert source == "runner@192.0.2.10:/srv/bioreservoir/results/optuna/optuna-P2-distance.db"
    assert destination.endswith("/results/optuna/")


@pytest.fixture
def quiet(monkeypatch):
    """main() configures logging for the whole process."""
    monkeypatch.setattr(deploy, "setup_logging", lambda **kwargs: None)
    monkeypatch.setattr(collect, "setup_logging", lambda **kwargs: None)


def _answer(text, returncode=0):
    """A subprocess.run stand-in that records commands and returns *text*."""
    commands = []

    def run(cmd, **kwargs):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, returncode, stdout=text, stderr="boom")

    run.commands = commands
    return run


class TestDeployer:
    def test_sync_replaces_the_code_and_adds_missing_data(self, settings):
        deployer = deploy.Deployer(settings, dry_run=True)
        deployer.sync("host1")
        mkdir, *copies = deployer.commands
        assert mkdir[-1].endswith("\nmkdir -p /srv/bioreservoir/logs /srv/bioreservoir/results")
        assert {"experiments/", "uv.lock"} <= set(deploy.SYNC_SOURCES)
        assert len(copies) == len(deploy.SYNC_SOURCES) + 1
        assert all("--delete" in cmd for cmd in copies[:-1])
        assert [cmd[-1] for cmd in copies[:-1]] == [
            f"runner@node1.example.org:/srv/bioreservoir/{source}"
            for source in deploy.SYNC_SOURCES]
        assert "--ignore-existing" in copies[-1]
        assert copies[-1][-1] == "runner@node1.example.org:/srv/bioreservoir/data/"

    def test_push_copies_results_to_the_same_place(self, settings, monkeypatch, tmp_path):
        monkeypatch.setattr(deploy.config, "PROJECT_ROOT", tmp_path)
        (tmp_path / "results" / "optuna").mkdir(parents=True)
        (tmp_path / "results" / "optuna" / "optuna-x.db").write_text("db")
        deployer = deploy.Deployer(settings, dry_run=True)
        deployer.push("host1", ["results/optuna/optuna-x.db", "results/optuna"])
        file_copy, folder_copy = (cmd for cmd in deployer.commands if cmd[0] == "rsync")
        assert file_copy[-1].endswith(":/srv/bioreservoir/results/optuna/optuna-x.db")
        assert folder_copy[-2].endswith("/results/optuna/")
        for path in ("results/missing", "/etc/hosts", "results/../config.env"):
            with pytest.raises(SystemExit):
                deployer.push("host1", [path])

    def test_uv_installs_python_and_the_locked_environment(self, settings, monkeypatch):
        monkeypatch.setenv("BIORESERVOIR_DEPLOY_PIP_PROXY", "http://proxy.example:3128")
        deployer = deploy.Deployer(deploy.load_settings(), dry_run=True)
        deployer.setup_venv("host1")
        copy = next(cmd for cmd in deployer.commands if cmd[0] == "rsync")
        assert copy[-2:] == ["/opt/uv/bin/uv", "runner@node1.example.org:/srv/bioreservoir/bin/uv"]
        script = deployer.commands[-1][-1]
        assert "bin/uv sync --frozen" in script and "pip install" not in script
        assert "UV_PYTHON_INSTALL_DIR=/srv/bioreservoir/python" in script
        assert "UV_PROJECT_ENVIRONMENT=/srv/bioreservoir/venv" in script
        assert "HTTPS_PROXY=http://proxy.example:3128" in script
        assert "--no-install-package pillow" in script
        monkeypatch.setattr(shutil, "which", lambda name: None)
        with pytest.raises(SystemExit, match="uv is not installed"):
            deployer.setup_venv("host1")

    def test_services_are_paused_and_started_only_when_configured(self, settings,
                                                                  monkeypatch):
        deployer = deploy.Deployer(settings, dry_run=True)
        deployer.stop_service("host1")
        deployer.start_service("host1")
        assert deployer.commands == []
        monkeypatch.setenv("BIORESERVOIR_DEPLOY_SERVICE", "worker.service")
        deployer = deploy.Deployer(deploy.load_settings(), dry_run=True)
        deployer.stop_service("host1")
        deployer.start_service("host1")
        stop, start = (cmd[-1] for cmd in deployer.commands)
        assert "\nsystemctl stop worker.service" in stop
        assert "\nsystemctl start worker.service" in start

    def test_cleanup_stops_everything_the_session_started(self, settings):
        deployer = deploy.Deployer(settings, dry_run=True)
        deployer.launch("host1", deploy.command_jobs(["sleep 60"], None))
        assert "setsid nohup bash" in deployer.commands[-1][-1]
        deployer.cleanup("host1")
        script = deployer.commands[-1][-1]
        assert 'kill -- -"$(cat /srv/bioreservoir/logs/session.pid)"' in script
        assert "rm -rf -- /srv/bioreservoir" in script
        # pkill -f would match the remote shell running this script and stop it
        assert "pkill" not in script

    def test_remote_commands_report_their_output(self, settings, monkeypatch):
        run = _answer("alive\n")
        monkeypatch.setattr(deploy.subprocess, "run", run)
        deployer = deploy.Deployer(settings)
        assert deployer.session_alive("host1") is True
        assert deployer.rsync("host1", "a", "b") is True
        assert run.commands[0][:2] == ["ssh", "-o"] and run.commands[1][0] == "rsync"
        assert {"BatchMode=yes", "ConnectTimeout=60"} <= set(run.commands[0])
        assert run.commands[0][-1].startswith("unset PYTHONPATH PYTHONHOME\n")
        assert deploy.Deployer(settings, dry_run=True).session_alive("host1") is False

    def test_failed_copies_stop_or_warn(self, settings, monkeypatch, caplog):
        monkeypatch.setattr(deploy.subprocess, "run", _answer("", returncode=1))
        deployer = deploy.Deployer(settings)
        with pytest.raises(SystemExit, match=r"\[host1\] rsync failed: boom"):
            deployer.rsync("host1", "a", "b")
        with caplog.at_level(logging.WARNING, logger="deploy"):
            assert deployer.rsync("host1", "a", "b", check=False) is False
        assert any("boom" in message for message in caplog.messages)


class TestJobs:
    def test_commands_are_named_after_the_label(self):
        assert deploy.command_jobs(["echo a"], None) == [("run", "echo a")]
        assert deploy.command_jobs(["a", "b"], "batch") == [("batch-1", "a"), ("batch-2", "b")]
        with pytest.raises(SystemExit, match="--label"):
            deploy.command_jobs(["a"], "bad label")

    def test_baseline_studies_are_named_with_study_name(self):
        assert deploy.study_jobs(["B1-balanced"], 8, [], study_type="baseline") == [(
            "B1-balanced", "python -m optimization.optimize research --type baseline "
            '--study_name B1-balanced --processes "8"')]
        with pytest.raises(SystemExit, match="Invalid study"):
            deploy.study_jobs(["bad;name"], None, [], study_type="readout")


class TestMonitoring:
    INFO = {"os_name": "Rocky Linux", "os_version": "9.6", "os_family": "rocky rhel centos",
            "kernel": "5.14.0", "arch": "x86_64", "glibc": "2.34", "cpu_model": "CPU",
            "sockets": 2, "physical_cores": 48, "logical_cpus": 96, "ram_gb": 754.3,
            "configured_python": "3.12.1", "pythons": {}}

    @staticmethod
    def _deployer(settings, monkeypatch, answer):
        deployer = deploy.Deployer(settings)
        monkeypatch.setattr(deployer, "ssh", lambda host, script, capture=False, check=True:
                            answer)
        return deployer

    def test_errors_are_logged_per_host(self, caplog):
        def action(host):
            if host == "bad":
                raise SystemExit("unreachable")
            return host.upper()

        with caplog.at_level(logging.ERROR, logger="deploy"):
            assert deploy.for_hosts(["good", "bad"], action) == {"good": "GOOD", "bad": None}
        assert any("[bad] failed" in message for message in caplog.messages)

    def test_the_inventory_is_recorded_per_host(self, settings, monkeypatch, capsys):
        deployer = self._deployer(settings, monkeypatch, "noise\n" + json.dumps(self.INFO))
        deploy.inventory(deployer, ["host1"])
        record = json.loads((config.metrics_dir() / "hosts.json").read_text())
        assert record["hosts"]["host1"]["problems"] == []
        assert record["hosts"]["host1"]["logical_cpus"] == 96
        assert "48 cores / 96 threads" in capsys.readouterr().out
        deploy.inventory(deploy.Deployer(settings, dry_run=True), ["host2"])
        assert set(json.loads((config.metrics_dir() / "hosts.json").read_text())["hosts"]) == {
            "host1"}

    def test_status_progress_logs_and_resets_are_printed(self, settings, monkeypatch, capsys):
        deploy.show_status(self._deployer(settings, monkeypatch, "running|1h05m|trial 12"),
                           ["host1"])
        assert "running" in capsys.readouterr().out
        deploy.show_progress(self._deployer(settings, monkeypatch,
                                            "P2-distance|10|2|1|0|0.9047\nnot a row"),
                             ["host1"], [])
        out = capsys.readouterr().out
        assert "P2-distance" in out and "0.9047" in out and "not a row" not in out
        deploy.show_logs(self._deployer(settings, monkeypatch, "--- run-a.log\nline"),
                         ["host1"], 5)
        assert "=== host1 ===" in capsys.readouterr().out
        deploy.reset_running(self._deployer(settings, monkeypatch, "P2-distance: 2 RUNNING"),
                             ["host1"], ["P2-distance"])
        assert "2 RUNNING" in capsys.readouterr().out

    def test_waiting_collects_when_the_sessions_end(self, settings, monkeypatch):
        collected = []
        monkeypatch.setattr(collect, "collect_host",
                            lambda deployer, host, **kwargs: collected.append(host))
        deploy.wait_and_collect(deploy.Deployer(settings, dry_run=True), ["host1", "host2"],
                                poll_s=0)
        assert sorted(collected) == ["host1", "host2"]


class TestMain:
    @pytest.mark.parametrize("argv", [
        ["--study", "P2-distance"],
        ["--host", "host1", "--host", "host2", "--study", "P2-distance"],
        ["--host", "host1"],
        ["--host", "host1", "--study", "P2-distance", "--run", "echo"],
        ["--reset-running"],
        ["--host", "host1", "--run", "echo", "--", "--trials", "3"],
    ])
    def test_incomplete_requests_are_rejected(self, settings, quiet, argv):
        with pytest.raises(SystemExit):
            deploy.main([*argv, "--dry-run"])

    def test_a_dry_run_launch_goes_through_every_step(self, settings, quiet, caplog):
        with caplog.at_level(logging.INFO, logger="deploy"):
            deploy.main(["--host", "host1", "--study", "P2-distance", "--dry-run", "--",
                         "--trials", "5"])
        assert any("launching: P2-distance" in message for message in caplog.messages)

    def test_cleanup_asks_first(self, settings, quiet, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda prompt: "n")
        with pytest.raises(SystemExit, match="aborted"):
            deploy.main(["--cleanup", "--host", "host1"])


class TestCollect:
    def test_runs_and_logs_are_collected_with_the_databases(self, settings):
        deployer = deploy.Deployer(settings, dry_run=True)
        collect.collect_host(deployer, "host1", runs=["P2-distance-best"], logs=True)
        databases, run, logs = deployer.commands
        assert "--include=optuna-*.db" in databases
        assert run[-2] == ("runner@node1.example.org:"
                           "/srv/bioreservoir/results/runs/P2-distance-best/")
        assert "--include=*.log" in logs and logs[-1].endswith("/logs/remote/host1/")
        with pytest.raises(SystemExit, match="Invalid"):
            collect.collect_host(deployer, "host1", runs=["../elsewhere"])

    def test_remote_databases_are_listed(self, settings, quiet, monkeypatch, capsys):
        monkeypatch.setattr(deploy.Deployer, "ssh",
                            lambda self, host, script, capture=False, check=True:
                            "optuna-P2-distance.db\nnotes.txt")
        collect.main(["--list", "--host", "host1"])
        out = capsys.readouterr().out
        assert "optuna-P2-distance.db" in out and "notes.txt" not in out

    def test_a_dry_run_collects_from_every_host(self, settings, quiet):
        collect.main(["--dry-run", "--study", "P2-distance"])
