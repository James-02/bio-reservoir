"""Unit tests for the deployment helpers: host details only come from config, dry runs
execute nothing, and remote runs are labelled with the host label and git SHA."""

import pytest

from optimization import collect, deploy


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("BIORESERVOIR_CONFIG", str(tmp_path / "config.env"))
    monkeypatch.setenv("BIORESERVOIR_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("BIORESERVOIR_LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("BIORESERVOIR_DEPLOY_HOSTS", "host1=node1.example.org, host2=192.0.2.10")
    monkeypatch.setenv("BIORESERVOIR_DEPLOY_USER", "runner")
    monkeypatch.setenv("BIORESERVOIR_DEPLOY_WORKSPACE", "/srv/bioreservoir/")
    for key in ("BIORESERVOIR_DEPLOY_PYTHON", "BIORESERVOIR_DEPLOY_PIP_PROXY",
                "BIORESERVOIR_DEPLOY_SERVICE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(deploy, "git_state", lambda: ("0" * 40, "false"))
    monkeypatch.setattr(deploy.subprocess, "run", _no_execution)
    return deploy.load_settings()


def _no_execution(*args, **kwargs):
    raise AssertionError("a dry run executed a command")


@pytest.mark.parametrize("python, supported", [("3.9.21", False), ("3.10.14", True),
                                               ("3.12.14", True), ("3.13.1", False)])
def test_host_python_must_be_3_10_to_3_12(python, supported):
    info = {"os_family": "rhel centos fedora", "os_version": "9.6", "configured_python": python}
    assert (deploy.check_requirements(info) == []) is supported


def test_settings_come_from_config(settings):
    assert settings.hosts == {"host1": "node1.example.org", "host2": "192.0.2.10"}
    assert settings.workspace == "/srv/bioreservoir"
    assert deploy.Deployer(settings).target("host2") == "runner@192.0.2.10"


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
