"""Unit tests for optimization.environment provenance collection."""

import getpass
import re
import socket
import subprocess

import pytest

import optimization.environment as env_mod
from optimization.environment import get_environment


@pytest.fixture
def fresh(monkeypatch):
    monkeypatch.setattr(env_mod, "_ENV_CACHE", None)
    for key in ("BIORESERVOIR_MACHINE_LABEL", "BIORESERVOIR_GIT_COMMIT", "BIORESERVOIR_GIT_DIRTY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("BIORESERVOIR_CONFIG", "/nonexistent/config.env")


class TestGetEnvironment:
    def test_returns_dict_with_expected_keys(self, fresh):
        env = get_environment()
        assert isinstance(env, dict)
        for key in ("python_version", "platform", "os_name", "os_version", "timestamp_utc",
                    "cpu_count_logical", "git_commit", "git_dirty", "machine_label"):
            assert key in env

    def test_records_dependency_versions(self, fresh):
        env = get_environment()
        for pkg in ("numpy", "reservoirpy", "optuna", "numba"):
            assert pkg + "_version" in env

    def test_result_is_cached(self, fresh):
        first = get_environment()
        second = get_environment()
        assert first is second

    def test_records_full_git_sha(self, fresh):
        env = get_environment()
        assert re.fullmatch(r"[0-9a-f]{40}", env["git_commit"])
        assert isinstance(env["git_dirty"], bool)


class TestPrivacy:
    """New trials must not store host names, IP addresses, user names or paths."""

    def test_host_identifying_keys_are_absent(self, fresh):
        env = get_environment()
        for key in ("hostname", "ip_address", "cwd", "script_path", "output_dir",
                    "python_executable"):
            assert key not in env

    def test_values_contain_no_host_names_ips_users_or_paths(self, fresh):
        values = [str(v) for v in get_environment().values()]
        forbidden = {socket.gethostname(), socket.getfqdn(), getpass.getuser()}
        for value in values:
            assert not re.search(r"(^|[\s=:'\"])(/|~|[A-Za-z]:\\)", value), value
            assert not re.search(r"\b\d{1,3}(\.\d{1,3}){3}\b", value), value
            for word in forbidden:
                assert len(word) < 3 or word not in value, value

    def test_platform_keeps_only_numeric_kernel_release(self, fresh, monkeypatch):
        import platform

        monkeypatch.setattr(platform, "release", lambda: "5.14.0-427.13.1.el9_4.site-build.x86_64")
        assert env_mod._platform_summary().split("-")[1] == "5.14.0"

    def test_machine_label_and_git_fallback_from_config(self, fresh, monkeypatch):
        def no_git(*args, **kwargs):
            raise subprocess.CalledProcessError(128, "git")

        monkeypatch.setattr(subprocess, "check_output", no_git)
        monkeypatch.setenv("BIORESERVOIR_MACHINE_LABEL", "Host 1")
        monkeypatch.setenv("BIORESERVOIR_GIT_COMMIT", "0" * 40)
        env = get_environment()
        assert env["machine_label"] == "Host 1"
        assert env["git_commit"] == "0" * 40
        assert env["git_dirty"] == "N/A"
