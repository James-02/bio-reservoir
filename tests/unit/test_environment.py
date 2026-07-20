"""Unit tests for optimization.environment provenance collection."""

import optimization.environment as env_mod
from optimization.environment import get_environment


class TestGetEnvironment:
    def test_returns_dict_with_expected_keys(self, monkeypatch):
        monkeypatch.setattr(env_mod, "_ENV_CACHE", None)
        env = get_environment()
        assert isinstance(env, dict)
        for key in ("python_version", "platform", "timestamp_utc", "cwd"):
            assert key in env

    def test_records_dependency_versions(self, monkeypatch):
        monkeypatch.setattr(env_mod, "_ENV_CACHE", None)
        env = get_environment()
        for pkg in ("numpy", "reservoirpy", "optuna", "numba"):
            assert pkg + "_version" in env

    def test_result_is_cached(self, monkeypatch):
        monkeypatch.setattr(env_mod, "_ENV_CACHE", None)
        first = get_environment()
        second = get_environment()
        assert first is second
