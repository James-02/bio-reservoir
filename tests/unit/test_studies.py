"""Unit tests for optimization.studies registry helpers."""

import pytest

from optimization.studies import (
    BASELINE,
    STUDY_REGISTRY,
    get_study,
    list_studies,
)


class TestStudyRegistry:
    def test_registry_non_empty(self):
        assert STUDY_REGISTRY

    def test_each_study_has_core_keys(self):
        required = {"description", "sampler", "n_trials", "folds"}
        for name, cfg in STUDY_REGISTRY.items():
            missing = required - cfg.keys()
            assert not missing, f"{name} missing keys: {missing}"

    def test_samplers_are_non_empty_strings(self):
        for name, cfg in STUDY_REGISTRY.items():
            assert isinstance(cfg["sampler"], str) and cfg["sampler"], name

    def test_optimize_and_fixed_are_dicts_when_present(self):
        for name, cfg in STUDY_REGISTRY.items():
            if "optimize" in cfg:
                assert isinstance(cfg["optimize"], dict), name
            if "fixed" in cfg:
                assert isinstance(cfg["fixed"], dict), name


class TestGetStudy:
    def test_returns_config_for_known_study(self):
        name = next(iter(STUDY_REGISTRY))
        assert get_study(name) is STUDY_REGISTRY[name]

    def test_unknown_raises_keyerror_listing_available(self):
        with pytest.raises(KeyError) as exc:
            get_study("does-not-exist")
        assert "Available" in str(exc.value)


class TestBaseline:
    def test_baseline_has_core_keys(self):
        for key in ("reservoir_type", "units", "topology_type"):
            assert key in BASELINE


class TestListStudies:
    def test_prints_all_study_names(self, capsys):
        list_studies()
        out = capsys.readouterr().out
        for name in STUDY_REGISTRY:
            assert name in out
