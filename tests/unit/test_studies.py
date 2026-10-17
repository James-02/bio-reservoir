"""Unit tests for optimization.studies registry helpers."""

import re

import pytest

from optimization.studies import (
    BASELINE,
    STUDY_REGISTRY,
    get_study,
    list_studies,
)

# Phase prefix (P1-P4), ESN comparator or baseline (B1-B3), then a lowercase descriptor.
STUDY_NAME_RE = re.compile(r"^(P[1-4]|ESN|B[1-3])-[a-z0-9]+(-[a-z0-9]+)*$")

# Studies in SI Table S5 that are defined in the registry (readout studies are created by
# optimization/readout.py and have no registry entry).
SI_TABLE_S5_REGISTERED = {
    "P1-scaler", "P1-scaling", "P1-integration", "P1-topology-distance", "P1-topology-random",
    "P1-noise-data", "P1-noise-reservoir", "P1-input", "P1-readout", "P1-units", "P1-coupling",
    "P2-distance", "P2-random", "P3-augmentation", "P3-imbalance", "P3-binary",
    "ESN-balanced", "B1-balanced", "B2-unbalanced", "B3-binary",
}


class TestStudyNames:
    @pytest.mark.parametrize("name", sorted(STUDY_REGISTRY))
    def test_names_follow_convention(self, name):
        assert STUDY_NAME_RE.match(name), name

    def test_manuscript_studies_are_registered(self):
        assert not SI_TABLE_S5_REGISTERED - set(STUDY_REGISTRY)


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
