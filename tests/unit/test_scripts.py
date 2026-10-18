"""Unit tests for the ablation and heterogeneity scripts (their settings; nothing is run)."""

import importlib.util
from pathlib import Path

import pytest

from training.run import PRESETS

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
V0_1 = PRESETS["v0.1"]["reservoir"]


def _load(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def capture(monkeypatch):
    """Record the topology and reservoir arguments of a script instead of building them."""

    def patch(module):
        calls = {}

        def fake_matrix(**kwargs):
            calls["W"] = kwargs
            return "W"

        def fake_node(**kwargs):
            calls["node"] = kwargs
            return "node"

        monkeypatch.setattr(module, "distance_matrix", fake_matrix)
        monkeypatch.setattr(module, "BioReservoir", fake_node)
        return calls

    return patch


class TestHeterogeneityScript:
    """The v0.1 sweep augmented the training folds, and its five CV = 0 runs were the
    same reservoir (only the parameter draws were seeded per point)."""

    def test_training_folds_are_not_augmented_by_default(self):
        het = _load("run_heterogeneity")
        assert het.NOISE_RATE == 0.0 and het.NOISE_RATIO == 0.0

    def test_each_grid_seed_has_its_own_array(self):
        het = _load("run_heterogeneity")
        points = het.grid_points(seed=6337)
        assert len(points) == len(het.CV_LEVELS) * len(het.HET_SEEDS)
        at_zero = [(het_seed, array_seed) for cv, het_seed, array_seed in points if cv == 0.0]
        assert sorted(array for _, array in at_zero) == sorted(het.HET_SEEDS)
        fixed = het.grid_points(seed=6337, fixed_array=True)
        assert {array for _, _, array in fixed} == {6337}

    def test_seeds_and_preset_reach_the_reservoir(self, capture):
        het = _load("run_heterogeneity")
        calls = capture(het)
        het._build_reservoir(0.2, het_seed=100, seed=300)
        assert calls["W"]["seed"] == calls["node"]["seed"] == 300
        assert calls["node"]["param_heterogeneity_seed"] == 100
        assert calls["node"]["param_heterogeneity_cv"] == 0.2
        assert calls["node"]["param_heterogeneity_mean_preserving"] is True
        assert calls["node"]["geometry"] == "per_node"
        assert calls["W"]["normalize_after_masks"] is True

        het._build_reservoir(0.2, het_seed=100, seed=300, overrides=V0_1)
        assert calls["node"]["param_heterogeneity_mean_preserving"] is False
        assert calls["node"]["geometry"] == "v0.1"
        assert calls["node"]["delay_scheme"] == "frozen"
        assert calls["W"]["normalize_after_masks"] is False


class TestAblationScript:
    """The v0.1 ablation script augmented the training folds unless told not to; the
    v0.1 ablations were run without augmentation."""

    def test_training_folds_are_not_augmented_by_default(self):
        ablation = _load("run_ablation")
        assert ablation.NOISE_RATE == 0.0 and ablation.NOISE_RATIO == 0.0

    @pytest.mark.parametrize("condition, rc_scaling, dde_kwargs", [
        ("full", None, {}), ("no-coupling", 0.0, {}), ("no-delay", None, {"delay": 0}),
    ])
    def test_conditions_and_preset_reach_the_reservoir(self, capture, condition, rc_scaling,
                                                       dde_kwargs):
        ablation = _load("run_ablation")
        calls = capture(ablation)
        ablation._build_reservoir(condition, seed=7)
        expected = ablation.RESERVOIR_PARAMS["rc_scaling"] if rc_scaling is None else rc_scaling
        assert calls["node"]["rc_scaling"] == expected
        assert calls["node"]["dde_kwargs"] == dde_kwargs
        assert calls["node"]["geometry"] == "per_node"
        assert calls["node"]["delay_scheme"] == "substep"
        assert calls["W"]["normalize_after_masks"] is True

        ablation._build_reservoir(condition, seed=7, overrides=V0_1)
        assert calls["node"]["geometry"] == "v0.1"
        assert calls["node"]["delay_scheme"] == "frozen"
        assert calls["W"]["normalize_after_masks"] is False
