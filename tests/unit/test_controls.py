"""Unit tests for the digital controls (optimization.controls) and their run-config path."""

import importlib
import json
import logging

import numpy as np
import pytest
import yaml

from optimization.controls import (
    CONTROL_DEFAULTS,
    LeakyBank,
    RandomFeatures,
    build_control,
    leaky_bank_size,
)
from optimization.reservoirs import build_reservoir, normalize_reservoir_type
from training import run as run_module
from training import study as study_module
from utils import provenance

cv_module = importlib.import_module("training.cross_validate")


def _beats(n=4, steps=187, seed=0):
    return np.random.default_rng(seed).normal(size=(n, steps, 1))


def _pool(n_per_class=12, steps=8, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, steps)
    X = np.concatenate([t + 0.3 * rng.normal(size=(n_per_class, steps)),
                        1.0 - t + 0.3 * rng.normal(size=(n_per_class, steps))])[..., None]
    return X, np.eye(2)[np.repeat([0, 1], n_per_class)][:, None, :]


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Results and registry under tmp_path; a synthetic two-class pool of 8-step beats."""
    monkeypatch.setenv("BIORESERVOIR_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("BIORESERVOIR_ARCHIVE_DIR", str(tmp_path / "archive"))
    monkeypatch.setattr(provenance, "REGISTRY_FILE", tmp_path / "registry.jsonl")
    monkeypatch.setattr(provenance, "git_info",
                        lambda: {"commit": "abc", "describe": "abc", "dirty": False})
    monkeypatch.setattr(run_module, "load_task", lambda task: (*_pool(), []))
    monkeypatch.setattr(cv_module, "setup_logging", lambda **kwargs: None)
    return tmp_path


def _direct_map(seed, units, length=187, input_scaling=1.0, bias_scaling=1.0):
    """W and b drawn as the node documents: W ~ N(0, g^2 / L), then b ~ U(-beta, beta)."""
    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, input_scaling / np.sqrt(length), size=(length, units))
    return W, rng.uniform(-bias_scaling, bias_scaling, size=units)


# ---------------------------------------------------------------------------
# Random features
# ---------------------------------------------------------------------------

class TestRandomFeatures:
    def test_the_final_state_is_tanh_of_the_random_map_of_the_whole_beat(self):
        beats = _beats()
        node = RandomFeatures(40, seed=11)
        W, b = _direct_map(11, 40)
        for z in beats:
            states = node.run(z, reset=True)
            assert states.shape == (187, 40)
            np.testing.assert_allclose(states[-1], np.tanh(z[:, 0] @ W + b), rtol=1e-12,
                                       atol=1e-13)

    def test_each_step_sees_only_the_beat_so_far(self):
        z = _beats(n=1)[0]
        states = RandomFeatures(30, seed=4).run(z, reset=True)
        W, b = _direct_map(4, 30)
        for t in (1, 47, 120):
            np.testing.assert_allclose(states[t - 1], np.tanh(z[:t, 0] @ W[:t] + b),
                                       rtol=1e-12, atol=1e-13)

    def test_the_linear_activation_gives_the_random_linear_map(self):
        z = _beats(n=1)[0]
        states = RandomFeatures(20, activation="linear", input_scaling=2.0, bias_scaling=0.5,
                                seed=7).run(z, reset=True)
        W, b = _direct_map(7, 20, input_scaling=2.0, bias_scaling=0.5)
        np.testing.assert_allclose(states[-1], z[:, 0] @ W + b, rtol=1e-12, atol=1e-13)

    def test_the_seed_fixes_the_map_and_beats_do_not_depend_on_earlier_beats(self):
        beats = _beats()
        first = RandomFeatures(25, seed=3)
        alone = [RandomFeatures(25, seed=3).run(z, reset=True) for z in beats]
        in_turn = [first.run(z, reset=True) for z in beats]
        for a, b in zip(alone, in_turn, strict=True):
            np.testing.assert_array_equal(a, b)
        other = RandomFeatures(25, seed=4).run(beats[0], reset=True)
        assert not np.allclose(other, alone[0])

    def test_a_copy_gives_the_same_features(self):
        node = RandomFeatures(10, seed=2)
        z = _beats(n=1)[0]
        np.testing.assert_array_equal(node.copy().run(z, reset=True), node.run(z, reset=True))

    @pytest.mark.parametrize("activation", ["tanh", "linear"])
    def test_leading_zero_inputs_without_bias_keep_the_beat_aligned_with_W(self, activation):
        # The state is exactly zero after the leading zeros; the node must not start over.
        z = np.array([0.0, 0.0, 1.0, 2.0, -1.0])
        node = RandomFeatures(4, length=5, activation=activation, bias_scaling=0.0, seed=0)
        states = node.run(z[:, None], reset=True)
        W, _ = _direct_map(0, 4, length=5, bias_scaling=0.0)
        f = np.tanh if activation == "tanh" else (lambda a: a)
        np.testing.assert_allclose(states[-1], f(z @ W), rtol=1e-12, atol=1e-15)

    def test_a_run_without_reset_continues_the_sequence(self):
        z = _beats(n=1, steps=20)[0]
        whole = RandomFeatures(8, length=20, seed=5).run(z, reset=True)
        node = RandomFeatures(8, length=20, seed=5)
        first = node.run(z[:7], reset=True)
        rest = node.run(z[7:])
        np.testing.assert_allclose(np.vstack([first, rest]), whole, rtol=1e-12)

    def test_several_input_channels_use_one_row_of_W_per_sample_and_channel(self):
        z = np.random.default_rng(1).normal(size=(5, 2))
        states = RandomFeatures(6, length=5, seed=9).run(z, reset=True)
        W, b = _direct_map(9, 6, length=10)
        np.testing.assert_allclose(states[-1], np.tanh(z.reshape(-1) @ W + b), rtol=1e-12)

    def test_a_sequence_longer_than_the_length_is_refused(self):
        with pytest.raises(ValueError, match="length"):
            RandomFeatures(5, length=3, seed=0).run(np.zeros((4, 1)), reset=True)

    @pytest.mark.parametrize("kwargs", [{"activation": "relu"}, {"units": 0},
                                        {"length": 0}])
    def test_invalid_settings_are_refused(self, kwargs):
        with pytest.raises(ValueError):
            RandomFeatures(**{"units": 5, **kwargs})


# ---------------------------------------------------------------------------
# Leaky integrators
# ---------------------------------------------------------------------------

class TestLeakyBank:
    TAUS = np.geomspace(1.0, 16.0, 5)

    def _bank(self, **kwargs):
        return LeakyBank(**{"units": 5, "tau_max": 16.0, **kwargs})

    def test_the_step_response_is_one_minus_a_geometric_decay(self):
        level = 2.5
        x = np.full((40, 1), -3.0 + level)  # encoded input clip(x, -3, 8) + 3 = level
        states = self._bank(band_pass=False).run(x, reset=True)
        t = np.arange(1, 41)[:, None]
        a = 1.0 / self.TAUS
        np.testing.assert_allclose(states, level * (1.0 - (1.0 - a) ** t), rtol=1e-12)

    def test_the_impulse_response_is_a_geometric_decay(self):
        x = np.full((30, 1), -3.0)
        x[0] = -3.0 + 4.0
        states = self._bank(band_pass=False).run(x, reset=True)
        t = np.arange(1, 31)[:, None]
        a = 1.0 / self.TAUS
        np.testing.assert_allclose(states, 4.0 * a * (1.0 - a) ** (t - 1), rtol=1e-12,
                                   atol=1e-15)

    def test_band_pass_features_are_differences_of_neighbouring_integrators(self):
        x = _beats(n=1, steps=50)[0]
        states = self._bank().run(x, reset=True)
        assert states.shape == (50, leaky_bank_size(5)) == (50, 9)
        np.testing.assert_allclose(states[:, 5:], states[:, 1:5] - states[:, :4])
        np.testing.assert_array_equal(states[:, :5], self._bank(band_pass=False).run(x))

    def test_the_input_is_clipped_to_the_encoding_range(self):
        x = np.array([[-10.0], [20.0], [0.0]])
        states = LeakyBank(2, tau_max=2.0, band_pass=False).run(x, reset=True)
        np.testing.assert_allclose(states[:, 0], [0.0, 11.0, 3.0])  # tau 1 follows the input
        shifted = LeakyBank(2, tau_max=2.0, band_pass=False, input_range=(-1.0, 1.0))
        np.testing.assert_allclose(shifted.run(x, reset=True)[:, 0], [0.0, 2.0, 1.0])

    def test_the_expansion_is_a_fixed_random_linear_map_of_the_bank(self):
        x = _beats(n=1, steps=60)[0]
        bank = self._bank().run(x, reset=True)
        expanded = self._bank(expansion=50, seed=8)
        states = expanded.run(x, reset=True)
        W = np.random.default_rng(8).normal(0.0, 1.0 / np.sqrt(9), size=(9, 50))
        assert states.shape == (60, 50)
        np.testing.assert_allclose(states, bank @ W, rtol=1e-10, atol=1e-12)

    def test_each_beat_starts_from_empty_integrators(self):
        beats = _beats(n=3, steps=40)
        node = self._bank()
        in_turn = [node.run(z, reset=True) for z in beats]
        for z, states in zip(beats, in_turn, strict=True):
            np.testing.assert_array_equal(states, self._bank().run(z, reset=True))

    def test_a_run_without_reset_continues_the_integration(self):
        x = _beats(n=1, steps=30)[0]
        whole = self._bank().run(x, reset=True)
        node = self._bank()
        first = node.run(x[:11], reset=True)
        np.testing.assert_allclose(np.vstack([first, node.run(x[11:])]), whole, rtol=1e-12)

    @pytest.mark.parametrize("kwargs", [{"tau_min": 0.5}, {"tau_min": 8.0, "tau_max": 4.0},
                                        {"units": 1}, {"expansion": 0},
                                        {"input_range": (1.0, -1.0)}])
    def test_invalid_settings_are_refused(self, kwargs):
        with pytest.raises(ValueError):
            LeakyBank(**{"units": 4, **kwargs})


# ---------------------------------------------------------------------------
# Builders and run configs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alias, kind", [("random_features", "random_features"),
                                         ("Random-Features", "random_features"),
                                         ("leaky_bank", "leaky_bank"),
                                         ("leaky-bank", "leaky_bank")])
def test_control_names_are_reservoir_types(alias, kind):
    assert normalize_reservoir_type(alias) == kind


def test_the_reservoir_builder_builds_the_controls_with_their_settings():
    node, attrs = build_reservoir({"reservoir_type": "random_features", "units": 12,
                                   "activation": "linear"}, seed=5)
    assert isinstance(node, RandomFeatures)
    assert attrs["reservoir_type"] == "random_features" and attrs["res_units"] == 12
    assert attrs["control_activation"] == "linear" and attrs["control_length"] == 187
    assert attrs["topo_seed"] == 5
    node, attrs = build_reservoir({"reservoir_type": "leaky_bank", "units": 24,
                                   "expansion": 100}, seed=5)
    assert isinstance(node, LeakyBank)
    assert attrs["control_expansion"] == 100 and attrs["control_input_range"] == [-3.0, 8.0]
    with pytest.raises(ValueError, match="Unknown control"):
        build_control("esn", {"units": 3}, seed=0)


def _control_config(kind, **params):
    return {"name": "control", "seed": 3,
            "reservoir": {"type": kind, "params": {"units": 6, **params}},
            "readout": {"classifier": "RidgeCV", "scale": True},
            "task": {"rows": 24}, "cv": {"folds": 3}}


class TestControlConfigs:
    @pytest.mark.parametrize("kind", ["random_features", "leaky_bank"])
    def test_configs_resolve_with_every_setting_explicit(self, kind):
        resolved = run_module.resolve(_control_config(kind))
        assert resolved["reservoir"]["type"] == kind
        assert resolved["reservoir"]["params"] == {"units": 6, **CONTROL_DEFAULTS[kind]}
        assert run_module.resolve(resolved) == resolved
        assert yaml.safe_load(run_module.dump_config(resolved)) == resolved

    def test_parameters_of_other_reservoirs_are_reported(self, caplog):
        cfg = _control_config("random_features", warmup=50)
        with caplog.at_level(logging.WARNING, logger="training.run"):
            resolved = run_module.resolve(cfg)
        assert "warmup" not in resolved["reservoir"]["params"]
        assert any("warmup" in r.getMessage() for r in caplog.records)

    @pytest.mark.parametrize("kind", ["random_features", "leaky_bank"])
    def test_observing_model_variables_is_refused_with_the_reason(self, kind):
        cfg = dict(_control_config(kind), features={"variables": ["I"]})
        with pytest.raises(ValueError, match=f"colony model.*the {kind} reservoir"):
            run_module.resolve(cfg)

    @pytest.mark.parametrize("kind, params, features", [
        ("random_features", {"length": 8}, 6), ("leaky_bank", {}, leaky_bank_size(6))])
    def test_a_run_observes_the_control_like_the_reservoir(self, tmp_path, kind, params,
                                                           features):
        cfg = dict(_control_config(kind, **params),
                   features={"aggregation": "images:2", "measurement_noise": 0.05},
                   outputs={"save_states": True})
        run_dir = run_module.execute(cfg, out=tmp_path / "noisy", allow_dirty=True)
        clean = run_module.execute(dict(cfg, features={"aggregation": "images:2"}),
                                   out=tmp_path / "clean", allow_dirty=True)
        noisy_states = np.load(run_dir / "fold-0.npz", allow_pickle=True)["train_states"]
        clean_states = np.load(clean / "fold-0.npz", allow_pickle=True)["train_states"]
        assert noisy_states.shape[-1] == 2 * features
        ratio = noisy_states / clean_states
        assert 0.02 < np.std(ratio) < 0.08 and abs(np.mean(ratio) - 1.0) < 0.02
        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert 0.0 <= metrics["mean"]["f1"] <= 1.0
        assert yaml.safe_load((run_dir / "config.yaml").read_text())["reservoir"]["type"] == kind

    def test_a_study_trial_records_the_control_settings(self):
        cfg = dict(_control_config("random_features", length=8),
                   study={"trials": 2, "sampler": {"type": "grid"},
                          "search": {"reservoir.params.activation": {
                              "type": "categorical", "choices": ["tanh", "linear"]}}})
        study = study_module.run_study(cfg)
        recorded = sorted(yaml.safe_load(t.user_attrs["config"])["reservoir"]["params"]
                          ["activation"] for t in study.trials)
        assert recorded == ["linear", "tanh"]
        params = yaml.safe_load(study.trials[0].user_attrs["config"])["reservoir"]["params"]
        assert set(params) == {"units", "length", "activation", "input_scaling", "bias_scaling"}

    def test_a_study_cannot_search_parameters_the_control_does_not_read(self):
        cfg = dict(_control_config("leaky_bank"),
                   study={"trials": 2, "sampler": {"type": "grid"},
                          "search": {"reservoir.params.activation": {
                              "type": "categorical", "choices": ["tanh"]}}})
        with pytest.raises(ValueError, match="leaky_bank builder has no such parameter"):
            study_module.run_study(cfg)
