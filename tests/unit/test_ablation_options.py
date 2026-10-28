"""Unit tests for the config options behind the ablations: the colonies' feedback delay
(``reservoir.params.delay``) and scrambled beats (``task.scramble``)."""

import numpy as np
import pytest

from bioreservoir.dde import DEFAULT_DDE_PARAMS
from optimization.reservoirs import build_reservoir
from training import run as run_module
from training import study as study_module

# A small P-like array (diffusive coupling on a trap lattice, the substep delay scheme) and a
# small v0.1-like one (abstract coupling, the frozen delay scheme).
P_LIKE = dict(units=16, warmup=5, rk4_substeps=16, input_scaling=0.155113,
              input_connectivity=0.66, input_bias=False, bias_scaling=0.1, dde_scaling=0.01,
              cell_coupling=12.75, noise_in=0.01442, output_variables="IHe",
              topology_type="lattice", coupling="diffusion", diffusion_rate=0.6,
              input_encoding="nonnegative", input_sign="positive")
V0_1_LIKE = dict(units=12, warmup=4, rk4_substeps=16, input_scaling=0.03,
                 input_connectivity=0.5, input_bias=False, bias_scaling=0.1, rc_scaling=0.007,
                 dde_scaling=0.0006, cell_coupling=12.7, noise_in=0.01442, noise_rc=0.2397,
                 output_variables="IHe", topology_type="distance", interaction_diameters=8.2,
                 fade_alpha=2.8e-5, sparsity=0.58, signed=True, p_excite=0.84,
                 geometry="v0.1", delay_scheme="frozen", normalize_after_masks=False,
                 heterogeneity_mean_preserving=False)
X = np.random.default_rng(1).normal(size=(15, 1))


def _states(params, **changes):
    node, _ = build_reservoir(dict(params, **changes), seed=3)
    return node.run(X, reset=True)


class TestDelay:
    @pytest.mark.parametrize("delay", [0, 2.5, 10])
    def test_the_builder_passes_and_records_the_delay(self, delay):
        node, attrs = build_reservoir(dict(P_LIKE, delay=delay), seed=3)
        assert node.dde_kwargs["delay"] == attrs["res_delay"] == float(delay)

    def test_without_the_key_the_model_default_applies(self):
        node, attrs = build_reservoir(dict(P_LIKE), seed=3)
        assert node.dde_kwargs["delay"] == attrs["res_delay"] == DEFAULT_DDE_PARAMS["delay"]
        assert DEFAULT_DDE_PARAMS["delay"] == 10.0
        np.testing.assert_array_equal(_states(P_LIKE), _states(P_LIKE, delay=10))

    def test_trials_record_the_delay_and_rebuild_it(self):
        from optimization.training import RECORDED, _resolve_params_from_trial

        _, attrs = build_reservoir(dict(P_LIKE, delay=0), seed=3)

        class Trial:
            params: dict = {}
            user_attrs = attrs

        assert RECORDED["res_delay"] == "delay"
        assert _resolve_params_from_trial(Trial(), {})["delay"] == 0.0

    @pytest.mark.parametrize("delay", [-1, -0.5, float("nan"), True, "10", [10]])
    def test_a_delay_that_is_not_a_non_negative_number_is_refused(self, delay):
        with pytest.raises(ValueError, match="delay is the feedback delay in input steps"):
            build_reservoir(dict(P_LIKE, delay=delay), seed=3)

    def test_run_configs_add_the_delay_only_when_it_is_set(self):
        cfg = {"name": "d", "reservoir": {"params": dict(P_LIKE)},
               "readout": {"classifier": "RidgeCV"}}
        assert "delay" not in run_module.resolve(cfg)["reservoir"]["params"]
        cfg["reservoir"]["params"]["delay"] = 0
        assert run_module.resolve(cfg)["reservoir"]["params"]["delay"] == 0

    @pytest.mark.parametrize("params", [P_LIKE, V0_1_LIKE], ids=["substep", "frozen"])
    def test_no_delay_integrates_and_changes_the_states(self, params):
        node, _ = build_reservoir(dict(params, delay=0), seed=3)
        assert node.states.shape[0] == 1  # the history holds the current state only
        if node.delay_scheme == "substep":
            assert node._ring.shape == (params["units"], 1)
        no_delay = node.run(X, reset=True)  # the state check runs at every step
        assert np.isfinite(no_delay).all()
        assert not np.allclose(no_delay, _states(params, delay=10))


def _ecg_pool(n=30, steps=40):
    rng = np.random.default_rng(0)
    return rng.normal(size=(n, steps, 1)), np.eye(5)[rng.integers(0, 5, n)][:, None, :]


ECG_TASK = {"labels": "five-class", "rows": 30, "balance_classes": True, "max_per_class": None,
            "cap_includes_test": True, "scaler": "sequence_zscore", "seed": 6337}
RAW_RUN = {"name": "s", "reservoir": {"type": "none"}, "readout": {"classifier": "RidgeCV"}}


class TestScramble:
    def test_each_beat_keeps_its_values_in_another_order(self):
        X = _ecg_pool()[0]
        scrambled = run_module.scramble_steps(X, seed=6337)
        np.testing.assert_array_equal(np.sort(scrambled, axis=1), np.sort(X, axis=1))
        assert (scrambled != X).any(axis=(1, 2)).all()

    def test_a_beat_depends_only_on_the_seed_and_its_position(self):
        X = _ecg_pool()[0]
        first = run_module.scramble_steps(X, seed=6337)
        np.testing.assert_array_equal(run_module.scramble_steps(X, seed=6337), first)
        assert not np.array_equal(run_module.scramble_steps(X, seed=6338), first)
        changed = X.copy()
        changed[3] = 0.0
        again = run_module.scramble_steps(changed, seed=6337)
        np.testing.assert_array_equal(np.delete(again, 3, axis=0), np.delete(first, 3, axis=0))
        np.testing.assert_array_equal(run_module.scramble_steps(X[:5], seed=6337), first[:5])
        steps = np.arange(X.shape[1])
        order = np.random.default_rng([6337, 7, 4]).permutation(len(steps))
        np.testing.assert_array_equal(run_module.scramble_steps(
            np.tile(steps, (8, 1))[..., None], seed=6337)[7, :, 0], order)

    def test_load_task_scrambles_the_pool_and_keeps_the_labels(self, monkeypatch):
        X, Y = _ecg_pool()
        monkeypatch.setattr("utils.preprocessing.load_ecg_data",
                            lambda **kwargs: (X[:20], Y[:20], X[20:], Y[20:]))
        plain, labels, _ = run_module.load_task(ECG_TASK)
        np.testing.assert_array_equal(plain, X)
        np.testing.assert_array_equal(run_module.load_task(dict(ECG_TASK, scramble=False))[0], X)
        scrambled, scrambled_labels, _ = run_module.load_task(dict(ECG_TASK, scramble=True))
        np.testing.assert_array_equal(scrambled, run_module.scramble_steps(X, seed=6337))
        np.testing.assert_array_equal(scrambled_labels, labels)

    def test_run_configs_add_the_option_only_when_it_is_set(self):
        assert "scramble" not in run_module.resolve(RAW_RUN)["task"]
        assert run_module.resolve({**RAW_RUN, "task": {"scramble": True}})["task"]["scramble"]

    @pytest.mark.parametrize("change, message", [
        ({"task": {"scramble": "yes"}}, "true or false"),
        ({"task": {"scramble": True}, "test": {"split": "validation-1"}}, "held-out split"),
        ({"task": {"scramble": True, "scaler": "standard"}, "cv": {"simulate": "per_fold"}},
         "scales each time step"),
    ])
    def test_invalid_settings_are_refused(self, change, message):
        with pytest.raises(ValueError, match=message):
            run_module.resolve({**RAW_RUN, "readout": {"classifier": "Ridge"},
                                "reservoir": {"params": dict(P_LIKE)}, **change})

    def test_a_study_may_search_the_option(self):
        study_module.check(RAW_RUN, {"search": {"task.scramble": {
            "type": "categorical", "choices": [False, True]}}})
