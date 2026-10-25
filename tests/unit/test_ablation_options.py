"""Unit tests for the config options behind the ablations: the colonies' feedback delay
(``reservoir.params.delay``)."""

import numpy as np
import pytest

from bioreservoir.dde import DEFAULT_DDE_PARAMS
from optimization.reservoirs import build_reservoir
from training import run as run_module

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

    @pytest.mark.parametrize("delay", [-1, -0.5, float("nan")])
    def test_a_negative_or_undefined_delay_is_refused(self, delay):
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
