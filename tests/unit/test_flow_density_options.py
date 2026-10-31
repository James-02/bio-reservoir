"""Unit tests for the operating-condition options: the AHL washout rate set by the flow
(``reservoir.params.washout_rate``, the model's ``mu``) and the cell density in a colony
(``reservoir.params.cell_density``, the model's ``d``)."""

import numpy as np
import pytest

from bioreservoir.dde import DEFAULT_DDE_PARAMS, compute_decay_term
from optimization.reservoirs import build_reservoir
from training import run as run_module
from training import study as study_module

# A small P-like array: diffusive coupling on a trap lattice, with colony-to-colony variation
# of the kinetic rates so the per-colony parameter table is used.
P_LIKE = dict(units=16, warmup=5, rk4_substeps=16, input_scaling=0.155113,
              input_connectivity=0.66, input_bias=False, bias_scaling=0.1, dde_scaling=0.01,
              cell_coupling=12.75, noise_in=0.01442, output_variables="IHe",
              topology_type="lattice", coupling="diffusion", diffusion_rate=0.6,
              input_encoding="nonnegative", input_sign="positive",
              param_heterogeneity_cv=0.1)
X = np.random.default_rng(1).normal(size=(15, 1))
OPTIONS = {"washout_rate": "mu", "cell_density": "d"}


def _states(**changes):
    node, _ = build_reservoir(dict(P_LIKE, **changes), seed=3)
    return node.run(X, reset=True)


class TestFlowAndDensity:
    @pytest.mark.parametrize("washout_rate, cell_density", [(0.3, 0.5), (1.2, 0.85)])
    def test_the_builder_passes_and_records_both(self, washout_rate, cell_density):
        node, attrs = build_reservoir(
            dict(P_LIKE, washout_rate=washout_rate, cell_density=cell_density), seed=3)
        assert node.dde_kwargs["mu"] == attrs["res_washout_rate"] == washout_rate
        assert node.dde_kwargs["d"] == attrs["res_cell_density"] == cell_density
        assert node.dde_kwargs["decay"] == compute_decay_term(cell_density, 0.88)

    def test_without_the_keys_the_model_defaults_apply(self):
        node, attrs = build_reservoir(dict(P_LIKE), seed=3)
        assert attrs["res_washout_rate"] == DEFAULT_DDE_PARAMS["mu"] == 0.6
        assert attrs["res_cell_density"] == DEFAULT_DDE_PARAMS["d"] == 0.7
        np.testing.assert_array_equal(_states(), _states(washout_rate=0.6, cell_density=0.7))

    @pytest.mark.parametrize("key", OPTIONS)
    def test_each_option_changes_the_states(self, key):
        changed = _states(**{key: 0.4})
        assert np.isfinite(changed).all()
        assert not np.allclose(changed, _states())

    @pytest.mark.parametrize("key", OPTIONS)
    def test_the_per_colony_table_uses_the_setting(self, key):
        node, _ = build_reservoir(dict(P_LIKE, **{key: 0.4}), seed=3)
        column = ["CA", "CI", "del_", "alpha", "k1", "f", "gammaA", "gammaI", "b", "k",
                  "gammaH", "g", "D", "d", "mu", "decay"].index(OPTIONS[key])
        np.testing.assert_array_equal(node.node_params[:, column], 0.4)

    def test_trials_record_both_and_rebuild_them(self):
        from optimization.training import RECORDED, _resolve_params_from_trial

        _, attrs = build_reservoir(dict(P_LIKE, washout_rate=0.9, cell_density=0.6), seed=3)

        class Trial:
            params: dict = {}
            user_attrs = attrs

        assert RECORDED["res_washout_rate"] == "washout_rate"
        assert RECORDED["res_cell_density"] == "cell_density"
        rebuilt = _resolve_params_from_trial(Trial(), {})
        assert (rebuilt["washout_rate"], rebuilt["cell_density"]) == (0.9, 0.6)

    @pytest.mark.parametrize("value", [0, -0.5, float("nan"), True, "0.6", [0.6]])
    def test_a_washout_rate_that_is_not_positive_is_refused(self, value):
        with pytest.raises(ValueError, match="washout_rate is the AHL washout rate"):
            build_reservoir(dict(P_LIKE, washout_rate=value), seed=3)

    @pytest.mark.parametrize("value", [0, 0.88, 1.2, -0.1, float("nan"), True, "0.7"])
    def test_a_density_outside_zero_to_d0_is_refused(self, value):
        with pytest.raises(ValueError, match="cell_density is the volume fraction"):
            build_reservoir(dict(P_LIKE, cell_density=value), seed=3)

    def test_run_configs_add_the_options_only_when_they_are_set(self):
        cfg = {"name": "f", "reservoir": {"params": dict(P_LIKE)},
               "readout": {"classifier": "RidgeCV"}}
        resolved = run_module.resolve(cfg)["reservoir"]["params"]
        assert not set(OPTIONS) & set(resolved)
        cfg["reservoir"]["params"].update(washout_rate=0.9, cell_density=0.6)
        resolved = run_module.resolve(cfg)["reservoir"]["params"]
        assert (resolved["washout_rate"], resolved["cell_density"]) == (0.9, 0.6)

    def test_a_study_may_search_both(self):
        cfg = {"name": "f", "reservoir": {"params": dict(P_LIKE)},
               "readout": {"classifier": "RidgeCV"}}
        study_module.check(cfg, {"search": {
            "reservoir.params.washout_rate": {"type": "float", "low": 0.3, "high": 1.2},
            "reservoir.params.cell_density": {"type": "float", "low": 0.5, "high": 0.85}}})
