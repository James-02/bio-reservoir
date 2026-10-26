"""Unit tests for the realism options (bioreservoir.realism): heterogeneity by group, input
levels and imperfect reset, and their hooks in the node, the builder and run configs."""

import numpy as np
import pytest
from scipy import sparse

from bioreservoir import BioReservoir, lattice_matrix
from bioreservoir.dde import (
    _NP_D_FRAC,
    _NP_DECAY,
    _NP_MU,
    _PARAM_ORDER,
    _VARY_SET,
    generate_node_params,
    get_dde_params,
    prepare_conductances,
    solve_dde_substep,
    substep_history,
)
from bioreservoir.realism import (
    DENSITY_STREAM,
    DILUTION_STREAM,
    INPUT_LEVEL_STREAM,
    RESET_STREAM,
    check_input_levels,
    level_factors,
    level_weights,
    lognormal_factors,
    perturb_start,
    stream,
    vary_colonies,
)
from bioreservoir.topology import scale_conductances
from optimization.reservoirs import build_reservoir
from training import run as run_module
from training import study as study_module
from training.simulate import reseed_noise, simulate_beats

PARAMS = get_dde_params()
D0 = PARAMS["d0"]
KINETIC_COLUMNS = [i for i, name in enumerate(_PARAM_ORDER) if name in _VARY_SET]

# A small P-like array: diffusive coupling on a trap lattice, non-negative input.
P_LIKE = dict(units=16, warmup=5, rk4_substeps=16, input_scaling=0.155113,
              input_connectivity=0.66, input_bias=False, bias_scaling=0.1, dde_scaling=0.01,
              cell_coupling=12.75, noise_in=0.01442, output_variables="IHe",
              topology_type="lattice", coupling="diffusion", diffusion_rate=0.6,
              input_encoding="nonnegative", input_sign="positive", param_heterogeneity_cv=0.065,
              param_heterogeneity_seed=7)
ZERO = dict(heterogeneity_density_cv=0.0, heterogeneity_dilution_cv=0.0, input_levels=1,
            reset_noise=0.0)
X = np.random.default_rng(1).normal(size=(15, 1))


def _beats(n=4, steps=12, seed=0):
    return np.random.default_rng(seed).normal(size=(n, steps, 1))


def _nominal_table(n):
    return generate_node_params(n, 0.0, PARAMS, seed=0)


# ---------------------------------------------------------------------------
# Factors and random streams
# ---------------------------------------------------------------------------

class TestFactors:
    @pytest.mark.parametrize("cv", [0.05, 0.3, 0.5])
    def test_factors_have_mean_one_and_the_requested_cv(self, cv):
        f = lognormal_factors(np.random.default_rng(0), cv, 400_000)
        assert f.min() > 0
        assert f.mean() == pytest.approx(1.0, abs=4 * cv / np.sqrt(len(f)))
        assert f.std() / f.mean() == pytest.approx(cv, rel=0.01)

    def test_a_zero_cv_gives_ones_and_a_negative_cv_is_refused(self):
        np.testing.assert_array_equal(lognormal_factors(np.random.default_rng(0), 0.0, 5), 1.0)
        with pytest.raises(ValueError, match="0 or more"):
            lognormal_factors(np.random.default_rng(0), -0.1, 5)

    def test_streams_repeat_and_are_independent_of_each_other_and_of_the_seed(self):
        n = 200_000
        draws = {key: stream(11, key).standard_normal(n)
                 for key in (DENSITY_STREAM, DILUTION_STREAM, INPUT_LEVEL_STREAM, RESET_STREAM)}
        draws["seed"] = np.random.default_rng(11).standard_normal(n)
        np.testing.assert_array_equal(stream(11, DENSITY_STREAM).standard_normal(n),
                                      draws[DENSITY_STREAM])
        names = list(draws)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                assert abs(np.corrcoef(draws[a], draws[b])[0, 1]) < 0.01, (a, b)
        assert not np.array_equal(stream(12, DENSITY_STREAM).standard_normal(5),
                                  draws[DENSITY_STREAM][:5])


# ---------------------------------------------------------------------------
# Heterogeneity by group
# ---------------------------------------------------------------------------

class TestGroups:
    def test_density_factor_has_the_requested_spread_and_the_growth_factor_follows(self):
        n = 200_000
        table = vary_colonies(_nominal_table(n), D0, density_cv=0.04, seed=3)
        d = table[:, _NP_D_FRAC]
        # At a CV of 0.04 no colony comes near 0.99 d0 (5.5 SD above the mean): no clipping.
        assert d.max() < 0.99 * D0
        assert d.mean() / PARAMS["d"] == pytest.approx(1.0, abs=1e-3)
        assert d.std() / d.mean() == pytest.approx(0.04, rel=0.02)
        np.testing.assert_array_equal(table[:, _NP_DECAY], 1.0 - (d / D0) ** 4)
        others = np.ones(table.shape[1], dtype=bool)
        others[[_NP_D_FRAC, _NP_DECAY]] = False
        np.testing.assert_array_equal(table[:, others], _nominal_table(n)[:, others])

    def test_density_is_clipped_below_the_carrying_capacity(self):
        table = vary_colonies(_nominal_table(50_000), D0, density_cv=3.0, seed=3)
        d = table[:, _NP_D_FRAC]
        assert d.min() == 0.05 and d.max() == pytest.approx(0.99 * D0)
        assert table[:, _NP_DECAY].min() == pytest.approx(1.0 - 0.99 ** 4)
        assert table[:, _NP_DECAY].max() == pytest.approx(1.0 - (0.05 / D0) ** 4)

    def test_dilution_factor_has_the_requested_spread(self):
        n = 200_000
        table = vary_colonies(_nominal_table(n), D0, dilution_cv=0.3, seed=3)
        mu = table[:, _NP_MU] / PARAMS["mu"]
        assert mu.mean() == pytest.approx(1.0, abs=4 * 0.3 / np.sqrt(n))
        assert mu.std() / mu.mean() == pytest.approx(0.3, rel=0.02)
        others = np.arange(table.shape[1]) != _NP_MU
        np.testing.assert_array_equal(table[:, others], _nominal_table(n)[:, others])

    def test_zero_cvs_return_the_table_and_negative_cvs_are_refused(self):
        table = generate_node_params(20, 0.2, PARAMS, seed=1)
        np.testing.assert_array_equal(vary_colonies(table, D0, seed=3), table)
        for key in ("density_cv", "dilution_cv"):
            with pytest.raises(ValueError, match=key):
                vary_colonies(table, D0, seed=3, **{key: -0.1})

    def test_each_group_keeps_its_draws_when_another_group_is_added(self):
        n = 50_000
        kinetic = generate_node_params(n, 0.2, PARAMS, seed=3)
        density = vary_colonies(_nominal_table(n), D0, density_cv=0.2, seed=3)
        dilution = vary_colonies(_nominal_table(n), D0, dilution_cv=0.2, seed=3)
        both = vary_colonies(kinetic, D0, density_cv=0.2, dilution_cv=0.2, seed=3)
        np.testing.assert_array_equal(both[:, KINETIC_COLUMNS], kinetic[:, KINETIC_COLUMNS])
        np.testing.assert_array_equal(both[:, _NP_D_FRAC], density[:, _NP_D_FRAC])
        np.testing.assert_array_equal(both[:, _NP_MU], dilution[:, _NP_MU])
        log = np.log(both)
        for col in (_NP_D_FRAC, _NP_MU):
            for other in [*KINETIC_COLUMNS, _NP_MU if col == _NP_D_FRAC else _NP_D_FRAC]:
                assert abs(np.corrcoef(log[:, col], log[:, other])[0, 1]) < 0.02

    def test_the_node_varies_density_and_washout_from_the_heterogeneity_seed(self):
        kinetic = BioReservoir(units=16, warmup=2, rk4_substeps=16, seed=5,
                               param_heterogeneity_cv=0.1, param_heterogeneity_seed=9)
        node = BioReservoir(units=16, warmup=2, rk4_substeps=16, seed=5,
                            param_heterogeneity_cv=0.1, param_heterogeneity_seed=9,
                            heterogeneity_density_cv=0.2, heterogeneity_dilution_cv=0.3)
        expected = vary_colonies(kinetic.node_params, D0, 0.2, 0.3, seed=9)
        np.testing.assert_array_equal(node.node_params, expected)
        assert node._param_table is node.node_params
        alone = BioReservoir(units=16, warmup=2, rk4_substeps=16, seed=5,
                             param_heterogeneity_seed=9, heterogeneity_density_cv=0.2)
        np.testing.assert_array_equal(alone.node_params[:, KINETIC_COLUMNS],
                                      _nominal_table(16)[:, KINETIC_COLUMNS])
        np.testing.assert_array_equal(alone.node_params[:, _NP_D_FRAC],
                                      expected[:, _NP_D_FRAC])

    def test_varied_density_and_washout_change_the_states(self):
        base = BioReservoir(units=8, warmup=20, rk4_substeps=16, seed=5, input_dim=1)
        for key in ("heterogeneity_density_cv", "heterogeneity_dilution_cv"):
            varied = BioReservoir(units=8, warmup=20, rk4_substeps=16, seed=5, input_dim=1,
                                  **{key: 0.2})
            assert not np.allclose(varied.run(X, reset=True), base.run(X, reset=True)), key

    def test_frozen_delay_scheme_reads_the_varied_table(self):
        node = BioReservoir(units=8, warmup=3, rk4_substeps=16, seed=5, input_dim=1,
                            delay_scheme="frozen", heterogeneity_density_cv=0.2)
        base = BioReservoir(units=8, warmup=3, rk4_substeps=16, seed=5, input_dim=1,
                            delay_scheme="frozen")
        assert not np.allclose(node.run(X, reset=True), base.run(X, reset=True))


class TestStabilityCheck:
    # 10 x 10 lattice, coupling rate 4, 8 substeps (limit 2.7 x 8 = 21.6 per time unit):
    # with the nominal density the fastest exchange mode decays at up to 8.33 + 0.6 + 2 x
    # 1.11 x 4 = 17.8, which the solver integrates; colonies at 0.99 d0 exchange through the
    # membrane at D/(1-d) = 19.4 instead of 8.33, so the bound reaches 28.9 and the solver
    # blows up. (No warm-up: the nominal kinetics need more than 8 substeps.)
    OPTIONS = dict(units=100, warmup=0, rk4_substeps=8, seed=5, coupling="diffusion",
                   diffusion_rate=4.0, W=lattice_matrix(), param_heterogeneity_seed=3)

    def test_uniform_density_is_accepted_and_varied_density_is_refused(self):
        BioReservoir(**self.OPTIONS)
        with pytest.raises(ValueError, match=r"up to 28\.9 per time unit"):
            BioReservoir(**self.OPTIONS, heterogeneity_density_cv=0.5)

    def test_the_refused_setting_blows_up_and_the_accepted_one_does_not(self):
        conductances = prepare_conductances(scale_conductances(lattice_matrix(100, 100), 4.0))
        history = np.abs(np.random.default_rng(0).normal(0.5, 0.1, size=(11, 100, 4)))
        peaks = []
        for cv in (0.0, 0.5):
            table = vary_colonies(_nominal_table(100), D0, density_cv=cv, seed=3)
            ring, pos = substep_history(history, PARAMS["delay"], 8)
            state, peak = history[-1], 0.0
            with np.errstate(all="ignore"):
                try:
                    for _ in range(60):
                        state, pos = solve_dde_substep(state, ring, pos, np.zeros(100), 8,
                                                       node_params=table,
                                                       conductance=conductances)
                        peak = max(peak, float(np.max(np.abs(state))))
                except ZeroDivisionError:  # a state blown up to a singular promoter term
                    peak = np.inf
            peaks.append(peak)
        assert peaks[0] < 1e5 and not peaks[1] < 1e10


# ---------------------------------------------------------------------------
# Input levels
# ---------------------------------------------------------------------------

class TestInputLevels:
    @pytest.mark.parametrize("value", [1, 4, np.int64(2), "continuous"])
    def test_valid_levels(self, value):
        assert check_input_levels(value) == value

    @pytest.mark.parametrize("value", [0, -1, True, 2.0, "2", "uniform", None])
    def test_invalid_levels_are_refused(self, value):
        with pytest.raises(ValueError, match="input_levels"):
            check_input_levels(value)

    @pytest.mark.parametrize("levels", [2, 3, 4])
    def test_k_levels_are_evenly_spaced_up_to_the_full_level(self, levels):
        f = level_factors(np.random.default_rng(0), levels, 20_000)
        np.testing.assert_allclose(np.unique(f), np.arange(1, levels + 1) / levels)
        counts = np.unique(f, return_counts=True)[1]
        assert counts.min() > 0.9 * len(f) / levels

    def test_continuous_levels_are_uniform_on_zero_to_one(self):
        f = level_factors(np.random.default_rng(0), "continuous", 100_000)
        assert 0.0 < f.min() and f.max() <= 1.0
        assert f.mean() == pytest.approx(0.5, abs=0.005)

    def test_one_level_leaves_the_weights_unchanged(self):
        Win = np.array([[0.2], [0.0], [-0.2]])
        assert level_weights(Win, 1, seed=0) is Win

    @pytest.mark.parametrize("levels", [3, "continuous"])
    def test_levels_scale_each_connection_and_keep_zeros_and_signs(self, levels):
        rng = np.random.default_rng(0)
        Win = 0.155 * rng.choice([-1.0, 0.0, 1.0], size=(500, 1))
        out = level_weights(Win, levels, seed=4)
        np.testing.assert_array_equal(out == 0, Win == 0)
        np.testing.assert_array_equal(np.sign(out), np.sign(Win))
        ratio = out[Win != 0] / Win[Win != 0]
        assert 0.0 < ratio.min() and ratio.max() <= 1.0
        if levels == 3:
            np.testing.assert_allclose(np.unique(ratio), [1 / 3, 2 / 3, 1.0])
        sparse_out = level_weights(sparse.csr_matrix(Win), levels, seed=4)
        np.testing.assert_array_equal(sparse_out.toarray(), out)
        np.testing.assert_array_equal(level_weights(Win, levels, seed=4), out)
        assert not np.array_equal(level_weights(Win, levels, seed=5), out)

    def test_today_every_induced_trap_receives_the_same_level(self):
        """Without levels, ReservoirPy's bernoulli weights are +-input_scaling and the
        positive sign makes them input_scaling: one level, or off."""
        node, _ = build_reservoir(dict(P_LIKE), seed=6337)
        node.run(X, reset=True)
        values = np.unique(node.Win.toarray())
        np.testing.assert_array_equal(values, [0.0, P_LIKE["input_scaling"]])

    @pytest.mark.parametrize("levels", [2, 4, "continuous"])
    def test_the_node_gives_the_induced_traps_levels_with_the_reservoir_seed(self, levels):
        one, _ = build_reservoir(dict(P_LIKE), seed=6337)
        node, _ = build_reservoir(dict(P_LIKE, input_levels=levels), seed=6337)
        for n in (one, node):
            n.run(X, reset=True)
        Win, base = node.Win.toarray(), one.Win.toarray()
        np.testing.assert_array_equal(Win != 0, base != 0)
        expected = level_weights(base, levels, seed=6337)
        np.testing.assert_allclose(Win, expected, rtol=1e-15)
        if levels != "continuous":
            np.testing.assert_allclose(np.unique(Win[Win > 0]) / P_LIKE["input_scaling"],
                                       np.arange(1, levels + 1) / levels)

    def test_levels_need_the_per_node_geometry(self):
        with pytest.raises(ValueError, match="input_levels other than 1"):
            BioReservoir(units=4, warmup=1, geometry="v0.1", input_levels=2)


# ---------------------------------------------------------------------------
# Imperfect reset
# ---------------------------------------------------------------------------

class TestImperfectReset:
    def test_each_colony_variable_gets_one_factor_for_its_whole_history(self):
        states = np.random.default_rng(0).uniform(0.5, 2.0, size=(11, 3000, 4))
        ring = np.random.default_rng(1).uniform(0.5, 2.0, size=(3000, 161))
        new_states, new_ring = perturb_start(states, ring, 0.3, np.random.default_rng(2))
        factors = new_states / states
        np.testing.assert_allclose(factors, factors[:1].repeat(11, axis=0), rtol=1e-12)
        np.testing.assert_allclose(new_ring / ring, factors[0, :, 2:3].repeat(161, axis=1),
                                   rtol=1e-12)
        assert factors[0].mean() == pytest.approx(1.0, abs=0.02)
        np.testing.assert_allclose(factors[0].std(axis=0) / factors[0].mean(axis=0), 0.3,
                                   rtol=0.08)
        assert perturb_start(states, None, 0.3, np.random.default_rng(2))[1] is None

    def _node(self, **kwargs):
        options = dict(units=6, warmup=10, rk4_substeps=16, seed=5, input_dim=1,
                       output_variables="IHe", noise_in=0.0, noise_rc=0.0)
        options.update(kwargs)
        return BioReservoir(**options)

    def test_without_reset_noise_every_sequence_starts_from_the_warm_up(self):
        node = self._node()
        first = node.run(X, reset=True)
        np.testing.assert_array_equal(node.run(X, reset=True), first)

    def test_each_reset_draws_a_new_start_and_reseeding_repeats_it(self):
        node = self._node(reset_noise=0.1)
        runs = []
        for seed in (3, 3, 4):
            node.reseed_reset(seed)
            runs.append(node.run(X, reset=True))
        np.testing.assert_array_equal(runs[0], runs[1])
        assert not np.allclose(runs[0], runs[2])
        assert not np.allclose(node.run(X, reset=True), node.run(X, reset=True))
        assert not np.allclose(runs[0], self._node().run(X, reset=True))

    def test_the_start_is_the_perturbed_warm_up_state(self):
        node = self._node(reset_noise=0.2)
        node.run(X[:1], reset=True)
        node.reseed_reset(7)
        node.reset(to_state=np.zeros((1, node.output_dim)))
        expected, ring = perturb_start(node._warmup_states, node._warmup_ring[0], 0.2,
                                       stream(7, RESET_STREAM))
        np.testing.assert_array_equal(node.states, expected)
        np.testing.assert_array_equal(node._ring, ring)
        np.testing.assert_array_equal(node._warmup_states[-1], self._node().states[-1])

    def test_the_first_sequence_of_a_fresh_node_gets_the_same_start_as_any_other(self):
        used = self._node(reset_noise=0.2)
        used.run(X, reset=True)
        fresh = self._node(reset_noise=0.2)
        for node in (used, fresh):
            node.reseed_reset(8)
        np.testing.assert_array_equal(fresh.run(X, reset=True), used.run(X, reset=True))

    def test_the_frozen_scheme_perturbs_its_history(self):
        node = self._node(reset_noise=0.2, delay_scheme="frozen")
        assert not np.allclose(node.run(X, reset=True),
                               self._node(delay_scheme="frozen").run(X, reset=True))

    def test_reseeding_the_noise_reseeds_the_reset(self):
        node = self._node(reset_noise=0.1)
        reseed_noise(node, 12)
        a = node.run(X, reset=True)
        reseed_noise(node, 12)
        np.testing.assert_array_equal(node.run(X, reset=True), a)

    def test_a_beat_gets_the_same_features_in_any_batch(self):
        reservoir, _ = build_reservoir(dict(P_LIKE, reset_noise=0.05), seed=2)
        X_beats = _beats()
        whole = simulate_beats(reservoir, X_beats, noise_seed=5)
        part = simulate_beats(reservoir, X_beats[2:4], noise_seed=5, first_index=2)
        np.testing.assert_array_equal(whole[2:4], part)
        plain, _ = build_reservoir(dict(P_LIKE), seed=2)
        assert not np.allclose(whole, simulate_beats(plain, X_beats, noise_seed=5))

    def test_a_negative_reset_noise_is_refused(self):
        with pytest.raises(ValueError, match="reset_noise"):
            self._node(reset_noise=-0.1)

    def test_the_per_fold_path_refuses_reset_noise(self):
        cfg = {"name": "r", "reservoir": {"params": dict(P_LIKE, reset_noise=0.1)},
               "readout": {"classifier": "RidgeCV"}, "cv": {"simulate": "per_fold"}}
        with pytest.raises(ValueError, match="reset_noise needs cv.simulate: once"):
            run_module.resolve(cfg)
        with pytest.raises(ValueError, match="reset_noise needs cv.simulate: once"):
            run_module.resolve(dict(cfg, preset="v0.1", cv={}))
        run_module.resolve(dict(cfg, cv={"simulate": "once"}))

    def test_a_fold_of_the_per_fold_path_refuses_reset_noise(self):
        from reservoirpy.nodes import ScikitLearnNode
        from sklearn.linear_model import RidgeClassifier

        from training.cross_validate import run_fold

        reservoir, _ = build_reservoir(dict(P_LIKE, reset_noise=0.1), seed=2)
        Y = np.eye(2)[[0, 1, 0, 1]]
        with pytest.raises(ValueError, match="reset_noise needs the beats simulated once"):
            run_fold(reservoir, ScikitLearnNode(RidgeClassifier), _beats(), Y, _beats(), Y,
                     fold_index=0)


# ---------------------------------------------------------------------------
# Options at zero
# ---------------------------------------------------------------------------

V0_1_LIKE = dict(units=12, warmup=4, rk4_substeps=16, input_scaling=0.03,
                 input_connectivity=0.5, input_bias=False, bias_scaling=0.1, rc_scaling=0.007,
                 dde_scaling=0.0006, cell_coupling=12.7, noise_in=0.01442, noise_rc=0.2397,
                 output_variables="IHe", topology_type="distance", interaction_diameters=8.2,
                 fade_alpha=2.8e-5, sparsity=0.58, signed=True, p_excite=0.84,
                 geometry="v0.1", delay_scheme="frozen", normalize_after_masks=False,
                 heterogeneity_mean_preserving=False)


class TestOptionsAtZero:
    @pytest.mark.parametrize("params", [P_LIKE, V0_1_LIKE], ids=["P", "v0.1"])
    def test_the_options_at_zero_give_the_states_without_them(self, params):
        base, _ = build_reservoir(dict(params), seed=6337)
        zero, _ = build_reservoir(dict(params, **ZERO), seed=6337)
        assert zero.node_params is None or np.array_equal(zero.node_params, base.node_params)
        np.testing.assert_array_equal(zero._warmup_states, base._warmup_states)
        for x in _beats(3, 20):
            np.testing.assert_array_equal(zero.run(x, reset=True), base.run(x, reset=True))
        assert (abs(zero.Win - base.Win) > 0).sum() == 0
        np.testing.assert_array_equal(simulate_beats(zero, _beats(), noise_seed=5),
                                      simulate_beats(base, _beats(), noise_seed=5))


# ---------------------------------------------------------------------------
# Builder, recorded trial attributes and configs
# ---------------------------------------------------------------------------

class TestConfigs:
    OPTIONS = dict(heterogeneity_density_cv=0.1, heterogeneity_dilution_cv=0.2,
                   input_levels=3, reset_noise=0.01)

    def test_the_builder_passes_and_records_the_options(self):
        node, attrs = build_reservoir(dict(P_LIKE, **self.OPTIONS), seed=1)
        assert (node.heterogeneity_density_cv, node.heterogeneity_dilution_cv,
                node.input_levels, node.reset_noise) == (0.1, 0.2, 3, 0.01)
        assert {k: attrs[f"res_{k}"] for k in self.OPTIONS} == self.OPTIONS
        _, defaults = build_reservoir(dict(P_LIKE), seed=1)
        assert {k: defaults[f"res_{k}"] for k in self.OPTIONS} == ZERO

    def test_trials_record_the_options_and_rebuild_them(self):
        from optimization.training import RECORDED, _resolve_params_from_trial

        assert {RECORDED[f"res_{k}"] for k in self.OPTIONS} == set(self.OPTIONS)
        _, attrs = build_reservoir(dict(P_LIKE, **self.OPTIONS), seed=1)

        class Trial:
            params: dict = {}
            user_attrs = attrs

        params = _resolve_params_from_trial(Trial(), {})
        assert {k: params[k] for k in self.OPTIONS} == self.OPTIONS

    def test_run_configs_keep_the_options(self):
        resolved = run_module.resolve({
            "name": "r", "reservoir": {"params": dict(P_LIKE, **self.OPTIONS)},
            "readout": {"classifier": "RidgeCV"}})
        assert {k: resolved["reservoir"]["params"][k] for k in self.OPTIONS} == self.OPTIONS
        continuous = run_module.resolve({
            "name": "r", "reservoir": {"params": dict(P_LIKE, input_levels="continuous")},
            "readout": {"classifier": "RidgeCV"}})
        assert continuous["reservoir"]["params"]["input_levels"] == "continuous"

    def test_studies_can_search_the_options(self):
        cfg = {"name": "s", "reservoir": {"params": dict(P_LIKE)},
               "readout": {"classifier": "RidgeCV"},
               "study": {"sampler": {"type": "grid"}, "search": {
                   f"reservoir.params.{k}": {"type": "categorical", "choices": [v]}
                   for k, v in self.OPTIONS.items()}}}
        study_module.check(*study_module.split(cfg))
