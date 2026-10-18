"""Unit tests for the bioreservoir.dde DDE solver."""

import itertools
import logging

import numba
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from bioreservoir import dde
from bioreservoir.dde import (
    DEFAULT_DDE_PARAMS,
    clear_prepared_cache,
    compute_decay_term,
    generate_node_params,
    get_dde_params,
    log_dde_params,
    solve_dde_rk4,
)

# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------

class TestGetDDEParams:
    """Tests for get_dde_params and related helpers."""

    def test_returns_all_default_keys(self):
        params = get_dde_params()
        for key in DEFAULT_DDE_PARAMS:
            assert key in params, f"Missing default key: {key}"

    def test_decay_is_derived(self):
        params = get_dde_params()
        expected = compute_decay_term(params["d"], params["d0"])
        assert params["decay"] == pytest.approx(expected)

    def test_overrides_apply(self):
        params = get_dde_params(delay=20, D=5.0)
        assert params["delay"] == 20
        assert params["D"] == 5.0
        # Other defaults unchanged
        assert params["CA"] == DEFAULT_DDE_PARAMS["CA"]

    def test_decay_recomputed_on_override(self):
        params = get_dde_params(d=0.5, d0=1.0)
        expected = compute_decay_term(0.5, 1.0)
        assert params["decay"] == pytest.approx(expected)


class TestComputeDecayTerm:
    def test_known_values(self):
        # decay = 1 - (d/d0)^4
        assert compute_decay_term(0.0, 1.0) == pytest.approx(1.0)
        assert compute_decay_term(1.0, 1.0) == pytest.approx(0.0)
        assert compute_decay_term(0.5, 1.0) == pytest.approx(1.0 - 0.0625)


# ---------------------------------------------------------------------------
# DDE solver
# ---------------------------------------------------------------------------

class TestSolveDDERK4:
    """Tests for the RK4 DDE integrator."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        clear_prepared_cache()
        yield
        clear_prepared_cache()

    @staticmethod
    def _make_inputs(n_nodes: int = 4, history_len: int = 20, substeps: int = 8):
        """Build minimal valid inputs for solve_dde_rk4."""
        rng = np.random.default_rng(42)
        history = rng.uniform(0.01, 0.5, size=(history_len, n_nodes, 4))
        times = np.linspace(0.0, 1.0, substeps + 1)
        input_vec = rng.uniform(-0.01, 0.01, size=n_nodes)
        return history, times, input_vec

    def test_output_shape(self):
        """solve_dde_rk4 should return (n_steps, n_nodes, 4)."""
        n_nodes, substeps = 4, 8
        history, times, inp = self._make_inputs(n_nodes=n_nodes, substeps=substeps)
        out = solve_dde_rk4(history, times, inp, parallel=False)
        assert out.shape == (substeps + 1, n_nodes, 4)

    def test_output_shape_parallel(self):
        """Parallel solver should produce the same shape."""
        n_nodes, substeps = 4, 8
        history, times, inp = self._make_inputs(n_nodes=n_nodes, substeps=substeps)
        out = solve_dde_rk4(history, times, inp, parallel=True)
        assert out.shape == (substeps + 1, n_nodes, 4)

    def test_sequential_and_parallel_agree(self):
        """Sequential and parallel solvers should produce identical results."""
        n_nodes, substeps = 6, 16
        history, times, inp = self._make_inputs(n_nodes=n_nodes, substeps=substeps)
        out_seq = solve_dde_rk4(history, times, inp, parallel=False)
        out_par = solve_dde_rk4(history, times, inp, parallel=True)
        np.testing.assert_allclose(out_seq, out_par, atol=1e-10)

    def test_initial_condition_preserved(self):
        """First timestep of output should equal last history state."""
        history, times, inp = self._make_inputs()
        out = solve_dde_rk4(history, times, inp, parallel=False)
        np.testing.assert_array_equal(out[0], history[-1])

    def test_output_is_finite(self):
        """Output should not contain NaN or Inf for default parameters."""
        history, times, inp = self._make_inputs()
        out = solve_dde_rk4(history, times, inp, parallel=False)
        assert np.all(np.isfinite(out))

    def test_zero_input_still_integrates(self):
        """Zero external input should still produce non-trivial dynamics."""
        n_nodes = 4
        history, times, _ = self._make_inputs(n_nodes=n_nodes)
        zero_inp = np.zeros(n_nodes)
        out = solve_dde_rk4(history, times, zero_inp, parallel=False)
        assert np.all(np.isfinite(out))
        # Should evolve from initial condition (not all identical to t=0)
        assert not np.allclose(out[0], out[-1])

    def test_dde_overrides_affect_output(self):
        """Changing a DDE parameter should change the trajectory."""
        history, times, inp = self._make_inputs()
        out_default = solve_dde_rk4(history, times, inp, parallel=False)
        out_modified = solve_dde_rk4(history, times, inp, parallel=False, D=10.0)
        assert not np.allclose(out_default, out_modified)

    def test_convergence_with_step_refinement(self):
        """Halving the step size should reduce the difference between solutions.

        This is a basic consistency check — not a formal O(h^4) test, but
        confirms that the solver converges as expected.
        """
        n_nodes = 3
        history = np.full((20, n_nodes, 4), 0.1)
        inp = np.zeros(n_nodes)

        # Coarse: 8 substeps
        times_coarse = np.linspace(0.0, 1.0, 9)
        out_coarse = solve_dde_rk4(history, times_coarse, inp, parallel=False)

        # Fine: 16 substeps
        times_fine = np.linspace(0.0, 1.0, 17)
        out_fine = solve_dde_rk4(history, times_fine, inp, parallel=False)

        # Very fine: 32 substeps
        times_vfine = np.linspace(0.0, 1.0, 33)
        out_vfine = solve_dde_rk4(history, times_vfine, inp, parallel=False)

        # Compare endpoints: |fine - vfine| should be smaller than |coarse - fine|
        diff_cf = np.linalg.norm(out_coarse[-1] - out_fine[-1])
        diff_fv = np.linalg.norm(out_fine[-1] - out_vfine[-1])
        assert diff_fv < diff_cf, (
            f"Expected finer step to converge: |fine-vfine|={diff_fv:.2e} "
            f"should be < |coarse-fine|={diff_cf:.2e}"
        )


# ---------------------------------------------------------------------------
# Numerical correctness: equations, integrator, delay lookup, parameter routing
# ---------------------------------------------------------------------------

# Order of the scalar parameters after the state arguments of the numba kernels.
KERNEL_PARAMS = ("CA", "CI", "del_", "alpha", "k1", "f", "gammaA", "gammaI", "b", "k",
                 "gammaH", "g", "D", "d", "mu", "decay")


def _model_rhs(y, delayed_hi, u, p):
    """The oscillator equations as implemented (Danino et al. 2010, input added to He)."""
    A, I, Hi, He = y  # noqa: E741 (LuxI, as in the model equations)
    promoter = (p["del_"] + p["alpha"] * delayed_hi ** 2) / (1.0 + p["k1"] * delayed_hi ** 2)
    denom = 1.0 + p["f"] * (A + I)
    return np.array([
        p["CA"] * p["decay"] * promoter - p["gammaA"] * A / denom,
        p["CI"] * p["decay"] * promoter - p["gammaI"] * I / denom,
        p["b"] * I / (1.0 + p["k"] * I) - p["gammaH"] * A * Hi / (1.0 + p["g"] * A)
        + p["D"] * (He + u - Hi),
        -p["d"] / (1.0 - p["d"]) * p["D"] * (He - Hi) - p["mu"] * He,
    ])


def _one_step(history, u, substeps, parallel=False, **overrides):
    times = np.linspace(0.0, 1.0, substeps + 1)
    return solve_dde_rk4(history, times, u, parallel=parallel, **overrides)


def _reference_step(y0, delayed_hi, u, params):
    """Integrate one external step (t = 0..1) with SciPy's DOP853 at tight tolerance."""
    sol = solve_ivp(lambda t, y: _model_rhs(y, delayed_hi, u, params), (0.0, 1.0), y0,
                    method="DOP853", rtol=1e-12, atol=1e-14)
    return sol.y[:, -1]


@pytest.fixture
def basal_history():
    """Two nodes at start-up levels (as BioReservoir initialises them); delay 10 reads the
    oldest of the 11 history entries."""
    rng = np.random.default_rng(7)
    return rng.uniform([0.10, 0.10, 0.01, 0.04], [0.20, 0.20, 0.03, 0.06], size=(11, 2, 4))


class TestModelEquations:
    def test_right_hand_side_matches_the_written_equations(self):
        rng = np.random.default_rng(0)
        params = get_dde_params()
        y = rng.uniform(0.0, 2.0, size=(4, 50))
        delayed = rng.uniform(0.0, 2.0, size=50)
        u = rng.normal(0.0, 0.1, size=50)
        got = dde._compute_derivatives(*y, delayed, u, *(params[k] for k in KERNEL_PARAMS))
        expected = _model_rhs(y, delayed, u, params)
        np.testing.assert_allclose(np.array(got), expected, rtol=1e-12, atol=1e-15)

    def test_derived_decay_term_enters_the_promoter_terms(self):
        params = get_dde_params(d=0.0)
        assert params["decay"] == 1.0
        params = get_dde_params(d=params["d0"])
        y = np.array([0.3, 0.4, 0.2, 0.1])
        dA, dI, _, _ = dde._compute_derivatives(*y, 0.5, 0.0,
                                                *(params[k] for k in KERNEL_PARAMS))
        assert dA == pytest.approx(-params["gammaA"] * 0.3 / (1.0 + params["f"] * 0.7))
        assert dI == pytest.approx(-params["gammaI"] * 0.4 / (1.0 + params["f"] * 0.7))


class TestIntegrator:
    @pytest.mark.parametrize("parallel", [False, True])
    def test_extracellular_signal_decays_exponentially_without_diffusion(self, parallel):
        history = np.full((11, 3, 4), 0.2)
        history[-1, :, 3] = [0.5, 1.0, 2.0]
        out = _one_step(history, np.zeros(3), substeps=64, parallel=parallel, D=0.0)
        mu = DEFAULT_DDE_PARAMS["mu"]
        np.testing.assert_allclose(out[-1, :, 3], history[-1, :, 3] * np.exp(-mu), rtol=1e-10)
        np.testing.assert_allclose(out[:, :, 3],
                                   history[-1, :, 3] * np.exp(-mu * np.linspace(0, 1, 65))[:, None],
                                   rtol=1e-10)

    @pytest.mark.parametrize("parallel", [False, True])
    def test_one_step_matches_a_high_accuracy_solver(self, basal_history, parallel):
        u = np.array([0.02, -0.01])
        params = get_dde_params()
        out = _one_step(basal_history, u, substeps=64, parallel=parallel)
        delayed = basal_history[0, :, 2]
        for node in range(2):
            expected = _reference_step(basal_history[-1, node], delayed[node], u[node], params)
            np.testing.assert_allclose(out[-1, node], expected, rtol=1e-6, atol=1e-9)

    def test_converges_at_fourth_order(self, basal_history):
        u = np.array([0.02, -0.01])
        params = get_dde_params()
        expected = _reference_step(basal_history[-1, 0], basal_history[0, 0, 2], u[0], params)
        errors = [np.abs(_one_step(basal_history, u, n)[-1, 0] - expected).max()
                  for n in (32, 64, 128)]
        orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
        assert np.all((orders > 3.7) & (orders < 4.6)), orders

    def test_trajectory_has_one_entry_per_substep(self, basal_history):
        out = _one_step(basal_history, np.zeros(2), substeps=5)
        assert out.shape == (6, 2, 4)
        np.testing.assert_array_equal(out[0], basal_history[-1])


class TestDelayLookup:
    """The delayed Hi is read once per call from the history buffer (v0.1 behaviour)."""

    @pytest.fixture
    def captured(self, monkeypatch):
        seen = {}

        def spy(Y_init, Hi_low, Hi_high, times, input_vec, *params):
            seen.update(Y_init=Y_init, Hi_low=Hi_low, Hi_high=Hi_high, times=times,
                        input_vec=input_vec, params=params)
            return np.zeros((len(times),) + Y_init.shape)

        monkeypatch.setattr(dde, "_solve_sequential", spy)
        return seen

    @staticmethod
    def _history():
        history = np.zeros((11, 2, 4))
        history[:, :, 2] = np.arange(11.0)[:, None]
        return history

    @pytest.mark.parametrize("delay, expected", [
        (10, 0.0), (3, 7.0), (2.5, 7.5), (0.5, 9.5), (0, 10.0), (25, 0.0),
    ])
    def test_delayed_signal_is_interpolated_from_the_history(self, captured, delay, expected):
        solve_dde_rk4(self._history(), np.linspace(0, 1, 3), np.zeros(2), parallel=False,
                      delay=delay)
        np.testing.assert_allclose(captured["Hi_low"], expected)
        np.testing.assert_array_equal(captured["Hi_high"], captured["Hi_low"])

    def test_kernel_receives_the_current_state_and_parameters_in_order(self, captured):
        history = self._history()
        solve_dde_rk4(history, np.linspace(0, 1, 3), np.array([0.1, 0.2]), parallel=False,
                      D=3.0)
        params = get_dde_params(D=3.0)
        np.testing.assert_array_equal(captured["Y_init"], history[-1])
        np.testing.assert_array_equal(captured["input_vec"], [0.1, 0.2])
        assert captured["params"] == tuple(params[k] for k in KERNEL_PARAMS)


class TestHeterogeneousSolver:
    def test_zero_variability_reproduces_the_homogeneous_solver(self, basal_history):
        node_params = generate_node_params(2, 0.0, get_dde_params(), seed=1)
        u = np.array([0.02, -0.01])
        het = solve_dde_rk4(basal_history, np.linspace(0, 1, 17), u, parallel=True,
                            node_params=node_params)
        hom = solve_dde_rk4(basal_history, np.linspace(0, 1, 17), u, parallel=True)
        np.testing.assert_allclose(het, hom, rtol=1e-13, atol=1e-15)

    def test_each_node_uses_its_own_parameters(self, basal_history):
        node_params = generate_node_params(2, 0.4, get_dde_params(), seed=3)
        u = np.array([0.02, -0.01])
        times = np.linspace(0, 1, 17)
        het = solve_dde_rk4(basal_history, times, u, parallel=True, node_params=node_params)
        for node in range(2):
            own = {k: node_params[node, i] for i, k in enumerate(dde._PARAM_ORDER)
                   if k != "decay"}
            alone = solve_dde_rk4(basal_history[:, [node]], times, u[[node]], parallel=True,
                                  **own)
            np.testing.assert_allclose(het[:, node], alone[:, 0], rtol=1e-12, atol=1e-15)


class TestGenerateNodeParams:
    def test_zero_variability_gives_the_nominal_parameters(self):
        base = get_dde_params()
        node_params = generate_node_params(5, 0.0, base, seed=0)
        assert node_params.shape == (5, dde.N_NODE_PARAMS)
        nominal = [base[k] for k in dde._PARAM_ORDER]
        np.testing.assert_allclose(node_params, np.tile(nominal, (5, 1)))

    def test_only_intracellular_kinetics_vary(self):
        base = get_dde_params()
        node_params = generate_node_params(200, 0.3, base, seed=0)
        for i, name in enumerate(dde._PARAM_ORDER):
            column = node_params[:, i]
            if name in dde._VARY_SET:
                assert column.std() > 0 and np.all(column > 0), name
            else:
                np.testing.assert_allclose(column, base[name])

    def test_coefficient_of_variation_matches_the_request(self):
        node_params = generate_node_params(20000, 0.25, get_dde_params(), seed=0)
        factors = node_params[:, 0] / get_dde_params()["CA"]
        assert factors.std() / factors.mean() == pytest.approx(0.25, rel=0.03)

    def test_factors_have_mean_one(self):
        # v0.1 drew exp(N(0, s^2)), whose mean is sqrt(1 + cv^2): 4.4% high at cv 0.3.
        base = get_dde_params()
        node_params = generate_node_params(200_000, 0.3, base, seed=0)
        for i, name in enumerate(dde._PARAM_ORDER):
            if name in dde._VARY_SET:
                assert node_params[:, i].mean() / base[name] == pytest.approx(1.0, abs=0.004)

    def test_v0_1_factors_are_biased_upwards(self):
        base = get_dde_params()
        node_params = generate_node_params(200_000, 0.3, base, seed=0, mean_preserving=False)
        factors = node_params[:, 0] / base["CA"]
        assert factors.mean() == pytest.approx(np.sqrt(1.09), abs=0.004)
        assert factors.std() / factors.mean() == pytest.approx(0.3, rel=0.02)

    def test_reproducible_with_a_seed(self):
        base = get_dde_params()
        first = generate_node_params(10, 0.2, base, seed=11)
        np.testing.assert_array_equal(first, generate_node_params(10, 0.2, base, seed=11))
        assert not np.array_equal(first, generate_node_params(10, 0.2, base, seed=12))


class TestPreparedParameters:
    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        clear_prepared_cache()
        yield
        clear_prepared_cache()

    def test_prepared_values_follow_the_parameters(self):
        prepared = dde._prepare_dde_params_cached(get_dde_params(D=3.5, delay=12))
        assert prepared.D == 3.5 and prepared.delay == 12.0
        assert prepared.decay == pytest.approx(compute_decay_term(0.7, 0.88))

    def test_prepared_parameters_are_cached_until_cleared(self):
        first = dde._prepare_dde_params_cached(get_dde_params())
        assert dde._prepare_dde_params_cached(get_dde_params()) is first
        assert dde._prepare_dde_params_cached(get_dde_params(mu=0.5)) is not first
        clear_prepared_cache()
        assert dde._prepare_dde_params_cached(get_dde_params()) is not first

    def test_parameters_are_logged_only_at_debug_level(self, caplog):
        with caplog.at_level(logging.INFO, logger="bioreservoir.dde"):
            log_dde_params(get_dde_params())
        assert not caplog.records
        with caplog.at_level(logging.DEBUG, logger="bioreservoir.dde"):
            log_dde_params(get_dde_params())
        assert any("delay: 10.0" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Delay resolution inside an input step
# ---------------------------------------------------------------------------

def _dde_reference(y0, hi_history, delay, inputs, params):
    """Method of steps with DOP853 for one oscillator.

    The history of Hi before t = 0 is linear between the stored points at
    t = -(H-1), ..., 0; the input is constant within each unit step. ``delay = 0`` is the
    ordinary differential equation (the promoter reads the current Hi).
    """
    t_hist = np.arange(-(len(hi_history) - 1), 1.0)
    segments = []

    def delayed(t, y):
        if delay == 0:
            return y[2]
        s = t - delay
        if s <= 0:
            return np.interp(s, t_hist, hi_history)
        for a, b, sol in segments:
            if a <= s <= b:
                return sol.sol(s)[2]
        raise AssertionError(f"no solution stored at t = {s}")

    y = np.asarray(y0, dtype=float)
    for step, u in enumerate(inputs):
        sol = solve_ivp(lambda t, y, u=u: _model_rhs(y, delayed(t, y), u, params),
                        (step, step + 1), y, method="DOP853", rtol=1e-12, atol=1e-14,
                        dense_output=True)
        segments.append((step, step + 1, sol))
        y = sol.y[:, -1]
    return y


def _run_node(delay, substeps, inputs, units=2, parallel=False, **options):
    """Step a BioReservoir (no warm-up, unit coupling) through ``inputs``.

    Returns the start state, the stored Hi history and the state after the last step.
    """
    from bioreservoir import BioReservoir

    node = BioReservoir(units=units, warmup=0, rk4_substeps=substeps, seed=3, input_dim=1,
                        output_variables="He", parallel=parallel, cell_coupling=1.0,
                        dde_kwargs={"delay": delay}, **options)
    y0, hi_history = node.states[-1].copy(), node.states[:, :, 2].copy()
    for u in inputs:
        node._step(np.full(units, u))
    return y0, hi_history, node.states[-1].copy()


class TestDelayResolution:
    """The delayed Hi follows the solution within an input step."""

    INPUTS = (0.02, -0.01, 0.015, 0.0)

    @staticmethod
    def _errors(delay, inputs, units):
        """Relative error at the end of ``inputs`` for 32, 64 and 128 substeps."""
        params = get_dde_params(delay=delay)
        errors, expected = [], None
        for substeps in (32, 64, 128):
            y0, hist, end = _run_node(delay, substeps, inputs, units=units)
            if expected is None:
                expected = np.array([_dde_reference(y0[i], hist[:, i], delay, inputs, params)
                                     for i in range(units)])
            errors.append(np.abs(end - expected).max() / np.abs(expected).max())
        return np.array(errors), np.log2(np.array(errors[:-1]) / np.array(errors[1:]))

    def test_zero_delay_is_the_ordinary_differential_equation(self):
        # RK4 on the ODE: fourth order. The frozen v0.1 scheme held Hi at its value at the
        # start of the step instead, an error of order one that substeps do not reduce.
        errors, orders = self._errors(0, self.INPUTS[:1], units=2)
        assert np.all(orders > 3.5), (errors, orders)
        assert errors[-1] < 1e-7, errors

    @pytest.mark.parametrize("delay", [2.0, 1.5, 1.3])
    def test_delayed_solution_converges_at_second_order(self, delay):
        # 1.3 input steps is not a whole number of substeps: the lags are interpolated, and
        # the kinks of the stored history fall between substeps, so the error ratio varies
        # from one refinement to the next; the order over the whole range is still two.
        errors, orders = self._errors(delay, self.INPUTS, units=1)
        assert np.all(orders > 1.5), (errors, orders)
        assert np.log2(errors[0] / errors[-1]) / 2 > 1.8, errors
        assert errors[-1] < 2e-5, errors

    def test_lags_between_substeps_are_interpolated(self):
        assert dde.substep_delay_lags(1.3, 64) == pytest.approx((83.2, 82.7, 82.2))
        # One substep up to rounding (1/49 * 49 < 1 in floating point) is one substep.
        assert 1 / 49 * 49 < 1.0
        assert dde.substep_delay_lags(1 / 49, 49) == (1.0, 0.5, 0.0)
        ring = np.array([[10.0, 11.0, 12.0, 13.0, 14.0]])  # entry 4 is the current one
        assert dde._ring_value(ring, 0, 4, 5, 1, 0.25) == pytest.approx(12.75)
        ring = np.array([[14.0, 15.0, 11.0, 12.0, 13.0]])  # current entry 1, older ones wrap
        assert dde._ring_value(ring, 0, 1, 5, 3, 0.5) == pytest.approx(11.5)

    def test_zero_variability_matches_the_homogeneous_solver(self):
        _, _, homogeneous = _run_node(2.0, 16, self.INPUTS, parallel=True)
        _, _, heterogeneous = _run_node(2.0, 16, self.INPUTS, parallel=True,
                                        param_heterogeneity_cv=1e-300)
        np.testing.assert_allclose(heterogeneous, homogeneous, rtol=1e-12)

    def test_frozen_scheme_is_the_v0_1_solver(self):
        from bioreservoir import BioReservoir

        node = BioReservoir(units=3, warmup=0, rk4_substeps=16, seed=3, input_dim=1,
                            output_variables="He", parallel=False, cell_coupling=1.0,
                            delay_scheme="frozen")
        history = node.states.copy()
        out = node._step(np.full(3, 0.02))
        expected = solve_dde_rk4(history, node.rk4_timespan, np.full(3, 0.02),
                                 parallel=False, **node.dde_kwargs)[-1]
        np.testing.assert_array_equal(out, expected)

    def test_delay_below_one_substep_is_rejected(self):
        with pytest.raises(ValueError, match="substep"):
            _run_node(0.01, 16, ())

    def test_the_history_must_cover_the_delay(self):
        with pytest.raises(ValueError, match="shorter than the delay"):
            dde.substep_history(np.zeros((2, 3, 4)), delay=2.0, substeps=4)

    def test_the_substep_solver_checks_the_history_and_defaults_to_nominal_rates(self):
        state = np.full((2, 4), 0.1)
        ring, pos = dde.substep_history(np.full((3, 2, 4), 0.1), delay=2.0, substeps=4)
        with pytest.raises(ValueError, match="delay history"):
            dde.solve_dde_substep(state, ring[:, :-1], pos, np.zeros(2), 4, delay=2.0)
        nominal = generate_node_params(2, 0.0, get_dde_params(delay=2.0))
        a, _ = dde.solve_dde_substep(state, ring.copy(), pos, np.zeros(2), 4, delay=2.0)
        b, _ = dde.solve_dde_substep(state, ring.copy(), pos, np.zeros(2), 4,
                                     node_params=nominal, delay=2.0)
        np.testing.assert_array_equal(a, b)

    def test_unknown_scheme_is_rejected(self):
        with pytest.raises(ValueError, match="delay_scheme"):
            _run_node(2.0, 16, (), delay_scheme="cubic")


# ---------------------------------------------------------------------------
# The substep kernel, compiled and as plain Python
# ---------------------------------------------------------------------------

KERNELS = ("python", "sequential", "parallel")

# No LuxI production of AHL and no AiiA degradation of it, and He constant: Hi relaxes
# linearly towards He + u, so the solver's Hi has a closed form.
LINEAR_AHL = {"b": 0.0, "gammaH": 0.0, "d": 0.0, "mu": 0.0}


def _python(func):
    """The plain-Python version of a numba function (the function itself without the JIT)."""
    return getattr(func, "py_func", func)


def _start(delay, substeps, n_nodes=3, seed=0):
    """Start-up states and their Hi history at substep resolution."""
    rng = np.random.default_rng(seed)
    history = rng.uniform([0.10, 0.10, 0.01, 0.04], [0.20, 0.20, 0.03, 0.06],
                          size=(12, n_nodes, 4))
    ring, pos = dde.substep_history(history, delay, substeps)
    return history[-1].copy(), ring, pos


def _advance(kernel, monkeypatch, state, ring, pos, inputs, substeps, **kwargs):
    """Step ``solve_dde_substep`` through ``inputs`` (one row per input step).

    ``kernel="python"`` runs the kernel and its ring lookup as plain Python, which is what
    coverage measures; "sequential" and "parallel" run the compiled kernels.
    """
    with monkeypatch.context() as patch:
        if kernel == "python":
            patch.setattr(dde, "_solve_substep_sequential", dde._substep_kernel)
            patch.setattr(dde, "_ring_value", _python(dde._ring_value))
        for u in inputs:
            state, pos = dde.solve_dde_substep(state, ring, pos, np.asarray(u, dtype=float),
                                               substeps, parallel=kernel == "parallel",
                                               **kwargs)
    return state, pos


def _rk4_factor(rate, dt):
    """RK4's factor per step for dx/dt = -rate x."""
    z = -rate * dt
    return 1.0 + z + z ** 2 / 2.0 + z ** 3 / 6.0 + z ** 4 / 24.0


def _promoter(params, hi):
    """AiiA production for the (delayed) Hi."""
    return (params["CA"] * params["decay"] * (params["del_"] + params["alpha"] * hi ** 2)
            / (1.0 + params["k1"] * hi ** 2))


class TestSubstepKernel:
    """The delay-resolving RK4 kernel, compiled (sequential, parallel) and as plain Python.

    Lags are in substeps: 0 is the ODE; 1 reads the current entry at the end of a substep;
    2.9 keeps fewer entries than a step writes, so the ring wraps within a step; 20.8 is
    interpolated; 32 is a whole number of substeps. RK4 needs about 16 substeps per input
    step to be stable for the default kinetics; with fewer, rounding differences grow.
    """

    @pytest.mark.parametrize("substeps", [16, 25])
    @pytest.mark.parametrize("lag", [0.0, 1.0, 2.9, 20.8, 32.0])
    def test_compiled_kernels_match_the_python_kernel(self, monkeypatch, substeps, lag):
        delay = lag / substeps
        node_params = generate_node_params(3, 0.3, get_dde_params(delay=delay), seed=5)
        inputs = np.random.default_rng(1).normal(0.0, 0.02, size=(5, 3))
        results = {}
        for kernel in KERNELS:
            state, ring, pos = _start(delay, substeps)
            state, pos = _advance(kernel, monkeypatch, state, ring, pos, inputs, substeps,
                                  node_params=node_params, delay=delay)
            results[kernel] = state, ring, pos
        state, ring, pos = results["python"]
        for kernel in ("sequential", "parallel"):  # fastmath may reorder: rounding only
            np.testing.assert_allclose(results[kernel][0], state, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(results[kernel][1], ring, rtol=1e-12, atol=1e-12)
            assert results[kernel][2] == pos

    @pytest.mark.parametrize("kernel", KERNELS)
    @pytest.mark.parametrize("lag, steps", [(0.0, 2), (1.0, 2), (2.9, 2), (20.8, 1), (20.8, 2)])
    def test_the_ring_holds_hi_after_every_substep(self, monkeypatch, kernel, lag, steps):
        substeps = 16
        delay = lag / substeps
        state, ring, pos = _start(delay, substeps)
        before, length = ring.copy(), ring.shape[1]
        u = np.array([0.02, -0.01, 0.0])
        end, end_pos = _advance(kernel, monkeypatch, state, ring, pos, [u] * steps, substeps,
                                delay=delay, **LINEAR_AHL)
        target = state[:, 3] + u
        written = steps * substeps
        factors = _rk4_factor(DEFAULT_DDE_PARAMS["D"], 1.0 / substeps) ** np.arange(1, written + 1)
        hi = target[:, None] + (state[:, 2] - target)[:, None] * factors
        assert end_pos == (pos + written) % length
        np.testing.assert_allclose(end[:, 2], hi[:, -1], rtol=1e-12)
        np.testing.assert_array_equal(end[:, 3], state[:, 3])
        for back in range(length):  # newest entry first
            entry = ring[:, (end_pos - back) % length]
            if back < written:
                np.testing.assert_allclose(entry, hi[:, written - 1 - back], rtol=1e-12)
            else:  # not overwritten yet: the stored history
                np.testing.assert_array_equal(entry, before[:, (pos + written - back) % length])

    @pytest.mark.parametrize("kernel", KERNELS)
    @pytest.mark.parametrize("delay", [1.0, 1.3, 2.0])
    def test_each_stage_reads_hi_one_delay_before_its_own_time(self, monkeypatch, kernel, delay):
        # Without AiiA decay (gammaA = 0) dA/dt is the production for the delayed Hi alone, so
        # a substep from t adds dt/6 (P(t - delay) + 4 P(t + dt/2 - delay) + P(t + dt - delay)).
        # A delay of at least one input step reads only the stored history in the first step.
        substeps = 16
        params = get_dde_params(gammaA=0.0, delay=delay, **LINEAR_AHL)
        state, ring, pos = _start(delay, substeps)
        times = (np.arange(ring.shape[1]) - pos) / substeps
        stored = ring.copy()

        def production(t):
            return _promoter(params, np.array([np.interp(t - delay, times, row) for row in stored]))

        dt = 1.0 / substeps
        expected = state[:, 0] + sum(
            dt / 6.0 * (production(t) + 4.0 * production(t + dt / 2.0) + production(t + dt))
            for t in np.arange(substeps) * dt)
        end, _ = _advance(kernel, monkeypatch, state, ring, pos, [np.zeros(3)], substeps,
                          gammaA=0.0, delay=delay, **LINEAR_AHL)
        np.testing.assert_allclose(end[:, 0], expected, rtol=1e-12)

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_a_zero_delay_gives_each_stage_its_own_hi(self, monkeypatch, kernel):
        substeps = 16
        params = get_dde_params(gammaA=0.0, delay=0.0, **LINEAR_AHL)
        state, ring, pos = _start(0.0, substeps)
        u = np.array([0.02, -0.01, 0.0])
        target, dt, rate = state[:, 3] + u, 1.0 / substeps, params["D"]
        a, hi = state[:, 0].copy(), state[:, 2].copy()
        for _ in range(substeps):
            hi2 = hi + 0.5 * dt * rate * (target - hi)
            hi3 = hi + 0.5 * dt * rate * (target - hi2)
            hi4 = hi + dt * rate * (target - hi3)
            a = a + dt / 6.0 * (_promoter(params, hi) + 2.0 * _promoter(params, hi2)
                                + 2.0 * _promoter(params, hi3) + _promoter(params, hi4))
            hi = hi + dt / 6.0 * rate * ((target - hi) + 2.0 * (target - hi2)
                                         + 2.0 * (target - hi3) + (target - hi4))
        end, _ = _advance(kernel, monkeypatch, state, ring, pos, [u], substeps,
                          gammaA=0.0, delay=0.0, **LINEAR_AHL)
        np.testing.assert_allclose(end[:, 0], a, rtol=1e-12)
        np.testing.assert_allclose(end[:, 2], hi, rtol=1e-12)

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_each_node_uses_its_own_rates_input_and_history(self, monkeypatch, kernel):
        substeps, delay = 16, 1.3
        node_params = generate_node_params(3, 0.4, get_dde_params(delay=delay), seed=2)
        inputs = np.random.default_rng(3).normal(0.0, 0.02, size=(3, 3))
        state, ring, pos = _start(delay, substeps)
        together, _ = _advance(kernel, monkeypatch, state, ring.copy(), pos, inputs, substeps,
                               node_params=node_params, delay=delay)
        for node in range(3):
            alone, _ = _advance(kernel, monkeypatch, state[[node]], ring[[node]], pos,
                                inputs[:, [node]], substeps, node_params=node_params[[node]],
                                delay=delay)
            np.testing.assert_allclose(alone[0], together[node], rtol=1e-13, atol=1e-16)

    def test_parallel_results_do_not_depend_on_the_number_of_threads(self, monkeypatch):
        substeps, delay, n_nodes = 16, 1.3, 16
        node_params = generate_node_params(n_nodes, 0.3, get_dde_params(delay=delay), seed=4)
        inputs = np.random.default_rng(5).normal(0.0, 0.02, size=(3, n_nodes))
        threads, runs = numba.get_num_threads(), []
        for n in (1, threads):
            numba.set_num_threads(n)
            try:
                state, ring, pos = _start(delay, substeps, n_nodes=n_nodes)
                runs.append(_advance("parallel", monkeypatch, state, ring, pos, inputs,
                                     substeps, node_params=node_params, delay=delay)[0])
            finally:
                numba.set_num_threads(threads)
        np.testing.assert_array_equal(runs[0], runs[1])

    def test_the_state_is_copied_and_the_ring_updated_in_place(self):
        state, ring, pos = _start(2.0, 16)
        start = state.copy()
        new_state, new_pos = dde.solve_dde_substep(state, ring, pos, np.zeros(3), 16, delay=2.0)
        np.testing.assert_array_equal(state, start)
        assert new_state is not state and new_pos == (pos + 16) % ring.shape[1]
        np.testing.assert_array_equal(ring[:, new_pos], new_state[:, 2])

    def test_ring_lookups_wrap_around_and_interpolate(self):
        ring = np.arange(10.0, 20.0).reshape(2, 5) ** 2
        length = ring.shape[1]
        cases = ([(back, 0.0) for back in range(length)]
                 + [(back, frac) for back in range(length - 1) for frac in (0.25, 0.5)])
        for lookup in (dde._ring_value, _python(dde._ring_value)):
            for node, pos, (back, frac) in itertools.product(range(2), range(length), cases):
                expected = ((1.0 - frac) * ring[node, (pos - back) % length]
                            + frac * ring[node, (pos - back - 1) % length])
                got = lookup(ring, node, pos, length, back, frac)
                assert got == pytest.approx(expected, rel=1e-14)

    def test_lags_at_the_start_middle_and_end_of_a_substep(self):
        assert dde.substep_delay_lags(0.0, 16) is None
        assert dde.substep_delay_lags(2.0, 4) == (8.0, 7.5, 7.0)


class TestSubstepHistory:
    def test_hi_is_interpolated_to_the_substep_times(self):
        history = np.zeros((4, 2, 4))
        history[:, :, 2] = (2.0 + 0.5 * np.arange(-3.0, 1.0))[:, None] * [1.0, 3.0]
        ring, pos = dde.substep_history(history, delay=1.3, substeps=4)
        assert ring.shape == (2, 7) and pos == 6 and ring.flags.c_contiguous
        times = (np.arange(7) - pos) / 4
        np.testing.assert_allclose(ring, (2.0 + 0.5 * times) * np.array([[1.0], [3.0]]),
                                   rtol=1e-14)

    @pytest.mark.parametrize("entries", [1, 5])
    def test_a_zero_delay_keeps_only_the_current_hi(self, entries):
        history = np.random.default_rng(0).uniform(size=(entries, 3, 4))
        ring, pos = dde.substep_history(history, delay=0.0, substeps=8)
        assert ring.shape == (3, 1) and pos == 0
        np.testing.assert_array_equal(ring[:, 0], history[-1, :, 2])
