"""Unit tests for the bioreservoir.dde DDE solver."""

import logging

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

    def test_unknown_scheme_is_rejected(self):
        with pytest.raises(ValueError, match="delay_scheme"):
            _run_node(2.0, 16, (), delay_scheme="cubic")
