"""Unit tests for the bioreservoir.dde DDE solver."""

import pytest
import numpy as np

from bioreservoir.dde import (
    get_dde_params,
    compute_decay_term,
    solve_dde_rk4,
    clear_prepared_cache,
    DEFAULT_DDE_PARAMS,
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
