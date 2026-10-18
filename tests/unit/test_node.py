"""Unit tests for bioreservoir.node.BioReservoir."""

import logging
from types import SimpleNamespace

import numpy as np
import pytest

import bioreservoir.node as node_module
from bioreservoir import BioReservoir
from bioreservoir.topology import distance_matrix


@pytest.fixture
def tiny_node():
    """BioReservoir with 4 units, suitable for fast unit tests."""
    return BioReservoir(
        units=4,
        warmup=2,
        rk4_substeps=4,
        seed=42,
        input_dim=1,
        output_variables="He",
        diagnostics=False,
        parallel=False,
    )


@pytest.fixture
def tiny_node_diagnostics():
    """BioReservoir with diagnostics enabled."""
    return BioReservoir(
        units=4,
        warmup=2,
        rk4_substeps=4,
        seed=42,
        input_dim=1,
        output_variables="He",
        diagnostics=True,
        parallel=False,
    )


class TestBioReservoirInit:
    def test_units_attribute(self, tiny_node):
        assert tiny_node.units == 4

    def test_output_dim_single_variable(self, tiny_node):
        # 4 units × 1 variable (He) = 4
        assert tiny_node.output_dim == 4

    def test_output_dim_subset(self):
        node = BioReservoir(units=4, warmup=1, rk4_substeps=4, seed=0,
                            input_dim=1, output_variables=("A",), parallel=False)
        assert node.output_dim == 4  # 4 units × 1 var

    def test_output_variables_preset_string(self):
        node = BioReservoir(units=3, warmup=1, rk4_substeps=4, seed=0,
                            input_dim=1, output_variables="He", parallel=False)
        assert node.output_dim == 3  # 3 units × 1 var

    def test_invalid_output_variables_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown output_variables preset"):
            BioReservoir(units=4, warmup=1, rk4_substeps=4,
                         output_variables="INVALID", parallel=False)

    def test_states_shape(self, tiny_node):
        """Internal history buffer has shape (history_len, units, 4)."""
        assert tiny_node.states.ndim == 3
        assert tiny_node.states.shape[1] == 4  # units
        assert tiny_node.states.shape[2] == 4  # 4 DDE variables


class TestBioReservoirRun:
    def test_single_step_output_shape(self, tiny_node):
        """run() with a (1, input_dim) input should return (1, output_dim)."""
        x = np.random.default_rng(0).uniform(size=(1, 1))
        out = tiny_node.run(x, reset=True)
        assert out.shape == (1, 4)  # 4 units × 1 var

    def test_multi_step_output_shape(self, tiny_node):
        """run() with (T, input_dim) should return (T, output_dim)."""
        T = 5
        x = np.random.default_rng(0).uniform(size=(T, 1))
        out = tiny_node.run(x, reset=True)
        assert out.shape == (T, 4)

    def test_output_is_finite(self, tiny_node):
        x = np.random.default_rng(0).uniform(size=(3, 1))
        out = tiny_node.run(x, reset=True)
        assert np.all(np.isfinite(out))

    def test_deterministic_with_same_seed(self):
        """Two nodes with same seed + same input → identical output."""
        x = np.random.default_rng(0).uniform(size=(3, 1))
        n1 = BioReservoir(units=4, warmup=2, rk4_substeps=4, seed=99,
                          input_dim=1, parallel=False, output_variables="He")
        n2 = BioReservoir(units=4, warmup=2, rk4_substeps=4, seed=99,
                          input_dim=1, parallel=False, output_variables="He")
        o1 = n1.run(x, reset=True)
        o2 = n2.run(x, reset=True)
        np.testing.assert_array_equal(o1, o2)

    def test_different_seed_different_output(self):
        x = np.random.default_rng(0).uniform(size=(3, 1))
        n1 = BioReservoir(units=4, warmup=2, rk4_substeps=4, seed=1,
                          input_dim=1, parallel=False, output_variables="He")
        n2 = BioReservoir(units=4, warmup=2, rk4_substeps=4, seed=2,
                          input_dim=1, parallel=False, output_variables="He")
        o1 = n1.run(x, reset=True)
        o2 = n2.run(x, reset=True)
        assert not np.allclose(o1, o2)

    def test_output_variables_select_subset(self):
        """Output with 'IHe' (2 vars) should have twice the features of 'He' (1 var)."""
        x = np.random.default_rng(0).uniform(size=(3, 1))
        n_he = BioReservoir(units=4, warmup=1, rk4_substeps=4, seed=42,
                             input_dim=1, parallel=False, output_variables="He")
        out_he = n_he.run(x, reset=True)
        assert out_he.shape[1] == 4  # 4 units × 1 var


class TestBioReservoirReset:
    def test_reset_restores_warmup_state(self, tiny_node):
        """After reset, states should match the warmup states."""
        x = np.random.default_rng(0).uniform(size=(3, 1))
        tiny_node.run(x, reset=True)
        post_run_states = tiny_node.states.copy()

        # Now mutate by running more
        tiny_node.run(x)
        assert not np.array_equal(tiny_node.states, post_run_states)

        # Reset should restore to warmup
        tiny_node.reset()
        np.testing.assert_array_equal(tiny_node.states, tiny_node._warmup_states)

    def test_reset_produces_same_output(self):
        """Running the same input after reset should yield very similar output.

        Small differences are expected due to stochastic noise injection
        in _compute_input (noise_in, noise_rc).
        """
        node = BioReservoir(
            units=4, warmup=2, rk4_substeps=4, seed=42, input_dim=1,
            output_variables="He", parallel=False,
            noise_rc=0.0, noise_in=0.0, noise_fb=0.0,
        )
        x = np.random.default_rng(0).uniform(size=(3, 1))
        out1 = node.run(x, reset=True)
        out2 = node.run(x, reset=True)
        np.testing.assert_array_equal(out1, out2)


class TestBioReservoirDiagnostics:
    def test_timing_summary_empty_when_disabled(self, tiny_node):
        assert tiny_node.get_timing_summary() == {}

    def test_timing_summary_populated_when_enabled(self, tiny_node_diagnostics):
        x = np.random.default_rng(0).uniform(size=(3, 1))
        tiny_node_diagnostics.run(x, reset=True)
        summary = tiny_node_diagnostics.get_timing_summary()
        assert "input_computation" in summary
        assert "dde_solve" in summary
        # At least 1 call recorded (ReservoirPy may reset between timesteps)
        assert summary["input_computation"]["calls"] >= 1
        assert summary["dde_solve"]["total_s"] > 0

    def test_timing_clears_on_reset(self, tiny_node_diagnostics):
        x = np.random.default_rng(0).uniform(size=(3, 1))
        tiny_node_diagnostics.run(x, reset=True)
        assert tiny_node_diagnostics.get_timing_summary()["dde_solve"]["calls"] >= 1
        # Manual reset should clear timing
        tiny_node_diagnostics.reset()
        assert tiny_node_diagnostics.get_timing_summary() == {}


# ---------------------------------------------------------------------------
# Construction, history buffer, input drive and forward step (v0.1 behaviour)
# ---------------------------------------------------------------------------

def _node(**kwargs):
    options = dict(units=4, warmup=2, rk4_substeps=4, seed=42, input_dim=1,
                   output_variables="He", parallel=False)
    options.update(kwargs)
    return BioReservoir(**options)


class TestConstruction:
    def test_units_are_required_without_a_weight_matrix(self):
        with pytest.raises(ValueError, match="units"):
            BioReservoir(units=None)

    @pytest.mark.parametrize("variables, indices", [
        ("all", [0, 1, 2, 3]), ((), [0, 1, 2, 3]), ("IHe", [1, 3]), ("AHi", [0, 2]),
        ("IHiHe", [1, 2, 3]), (("Hi", "A"), [2, 0]),
    ])
    def test_output_variables_select_state_columns(self, variables, indices):
        node = _node(output_variables=variables, warmup=0)
        np.testing.assert_array_equal(node.output_indices, indices)

    def test_solver_time_grid_spans_one_input_step(self):
        node = _node(rk4_substeps=8, warmup=0)
        np.testing.assert_allclose(node.rk4_timespan, np.linspace(0.0, 1.0, 9))

    def test_warmup_runs_zero_input_steps(self, monkeypatch):
        inputs = []
        original = BioReservoir._step

        def counting_step(self, x):
            inputs.append(np.array(x))
            return original(self, x)

        monkeypatch.setattr(BioReservoir, "_step", counting_step)
        node = _node(warmup=3)
        assert len(inputs) == 3 and all(np.all(x == 0) for x in inputs)
        np.testing.assert_array_equal(node._warmup_states, node.states)
        assert node.t == 0

    def test_heterogeneity_creates_seeded_node_parameters(self):
        assert _node(warmup=0).node_params is None
        node = _node(warmup=0, param_heterogeneity_cv=0.2, param_heterogeneity_seed=5)
        assert node.node_params.shape == (4, 16)
        same = _node(warmup=0, param_heterogeneity_cv=0.2, param_heterogeneity_seed=5)
        other = _node(warmup=0, param_heterogeneity_cv=0.2, param_heterogeneity_seed=6)
        np.testing.assert_array_equal(node.node_params, same.node_params)
        assert not np.array_equal(node.node_params, other.node_params)

    def test_hyperparameters_are_logged_at_debug_level(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="bioreservoir.node"):
            _node(warmup=0)
        assert any("Reservoir Hyperparameters" in r.getMessage() for r in caplog.records)


class TestInitialHistory:
    def test_history_covers_the_delay(self):
        node = _node(warmup=0, dde_kwargs={"delay": 4.5})
        assert node._initialize_states().shape == (6, 4, 4)

    def test_initial_levels_are_log_normal_as_documented(self):
        node = _node(units=4000, warmup=0)
        logs = np.log(node._initialize_states())
        for var, (mean, sd) in enumerate([(-2.0, 0.5), (-2.0, 0.5), (-4.0, 0.3), (-3.0, 0.3)]):
            assert logs[..., var].mean() == pytest.approx(mean, abs=0.01)
            assert logs[..., var].std() == pytest.approx(sd, rel=0.02)

    def test_initial_levels_depend_on_the_seed(self):
        a, b, c = (_node(warmup=0, seed=s)._initialize_states() for s in (1, 1, 2))
        np.testing.assert_array_equal(a, b)
        assert not np.array_equal(a, c)

    def test_history_shifts_by_one_step(self):
        node = _node(warmup=0)
        before = node.states.copy()
        new = np.full((4, 4), 9.0)
        node._update_history(new)
        np.testing.assert_array_equal(node.states[:-1], before[1:])
        np.testing.assert_array_equal(node.states[-1], new)


class TestStep:
    def test_step_solves_one_input_step_and_appends_the_final_state(self, monkeypatch):
        node = _node(warmup=0, cell_coupling=3.0, delay_scheme="frozen")
        history = node.states.copy()
        calls = {}

        def fake_solver(states, times, input_vec, parallel, node_params, **dde_kwargs):
            calls.update(states=states.copy(), times=times, input_vec=input_vec,
                         parallel=parallel, node_params=node_params, dde=dde_kwargs)
            return np.stack([np.zeros((4, 4)), np.full((4, 4), 0.5)])

        monkeypatch.setattr(node_module, "solve_dde_rk4", fake_solver)
        out = node._step(np.array([[0.1], [0.2], [0.3], [0.4]]))
        np.testing.assert_array_equal(calls["states"], history)
        np.testing.assert_allclose(calls["input_vec"], [0.3, 0.6, 0.9, 1.2])
        np.testing.assert_array_equal(calls["times"], node.rk4_timespan)
        assert calls["parallel"] is False and calls["node_params"] is None
        assert calls["dde"] == node.dde_kwargs
        np.testing.assert_array_equal(out, np.full((4, 4), 0.5))
        np.testing.assert_array_equal(node.states[-1], out)
        np.testing.assert_array_equal(node.states[:-1], history[1:])

    def test_substep_step_advances_the_state_and_the_delay_history(self, monkeypatch):
        node = _node(warmup=0, cell_coupling=3.0)
        history, ring, pos = node.states.copy(), node._ring, node._ring_pos
        calls = {}

        def fake_solver(state, ring_, pos_, input_vec, substeps, parallel, node_params,
                        **dde_kwargs):
            calls.update(state=state.copy(), ring=ring_, pos=pos_, input_vec=input_vec,
                         substeps=substeps, parallel=parallel, node_params=node_params,
                         dde=dde_kwargs)
            return np.full((4, 4), 0.5), 7

        monkeypatch.setattr(node_module, "solve_dde_substep", fake_solver)
        out = node._step(np.array([[0.1], [0.2], [0.3], [0.4]]))
        np.testing.assert_array_equal(calls["state"], history[-1])
        assert calls["ring"] is ring and calls["pos"] == pos
        np.testing.assert_allclose(calls["input_vec"], [0.3, 0.6, 0.9, 1.2])
        assert calls["substeps"] == 4 and calls["parallel"] is False
        assert calls["node_params"] is node._param_table and calls["dde"] == node.dde_kwargs
        np.testing.assert_array_equal(out, np.full((4, 4), 0.5))
        np.testing.assert_array_equal(node.states[-1], out)
        assert node._ring_pos == 7

    def test_delay_history_starts_from_the_stored_history(self):
        node = _node(warmup=0, rk4_substeps=4, dde_kwargs={"delay": 2.0})
        assert node._ring.shape == (4, 9) and node._ring_pos == 8
        hi = node._initialize_states()[:, :, 2]
        np.testing.assert_allclose(node._ring[:, ::4], hi.T)
        np.testing.assert_allclose(node._ring[:, 2], 0.5 * (hi[0] + hi[1]))

    def test_reset_restores_the_delay_history(self):
        node = _node(noise_rc=0.0, noise_in=0.0)
        warm = node._ring.copy(), node._ring_pos
        x = np.random.default_rng(0).uniform(size=(3, 1))
        first = node.run(x, reset=True)
        assert not np.array_equal(node._ring, warm[0])
        node.reset()
        np.testing.assert_array_equal(node._ring, warm[0])
        assert node._ring_pos == warm[1]
        np.testing.assert_array_equal(node.run(x, reset=True), first)

    def test_frozen_scheme_keeps_no_substep_history(self):
        node = _node(delay_scheme="frozen")
        assert node._warmup_ring is None and not hasattr(node, "_ring")


class TestInputDrive:
    def test_drive_combines_input_bias_and_recurrent_terms(self):
        node = _node(noise_in=0.0, noise_rc=0.0, input_bias=True, rc_scaling=0.3,
                     dde_scaling=0.01)
        x = np.array([[0.7]])
        node.initialize(x)
        r = node.state().T
        expected = 0.01 * (node.Win @ x.T + node.bias + 0.3 * (node.W @ r))
        drive = node_module._compute_input(node, x)
        assert drive.shape == (4, 1)
        np.testing.assert_allclose(drive, expected)

    def test_noise_changes_the_drive(self):
        node = _node(noise_in=0.5, noise_rc=0.5)
        x = np.array([[0.7]])
        node.initialize(x)
        assert not np.allclose(node_module._compute_input(node, x),
                               node_module._compute_input(node, x))

    def test_diagnostics_keep_the_last_terms(self, caplog):
        node = _node(diagnostics=True, noise_in=0.0, noise_rc=0.0, dde_scaling=0.02)
        with caplog.at_level(logging.DEBUG, logger="bioreservoir.node"):
            node.run(np.array([[0.5], [0.1]]), reset=True)
        np.testing.assert_allclose(node._last_drive, node._last_inp_term + node._last_rec_term)
        np.testing.assert_allclose(node._last_final_input, 0.02 * node._last_drive)
        assert any("std(inp)" in r.getMessage() for r in caplog.records)


class TestForward:
    def test_output_lists_the_selected_variables_unit_by_unit(self, monkeypatch):
        state = np.arange(12.0).reshape(3, 4)
        stub = SimpleNamespace(t=0, diagnostics=False, rk4_substep_aggregation_window=1,
                               output_indices=np.array([1, 3]), _step=lambda x: state)
        monkeypatch.setattr(node_module, "_compute_input", lambda reservoir, x: x)
        out = node_module._forward(stub, np.zeros((1, 1)))
        np.testing.assert_array_equal(out, [[1.0, 3.0, 5.0, 7.0, 9.0, 11.0]])
        assert stub.t == 1

    def test_run_output_is_the_last_solver_state(self):
        node = _node(noise_in=0.0, noise_rc=0.0)
        out = node.run(np.array([[0.2], [0.4]]), reset=True)
        np.testing.assert_array_equal(out[-1], node.states[-1][:, 3])


class TestGeometry:
    """With two observed variables the v0.1 node built its kernel over 2N points,
    became 2N units wide, and only the first N drives reached the oscillators."""

    KERNEL = dict(spatial_dim=2, interaction_diameters=3.0, fade_alpha=0.05, sparsity=1.0,
                  signed=True, p_excite=0.7, cutoff_radius=None, seed=4)

    def _built(self, **kwargs):
        node = _node(output_variables="IHe", warmup=0, W=distance_matrix(**self.KERNEL),
                     seed=4, **kwargs)
        node.initialize(np.zeros((1, 1)))
        return node

    def test_the_v0_1_geometry_doubles_the_units(self):
        node = self._built(geometry="v0.1")
        assert node.units == 8 and node.W.shape == (8, 8) and node.Win.shape == (8, 1)

    def test_one_kernel_row_per_oscillator(self):
        node = self._built()
        K = np.asarray(distance_matrix(4, 4, **self.KERNEL))
        assert node.units == 4 and node.W.shape == (4, 8)
        assert node.Win.shape == (4, 1) and node.bias.shape == (4, 1)
        np.testing.assert_allclose(node.W.toarray(), np.kron(K, [[1.0, 1.0]]))

    def test_drives_match_the_geometry_corrected_variant(self):
        node = self._built()
        K = node.W.toarray()[:, 0::2]
        W_full = np.zeros((8, 8))
        W_full[:4, 0::2] = K
        W_full[:4, 1::2] = K
        r = np.random.default_rng(0).normal(size=(8, 1))
        np.testing.assert_allclose(node.W @ r, (W_full @ r)[:4])

    def test_observed_variables_can_be_weighted(self):
        node = self._built(couple_vars=(1.0, 0.0))
        np.testing.assert_array_equal(node.W.toarray()[:, 1::2], 0.0)
        with pytest.raises(ValueError, match="couple_vars"):
            _node(output_variables="IHe", couple_vars=(1.0,))

    def test_one_observed_variable_is_the_same_in_both_geometries(self):
        x = np.linspace(0, 1, 6)[:, None]
        a, b = (_node(warmup=2, geometry=g).run(x, reset=True)
                for g in ("per_node", "v0.1"))
        np.testing.assert_array_equal(a, b)

    def test_unknown_geometry_is_rejected(self):
        with pytest.raises(ValueError, match="geometry"):
            _node(geometry="grid")


class TestSequentialSolver:
    """The v0.1 sequential solver crashed when two variables were observed."""

    @pytest.mark.parametrize("geometry", ["per_node", "v0.1"])
    def test_sequential_and_parallel_runs_agree(self, geometry):
        x = np.linspace(0, 1, 6)[:, None]
        runs = [_node(output_variables="IHe", warmup=2, geometry=geometry, parallel=p)
                .run(x, reset=True) for p in (True, False)]
        np.testing.assert_allclose(runs[0], runs[1], rtol=1e-10, atol=1e-12)


class TestResetToState:
    def test_new_state_restores_the_warmup_history(self):
        node = _node()
        node.run(np.array([[0.2], [0.4]]), reset=True)
        node.t = 5
        node.reset(to_state=np.zeros((1, 4)))
        np.testing.assert_array_equal(node.states, node._warmup_states)
        assert node.t == 0

    def test_same_state_keeps_the_history(self):
        node = _node()
        node.run(np.array([[0.2], [0.4]]), reset=True)
        history = node.states.copy()
        node.reset(to_state=node.state().copy())
        np.testing.assert_array_equal(node.states, history)
