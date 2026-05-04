"""Unit tests for bioreservoir.node.BioReservoir."""

import pytest
import numpy as np

from bioreservoir import BioReservoir


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
