"""Unit tests for training.pipeline helpers."""

import pytest
import numpy as np

from training.pipeline import _aggregate_states, _stack_runs
from training.profiling import Profiler


# ---------------------------------------------------------------------------
# _aggregate_states
# ---------------------------------------------------------------------------

class TestAggregateStates:
    """Tests for the state aggregation helper."""

    @pytest.fixture
    def states(self):
        """A (10, 4) trajectory — 10 timesteps, 4 state dims."""
        return np.arange(40, dtype=float).reshape(10, 4)

    @pytest.fixture
    def x_input(self):
        """A (10, 1) input array with last 3 timesteps zero-padded."""
        x = np.ones((10, 1))
        x[7:] = 0.0
        return x

    def test_final(self, states, x_input):
        out = _aggregate_states(states, "final", x_input)
        assert out.shape == (1, 4)
        np.testing.assert_array_equal(out[0], states[-1])

    def test_mean(self, states, x_input):
        out = _aggregate_states(states, "mean", x_input)
        assert out.shape == (1, 4)
        np.testing.assert_allclose(out[0], states.mean(axis=0))

    def test_signal_mean_excludes_padding(self, states, x_input):
        out = _aggregate_states(states, "signal_mean", x_input)
        assert out.shape == (1, 4)
        # x_input has signal up to index 6 (inclusive), so indices 0-6
        expected = states[:7].mean(axis=0)
        np.testing.assert_allclose(out[0], expected)

    def test_first_N(self, states, x_input):
        out = _aggregate_states(states, "first_N:3", x_input)
        assert out.shape == (1, 4)
        np.testing.assert_allclose(out[0], states[:3].mean(axis=0))

    def test_last_N(self, states, x_input):
        out = _aggregate_states(states, "last_N:3", x_input)
        assert out.shape == (1, 4)
        np.testing.assert_allclose(out[0], states[-3:].mean(axis=0))

    def test_last_N_bare_fallback(self, states, x_input):
        """Bare 'last_N' without a window should fall back to 'final'."""
        out = _aggregate_states(states, "last_N", x_input)
        assert out.shape == (1, 4)
        np.testing.assert_array_equal(out[0], states[-1])

    def test_invalid_mode_raises(self, states, x_input):
        with pytest.raises(ValueError, match="Unknown readout_aggregation mode"):
            _aggregate_states(states, "BOGUS", x_input)

    def test_first_N_clamps_to_length(self, states, x_input):
        """first_N:999 should not crash — just use all timesteps."""
        out = _aggregate_states(states, "first_N:999", x_input)
        assert out.shape == (1, 4)
        np.testing.assert_allclose(out[0], states.mean(axis=0))


# ---------------------------------------------------------------------------
# _stack_runs
# ---------------------------------------------------------------------------

class TestStackRuns:
    def test_stack_1_1_C(self):
        outputs = [np.zeros((1, 1, 5)) for _ in range(3)]
        stacked = _stack_runs(outputs)
        assert stacked.shape == (3, 1, 5)

    def test_stack_1_C(self):
        outputs = [np.zeros((1, 5)) for _ in range(4)]
        stacked = _stack_runs(outputs)
        assert stacked.shape == (4, 1, 5)

    def test_empty_list(self):
        stacked = _stack_runs([])
        assert stacked.shape[0] == 0


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------

class TestProfiler:
    def test_enabled_records_time(self):
        p = Profiler(enabled=True)
        with p.section("a"):
            _ = sum(range(100))
        with p.section("a"):
            _ = sum(range(100))
        summary = p.summary()
        assert "a" in summary
        assert summary["a"]["calls"] == 2
        assert summary["a"]["total_s"] > 0
        assert summary["a"]["mean_s"] > 0

    def test_disabled_is_noop(self):
        p = Profiler(enabled=False)
        with p.section("a"):
            _ = sum(range(100))
        assert p.summary() == {}

    def test_flat_summary(self):
        p = Profiler(enabled=True)
        with p.section("train_reservoir"):
            _ = 1 + 1
        flat = p.flat_summary(prefix="profile_")
        assert "profile_train_reservoir_total_s" in flat
        assert "profile_train_reservoir_calls" in flat
