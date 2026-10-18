"""Unit tests for training.pipeline helpers."""

import logging
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
from reservoirpy.nodes import Ridge, ScikitLearnNode
from sklearn.linear_model import LinearRegression, LogisticRegression

from training.pipeline import (
    _aggregate_states,
    _is_sklearn_classifier,
    _stack_runs,
    fit_readout,
    predict,
    train,
)
from training.profiling import Profiler, ResourceMonitor, _merge_profiles

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

    def test_stack_C(self):
        stacked = _stack_runs([np.zeros(5) for _ in range(2)])
        assert stacked.shape == (2, 1, 5)

    def test_other_shapes_are_rejected(self):
        with pytest.raises(ValueError, match="Unexpected output stack shape"):
            _stack_runs([1.0, 2.0])


# ---------------------------------------------------------------------------
# train, fit_readout and predict
# ---------------------------------------------------------------------------

class _StubReservoir:
    """Returns the input and twice the input as a two-unit state trajectory."""

    units = 2

    def __init__(self):
        self.resets = []

    def run(self, x, reset=False):
        self.resets.append(reset)
        return np.hstack([x, 2 * x])


class TestTrain:
    @pytest.fixture
    def X(self):
        return np.random.default_rng(0).normal(size=(3, 6, 1))

    def test_each_instance_runs_from_a_reset_state(self, X):
        reservoir = _StubReservoir()
        out = train(reservoir, X, readout_aggregation="mean")
        assert reservoir.resets == [True, True, True]
        for x, o in zip(X, out, strict=True):
            np.testing.assert_allclose(o, np.hstack([x, 2 * x]).mean(axis=0, keepdims=True))

    def test_a_readout_window_overrides_the_aggregation(self, X):
        out = train(_StubReservoir(), X, readout_aggregation="mean", readout_window=3)
        for x, o in zip(X, out, strict=True):
            np.testing.assert_allclose(o, np.hstack([x, 2 * x])[-3:].mean(axis=0, keepdims=True))

    def test_profiling_does_not_change_the_states(self, X):
        profiler = Profiler(enabled=True)
        plain = train(_StubReservoir(), X)
        timed = train(_StubReservoir(), X, profiler=profiler)
        np.testing.assert_array_equal(np.vstack(plain), np.vstack(timed))
        assert profiler.summary()["instance/reservoir_run"]["calls"] == 3
        assert profiler.summary()["instance/state_aggregation"]["calls"] == 3


def _clusters(n=60, seed=0):
    """Three separated clusters of (1, 2) state vectors with one-hot labels (N, 1, 3)."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 3, size=n)
    centres = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]])
    states = [(centres[c] + 0.5 * rng.normal(size=2)).reshape(1, 2) for c in labels]
    return states, np.eye(3)[labels][:, None, :], labels


class TestReadouts:
    def test_sklearn_classifiers_are_recognised(self):
        assert _is_sklearn_classifier(ScikitLearnNode(model=LogisticRegression))
        assert not _is_sklearn_classifier(ScikitLearnNode(model=LinearRegression))
        assert not _is_sklearn_classifier(Ridge(ridge=1e-3))

    def test_sklearn_classifier_is_fitted_on_class_indices(self):
        states, Y, labels = _clusters()
        readout = ScikitLearnNode(model=LogisticRegression,
                                  model_hypers={"C": 0.5, "max_iter": 500})
        fit_readout(readout, states, Y)
        assert readout._n_classes == 3 and readout._fitted_sklearn.C == 0.5
        reference = LogisticRegression(C=0.5, max_iter=500).fit(np.vstack(states), labels)
        predictions = predict(readout, states)
        assert all(p.shape == (1, 3) for p in predictions)
        np.testing.assert_array_equal(np.vstack(predictions),
                                      np.eye(3)[reference.predict(np.vstack(states))])

    def test_older_fitted_nodes_return_class_indices(self):
        states, _, labels = _clusters()
        fitted = LogisticRegression(max_iter=500).fit(np.vstack(states), labels)
        readout = SimpleNamespace(hypers={"model": LogisticRegression},
                                  params={"instances": [fitted]})
        predictions = predict(readout, states)
        np.testing.assert_array_equal(np.concatenate(predictions),
                                      fitted.predict(np.vstack(states)))

    def test_ridge_readout_fits_a_linear_map(self):
        rng = np.random.default_rng(1)
        states = [rng.normal(size=(1, 3)) for _ in range(40)]
        A = rng.normal(size=(3, 2))
        Y = np.stack([s @ A + 0.5 for s in states])
        readout = Ridge(ridge=1e-9)
        fit_readout(readout, states, Y)
        predictions = predict(readout, states)
        assert all(p.shape == (1, 2) for p in predictions)
        np.testing.assert_allclose(np.vstack(predictions), Y[:, 0, :], atol=1e-6)


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

    def test_flat_keys_replace_slashes(self):
        p = Profiler(enabled=True)
        with p.section("instance/reservoir_run"):
            pass
        assert set(p.flat_summary()) == {"instance_reservoir_run_total_s",
                                         "instance_reservoir_run_calls",
                                         "instance_reservoir_run_mean_s"}

    def test_failing_sections_are_still_timed(self):
        p = Profiler(enabled=True)
        with pytest.raises(RuntimeError):
            with p.section("boom"):
                raise RuntimeError
        assert p.summary()["boom"]["calls"] == 1


class TestMergeProfiles:
    def test_sections_are_combined_across_folds(self):
        profiles = [
            {"fit": {"total_s": 1.0, "calls": 2, "mean_s": 0.5},
             "run": {"total_s": 4.0, "calls": 10, "mean_s": 0.4}},
            None,
            {"fit": {"total_s": 3.0, "calls": 2, "mean_s": 1.5}},
        ]
        merged = _merge_profiles(profiles)
        assert merged["fit"] == {"total_s": 4.0, "mean_s": 2.0, "std_s": round(np.sqrt(2.0), 6),
                                 "calls_total": 4, "calls_per_fold": 2.0, "n_folds": 2}
        # A section missing from a fold is still averaged over every fold.
        assert merged["run"]["mean_s"] == 2.0 and merged["run"]["n_folds"] == 2

    def test_one_fold_has_no_spread(self):
        merged = _merge_profiles([{"fit": {"total_s": 2.0, "calls": 1, "mean_s": 2.0}}])
        assert merged["fit"]["std_s"] == 0.0

    def test_no_profiles_give_an_empty_summary(self):
        assert _merge_profiles([None, {}]) == {}


class TestResourceMonitor:
    def test_samples_cpu_and_memory(self):
        monitor = ResourceMonitor(interval=0.01).start()
        deadline = time.perf_counter() + 0.2
        while time.perf_counter() < deadline:
            pass
        stats = monitor.stop()
        assert 0 < stats["runtime_rss_mb_min"] <= stats["runtime_rss_mb_mean"] \
            <= stats["runtime_rss_mb_max"]
        assert isinstance(stats["runtime_cpu_pct_mean"], float)

    def test_zero_cpu_samples_are_ignored(self):
        monitor = ResourceMonitor()
        monitor._cpu, monitor._rss_mb = [0.0, 50.0, 100.0], [10.0, 30.0]
        stats = monitor.stop()
        assert (stats["runtime_cpu_pct_min"], stats["runtime_cpu_pct_mean"],
                stats["runtime_cpu_pct_max"]) == (50.0, 75.0, 100.0)
        assert stats["runtime_rss_mb_mean"] == 20.0

    def test_without_psutil_nothing_is_measured(self, monkeypatch, caplog):
        monkeypatch.setitem(sys.modules, "psutil", None)
        with caplog.at_level(logging.WARNING):
            monitor = ResourceMonitor().start()
        assert "psutil not installed" in caplog.text
        assert set(monitor.stop().values()) == {"N/A"}
