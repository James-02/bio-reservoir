"""Unit tests for bioreservoir.diagnostics."""

import numpy as np
import pytest

import bioreservoir.node as node_module
from bioreservoir import BioReservoir
from bioreservoir.diagnostics import STATE_NAMES, drive_and_state_ranges

DRIVE = np.array([[1e-2], [1e-3], [-1e-2], [0.0]])  # two of four oscillators above 5e-3


def _node():
    return BioReservoir(units=4, warmup=2, rk4_substeps=16, seed=1, input_dim=1,
                        output_variables="all", parallel=False, noise_in=0.0, noise_rc=0.0)


@pytest.fixture
def fixed_drive(monkeypatch):
    monkeypatch.setattr(node_module, "_compute_input", lambda reservoir, x: DRIVE)


def test_drive_beyond_the_bound_is_counted(fixed_drive):
    summary = drive_and_state_ranges(_node(), [np.zeros((3, 1)), np.zeros((2, 1))], bound=5e-3)
    assert summary["oscillator_steps"] == 4 * 5
    assert summary["drive_fraction_above"] == 0.5
    assert summary["drive_abs_max"] == 1e-2


def test_state_ranges_are_those_of_the_run(fixed_drive):
    node, inputs = _node(), [np.zeros((3, 1)), np.zeros((2, 1))]
    summary = drive_and_state_ranges(node, inputs)
    copy = node.copy()
    states = np.concatenate([copy.run(x, reset=True).reshape(len(x), 4, 4) for x in inputs])
    for i, name in enumerate(STATE_NAMES):
        assert summary["state_min"][name] == pytest.approx(states[..., i].min(), rel=1e-12)
        assert summary["state_max"][name] == pytest.approx(states[..., i].max(), rel=1e-12)
        assert summary["state_negative_fraction"][name] == np.mean(states[..., i] < 0)


def test_the_template_reservoir_is_left_untouched(fixed_drive):
    node = _node()
    before = node.states.copy()
    drive_and_state_ranges(node, [np.zeros((3, 1))])
    np.testing.assert_array_equal(node.states, before)
    assert not node.is_initialized
