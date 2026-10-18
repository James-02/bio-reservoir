"""Unit tests for bioreservoir.diagnostics."""

import numpy as np
import pytest

import bioreservoir.node as node_module
from bioreservoir import BioReservoir
from bioreservoir.diagnostics import STATE_NAMES, drive_and_state_ranges, drive_shares

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


class TestDriveShares:
    """How much of the drive comes from the input and from each coupled variable."""

    INPUTS = [np.random.default_rng(0).normal(size=(6, 1)) for _ in range(3)]

    @staticmethod
    def _node(**kwargs):
        options = dict(units=5, warmup=2, rk4_substeps=16, seed=2, input_dim=1,
                       output_variables="IHe", parallel=False, noise_in=0.0, noise_rc=0.0,
                       rc_scaling=0.5, rc_connectivity=1.0)
        options.update(kwargs)
        return BioReservoir(**options)

    def test_shares_cover_the_input_and_each_observed_variable(self):
        shares = drive_shares(self._node(), self.INPUTS)
        assert list(shares) == ["input", "I", "He"]
        assert sum(shares.values()) == pytest.approx(1.0)
        assert all(share > 0 for share in shares.values())

    @pytest.mark.parametrize("couple_vars, silent", [((1.0, 0.0), "He"), ((0.0, 1.0), "I")])
    def test_an_uncoupled_variable_has_no_share(self, couple_vars, silent):
        shares = drive_shares(self._node(couple_vars=couple_vars), self.INPUTS)
        assert shares[silent] == 0.0

    def test_without_coupling_the_drive_is_the_input(self):
        shares = drive_shares(self._node(rc_scaling=0.0), self.INPUTS)
        assert shares == {"input": 1.0, "I": 0.0, "He": 0.0}

    def test_the_v0_1_geometry_and_sparse_weights_are_supported(self):
        shares = drive_shares(self._node(geometry="v0.1", input_connectivity=0.5,
                                         input_bias=True), self.INPUTS)
        assert sum(shares.values()) == pytest.approx(1.0)
