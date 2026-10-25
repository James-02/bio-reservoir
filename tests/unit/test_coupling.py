"""Diffusive coupling: colonies exchange AHL through a graph Laplacian inside every RK4 stage."""

import numpy as np
import pytest
from scipy import sparse
from scipy.integrate import solve_ivp

from bioreservoir import BioReservoir, dde, distance_matrix, lattice_matrix
from bioreservoir.diagnostics import drive_and_state_ranges, drive_shares
from bioreservoir.topology import relative_asymmetry, scale_conductances
from optimization.reservoirs import build_reservoir
from training import run as run_module

KERNEL = dict(interaction_diameters=3.0, fade_alpha=0.01, sparsity=1.0, signed=False, seed=5)


def _model_rhs(y, delayed_hi, u, p):
    """One oscillator's equations (Danino et al. 2010, input added to He in the Hi flux)."""
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


def _coupled_reference(history, delay, inputs, params, G):
    """Method of steps with DOP853 for coupled oscillators.

    ``history`` (H, n, 4) holds the states at t = -(H-1), ..., 0 (Hi linear in between);
    each row of ``inputs`` is the drive of every oscillator during one unit step.
    """
    n = history.shape[1]
    t_hist = np.arange(-(len(history) - 1), 1.0)
    G = np.asarray(G, dtype=float)
    segments = []

    def delayed(t, Y):
        if delay == 0:
            return Y[:, 2]
        s = t - delay
        if s <= 0:
            return np.array([np.interp(s, t_hist, history[:, i, 2]) for i in range(n)])
        for a, b, sol in segments:
            if a <= s <= b:
                return sol.sol(s).reshape(n, 4)[:, 2]
        raise AssertionError(f"no solution stored at t = {s}")

    def rhs(t, y, u):
        Y = y.reshape(n, 4)
        lag = delayed(t, Y)
        dY = np.array([_model_rhs(Y[i], lag[i], u[i], params) for i in range(n)])
        dY[:, 3] += G @ Y[:, 3] - G.sum(axis=1) * Y[:, 3]
        return dY.ravel()

    y = history[-1].ravel().copy()
    for step, u in enumerate(inputs):
        sol = solve_ivp(lambda t, y, u=u: rhs(t, y, u), (step, step + 1), y, method="DOP853",
                        rtol=1e-12, atol=1e-14, dense_output=True)
        segments.append((step, step + 1, sol))
        y = sol.y[:, -1]
    return y.reshape(n, 4)


def _history(n=3, length=11, seed=0):
    """Start-up levels like those BioReservoir draws (A, I, Hi, He), constant in time."""
    rng = np.random.default_rng(seed)
    levels = np.column_stack([rng.lognormal(-2.0, 0.5, n), rng.lognormal(-2.0, 0.5, n),
                              rng.lognormal(-4.0, 0.3, n), rng.lognormal(-3.0, 0.3, n)])
    return np.repeat(levels[None], length, axis=0)


def _solve(history, delay, substeps, inputs, conductance=None, **overrides):
    ring, pos = dde.substep_history(history, delay, substeps)
    state = history[-1]
    for u in inputs:
        state, pos = dde.solve_dde_substep(state, ring, pos, u, substeps,
                                           conductance=conductance, delay=delay, **overrides)
    return state


G3 = np.array([[0.0, 1.5, 0.4], [1.5, 0.0, 0.8], [0.4, 0.8, 0.0]])
INPUTS = np.array([[0.02, 0.0, 0.01], [0.0, 0.015, 0.0], [0.01, 0.01, 0.02], [0.0, 0.0, 0.0]])


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class TestCoupledSolver:
    @pytest.mark.parametrize("delay", [10.0, 0.0])
    def test_zero_conductances_give_the_uncoupled_solution(self, delay):
        history = _history(n=4)
        inputs = np.full((3, 4), 0.01)
        uncoupled = _solve(history, delay, 16, inputs)
        coupled = _solve(history, delay, 16, inputs, conductance=np.zeros((4, 4)))
        np.testing.assert_allclose(coupled, uncoupled, rtol=1e-12, atol=1e-15)

    def test_identical_colonies_exchange_nothing(self):
        history = np.repeat(_history(n=1), 3, axis=1)
        inputs = np.full((3, 3), 0.01)
        np.testing.assert_allclose(_solve(history, 10.0, 16, inputs, conductance=G3),
                                   _solve(history, 10.0, 16, inputs), rtol=1e-12)

    def test_symmetric_exchange_only_moves_ahl(self):
        # Without membrane exchange (D = 0) and washout (mu = 0), only the coupling moves He.
        history = _history(n=3)
        total = history[-1][:, 3].sum()
        moved = _solve(history, 10.0, 16, np.zeros((3, 3)), conductance=G3, D=0.0, mu=0.0)
        assert moved[:, 3].sum() == pytest.approx(total, rel=1e-12)
        assert not np.allclose(moved[:, 3], history[-1][:, 3])
        one_way = np.triu(G3)
        leaked = _solve(history, 10.0, 16, np.zeros((3, 3)), conductance=one_way, D=0.0, mu=0.0)
        assert abs(leaked[:, 3].sum() - total) > 1e-3 * total

    def test_the_coupled_ode_converges_at_fourth_order(self):
        history = _history(n=3)
        params = dde.get_dde_params(delay=0.0)
        expected = _coupled_reference(history, 0.0, INPUTS[:1], params, G3)
        errors = np.array([np.abs(_solve(history, 0.0, s, INPUTS[:1], conductance=G3)
                                  - expected).max() for s in (32, 64, 128)])
        orders = np.log2(errors[:-1] / errors[1:])
        assert np.all(orders > 3.5), (errors, orders)
        assert errors[-1] < 1e-7 * np.abs(expected).max(), errors

    def test_the_coupled_delay_equation_converges_at_second_order(self):
        history = _history(n=3)
        params = dde.get_dde_params(delay=2.0)
        expected = _coupled_reference(history, 2.0, INPUTS, params, G3)
        errors = np.array([np.abs(_solve(history, 2.0, s, INPUTS, conductance=G3)
                                  - expected).max() for s in (32, 64, 128)])
        assert np.all(np.log2(errors[:-1] / errors[1:]) > 1.5), errors
        assert errors[-1] < 2e-5 * np.abs(expected).max(), errors

    def test_prepared_conductances_are_padded_rows(self):
        G = sparse.csr_matrix([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        prepared = dde.prepare_conductances(G)
        assert prepared.index.shape == prepared.value.shape == (3, 2)
        np.testing.assert_array_equal(prepared.row_sum, [2.0, 3.0, 1.0])
        np.testing.assert_array_equal(prepared.value.sum(axis=1), [2.0, 3.0, 1.0])
        history = _history(n=3)
        np.testing.assert_array_equal(_solve(history, 10.0, 8, INPUTS, conductance=G),
                                      _solve(history, 10.0, 8, INPUTS, conductance=prepared))

    def test_conductances_are_checked(self):
        history = _history(n=3)
        with pytest.raises(ValueError, match="non-negative"):
            _solve(history, 10.0, 8, INPUTS[:1], conductance=-G3)
        with pytest.raises(ValueError, match="square"):
            dde.prepare_conductances(np.ones((3, 2)))
        with pytest.raises(ValueError, match="not 3"):
            _solve(history, 10.0, 8, INPUTS[:1], conductance=np.zeros((2, 2)))

    @pytest.mark.parametrize("delay", [2.0, 0.0])
    def test_the_compiled_solver_matches_its_python_source(self, monkeypatch, delay):
        # The uncompiled source runs the same arithmetic where coverage can trace it.
        history = _history(n=3)
        compiled = _solve(history, delay, 8, INPUTS, conductance=G3)
        for name in ("_stage_state", "_stage_derivatives", "_solve_diffusion"):
            kernel = getattr(dde, name)
            monkeypatch.setattr(dde, name, getattr(kernel, "py_func", kernel))
        np.testing.assert_allclose(_solve(history, delay, 8, INPUTS, conductance=G3),
                                   compiled, rtol=1e-12, atol=1e-15)


# ---------------------------------------------------------------------------
# Layouts and conductances
# ---------------------------------------------------------------------------

class TestLayouts:
    def test_the_lattice_links_grid_neighbours(self):
        G = lattice_matrix(12, 12)  # 4 x 3 grid
        assert (G != G.T).nnz == 0 and set(G.data) == {1.0}
        assert sorted(np.diff(G.indptr)) == [2] * 4 + [3] * 6 + [4] * 2
        assert G[0, 1] == G[0, 3] == 1.0 and G[0, 4] == 0.0
        assert (lattice_matrix()(12, 12, sr=None, seed=3) != G).nnz == 0

    def test_long_thin_lattices_are_refused(self):
        assert sorted(np.diff(lattice_matrix(8, 8).indptr)) == [2] * 4 + [3] * 4  # 4 x 2
        assert lattice_matrix(1000, 1000).nnz == 2 * (40 * 24 + 39 * 25)  # 40 x 25
        with pytest.raises(ValueError, match="7 x 1 lattice.*such as 9"):
            lattice_matrix(7, 7)
        with pytest.raises(ValueError, match="25 x 10 lattice"):
            lattice_matrix(250, 250)
        with pytest.raises(ValueError, match="square"):
            lattice_matrix(4, 5)

    def test_conductances_are_scaled_to_the_mean_rate(self):
        rng = np.random.default_rng(0)
        W = rng.random((6, 6))
        G = scale_conductances(W + W.T, 0.7)
        assert G.diagonal().sum() == 0.0
        assert G.sum() / 6 == pytest.approx(0.7)
        assert scale_conductances(np.zeros((3, 3)), 0.7).nnz == 0
        with pytest.raises(ValueError, match="signed=False"):
            scale_conductances(W - 0.5, 1.0)
        with pytest.raises(ValueError, match="square"):
            scale_conductances(np.ones((3, 2)), 1.0)

    def test_asymmetry_of_symmetric_and_sparsified_kernels(self):
        assert relative_asymmetry(distance_matrix(30, 30, **KERNEL)) == 0.0
        sparsified = distance_matrix(30, 30, **dict(KERNEL, sparsity=0.5))
        assert relative_asymmetry(sparsified) > 0.3
        assert relative_asymmetry(np.zeros((3, 3))) == 0.0


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def _node(**kwargs):
    options = dict(units=6, warmup=4, rk4_substeps=16, seed=5, input_dim=1,
                   output_variables="IHe", input_scaling=0.05, dde_scaling=0.05,
                   cell_coupling=1.0, noise_in=0.0, noise_rc=0.0, parallel=False)
    options.update(kwargs)
    return BioReservoir(**options)


X = np.random.default_rng(1).normal(size=(12, 1))


class TestNodeCoupling:
    def test_the_conductances_are_the_scaled_kernel(self):
        node = _node(coupling="diffusion", diffusion_rate=0.4, W=distance_matrix(**KERNEL))
        node.run(X, reset=True)
        expected = scale_conductances(distance_matrix(6, 6, **KERNEL), 0.4)
        assert (node.W != expected).nnz == 0 and node.W.shape == (6, 6)
        assert node.conductance.sum() / 6 == pytest.approx(0.4)

    def test_a_zero_rate_is_the_uncoupled_array(self):
        zero = _node(coupling="diffusion", diffusion_rate=0.0, W=distance_matrix(**KERNEL))
        np.testing.assert_allclose(zero.run(X, reset=True), _node(coupling="none").run(X),
                                   rtol=1e-10)

    def test_uncoupled_is_the_abstract_model_without_recurrent_gain(self):
        np.testing.assert_array_equal(_node(coupling="none").run(X),
                                      _node(coupling="abstract", rc_scaling=0.0).run(X))

    def test_the_warm_up_is_coupled(self):
        coupled = _node(coupling="diffusion", diffusion_rate=2.0, W=distance_matrix(**KERNEL))
        uncoupled = _node(coupling="none")
        assert not np.allclose(coupled._warmup_states[-1], uncoupled._warmup_states[-1])
        spread = [np.ptp(n._warmup_states[-1][:, 3]) for n in (coupled, uncoupled)]
        assert spread[0] < spread[1]

    def test_coupling_changes_the_states(self):
        coupled = _node(coupling="diffusion", diffusion_rate=2.0, W=lattice_matrix())
        assert not np.allclose(coupled.run(X), _node(coupling="none").run(X))

    def test_an_exchange_too_fast_for_the_solver_step_is_refused(self):
        # 10 x 10 lattice, 16 substeps: an interior colony's conductances sum to 1.11 x the
        # rate, so He's fastest mode decays at 5.83 + 0.6 + 2.22 x the rate; at a rate of 18
        # (46.4 per time unit) the explicit solver blows up, at 10 (28.7) it is stable.
        with pytest.raises(ValueError, match="needs at least 18 substeps"):
            _node(units=100, coupling="diffusion", diffusion_rate=18.0, W=lattice_matrix())
        stable = _node(units=100, coupling="diffusion", diffusion_rate=10.0, W=lattice_matrix())
        assert np.abs(stable.run(X, reset=True)).max() < 1e3

    def test_drive_shares_need_the_abstract_coupling(self):
        node = _node(coupling="diffusion", W=distance_matrix(**KERNEL))
        with pytest.raises(ValueError, match="abstract"):
            drive_shares(node, [X])

    @pytest.mark.parametrize("options, message", [
        ({"coupling": "global"}, "coupling must be"),
        ({"coupling": "diffusion", "geometry": "v0.1"}, "geometry='per_node'"),
        ({"coupling": "diffusion", "delay_scheme": "frozen"}, "delay_scheme='substep'"),
        ({"coupling": "diffusion", "W": distance_matrix(**dict(KERNEL, signed=True))},
         "non-negative"),
        ({"input_encoding": "offset"}, "input_encoding"),
        ({"input_sign": "plus"}, "input_sign"),
        ({"input_sign": "positive", "geometry": "v0.1"}, "input_sign='positive'"),
        ({"input_range": (2.0, 1.0)}, "input_range"),
    ])
    def test_invalid_options_are_rejected(self, options, message):
        with pytest.raises(ValueError, match=message):
            _node(**options)


class TestNonNegativeInput:
    def test_the_input_is_clipped_and_shifted_by_a_fixed_offset(self):
        node = _node(input_encoding="nonnegative", input_range=(-3.0, 8.0),
                     input_sign="positive", diagnostics=True)
        node.run(np.array([[-5.0], [0.0], [9.0]]), reset=True)
        Win = node.Win.toarray() if sparse.issparse(node.Win) else node.Win
        assert Win.min() >= 0.0 and Win.max() > 0.0
        np.testing.assert_allclose(node._last_inp_term, Win * 11.0)  # 9 is clipped to 8
        node.run(np.array([[-5.0]]), reset=True)
        np.testing.assert_array_equal(node._last_inp_term, 0.0)

    def test_input_noise_cannot_make_the_input_negative(self):
        node = _node(input_encoding="nonnegative", input_sign="positive", noise_in=5.0,
                     diagnostics=True)
        for _ in range(5):
            node.run(np.array([[-3.0]]), reset=True)
            assert node._last_inp_term.min() >= 0.0

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_colonies_stay_non_negative(self, seed):
        node = _node(units=8, coupling="diffusion", diffusion_rate=1.0, W=lattice_matrix(),
                     input_encoding="nonnegative", input_sign="positive",
                     input_scaling=0.5, noise_in=0.1, seed=seed)
        beats = [np.random.default_rng(seed + k).normal(scale=2.0, size=(30, 1))
                 for k in range(3)]
        ranges = drive_and_state_ranges(node, beats)
        assert min(ranges["state_min"].values()) > 0.0, ranges["state_min"]


# ---------------------------------------------------------------------------
# Builder and run configs
# ---------------------------------------------------------------------------

PARAMS = dict(units=6, warmup=2, rk4_substeps=16, input_scaling=0.05, input_connectivity=1.0,
              input_bias=False, bias_scaling=0.0, dde_scaling=0.05, cell_coupling=1.0,
              noise_in=0.0, output_variables="IHe")


class TestBuilder:
    def test_a_diffusive_lattice_needs_no_recurrent_gain(self):
        node, attrs = build_reservoir(dict(PARAMS, coupling="diffusion", diffusion_rate=0.3,
                                           topology_type="lattice", input_sign="positive",
                                           input_encoding="nonnegative",
                                           input_range=[-2.0, 6.0]), seed=1)
        assert (node.coupling, node.diffusion_rate, node.input_range) == (
            "diffusion", 0.3, (-2.0, 6.0))
        assert (node.conductance != scale_conductances(lattice_matrix(6, 6), 0.3)).nnz == 0
        assert attrs["res_coupling"] == "diffusion" and attrs["res_diffusion_rate"] == 0.3
        assert attrs["res_input_range"] == [-2.0, 6.0] and attrs["topo_type"] == "lattice"
        assert attrs["res_rc_scaling"] == 0.0 and attrs["res_noise_rc"] == 0.0

    def test_the_abstract_coupling_still_needs_its_recurrent_gain(self):
        with pytest.raises(KeyError, match="rc_scaling"):
            build_reservoir(dict(PARAMS), seed=1)

    def test_the_abstract_coupling_refuses_the_unnormalised_lattice(self):
        with pytest.raises(ValueError, match="'lattice' needs coupling"):
            build_reservoir(dict(PARAMS, topology_type="lattice", rc_scaling=0.1,
                                 noise_rc=0.0), seed=1)
        node, _ = build_reservoir(dict(PARAMS, topology_type="lattice", coupling="none"),
                                  seed=1)
        assert node.coupling == "none"

    def test_trials_record_the_coupling_options(self):
        from optimization.training import RECORDED

        assert {RECORDED[f"res_{k}"] for k in ("coupling", "diffusion_rate", "input_encoding",
                                               "input_range", "input_sign")} == {
            "coupling", "diffusion_rate", "input_encoding", "input_range", "input_sign"}

    def test_the_qs_diffusion_preset_sets_the_model_structure(self):
        resolved = run_module.resolve({
            "name": "p", "preset": "qs_diffusion",
            "reservoir": {"params": dict(PARAMS, interaction_diameters=3.0, fade_alpha=0.01,
                                         diffusion_rate=0.5)},
            "readout": {"classifier": "Ridge"}})
        params = resolved["reservoir"]["params"]
        assert (params["coupling"], params["signed"], params["sparsity"]) == (
            "diffusion", False, 1.0)
        assert (params["input_encoding"], params["input_sign"]) == ("nonnegative", "positive")
        node, _ = build_reservoir(dict(params, reservoir_type="bioreservoir"), seed=1)
        assert relative_asymmetry(node.conductance) == 0.0
        lattice = run_module.resolve({
            "name": "p", "preset": "qs_diffusion",
            "reservoir": {"params": dict(PARAMS, topology_type="lattice")},
            "readout": {"classifier": "Ridge"}})
        assert lattice["reservoir"]["params"]["topology_type"] == "lattice"
