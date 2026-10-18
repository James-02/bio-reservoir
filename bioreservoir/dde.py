from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numba import njit, prange

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Public API: defaults + preparation
# -----------------------------------------------------------------------------

# NOTE: These are the “biological model” parameters (i.e., independent of
# solver settings like substeps). Keeping them here ensures all DDE logic and
# defaults live in one module.
DEFAULT_DDE_PARAMS: dict[str, float] = {
    "CA": 1.0,
    "CI": 4.0,
    "del_": 1e-3,
    "alpha": 2500.0,
    "k": 1.0,
    "k1": 0.1,
    "b": 0.06,
    "gammaA": 15.0,
    "gammaI": 24.0,
    "gammaH": 0.01,
    "f": 0.3,
    "g": 0.01,
    "d": 0.7,
    "d0": 0.88,
    "D": 2.5,
    "mu": 0.6,
    # delay is expressed in history-buffer “steps” (see solve_dde_rk4)
    "delay": 10.0,
}


def get_dde_params(**overrides: Any) -> dict[str, Any]:
    """Return a *full* DDE parameter dict with defaults overridden by kwargs.

    Example
    -------
    params = get_dde_params(delay=12, D=3.0)

    Notes
    -----
    - Adds/updates the derived parameter `decay`.
    - Does not include the time-varying `input_vec`.
    """
    params: dict[str, Any] = dict(DEFAULT_DDE_PARAMS)
    if overrides:
        params.update(overrides)
    params["decay"] = compute_decay_term(float(params["d"]), float(params["d0"]))
    return params


def log_dde_params(params: Mapping[str, Any], level: int = logging.DEBUG) -> None:
    """Log a resolved DDE parameter dictionary.

    This is intentionally *not* called automatically by :func:`get_dde_params`.
    The caller (typically :class:`reservoir.reservoir.BioReservoir`) decides when
    to emit configuration.
    """
    if not logger.isEnabledFor(level):
        return

    logger.log(level, "----- DDE Parameters -----")
    for k, v in params.items():
        logger.log(level, "%s: %s", k, v)
    logger.log(level, "--------------------------")


def compute_decay_term(d: float, d0: float) -> float:
    """Compute the decay term used in the gene promoter equations.

    This term only depends on parameters and does not change during integration.
    """
    return 1.0 - (d / d0) ** 4


def _freeze_param_key(params: Mapping[str, Any]) -> tuple[float, ...]:
    """Build a hashable cache key from the constant parameters."""
    # Keep order stable and explicit.
    return (
        float(params["CA"]),
        float(params["CI"]),
        float(params["del_"]),
        float(params["alpha"]),
        float(params["k1"]),
        float(params["f"]),
        float(params["gammaA"]),
        float(params["gammaI"]),
        float(params["b"]),
        float(params["k"]),
        float(params["gammaH"]),
        float(params["g"]),
        float(params["D"]),
        float(params["d"]),
        float(params["mu"]),
        float(params["d0"]),
        float(params["delay"]),
    )


_PREPARED_CACHE: dict[tuple[float, ...], PreparedDDEParams] = {}


def clear_prepared_cache() -> None:
    """Clear internal prepared-parameter cache (useful in experiments/tests)."""
    if _PREPARED_CACHE:
        logger.info("Clearing PreparedDDEParams cache (size=%s)", len(_PREPARED_CACHE))
    _PREPARED_CACHE.clear()


@dataclass(frozen=True)
class PreparedDDEParams:
    """A numba-friendly container of constant DDE scalars.

    `input_vec` is intentionally excluded because it’s time-varying at the
    reservoir level.
    """

    CA: float
    CI: float
    del_: float
    alpha: float
    k1: float
    f: float
    gammaA: float
    gammaI: float
    b: float
    k: float
    gammaH: float
    g: float
    D: float
    d: float
    mu: float
    delay: float
    decay: float


def _prepare_dde_params_cached(params: Mapping[str, Any]) -> PreparedDDEParams:
    """Prepare/cast DDE params and cache the result (internal)."""
    key = _freeze_param_key(params)
    cached = _PREPARED_CACHE.get(key)
    if cached is not None:
        return cached

    d = float(params["d"])
    d0 = float(params["d0"])
    decay = float(params.get("decay", compute_decay_term(d, d0)))

    prepared = PreparedDDEParams(
        CA=float(params["CA"]),
        CI=float(params["CI"]),
        del_=float(params["del_"]),
        alpha=float(params["alpha"]),
        k1=float(params["k1"]),
        f=float(params["f"]),
        gammaA=float(params["gammaA"]),
        gammaI=float(params["gammaI"]),
        b=float(params["b"]),
        k=float(params["k"]),
        gammaH=float(params["gammaH"]),
        g=float(params["g"]),
        D=float(params["D"]),
        d=d,
        mu=float(params["mu"]),
        delay=float(params["delay"]),
        decay=decay,
    )
    _PREPARED_CACHE[key] = prepared
    return prepared


@njit(fastmath=True, inline='always')
def _compute_derivatives(A, I, Hi, He, delayed_Hi, input_val,
                        CA, CI, del_, alpha, k1, f, gammaA, gammaI,
                        b, k, gammaH, g, D, d, mu, decay):
    """Core DDE equations - works with both scalars and arrays."""
    dH_sq = delayed_Hi * delayed_Hi
    gene_promoter = (del_ + alpha * dH_sq) / (1.0 + k1 * dH_sq)
    denom = 1.0 + f * (A + I)
    total_He = He + input_val

    dA = CA * decay * gene_promoter - gammaA * A / denom
    dI = CI * decay * gene_promoter - gammaI * I / denom
    dHi = b * I / (1.0 + k * I) - gammaH * A * Hi / (1.0 + g * A) + D * (total_He - Hi)
    dHe = -d / (1.0 - d) * D * (He - Hi) - mu * He
    
    return dA, dI, dHi, dHe

@njit(fastmath=True)
def _rk4_single_node(A, I, Hi, He, tau_local, Hi_low, Hi_high, input_val,
                     CA, CI, del_, alpha, k1, f, gammaA, gammaI,
                     b, k, gammaH, g, D, d, mu, decay):
    """Single-node RHS - used in parallel loop.
    
    Note: Hi_low and Hi_high are now the same (pre-interpolated delayed value),
    but we keep both parameters for signature compatibility.
    """
    delayed_Hi = Hi_low  # Use pre-interpolated value (Hi_low == Hi_high)
    return _compute_derivatives(A, I, Hi, He, delayed_Hi, input_val,
                               CA, CI, del_, alpha, k1, f, gammaA, gammaI,
                               b, k, gammaH, g, D, d, mu, decay)


@njit(fastmath=True)
def _rk4_vectorized(state, tau_local, Hi_low, Hi_high, input_vec,
                    CA, CI, del_, alpha, k1, f, gammaA, gammaI,
                    b, k, gammaH, g, D, d, mu, decay):
    """Vectorized RHS - used in sequential solver.
    
    Note: Hi_low and Hi_high are now the same (pre-interpolated delayed value),
    but we keep both parameters for signature compatibility.
    """
    A, I, Hi, He = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    delayed_Hi = Hi_low  # Use pre-interpolated value (Hi_low == Hi_high)
    
    dA, dI, dHi, dHe = _compute_derivatives(A, I, Hi, He, delayed_Hi, input_vec,
                                           CA, CI, del_, alpha, k1, f, gammaA, gammaI,
                                           b, k, gammaH, g, D, d, mu, decay)
    
    deriv = np.empty_like(state)
    deriv[:, 0], deriv[:, 1], deriv[:, 2], deriv[:, 3] = dA, dI, dHi, dHe
    return deriv


@njit(fastmath=True, inline='always')
def _rk4_step_single_node(A, I, Hi, He, dt, tau, Hi_low, Hi_high, input_val,
                         CA, CI, del_, alpha, k1, f, gammaA, gammaI,
                         b, k, gammaH, g, D, d, mu, decay):
    """Single RK4 step for one node."""
    # Stage 1
    dA1, dI1, dHi1, dHe1 = _rk4_single_node(
        A, I, Hi, He, tau, Hi_low, Hi_high, input_val,
        CA, CI, del_, alpha, k1, f, gammaA, gammaI,
        b, k, gammaH, g, D, d, mu, decay)
    
    # Stage 2
    dA2, dI2, dHi2, dHe2 = _rk4_single_node(
        A + 0.5*dt*dA1, I + 0.5*dt*dI1, Hi + 0.5*dt*dHi1, He + 0.5*dt*dHe1,
        tau + 0.5*dt, Hi_low, Hi_high, input_val,
        CA, CI, del_, alpha, k1, f, gammaA, gammaI,
        b, k, gammaH, g, D, d, mu, decay)
    
    # Stage 3
    dA3, dI3, dHi3, dHe3 = _rk4_single_node(
        A + 0.5*dt*dA2, I + 0.5*dt*dI2, Hi + 0.5*dt*dHi2, He + 0.5*dt*dHe2,
        tau + 0.5*dt, Hi_low, Hi_high, input_val,
        CA, CI, del_, alpha, k1, f, gammaA, gammaI,
        b, k, gammaH, g, D, d, mu, decay)
    
    # Stage 4
    dA4, dI4, dHi4, dHe4 = _rk4_single_node(
        A + dt*dA3, I + dt*dI3, Hi + dt*dHi3, He + dt*dHe3,
        tau + dt, Hi_low, Hi_high, input_val,
        CA, CI, del_, alpha, k1, f, gammaA, gammaI,
        b, k, gammaH, g, D, d, mu, decay)
    
    # Combine
    A_new = A + (dt/6.0) * (dA1 + 2.0*dA2 + 2.0*dA3 + dA4)
    I_new = I + (dt/6.0) * (dI1 + 2.0*dI2 + 2.0*dI3 + dI4)
    Hi_new = Hi + (dt/6.0) * (dHi1 + 2.0*dHi2 + 2.0*dHi3 + dHi4)
    He_new = He + (dt/6.0) * (dHe1 + 2.0*dHe2 + 2.0*dHe3 + dHe4)
    
    return A_new, I_new, Hi_new, He_new


@njit(fastmath=True)
def _solve_sequential(Y_init, Hi_low, Hi_high, times, input_vec,
                      CA, CI, del_, alpha, k1, f, gammaA, gammaI,
                      b, k, gammaH, g, D, d, mu, decay):
    """Sequential RK4 - processes all nodes together (vectorized)."""
    n_nodes, n_vars = Y_init.shape
    n_times = len(times)
    
    Y = Y_init.copy()
    out = np.empty((n_times, n_nodes, n_vars), dtype=Y.dtype)
    out[0] = Y
    dt = times[1] - times[0]

    for step_idx in range(n_times - 1):
        tau = times[step_idx]
        
        # RK4 stages - vectorized over all nodes
        k1_stage = _rk4_vectorized(Y, tau, Hi_low, Hi_high, input_vec,
                                   CA, CI, del_, alpha, k1, f, gammaA, gammaI,
                                   b, k, gammaH, g, D, d, mu, decay)
        
        k2_stage = _rk4_vectorized(Y + 0.5*dt*k1_stage, tau + 0.5*dt,
                                   Hi_low, Hi_high, input_vec,
                                   CA, CI, del_, alpha, k1, f, gammaA, gammaI,
                                   b, k, gammaH, g, D, d, mu, decay)
        
        k3_stage = _rk4_vectorized(Y + 0.5*dt*k2_stage, tau + 0.5*dt,
                                   Hi_low, Hi_high, input_vec,
                                   CA, CI, del_, alpha, k1, f, gammaA, gammaI,
                                   b, k, gammaH, g, D, d, mu, decay)
        
        k4_stage = _rk4_vectorized(Y + dt*k3_stage, tau + dt,
                                   Hi_low, Hi_high, input_vec,
                                   CA, CI, del_, alpha, k1, f, gammaA, gammaI,
                                   b, k, gammaH, g, D, d, mu, decay)
        
        Y = Y + (dt/6.0) * (k1_stage + 2.0*k2_stage + 2.0*k3_stage + k4_stage)
        out[step_idx + 1] = Y
    
    return out


@njit(parallel=True, fastmath=True)
def _solve_parallel(Y_init, Hi_low, Hi_high, times, input_vec,
                    CA, CI, del_, alpha, k1, f, gammaA, gammaI,
                    b, k, gammaH, g, D, d, mu, decay):
    """Parallel RK4 - each node processed independently with scalar math."""
    n_nodes, n_vars = Y_init.shape
    n_times = len(times)
    dt = times[1] - times[0]
    
    out = np.empty((n_times, n_nodes, n_vars), dtype=Y_init.dtype)
    out[0] = Y_init
    
    # Parallel loop over nodes
    for node_idx in prange(n_nodes):
        A, I, Hi, He = Y_init[node_idx, 0], Y_init[node_idx, 1], Y_init[node_idx, 2], Y_init[node_idx, 3]
        Hi_low_node, Hi_high_node, input_node = Hi_low[node_idx], Hi_high[node_idx], input_vec[node_idx]
        
        # Time-stepping loop
        for step_idx in range(n_times - 1):
            A, I, Hi, He = _rk4_step_single_node(
                A, I, Hi, He, dt, times[step_idx], Hi_low_node, Hi_high_node, input_node,
                CA, CI, del_, alpha, k1, f, gammaA, gammaI,
                b, k, gammaH, g, D, d, mu, decay)
            
            out[step_idx + 1, node_idx, 0] = A
            out[step_idx + 1, node_idx, 1] = I
            out[step_idx + 1, node_idx, 2] = Hi
            out[step_idx + 1, node_idx, 3] = He
    
    return out


# ---------------------------------------------------------------------------
# Per-node kinetic heterogeneity
# ---------------------------------------------------------------------------

# Column indices for the node_params array (N, 16).
_NP_CA, _NP_CI, _NP_DEL, _NP_ALPHA, _NP_K1 = 0, 1, 2, 3, 4
_NP_F, _NP_GAMMA_A, _NP_GAMMA_I, _NP_B, _NP_K = 5, 6, 7, 8, 9
_NP_GAMMA_H, _NP_G, _NP_D_COEFF, _NP_D_FRAC, _NP_MU, _NP_DECAY = 10, 11, 12, 13, 14, 15
N_NODE_PARAMS = 16

# Parameters representing intracellular kinetics that vary between cells
# (gene dosage, enzyme activity, degradation rates, membrane permeability).
_VARY_SET = {
    'CA', 'CI', 'del_', 'alpha', 'k1', 'f',
    'gammaA', 'gammaI', 'b', 'k', 'gammaH', 'g', 'D',
}

# Ordered parameter names matching node_params column indices.
_PARAM_ORDER = [
    'CA', 'CI', 'del_', 'alpha', 'k1', 'f', 'gammaA', 'gammaI',
    'b', 'k', 'gammaH', 'g', 'D', 'd', 'mu', 'decay',
]


def generate_node_params(
    n_nodes: int,
    cv: float,
    base_params: Mapping[str, Any],
    seed: int | None = None,
    mean_preserving: bool = True,
) -> np.ndarray:
    """Generate per-node kinetic parameter arrays with log-normal heterogeneity.

    Simulates cell-to-cell variability in gene expression and enzyme activity.
    Each node draws kinetic parameters from log-normal distributions centered
    on the nominal Danino et al. (2010) values.

    Parameters that are physical properties of the colony (cell density
    fraction ``d``, derived ``decay``) or the environment (extracellular
    decay ``mu``) are NOT varied—only intracellular kinetic rates are
    heterogeneous.

    Parameters
    ----------
    n_nodes : int
        Number of reservoir nodes.
    cv : float
        Coefficient of variation for the log-normal distribution.
        0.0 = homogeneous (all nodes identical).
        0.1–0.3 = realistic gene expression noise.
        0.5 = high heterogeneity.
    base_params : dict
        Nominal DDE parameters (from get_dde_params()).
    seed : int, optional
        Random seed for reproducibility.
    mean_preserving : bool, default=True
        Draw the factors as ``exp(N(-s^2/2, s^2))``, whose mean is 1, so the nominal
        values are the population means. False draws ``exp(N(0, s^2))`` (v0.1), whose
        mean is ``exp(s^2/2) = sqrt(1 + cv^2)``: every varied rate is biased upwards.

    Returns
    -------
    node_params : np.ndarray, shape (n_nodes, 16)
        Per-node kinetic parameters.
    """
    prepared = _prepare_dde_params_cached(base_params)
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(np.log(1.0 + cv ** 2)) if cv > 0 else 0.0
    log_mean = -0.5 * sigma ** 2 if mean_preserving else 0.0

    node_params = np.empty((n_nodes, N_NODE_PARAMS), dtype=np.float64)
    for i, name in enumerate(_PARAM_ORDER):
        nominal = getattr(prepared, name)
        if name in _VARY_SET and cv > 0:
            factors = rng.lognormal(mean=log_mean, sigma=sigma, size=n_nodes)
            node_params[:, i] = nominal * factors
        else:
            node_params[:, i] = nominal
    return node_params


@njit(parallel=True, fastmath=True)
def _solve_parallel_heterogeneous(Y_init, Hi_low, Hi_high, times, input_vec,
                                  node_params):
    """Parallel RK4 with per-node kinetic parameters.

    Identical to ``_solve_parallel`` except each node indexes its own row of
    ``node_params`` (shape ``(n_nodes, 16)``).
    """
    n_nodes = Y_init.shape[0]
    n_times = len(times)
    dt = times[1] - times[0]

    out = np.empty((n_times, n_nodes, 4), dtype=Y_init.dtype)
    out[0] = Y_init

    for node_idx in prange(n_nodes):
        A  = Y_init[node_idx, 0]
        I  = Y_init[node_idx, 1]
        Hi = Y_init[node_idx, 2]
        He = Y_init[node_idx, 3]
        Hi_low_node  = Hi_low[node_idx]
        Hi_high_node = Hi_high[node_idx]
        input_node   = input_vec[node_idx]

        # Per-node kinetic parameters (scalar per node).
        CA_n     = node_params[node_idx, 0]
        CI_n     = node_params[node_idx, 1]
        del_n    = node_params[node_idx, 2]
        alpha_n  = node_params[node_idx, 3]
        k1_n     = node_params[node_idx, 4]
        f_n      = node_params[node_idx, 5]
        gammaA_n = node_params[node_idx, 6]
        gammaI_n = node_params[node_idx, 7]
        b_n      = node_params[node_idx, 8]
        k_n      = node_params[node_idx, 9]
        gammaH_n = node_params[node_idx, 10]
        g_n      = node_params[node_idx, 11]
        D_n      = node_params[node_idx, 12]
        d_n      = node_params[node_idx, 13]
        mu_n     = node_params[node_idx, 14]
        decay_n  = node_params[node_idx, 15]

        for step_idx in range(n_times - 1):
            A, I, Hi, He = _rk4_step_single_node(
                A, I, Hi, He, dt, times[step_idx],
                Hi_low_node, Hi_high_node, input_node,
                CA_n, CI_n, del_n, alpha_n, k1_n, f_n,
                gammaA_n, gammaI_n, b_n, k_n, gammaH_n, g_n,
                D_n, d_n, mu_n, decay_n)

            out[step_idx + 1, node_idx, 0] = A
            out[step_idx + 1, node_idx, 1] = I
            out[step_idx + 1, node_idx, 2] = Hi
            out[step_idx + 1, node_idx, 3] = He

    return out


def solve_dde_rk4(
    history_states: np.ndarray,
    times: np.ndarray,
    input_vec: np.ndarray,
    parallel: bool = False,
    node_params: np.ndarray | None = None,
    **dde_overrides: Any,
) -> np.ndarray:
    """RK4 integrator for delay differential equations.
    
    Parameters
    ----------
    history_states : np.ndarray, shape (history_size, n_nodes, 4)
        History buffer of reservoir states
    times : np.ndarray
        Time points to evaluate
    input_vec : np.ndarray
        Per-node input vector (time-varying forcing for this step).
    parallel : bool, default=False
        Use parallel execution (21x faster but breaks with multiprocessing)
    node_params : np.ndarray, shape (n_nodes, 16), optional
        Per-node kinetic parameter array from ``generate_node_params``.
        When provided, the heterogeneous solver is used.
    **dde_overrides
        Keyword overrides for the DDE model parameters.
    
    Returns
    -------
    np.ndarray, shape (len(times), n_nodes, 4)
        Trajectory over time
    """
    Y_init = history_states[-1]

    # Resolve constant parameters via kwargs and use internal caching.
    params = get_dde_params(**dde_overrides)
    prepared = _prepare_dde_params_cached(params)

    input_vec = np.asarray(input_vec)

    delay = prepared.delay
    history_len = len(history_states)
    
    # Calculate proper delay index for interpolation
    # history_states[-1] is current state (t=0), history_states[0] is oldest (t=-history_len+1)
    # We want state at time t=-delay
    delay_index_float = history_len - 1 - delay
    delay_index = int(np.floor(delay_index_float))
    delay_frac = delay_index_float - delay_index  # Fractional part for interpolation
    
    # Clamp to valid range and interpolate
    if delay_index < 0:
        # Delay exceeds history buffer - use oldest available state
        delayed_Hi = history_states[0, :, 2]
    elif delay_index >= history_len - 1:
        # Delay is less than 1 timestep - use current state
        delayed_Hi = history_states[-1, :, 2]
    else:
        # Interpolate between the two states around the delay point
        Hi_low = history_states[delay_index, :, 2]
        Hi_high = history_states[delay_index + 1, :, 2]
        delayed_Hi = (1.0 - delay_frac) * Hi_low + delay_frac * Hi_high

    # Heterogeneous per-node parameters — use specialised solver.
    if node_params is not None:
        het_args = (Y_init, delayed_Hi, delayed_Hi, times, input_vec, node_params)
        if parallel:
            return _solve_parallel_heterogeneous(*het_args)
        # Sequential fallback: extract (N,) column vectors and pass to the
        # existing vectorized solver which numba will re-specialize for arrays.
        return _solve_sequential(
            Y_init, delayed_Hi, delayed_Hi, times, input_vec,
            node_params[:, _NP_CA],      node_params[:, _NP_CI],
            node_params[:, _NP_DEL],     node_params[:, _NP_ALPHA],
            node_params[:, _NP_K1],      node_params[:, _NP_F],
            node_params[:, _NP_GAMMA_A], node_params[:, _NP_GAMMA_I],
            node_params[:, _NP_B],       node_params[:, _NP_K],
            node_params[:, _NP_GAMMA_H], node_params[:, _NP_G],
            node_params[:, _NP_D_COEFF], node_params[:, _NP_D_FRAC],
            node_params[:, _NP_MU],      node_params[:, _NP_DECAY],
        )
    
    # Homogeneous (original scalar path).
    jit_args = (
        Y_init,
        delayed_Hi,
        delayed_Hi,
        times,
        input_vec,
        prepared.CA,
        prepared.CI,
        prepared.del_,
        prepared.alpha,
        prepared.k1,
        prepared.f,
        prepared.gammaA,
        prepared.gammaI,
        prepared.b,
        prepared.k,
        prepared.gammaH,
        prepared.g,
        prepared.D,
        prepared.d,
        prepared.mu,
        prepared.decay,
    )
    
    if parallel:
        return _solve_parallel(*jit_args)
    return _solve_sequential(*jit_args)


# ---------------------------------------------------------------------------
# Delay resolved at every substep (the default from v0.2)
# ---------------------------------------------------------------------------
#
# ``solve_dde_rk4`` reads the delayed Hi once per input step and holds it for all the
# substeps (the v0.1 scheme, ``delay_scheme="frozen"``). The solver below keeps Hi at
# every substep in a ring buffer, so each RK4 stage reads Hi(t + c*h - delay): the stored
# value when the delay is a whole number of substeps, otherwise the linear interpolation
# between the two neighbouring substeps (second order overall). A delay of 0 is the
# ordinary differential equation: each stage uses its own Hi.


def substep_delay_lags(delay: float, substeps: int) -> tuple[float, float, float] | None:
    """Delay, in substeps, seen by the RK4 stages at the start, middle and end of a substep.

    Returns ``None`` for a zero delay (the ordinary differential equation).
    """
    lag = float(delay) * int(substeps)
    if lag == 0.0:
        return None
    if lag < 1.0 - 1e-9:
        raise ValueError(
            f"A delay of {delay} input steps is shorter than one substep (1/{substeps} "
            "of a step); use 0 or a delay of at least one substep."
        )
    lag = max(lag, 1.0)  # a delay of one substep up to rounding
    return lag, lag - 0.5, lag - 1.0


def substep_history(
    history_states: np.ndarray, delay: float, substeps: int
) -> tuple[np.ndarray, int]:
    """Hi at substep resolution for :func:`solve_dde_substep`.

    The stored history (one entry per input step, the last at t = 0) is interpolated
    linearly to the substep times. Returns a ring buffer of shape ``(n_nodes, L)`` and the
    index of the current value in it.
    """
    lags = substep_delay_lags(delay, substeps)
    hi = np.asarray(history_states, dtype=np.float64)[:, :, 2]
    n_hist = hi.shape[0]
    length = 1 if lags is None else int(np.ceil(lags[0])) + 1
    # Positions of the substep times in history-index units (the last entry is t = 0).
    t = (np.arange(length) - (length - 1)) / int(substeps) + (n_hist - 1)
    if t[0] < -1e-9:
        raise ValueError(f"The history ({n_hist} steps) is shorter than the delay {delay}.")
    if n_hist == 1:
        ring = np.repeat(hi[0][:, None], length, axis=1)
    else:
        t = np.clip(t, 0.0, n_hist - 1)
        low = np.minimum(np.floor(t).astype(np.int64), n_hist - 2)
        w = (t - low)[:, None]
        ring = ((1.0 - w) * hi[low] + w * hi[low + 1]).T
    return np.ascontiguousarray(ring), length - 1


@njit(fastmath=True, inline='always')
def _ring_value(ring, node, pos, length, back, frac):
    """Hi ``back + frac`` substeps before the entry at ``pos`` (linear in between)."""
    i = pos - back
    if i < 0:
        i += length
    value = ring[node, i]
    if frac > 0.0:
        i -= 1
        if i < 0:
            i += length
        value = (1.0 - frac) * value + frac * ring[node, i]
    return value


def _substep_kernel(state, ring, pos, instantaneous, back0, frac0, back_mid, frac_mid,
                    back1, frac1, substeps, input_vec, node_params):
    """RK4 over one input step with the delayed Hi read at each stage.

    ``state`` (n_nodes, 4) and ``ring`` (n_nodes, L) are updated in place.
    """
    n_nodes = state.shape[0]
    length = ring.shape[1]
    dt = 1.0 / substeps
    for node in prange(n_nodes):
        A = state[node, 0]
        I = state[node, 1]  # noqa: E741 (LuxI, as in the model equations)
        Hi = state[node, 2]
        He = state[node, 3]
        u = input_vec[node]
        CA = node_params[node, 0]
        CI = node_params[node, 1]
        del_ = node_params[node, 2]
        alpha = node_params[node, 3]
        k1 = node_params[node, 4]
        f = node_params[node, 5]
        gammaA = node_params[node, 6]
        gammaI = node_params[node, 7]
        b = node_params[node, 8]
        k = node_params[node, 9]
        gammaH = node_params[node, 10]
        g = node_params[node, 11]
        D = node_params[node, 12]
        d = node_params[node, 13]
        mu = node_params[node, 14]
        decay = node_params[node, 15]
        p = pos
        for _ in range(substeps):
            if instantaneous:
                lag0 = Hi
                lag_mid = 0.0
                lag1 = 0.0
            else:
                lag0 = _ring_value(ring, node, p, length, back0, frac0)
                lag_mid = _ring_value(ring, node, p, length, back_mid, frac_mid)
                lag1 = _ring_value(ring, node, p, length, back1, frac1)

            dA1, dI1, dHi1, dHe1 = _compute_derivatives(
                A, I, Hi, He, lag0, u,
                CA, CI, del_, alpha, k1, f, gammaA, gammaI, b, k, gammaH, g, D, d, mu, decay)

            A2 = A + 0.5 * dt * dA1
            I2 = I + 0.5 * dt * dI1
            Hi2 = Hi + 0.5 * dt * dHi1
            He2 = He + 0.5 * dt * dHe1
            dA2, dI2, dHi2, dHe2 = _compute_derivatives(
                A2, I2, Hi2, He2, Hi2 if instantaneous else lag_mid, u,
                CA, CI, del_, alpha, k1, f, gammaA, gammaI, b, k, gammaH, g, D, d, mu, decay)

            A3 = A + 0.5 * dt * dA2
            I3 = I + 0.5 * dt * dI2
            Hi3 = Hi + 0.5 * dt * dHi2
            He3 = He + 0.5 * dt * dHe2
            dA3, dI3, dHi3, dHe3 = _compute_derivatives(
                A3, I3, Hi3, He3, Hi3 if instantaneous else lag_mid, u,
                CA, CI, del_, alpha, k1, f, gammaA, gammaI, b, k, gammaH, g, D, d, mu, decay)

            A4 = A + dt * dA3
            I4 = I + dt * dI3
            Hi4 = Hi + dt * dHi3
            He4 = He + dt * dHe3
            dA4, dI4, dHi4, dHe4 = _compute_derivatives(
                A4, I4, Hi4, He4, Hi4 if instantaneous else lag1, u,
                CA, CI, del_, alpha, k1, f, gammaA, gammaI, b, k, gammaH, g, D, d, mu, decay)

            A = A + (dt / 6.0) * (dA1 + 2.0 * dA2 + 2.0 * dA3 + dA4)
            I = I + (dt / 6.0) * (dI1 + 2.0 * dI2 + 2.0 * dI3 + dI4)  # noqa: E741
            Hi = Hi + (dt / 6.0) * (dHi1 + 2.0 * dHi2 + 2.0 * dHi3 + dHi4)
            He = He + (dt / 6.0) * (dHe1 + 2.0 * dHe2 + 2.0 * dHe3 + dHe4)

            p += 1
            if p == length:
                p = 0
            ring[node, p] = Hi

        state[node, 0] = A
        state[node, 1] = I
        state[node, 2] = Hi
        state[node, 3] = He


_solve_substep_parallel = njit(parallel=True, fastmath=True)(_substep_kernel)
_solve_substep_sequential = njit(fastmath=True)(_substep_kernel)


def solve_dde_substep(
    state: np.ndarray,
    ring: np.ndarray,
    pos: int,
    input_vec: np.ndarray,
    substeps: int,
    parallel: bool = False,
    node_params: np.ndarray | None = None,
    **dde_overrides: Any,
) -> tuple[np.ndarray, int]:
    """Advance the oscillators by one input step with the delay resolved at every substep.

    Parameters
    ----------
    state : np.ndarray, shape (n_nodes, 4)
        Current A, I, Hi, He.
    ring, pos
        Hi history at substep resolution and the index of the current value, from
        :func:`substep_history` or the previous call. ``ring`` is updated in place.
    input_vec : np.ndarray, shape (n_nodes,)
        Drive for this step (added to He in the Hi equation).
    substeps : int
        RK4 steps per input step.
    node_params : np.ndarray, shape (n_nodes, 16), optional
        Per-node parameters (:func:`generate_node_params`); the nominal ones if omitted.

    Returns
    -------
    The state after the step and the new index of the current value in ``ring``.
    """
    params = get_dde_params(**dde_overrides)
    lags = substep_delay_lags(params["delay"], substeps)
    length = 1 if lags is None else int(np.ceil(lags[0])) + 1
    if ring.shape != (state.shape[0], length):
        raise ValueError(f"The delay history has shape {ring.shape}; the delay "
                         f"{params['delay']} needs {(state.shape[0], length)}.")
    if node_params is None:
        node_params = generate_node_params(state.shape[0], 0.0, params, seed=0)
    offsets = []
    for lag in lags or (0.0, 0.0, 0.0):
        back = int(np.floor(lag))
        offsets += [back, float(lag - back)]

    new_state = np.array(state, dtype=np.float64)
    solver = _solve_substep_parallel if parallel else _solve_substep_sequential
    solver(new_state, ring, int(pos), lags is None, *offsets, int(substeps),
           np.ascontiguousarray(input_vec, dtype=np.float64).reshape(-1),
           np.ascontiguousarray(node_params, dtype=np.float64))
    return new_state, (int(pos) + int(substeps)) % length
