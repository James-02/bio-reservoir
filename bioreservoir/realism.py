"""Realism options: how real colonies, traps and resets depart from the nominal model.

Three options of :class:`bioreservoir.node.BioReservoir`, each off at 0 (or 1 level):

* **Heterogeneity by group.** Colonies differ in their kinetics (``param_heterogeneity_cv``,
  the 13 intracellular constants of :func:`bioreservoir.dde.generate_node_params`), in their
  cell density ``d`` (``heterogeneity_density_cv``) and in the washout of their trap
  (``heterogeneity_dilution_cv``, a factor on ``mu``). Density enters the model twice: the
  growth factor ``decay = 1 - (d/d0)^4`` of both promoters, and the cell/medium volume
  ratio ``d/(1-d)`` that converts the membrane flux ``D (He - Hi)`` into a change of He
  (Danino et al. 2010). Each group draws from its own random stream derived from the
  heterogeneity seed (:func:`stream`), so adding a group leaves the draws of the others
  unchanged.
* **Input levels.** The induced traps receive ``K`` inducer levels, evenly spaced in
  ``(0, 1]`` times ``input_scaling``, or a level uniform on ``(0, 1]``
  (``"continuous"``); see :func:`level_weights`.
* **Imperfect reset.** Each input sequence starts from the warm-up state multiplied, colony
  by colony and variable by variable, by log-normal factors of mean 1 and coefficient of
  variation ``reset_noise`` (:func:`perturb_start`).

All factors are mean-preserving log-normal: ``exp(N(-s^2/2, s^2))`` with
``s^2 = log(1 + cv^2)``, whose mean is 1 and coefficient of variation ``cv``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from bioreservoir.dde import _NP_D_FRAC, _NP_DECAY, _NP_MU

# Spawn keys of the random streams derived from a seed (numpy SeedSequence children). The
# kinetic heterogeneity, the input weights and the per-beat input noise use the seed itself.
DENSITY_STREAM = 1
DILUTION_STREAM = 2
INPUT_LEVEL_STREAM = 3
RESET_STREAM = 4

# Cell density is kept in [DENSITY_MIN, DENSITY_MAX_FRACTION * d0]: the growth factor
# 1 - (d/d0)^4 vanishes at the carrying capacity d0 and is negative above it.
DENSITY_MIN = 0.05
DENSITY_MAX_FRACTION = 0.99


def stream(seed: int | None, key: int) -> np.random.Generator:
    """The random stream ``key`` derived from ``seed``.

    The child of ``SeedSequence(seed)`` with spawn key ``(key,)``: independent of
    ``np.random.default_rng(seed)`` and of the other keys. ``seed=None`` draws fresh entropy.
    """
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(int(key),)))


def lognormal_factors(rng: np.random.Generator, cv: float, size) -> np.ndarray:
    """Mean-preserving log-normal factors with coefficient of variation ``cv`` (>= 0)."""
    cv = float(cv)
    if cv < 0:
        raise ValueError(f"A coefficient of variation must be 0 or more, got {cv}.")
    sigma = np.sqrt(np.log(1.0 + cv ** 2))
    return rng.lognormal(mean=-0.5 * sigma ** 2, sigma=sigma, size=size)


def vary_colonies(node_params: np.ndarray, d0: float, density_cv: float = 0.0,
                  dilution_cv: float = 0.0, seed: int | None = None) -> np.ndarray:
    """Per-colony cell density and washout: a copy of ``node_params`` with them varied.

    Parameters
    ----------
    node_params : np.ndarray, shape (n_colonies, 16)
        Per-colony parameters (:func:`bioreservoir.dde.generate_node_params`).
    d0 : float
        Carrying capacity (volume fraction) of the growth factor ``1 - (d/d0)^4``.
    density_cv : float
        Coefficient of variation of a factor on the density ``d`` (volume fraction of cells
        in the trap). The density is then clipped to ``[0.05, 0.99 d0]`` and the growth
        factor recomputed for each colony; clipping lowers the mean density at large CVs.
    dilution_cv : float
        Coefficient of variation of a factor on the washout rate ``mu`` (per time unit).
    seed : int, optional
        Heterogeneity seed; density and washout draw from its streams ``DENSITY_STREAM``
        and ``DILUTION_STREAM``.

    Returns
    -------
    np.ndarray, shape (n_colonies, 16)
    """
    for name, cv in (("density_cv", density_cv), ("dilution_cv", dilution_cv)):
        if float(cv) < 0:
            raise ValueError(f"{name} must be 0 or more, got {cv}.")
    table = np.array(node_params, dtype=np.float64, copy=True)
    n = table.shape[0]
    if float(density_cv) > 0:
        d = table[:, _NP_D_FRAC] * lognormal_factors(stream(seed, DENSITY_STREAM),
                                                     density_cv, n)
        d = np.clip(d, DENSITY_MIN, DENSITY_MAX_FRACTION * float(d0))
        table[:, _NP_D_FRAC] = d
        table[:, _NP_DECAY] = 1.0 - (d / float(d0)) ** 4
    if float(dilution_cv) > 0:
        table[:, _NP_MU] *= lognormal_factors(stream(seed, DILUTION_STREAM), dilution_cv, n)
    return table


def check_input_levels(levels: Any) -> int | str:
    """``levels`` as an int of 1 or more, or ``"continuous"``; refuse anything else."""
    if levels == "continuous":
        return "continuous"
    if isinstance(levels, (int, np.integer)) and not isinstance(levels, bool) and levels >= 1:
        return int(levels)
    raise ValueError(f"input_levels must be an integer of 1 or more or 'continuous', "
                     f"got {levels!r}.")


def level_factors(rng: np.random.Generator, levels: int | str, size: int) -> np.ndarray:
    """Input levels as fractions of the full level: ``k/K``, k uniform on 1..K, or uniform
    on (0, 1] for ``"continuous"``."""
    levels = check_input_levels(levels)
    if levels == "continuous":
        return 1.0 - rng.random(size)
    return rng.integers(1, levels + 1, size=size) / levels


def level_weights(Win, levels: int | str, seed: int | None):
    """Input weights with each non-zero weight scaled by a random level.

    Every connection of ``Win`` (one per induced trap with a one-dimensional input) keeps
    its sign and is multiplied by a level from :func:`level_factors`, drawn in row-major
    order from the reservoir seed's stream ``INPUT_LEVEL_STREAM``, so a dense and a sparse
    ``Win`` get the same levels. One level returns ``Win`` unchanged.
    """
    levels = check_input_levels(levels)
    if levels == 1:
        return Win
    rng = stream(seed, INPUT_LEVEL_STREAM)
    if sparse.issparse(Win):
        out = sparse.csr_matrix(Win, dtype=np.float64, copy=True)
        out.eliminate_zeros()
        out.sort_indices()
        out.data *= level_factors(rng, levels, out.nnz)
        return out
    out = np.array(Win, dtype=np.float64, copy=True)
    rows, cols = np.nonzero(out)
    out[rows, cols] *= level_factors(rng, levels, len(rows))
    return out


def perturb_start(states: np.ndarray, ring: np.ndarray | None, cv: float,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray | None]:
    """An imperfect reset: the starting state multiplied by log-normal factors.

    Each colony draws one factor per variable (A, I, Hi, He; mean 1, coefficient of
    variation ``cv``). A variable's factor scales its whole history, and Hi's factor also
    scales the delayed-Hi history, so the initial function of the delay equation stays
    continuous.

    Parameters
    ----------
    states : np.ndarray, shape (history, n_colonies, 4)
        State history at input-step resolution (the last row is the current state).
    ring : np.ndarray, shape (n_colonies, L), optional
        Hi history at substep resolution (:func:`bioreservoir.dde.substep_history`).
    cv : float
        Coefficient of variation of the factors.
    rng : np.random.Generator

    Returns
    -------
    The perturbed copies of ``states`` and ``ring``.
    """
    factors = lognormal_factors(rng, cv, (states.shape[1], 4))
    return states * factors[None], None if ring is None else ring * factors[:, 2:3]
