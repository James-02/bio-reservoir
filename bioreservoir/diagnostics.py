"""Diagnostics of the drive and states of a BioReservoir over a set of input sequences."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from scipy import sparse

STATE_NAMES = ("A", "I", "Hi", "He")


def drive_and_state_ranges(reservoir, inputs: Iterable[np.ndarray],
                           bound: float = 5e-3) -> dict[str, Any]:
    """Run ``inputs`` through a copy of ``reservoir`` and summarise its drive and states.

    Each sequence starts from the warm-up state, as in training. The drive is ``x_t``, the
    input to each oscillator's Hi equation before ``cell_coupling`` (the quantity the
    model description bounds by ``bound``); the states are A, I, Hi and He after each
    input step. Fractions count oscillator-steps.
    """
    node = reservoir.copy()
    n_nodes = node.states.shape[1]
    step = node._step
    totals = {"count": 0, "above": 0, "abs_max": 0.0, "min": np.full(4, np.inf),
              "max": np.full(4, -np.inf), "negative": np.zeros(4, dtype=np.int64)}

    def recording_step(x):
        drive = np.asarray(x, dtype=float).reshape(-1)[:n_nodes]
        state = step(x)
        totals["count"] += n_nodes
        totals["above"] += int(np.count_nonzero(np.abs(drive) > bound))
        totals["abs_max"] = max(totals["abs_max"], float(np.max(np.abs(drive))))
        totals["min"] = np.minimum(totals["min"], state.min(axis=0))
        totals["max"] = np.maximum(totals["max"], state.max(axis=0))
        totals["negative"] += np.count_nonzero(state < 0, axis=0)
        return state

    node._step = recording_step
    for x in inputs:
        node.run(np.asarray(x), reset=True)
    count = max(totals["count"], 1)
    return {
        "bound": bound,
        "oscillator_steps": totals["count"],
        "drive_abs_max": totals["abs_max"],
        "drive_fraction_above": totals["above"] / count,
        "state_min": dict(zip(STATE_NAMES, totals["min"].tolist(), strict=True)),
        "state_max": dict(zip(STATE_NAMES, totals["max"].tolist(), strict=True)),
        "state_negative_fraction": dict(zip(STATE_NAMES,
                                            (totals["negative"] / count).tolist(),
                                            strict=True)),
    }


def drive_shares(reservoir, inputs: Iterable[np.ndarray]) -> dict[str, float]:
    """Share of each term in the drive that reaches the oscillators, over ``inputs``.

    The drive of oscillator i at step t is ``Win u_t + bias + rc_scaling * W r_{t-1}``
    (before noise and ``dde_scaling``); the recurrent term is split by observed variable.
    Each share is the summed absolute value of its term over oscillators and steps,
    divided by the sum over all terms. Keys: ``"input"`` and one per observed variable.
    Only the abstract coupling acts through the drive.
    """
    if getattr(reservoir, "coupling", "abstract") != "abstract":
        raise ValueError("Drive shares need coupling='abstract'; with diffusive coupling the "
                         "colonies interact through He, not through the drive.")
    node = reservoir.copy()
    n_nodes = node.states.shape[1]
    names = [STATE_NAMES[i] for i in node.output_indices]
    n_obs = len(names)
    sums = np.zeros(1 + n_obs)
    for x in inputs:
        x = np.asarray(x, dtype=float)
        outputs = node.run(x, reset=True)
        warm = node._warmup_states[-1][:, node.output_indices].reshape(1, -1)
        previous = np.vstack([warm, outputs[:-1]])
        W = node.W[:n_nodes]
        Win, bias = (_dense(m[:n_nodes]) for m in (node.Win, node.bias))
        inp = x @ Win.T + bias.T
        sums[0] += np.abs(inp).sum()
        for v in range(n_obs):
            rec = node.rc_scaling * (W[:, v::n_obs] @ previous[:, v::n_obs].T)
            sums[1 + v] += np.abs(np.asarray(rec)).sum()
    total = sums.sum() or 1.0
    return dict(zip(["input", *names], (sums / total).tolist(), strict=True))


def _dense(matrix) -> np.ndarray:
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def oscillation_amplitudes(reservoir, steps: int = 200, variable: str = "I") -> np.ndarray:
    """Relative peak-to-peak amplitude of each colony's ``variable`` without input.

    Continues the warm-up of a copy of ``reservoir`` (zero drive, from its warm-up state) for
    ``steps`` input steps and returns ``(max - min) / mean`` of ``variable`` over those steps,
    one value per colony. A colony that oscillates has an amplitude of order 1; one that
    has settled to a steady state has an amplitude near 0. NaN marks a colony whose mean is
    not positive (or a state that is not finite).
    """
    node = reservoir.copy()
    node.states = node._warmup_states.copy()
    if getattr(node, "_warmup_ring", None) is not None:
        node._ring, node._ring_pos = node._warmup_ring[0].copy(), node._warmup_ring[1]
    index = STATE_NAMES.index(variable)
    zero = np.zeros(node.states.shape[1])
    trace = np.empty((int(steps), node.states.shape[1]))
    with np.errstate(all="ignore"):
        for t in range(int(steps)):
            trace[t] = node._step(zero)[:, index]
        mean = trace.mean(axis=0)
        return np.where(mean > 0, np.ptp(trace, axis=0) / mean, np.nan)
