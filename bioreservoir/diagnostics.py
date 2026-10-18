"""Diagnostics of the drive and states of a BioReservoir over a set of input sequences."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

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
