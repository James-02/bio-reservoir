"""A run config's reservoir as a run builds and drives it, for the figures in scripts/figures.

:func:`run_config` reads a run config and :func:`task_beat` takes one beat of its task as a
run simulates it. :func:`build` builds the config's reservoir as a run does
(:func:`optimization.reservoirs.build_reservoir`). :func:`colony_states` runs one colony of
the design on its own, :func:`array_states` steps the whole array through a beat as
``node.run(beat, reset=True)`` does and keeps the terms of its drive, and
:func:`coupling` and :func:`raw_kernel` give where the colonies sit and how they are coupled.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import sparse

from bioreservoir import distance_matrix
from optimization.reservoirs import _build_bioreservoir_topology, build_reservoir
from training.run import load_config, load_task, resolve
from training.simulate import beat_seed, reseed_noise
from utils.preprocessing import class_indices, standardize_data
from utils.visualisation import class_names

#: The oscillator's variables, in the order of its state: AiiA, LuxI, and AHL in the cells
#: and in the medium.
VARIABLES = ("A", "I", "Hi", "He")
#: The terms of the drive that a node keeps with ``diagnostics`` on: the input term, the
#: recurrent term (the abstract coupling only) and their sum.
TERMS = ("inp_term", "rec_term", "drive")


def run_config(path: str | Path) -> dict:
    """The resolved run config at ``path`` (the ``study`` section of a study config is
    ignored); its reservoir must be a bio-reservoir."""
    cfg = load_config(path)
    cfg.pop("study", None)
    cfg = resolve(cfg)
    if cfg["reservoir"].get("type") != "bioreservoir":
        raise ValueError(f"{path} is not a bio-reservoir config.")
    return cfg


def task_beat(cfg: dict, index: int) -> tuple[np.ndarray, str]:
    """Beat ``index`` of the config's beat pool, scaled as a run scales it, as a
    (steps, 1) array, and the name of its class."""
    X, Y, _ = load_task(cfg["task"])
    X, _ = standardize_data(X, X[:1], scaler_type=str(cfg["task"]["scaler"]))
    names = class_names(2 if cfg["task"]["labels"] == "binary" else 5)
    return np.asarray(X[index], dtype=float).reshape(-1, 1), names[class_indices(Y)[index]]


def build(cfg: dict, **overrides):
    """The config's reservoir with ``overrides`` to its parameters, observing all four
    variables."""
    params = dict(cfg["reservoir"]["params"], reservoir_type="bioreservoir",
                  output_variables="all")
    params.update(overrides)
    node, _ = build_reservoir(params, seed=int(cfg["reservoir"]["seed"]))
    return node


def colony_states(cfg: dict, beat: np.ndarray, index: int, scalings: list[float]
                  ) -> tuple[np.ndarray, np.ndarray, dict[float, np.ndarray]]:
    """One colony of the design on its own: its state (steps, 4) during the warm-up, then
    during the beat without input, and during the beat at each input scaling.

    The colony has the design's kinetics without heterogeneity, no neighbours, and receives
    the beat with a positive weight, the input scaling; the beat is encoded, and its input
    noise drawn, as a run of beat ``index`` draws it. Each driven state starts from the
    state at the end of the warm-up, as in a run.
    """
    alone = {"units": 1, "coupling": "none", "topology_type": "lattice",
             "param_heterogeneity_cv": 0.0, "input_connectivity": 1.0,
             "input_sign": "positive"}
    warmup = int(cfg["reservoir"]["params"]["warmup"])
    free = build(cfg, **alone, warmup=0)
    rest = np.array([free._step(np.zeros(1))[0] for _ in range(warmup + len(beat))])
    driven = {}
    for scaling in scalings:
        node = build(cfg, **alone, input_scaling=scaling)
        reseed_noise(node, beat_seed(int(cfg["reservoir"]["seed"]), index))
        driven[scaling] = np.asarray(node.run(beat, reset=True)).reshape(len(beat), 4)
    return rest[:warmup], rest[warmup:], driven


def array_states(cfg: dict, beat: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray]:
    """The array's state (steps, colonies, 4) during beat ``index``, and the mean size over
    the colonies of each of :data:`TERMS` at each step (steps, 3): the array is stepped
    through the beat exactly as a run runs it."""
    node = build(cfg)
    node.initialize(beat[:1])
    reseed_noise(node, beat_seed(int(cfg["reservoir"]["seed"]), index))
    node.reset(to_state=node.zero_state())  # as run(beat, reset=True) does
    node.diagnostics = True  # the node keeps the terms of its last step
    states, terms = [], []
    for x in beat:
        states.append(np.asarray(node(x.reshape(1, 1))).reshape(-1, 4))
        terms.append([np.mean(np.abs(getattr(node, f"_last_{name}"))) for name in TERMS])
    return np.array(states), np.array(terms)


def coupling(cfg: dict) -> tuple[np.ndarray, sparse.coo_matrix | None]:
    """Where the config's colonies sit (colonies, 2), in units of the array's width, and
    their coupling weights: the conductances of the diffusive coupling, the kernel of the
    abstract one (which couples each pair of colonies through all four variables), or None
    without coupling."""
    params = cfg["reservoir"]["params"]
    seed, units = int(cfg["reservoir"]["seed"]), int(params["units"])
    topology = params.get("topology_type", "distance")
    kernel = None
    if topology == "lattice":  # the grid of bioreservoir.topology.lattice_matrix
        columns = max(int(round(np.sqrt(units))), 1)
        while units % columns:
            columns -= 1
        positions = np.column_stack([np.arange(units) % columns,
                                     np.arange(units) // columns]) / max(columns - 1, 1)
    elif topology == "distance":  # the node builds its kernel with this call and seed
        kernel, positions = _build_bioreservoir_topology(params, seed)(units, units,
                                                                     return_coords=True)
    else:
        raise ValueError(f"A {topology} topology places no colonies in space.")
    weights = {"diffusion": lambda: build(cfg, warmup=0).conductance,
               "abstract": lambda: kernel}.get(params.get("coupling", "abstract"), lambda: None)()
    return (np.asarray(positions, dtype=float),
            None if weights is None else sparse.coo_matrix(weights))


def raw_kernel(cfg: dict) -> tuple[np.ndarray, np.ndarray, float, float]:
    """The distance kernel of the config before its signs, sparsity and normalisation: the
    distance and the weight of every pair of colonies (both flat), the mean distance to the
    nearest colony, and the interaction distance, ``interaction_diameters`` times that mean,
    at which the weight has fallen to ``fade_alpha`` and beyond which it is 0."""
    params = cfg["reservoir"]["params"]
    if params.get("topology_type", "distance") != "distance":
        raise ValueError("Only a distance topology has a kernel.")
    units, diameters = int(params["units"]), float(params["interaction_diameters"])
    weights, positions = distance_matrix(
        units, units, interaction_diameters=diameters, fade_alpha=float(params["fade_alpha"]),
        normalize="none", signed=False, sparsity=1.0, seed=int(cfg["reservoir"]["seed"]),
        return_coords=True)
    weights = weights.toarray() if sparse.issparse(weights) else np.asarray(weights)
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    others = ~np.eye(units, dtype=bool)
    nearest = float(np.where(others, distances, np.inf).min(axis=1).mean())
    return distances[others], weights[others], nearest, diameters * nearest


def strongest(weights: sparse.coo_matrix, links: int) -> np.ndarray:
    """The positions in ``weights.data`` of the ``links`` largest weights (by size) of each
    colony, other than its own."""
    order = np.lexsort((-np.abs(weights.data), weights.row))
    order = order[weights.row[order] != weights.col[order]]
    rows = weights.row[order]
    rank = np.arange(len(order)) - np.searchsorted(rows, rows)
    return order[rank < links]
