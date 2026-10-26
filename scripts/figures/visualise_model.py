"""Draw a run config's model: one colony during a beat, the whole array during a beat, and
its coupling.

The model is built as a run builds it (:func:`optimization.reservoirs.build_reservoir`), and
the beat is one of the beats a run of the config reads (``--beat``, an index into the task's
pool), scaled and with its input noise drawn as a run draws them. ``--plots`` chooses among:

colony
    One colony with the config's kinetics, input encoding, input noise and warm-up, but no
    neighbours and no heterogeneity: LuxI and AiiA without the beat (the warm-up continued)
    and during the beat at multiples of the config's input gain (``--gains``).
array
    The beat through the whole array: the beat; the mean size over the colonies of the input
    term and, with the abstract coupling, of the recurrent term and their sum, the drive; and
    ``--var`` of every colony (default LuxI, the reporter that the readout images).
coupling
    Where the colonies sit, with the strongest links of each (``--links``; negative weights
    dashed), and every coupling weight against the distance between its two colonies.

    python scripts/figures/visualise_model.py experiments/v0.2/designs/centre.yaml --beat 0

It writes ``results/figures/model/<name>-colony-beat<N>``, ``<name>-array-beat<N>-<var>``
and ``<name>-coupling``, each as PNG and PDF.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from scipy import sparse

from optimization.reservoirs import _build_bioreservoir_topology, build_reservoir
from training.run import load_config, load_task, resolve
from training.simulate import beat_seed, reseed_noise
from utils.preprocessing import class_indices, standardize_data
from utils.visualisation import (
    DOUBLE_COLUMN,
    OKABE_ITO,
    SINGLE_COLUMN,
    STATE_NAMES,
    STYLE,
    class_names,
    save_figure,
)

VARIABLES = ("A", "I", "Hi", "He")  # the order of the oscillator's state
PLOTS = ("colony", "array", "coupling")
TERMS = (("inp_term", "Input term", "-"), ("rec_term", "Recurrent term", "--"),
         ("drive", "Drive", ":"))


def run_config(path: str | Path) -> dict:
    """The resolved run config at ``path``; its model must be a bio-reservoir."""
    cfg = load_config(path)
    cfg.pop("study", None)
    cfg = resolve(cfg)
    if cfg["reservoir"].get("type") != "bioreservoir":
        raise SystemExit(f"{path} is not a bio-reservoir config.")
    return cfg


def task_beat(cfg: dict, index: int) -> tuple[np.ndarray, str]:
    """Beat ``index`` of the config's task as a run simulates it (steps, 1), and its class."""
    X, Y, _ = load_task(cfg["task"])
    X, _ = standardize_data(X, X[:1], scaler_type=str(cfg["task"]["scaler"]))
    names = class_names(2 if cfg["task"]["labels"] == "binary" else 5)
    return np.asarray(X[index], dtype=float).reshape(-1, 1), names[class_indices(Y)[index]]


def build(cfg: dict, **overrides):
    """The config's reservoir with ``overrides``, observing all four variables."""
    params = dict(cfg["reservoir"]["params"], reservoir_type="bioreservoir",
                  output_variables="all")
    params.update(overrides)
    node, _ = build_reservoir(params, seed=int(cfg["reservoir"]["seed"]))
    return node


# --- colony -----------------------------------------------------------------------------

def colony_states(cfg: dict, beat: np.ndarray, index: int,
                  gains: list[float]) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    """One colony's state (steps, 4) without the beat, and with it at each multiple of the
    config's input gain. The colony is alone, with nominal kinetics, and receives the input."""
    alone = {"units": 1, "coupling": "none", "topology_type": "lattice",
             "param_heterogeneity_cv": 0.0, "input_connectivity": 1.0}
    rest = build(cfg, **alone)
    no_beat = np.array([rest._step(np.zeros(1))[0] for _ in range(len(beat))])
    gain = float(cfg["reservoir"]["params"]["input_scaling"])
    driven = {}
    for factor in gains:
        node = build(cfg, **alone, input_scaling=factor * gain)
        reseed_noise(node, beat_seed(int(cfg["reservoir"]["seed"]), index))
        driven[factor] = np.asarray(node.run(beat, reset=True)).reshape(len(beat), 4)
    return no_beat, driven


def draw_colony(no_beat: np.ndarray, driven: dict[float, np.ndarray], beat: np.ndarray,
                beat_label: str) -> plt.Figure:
    """One row of LuxI and AiiA per input gain under a row without the beat; the beat below."""
    rows = [("No beat", no_beat)] + [(f"Input gain \u00d7 {factor:g}", states)
                                     for factor, states in driven.items()]
    fig, axes = plt.subplots(len(rows) + 1, 1, sharex=True,
                             figsize=(SINGLE_COLUMN, 0.8 * len(rows) + 1.1),
                             gridspec_kw={"height_ratios": [1.0] * len(rows) + [0.7]})
    t = np.arange(len(beat))
    for ax, (title, states) in zip(axes[:-1], rows, strict=True):
        ax.sharey(axes[0])
        for var, colour, style in (("I", OKABE_ITO[0], "-"), ("A", OKABE_ITO[1], "--")):
            ax.plot(t, states[:, VARIABLES.index(var)], color=colour, ls=style,
                    label=STATE_NAMES[var])
        ax.set_yscale("log")
        ax.set_title(title, loc="left")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncols=2)
    axes[-1].plot(t, beat[:, 0], color="0.3")
    axes[-1].set_ylabel(beat_label)
    axes[-1].set_xlabel("Time from the start of the beat (steps)")
    fig.supylabel("Level (model units)", fontsize=STYLE["axes.labelsize"])
    return fig


# --- array ------------------------------------------------------------------------------

def array_states(cfg: dict, beat: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray]:
    """The array's state (steps, colonies, 4) during the beat, and the mean size over the
    colonies of each of :data:`TERMS` at each step (steps, 3)."""
    node = build(cfg)
    node.initialize(beat[:1])
    reseed_noise(node, beat_seed(int(cfg["reservoir"]["seed"]), index))
    node.reset(to_state=node.zero_state())  # as run(beat, reset=True) does
    node.diagnostics = True  # the node keeps the terms of its last step
    states, terms = [], []
    for x in beat:
        states.append(np.asarray(node(x.reshape(1, 1))).reshape(-1, 4))
        terms.append([np.mean(np.abs(getattr(node, f"_last_{name}"))) for name, *_ in TERMS])
    return np.array(states), np.array(terms)


def draw_array(states: np.ndarray, terms: np.ndarray, beat: np.ndarray, beat_label: str,
               var: str) -> plt.Figure:
    """The beat, the drive terms and a heatmap of ``var``, one row per colony, both on log
    scales (the colours span the 1st percentile of the positive levels to the largest).
    Without a recurrent term the drive is the input term, and only that is drawn."""
    fig, (ax_beat, ax_drive, ax_states) = plt.subplots(
        3, 1, sharex=True, figsize=(DOUBLE_COLUMN, 4.2),
        gridspec_kw={"height_ratios": [0.7, 1.0, 2.0]})
    t = np.arange(len(beat))
    ax_beat.plot(t, beat[:, 0], color="0.3")
    ax_beat.set_ylabel(beat_label)
    recurrent = np.any(terms[:, 1] > 0)
    for column, (_, name, style) in enumerate(TERMS if recurrent else TERMS[:1]):
        ax_drive.plot(t, terms[:, column], color=OKABE_ITO[column], ls=style, label=name)
    ax_drive.set(yscale="log", ylabel="Mean size over\nthe colonies")
    ax_drive.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    levels = states[:, :, VARIABLES.index(var)].T
    positive = levels[levels > 0]
    image = ax_states.imshow(levels, aspect="auto", origin="lower", interpolation="nearest",
                             cmap="viridis", extent=(-0.5, len(beat) - 0.5, -0.5,
                                                     states.shape[1] - 0.5),
                             norm=LogNorm(np.percentile(positive, 1), positive.max(), clip=True)
                             if positive.size else None)
    ax_states.set(ylabel="Colony", xlabel="Time from the start of the beat (steps)")
    fig.colorbar(image, ax=ax_states, label=f"{STATE_NAMES.get(var, var)} (model units)")
    return fig


# --- coupling ---------------------------------------------------------------------------

def coupling(cfg: dict) -> tuple[np.ndarray, sparse.coo_matrix | None]:
    """The positions (colonies, 2) of the config's colonies, in units of the array's width,
    and their coupling weights: the conductances of the diffusive coupling, the kernel of
    the abstract one (which couples each pair of colonies through all four variables), or
    None without coupling."""
    params = cfg["reservoir"]["params"]
    seed, units = int(cfg["reservoir"]["seed"]), int(params["units"])
    topology = params.get("topology_type", "distance")
    if topology == "lattice":  # the grid of bioreservoir.topology.lattice_matrix
        columns = max(int(round(np.sqrt(units))), 1)
        while units % columns:
            columns -= 1
        positions = np.column_stack([np.arange(units) % columns,
                                     np.arange(units) // columns]) / max(columns - 1, 1)
        kernel = None
    elif topology == "distance":  # the node builds its kernel with this call and seed
        kernel, positions = _build_bioreservoir_topology(params, seed)(units, units,
                                                                     return_coords=True)
    else:
        raise SystemExit(f"A {topology} topology places no colonies in space.")
    weights = {"diffusion": lambda: build(cfg, warmup=0).conductance,
               "abstract": lambda: kernel}.get(params.get("coupling", "abstract"), lambda: None)()
    return (np.asarray(positions, dtype=float),
            None if weights is None else sparse.coo_matrix(weights))


def strongest(weights: sparse.coo_matrix, links: int) -> np.ndarray:
    """The positions in ``weights.data`` of the ``links`` largest weights (by size) of each
    colony, other than its own."""
    order = np.lexsort((-np.abs(weights.data), weights.row))
    order = order[weights.row[order] != weights.col[order]]
    rows = weights.row[order]
    rank = np.arange(len(order)) - np.searchsorted(rows, rows)
    return order[rank < links]


def draw_coupling(positions: np.ndarray, weights: sparse.coo_matrix | None,
                  links: int) -> plt.Figure:
    """Left, the colonies and their strongest links; right, each weight against distance."""
    fig, (ax_map, ax_fall) = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN, 3.2),
                                          gridspec_kw={"width_ratios": [1.0, 1.2]})
    if weights is None:
        ax_fall.set_axis_off()
        ax_map.set_title("No coupling", loc="left")
    else:
        keep = strongest(weights, links)
        values = weights.data[keep]
        segments = np.stack([positions[weights.row[keep]], positions[weights.col[keep]]], 1)
        others = weights.row != weights.col
        distance = np.linalg.norm(positions[weights.row] - positions[weights.col], axis=1)
        for sign, colour, style, name in ((1, OKABE_ITO[0], "solid", "Positive"),
                                          (-1, OKABE_ITO[1], "dashed", "Negative")):
            if np.any(np.sign(values) == sign):
                ax_map.add_collection(LineCollection(
                    segments[np.sign(values) == sign], colors=colour, linestyles=style,
                    linewidths=1.0, alpha=0.6, label=name))
            mask = others & (np.sign(weights.data) == sign)
            if np.any(mask):
                ax_fall.scatter(distance[mask], weights.data[mask], s=3, alpha=0.4,
                                color=colour, edgecolors="none", label=name)
        ax_fall.legend(markerscale=3)
        ax_map.set_title(f"Strongest {links} links of each colony", loc="left")
        ax_fall.set(xlabel="Distance between the colonies (array width = 1)",
                    ylabel="Coupling weight")
    ax_map.scatter(*positions.T, s=3, color="black", edgecolors="none", zorder=3)
    ax_map.set(aspect="equal", xticks=[], yticks=[])
    ax_map.spines[:].set_visible(False)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw a run config's model.")
    parser.add_argument("config", type=Path,
                        help="A run config, e.g. experiments/v0.2/designs/centre.yaml.")
    parser.add_argument("--plots", default=",".join(PLOTS),
                        help=f"Figures to draw, comma-separated, from {', '.join(PLOTS)}.")
    parser.add_argument("--beat", type=int, default=0, help="Index of the beat in the task's pool.")
    parser.add_argument("--gains", default="0.5,1,2",
                        help="Colony: multiples of the config's input gain, comma-separated.")
    parser.add_argument("--var", default="I", choices=VARIABLES,
                        help="Array: the variable of the heatmap (default: I, LuxI).")
    parser.add_argument("--links", type=int, default=4,
                        help="Coupling: links drawn for each colony (default: 4).")
    args = parser.parse_args()
    plots = [p.strip() for p in args.plots.split(",") if p.strip()]
    if set(plots) - set(PLOTS):
        raise SystemExit(f"Unknown plots {sorted(set(plots) - set(PLOTS))}; choose from {PLOTS}.")

    cfg = run_config(args.config)
    name = cfg["name"]
    plt.style.use(STYLE)
    figures = []
    if {"colony", "array"} & set(plots):
        beat, beat_class = task_beat(cfg, args.beat)
        label = f"{beat_class} beat\n(z-score)"
        print(f"Beat {args.beat}: {beat_class}")
    if "colony" in plots:
        gains = [float(g) for g in args.gains.split(",") if g.strip()]
        figures.append((draw_colony(*colony_states(cfg, beat, args.beat, gains), beat, label),
                        f"{name}-colony-beat{args.beat}"))
    if "array" in plots:
        figures.append((draw_array(*array_states(cfg, beat, args.beat), beat, label, args.var),
                        f"{name}-array-beat{args.beat}-{args.var}"))
    if "coupling" in plots:
        figures.append((draw_coupling(*coupling(cfg), args.links), f"{name}-coupling"))
    for fig, figure_name in figures:
        print("Saved:", ", ".join(str(p) for p in save_figure(fig, figure_name, topic="model")))


if __name__ == "__main__":
    main()
