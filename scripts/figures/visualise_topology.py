"""Draw where a run config's colonies sit and how they are coupled: the kernel against
distance (the design of Fig. 4b of ``v0.1``) and the colonies with their strongest links
(Fig. 4d).

The kernel figure shows the weight of every pair of colonies against the distance between
them before the kernel's signs, sparsity and normalisation
(:func:`training.figures.raw_kernel`), with the mean distance to the nearest colony and the
interaction distance, beyond which the weight is 0; only a distance topology has a kernel.
The topology figure shows each colony and its ``--links`` strongest links: with the
abstract coupling the kernel's weights, positive (solid) or negative (dashed); with the
diffusive coupling the conductances through which neighbouring traps exchange AHL (all of
them by default: a lattice gives each trap at most four). ``--units`` draws the design with
fewer colonies, as ``v0.1`` drew 250 of its 1,000.

    python scripts/figures/visualise_topology.py experiments/v0.2/designs/abstract.yaml --units 250

It writes ``results/figures/topology/<name>-kernel.png`` and ``<name>-topology.png`` (and
``.pdf``), with ``-<units>`` after the name when ``--units`` is given.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy import sparse

from training.figures import coupling, raw_kernel, run_config, strongest
from utils.visualisation import STYLE, save_figure


def draw_kernel(distances: np.ndarray, weights: np.ndarray, nearest: float, reach: float,
                diameters: float) -> plt.Figure:
    """Each pair's kernel weight against its distance, with the nearest-colony distance and
    the interaction distance marked."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(distances, weights, s=3, alpha=0.2, rasterized=True)
    ax.axvline(nearest, color="g", ls="dotted", label="~1 diameter (mean NN)")
    ax.axvline(reach, color="r", ls="dashed", label=f"~{diameters:.1f} diameters")
    ax.set(xlabel="Distance", ylabel="Weight (raw)")
    ax.legend()
    fig.tight_layout()
    return fig


def draw_topology(positions: np.ndarray, weights: sparse.coo_matrix | None, links: int,
                  diffusive: bool = False) -> plt.Figure:
    """The colonies and the ``links`` strongest links of each."""
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.scatter(*positions.T, s=14, alpha=0.9, c="black")
    handles = [Line2D([], [], marker="o", color="w", markerfacecolor="black", markersize=6,
                      label="colony")]
    if weights is not None:
        keep = strongest(weights, links)
        values = weights.data[keep]
        segments = np.stack([positions[weights.row[keep]], positions[weights.col[keep]]], 1)
        kinds = ((1, "tab:blue", "solid", "AHL exchange" if diffusive else "excitation (+)"),
                 (-1, "tab:red", "dashed", "inhibition (-)"))
        for sign, colour, style, label in kinds:
            if np.any(np.sign(values) == sign):
                ax.add_collection(LineCollection(segments[np.sign(values) == sign],
                                                 colors=colour, linestyles=style, alpha=0.25,
                                                 linewidths=0.8))
                handles.append(Line2D([], [], color=colour, lw=2, ls=style, label=label))
    ax.legend(handles=handles, loc="upper right", fontsize=9)
    ax.set(xlabel="x", ylabel="y")
    ax.axis("equal")
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw where a run config's colonies sit and "
                                                 "how they are coupled.")
    parser.add_argument("config", type=Path,
                        help="A run config, e.g. experiments/v0.2/designs/abstract.yaml.")
    parser.add_argument("--units", type=int, help="Draw the design with this many colonies.")
    parser.add_argument("--links", type=int, help="Links drawn for each colony (default: 3, "
                                                  "or all with the diffusive coupling).")
    args = parser.parse_args()

    cfg = run_config(args.config)
    params = cfg["reservoir"]["params"]
    name = cfg["name"]
    if args.units:
        params["units"] = args.units
        name += f"-{args.units}"
    diffusive = params.get("coupling") == "diffusion"
    links = args.links or (4 if diffusive else 3)
    plt.style.use(STYLE)
    figures = []
    if params.get("topology_type", "distance") == "distance":
        figures.append((draw_kernel(*raw_kernel(cfg), float(params["interaction_diameters"])),
                        f"{name}-kernel"))
    figures.append((draw_topology(*coupling(cfg), links, diffusive), f"{name}-topology"))
    for fig, figure_name in figures:
        print("Saved:", ", ".join(str(p) for p in save_figure(fig, figure_name,
                                                               topic="topology")))


if __name__ == "__main__":
    main()
