"""Draw the states of a run config's array during one beat (the design of Fig. 4c of
``v0.1``) and, with ``--drive``, the terms of the drive its colonies receive (Fig. 4a).

The array is built and stepped through the beat as a run steps it
(:func:`training.figures.array_states`). The heatmap shows ``--var`` (default LuxI, the
reporter that the readout images) of every colony at every step, on a linear colour scale,
or on a logarithmic one with ``--log`` (from the 1st percentile of the positive levels, for
arrays in which a few colonies run far above the rest). The drive figure shows the mean
size over the colonies of the input term, the recurrent term and their sum, the drive, on a
log scale, above the beat as the colonies receive it
(:func:`training.figures.encoded_input`). Only the abstract coupling has a recurrent term;
without it, the drive is the input term and only that is drawn.

    python scripts/figures/visualise_reservoir_states.py \\
        experiments/v0.2/designs/abstract.yaml --drive

It writes ``results/figures/states/<name>-states-beat<N>-<var>.png`` and
``<name>-drive-beat<N>.png`` (``--formats pdf,svg`` adds a PDF and an SVG).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from training.figures import (
    VARIABLES,
    array_states,
    encoded_input,
    input_name,
    run_config,
    task_beat,
)
from utils.visualisation import STATE_NAMES, STYLE, add_formats_option, save_figure

#: The drive terms: label, Okabe-Ito colour, line width and style (readable in greyscale).
TERMS = ((r"Input term: $\alpha_u \mathbf{W}_{in} u_t + b$", "#E69F00", 1.8, "solid"),
         (r"Recurrent term: $\alpha_r \mathbf{W} r_{t-1}$", "#56B4E9", 1.8, "dashed"),
         (r"Drive: input $+$ recurrent", "#009E73", 1.4, "dashdot"))
BOX = {"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "grey", "alpha": 0.85}


def draw_states(states: np.ndarray, var: str = "I", log: bool = False) -> plt.Figure:
    """A heatmap of ``var`` of every colony (rows) at every step (columns)."""
    levels = states[:, :, VARIABLES.index(var)].T
    positive = levels[levels > 0]
    norm = (LogNorm(np.percentile(positive, 1), positive.max(), clip=True)
            if log and positive.size else None)
    fig, ax = plt.subplots(figsize=(14, 6))
    image = ax.imshow(levels, aspect="auto", origin="lower", interpolation="nearest",
                      cmap="viridis", norm=norm)
    fig.colorbar(image, ax=ax, label=f"{STATE_NAMES[var]} concentration")
    ax.set(xlabel="Timestep", ylabel="Colony index")
    fig.tight_layout()
    return fig


def draw_drive(terms: np.ndarray, beat_input: np.ndarray, beat_class: str,
               input_scaling: float, rc_scaling: float, source: str = "ECG") -> plt.Figure:
    """The mean size over the colonies of each drive term (``terms``, steps x 3) above the
    beat's input (``source``: ECG or inducer)."""
    fig, (ax, ax_ecg) = plt.subplots(2, 1, figsize=(14, 6), sharex=True, layout="constrained",
                                     gridspec_kw={"height_ratios": [2.2, 1.2]})
    t = np.arange(len(terms))
    recurrent = bool(np.any(terms[:, 1] > 0))
    for column, (label, colour, width, style) in enumerate(TERMS if recurrent else TERMS[:1]):
        ax.plot(t, terms[:, column], label=label, color=colour, lw=width, ls=style,
                alpha=0.9 if column == 2 else 1.0)
    ax.set_yscale("log")
    ax.set_ylabel("Mean per colony (log)")
    ax.legend(ncol=3, fontsize=9, loc="upper right")
    scalings = [rf"$\alpha_u = {input_scaling:.2g}$"]
    if recurrent:
        scalings.append(rf"$\alpha_r = {rc_scaling:.2g}$")
    ax.annotate(",\u2002".join(scalings), xy=(0.02, 0.05), xycoords="axes fraction",
                fontsize=9, bbox=BOX)
    ax_ecg.plot(t, beat_input, color="#CC0000", lw=1.6, label=f"{source}: {beat_class}")
    ax_ecg.set(ylabel=f"{source} input", xlabel="Timestep")
    ax_ecg.legend(loc="upper right", fontsize=8)
    for a, grid in ((ax, 0.25), (ax_ecg, 0.18)):
        a.grid(True, alpha=grid)
        a.spines[["top", "right"]].set_visible(False)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the states of a run config's array "
                                                 "during one beat, and the drive terms.")
    parser.add_argument("config", type=Path,
                        help="A run config, e.g. experiments/v0.2/designs/centre.yaml.")
    parser.add_argument("--beat", type=int, default=0,
                        help="Index of the beat in the task's pool (default: 0).")
    parser.add_argument("--var", default="I", choices=VARIABLES,
                        help="The variable of the heatmap (default: I, LuxI).")
    parser.add_argument("--log", action="store_true", help="A logarithmic colour scale.")
    parser.add_argument("--drive", action="store_true",
                        help="Also draw the terms of the drive (Fig. 4a of v0.1).")
    add_formats_option(parser)
    args = parser.parse_args()

    cfg = run_config(args.config)
    beat, beat_class = task_beat(cfg, args.beat)
    states, terms = array_states(cfg, beat, args.beat)
    plt.style.use(STYLE)
    name, params = cfg["name"], cfg["reservoir"]["params"]
    figures = [(draw_states(states, args.var, args.log),
                f"{name}-states-beat{args.beat}-{args.var}")]
    if args.drive:
        figures.append((draw_drive(terms, encoded_input(cfg, beat), beat_class,
                                   float(params["input_scaling"]),
                                   float(params.get("rc_scaling", 0.0)), input_name(cfg)),
                        f"{name}-drive-beat{args.beat}"))
    print(f"Beat {args.beat}: {beat_class}")
    for fig, figure_name in figures:
        paths = save_figure(fig, figure_name, topic="states", formats=args.formats)
        print("Saved:", ", ".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
