"""Draw one colony of a run config's design driven by a beat at several input scalings: the
design of Fig. 3b of ``v0.1``.

The colony has the design's kinetics without heterogeneity, no neighbours, and the
design's input encoding and input noise (:func:`training.figures.colony_states`). Rows, top
to bottom: the colony without input; the colony driven by the beat at each input scaling
(``--scales``; default 0.1, 1 and 10 times the config's); and the beat as the colony
receives it (:func:`training.figures.encoded_input`). The end of the warm-up, without
input, is drawn dashed (its last ``--warmup-steps``), and the beat starts at timestep 0.

    python scripts/figures/visualise_dde.py experiments/v0.2/designs/centre.yaml --beat 0

It writes ``results/figures/states/<name>-dde-beat<N>.png`` (and ``.pdf``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from training.figures import VARIABLES, colony_states, encoded_input, run_config, task_beat
from utils.visualisation import OKABE_ITO, STATE_NAMES, STYLE, save_figure

LINES = {"I": {"color": OKABE_ITO[0], "lw": 1.8, "ls": "-"},
         "A": {"color": OKABE_ITO[1], "lw": 1.8, "ls": (0, (4, 2))}}
WARMUP = {"color": "black", "lw": 1.4, "ls": "--"}
ECG = {"color": "#CC0000", "lw": 1.6}
BOX = {"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2}


def draw(warm: np.ndarray, rest: np.ndarray, driven: dict[float, np.ndarray],
         beat_input: np.ndarray, beat_class: str, config_scaling: float | None = None
         ) -> plt.Figure:
    """The colony's LuxI and AiiA without input and at each scaling in ``driven``, after the
    ``warm`` steps of the warm-up, and the beat's input below."""
    w = len(warm)
    t = np.arange(-w, len(beat_input))
    start = max(w - 1, 0)  # each trace continues from the last step of the warm-up

    def label(scaling):
        mark = " (config)" if config_scaling and np.isclose(scaling, config_scaling) else ""
        return f"Scale={scaling:.3g}{mark}"

    rows = [("Baseline (0 input)", rest)] + [(label(s), states) for s, states in driven.items()]
    n = len(rows) + 1
    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(12.0, 2.0 * n), layout="constrained",
                             gridspec_kw={"height_ratios": [2.2] * len(rows) + [1.2]})
    for ax, (text, states) in zip(axes[:-1], rows, strict=True):
        for var in ("I", "A"):
            y = np.concatenate([warm[:, VARIABLES.index(var)], states[:, VARIABLES.index(var)]])
            if w:
                ax.plot(t[:w], y[:w], **WARMUP, alpha=0.75)
            ax.plot(t[start:], y[start:], **LINES[var])
        if w:
            ax.axvline(-1, color="black", alpha=0.25, lw=1.0)
        ax.grid(True, alpha=0.22)
        ax.text(0.01, 0.86, text, transform=ax.transAxes, fontsize=9, va="top", ha="left",
                bbox=BOX)
    warm_handle = Line2D([], [], **WARMUP, label="Warmup (0 input)")
    axes[0].legend(handles=[Line2D([], [], **LINES[v], label=STATE_NAMES[v]) for v in "IA"]
                   + [warm_handle], loc="upper right", fontsize=8)

    ax = axes[-1]
    u = np.concatenate([np.zeros(w), beat_input])
    if w:
        ax.plot(t[:w], u[:w], **WARMUP, alpha=0.85)
        ax.axvline(-1, color="black", alpha=0.35, lw=1.0)
    ax.plot(t[start:], u[start:], **ECG)
    ax.grid(True, alpha=0.18)
    ax.set_ylabel("ECG Input")
    ax.legend(handles=[Line2D([], [], **ECG, label=f"ECG Class: {beat_class}"), warm_handle],
              loc="upper right", fontsize=8)
    fig.supxlabel("Timestep")
    fig.supylabel("Genetic Biopixel States")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw one colony of a run config's design "
                                                 "driven by a beat at several input scalings.")
    parser.add_argument("config", type=Path,
                        help="A run config, e.g. experiments/v0.2/designs/centre.yaml.")
    parser.add_argument("--beat", type=int, default=0,
                        help="Index of the beat in the task's pool (default: 0).")
    parser.add_argument("--scales", help="Input scalings, comma-separated (default: 0.1, 1 and "
                                         "10 times the config's input_scaling).")
    parser.add_argument("--warmup-steps", type=int, default=100,
                        help="Steps at the end of the warm-up to draw (default: 100).")
    args = parser.parse_args()

    cfg = run_config(args.config)
    gain = float(cfg["reservoir"]["params"]["input_scaling"])
    scales = ([float(s) for s in args.scales.split(",") if s.strip()] if args.scales
              else [0.1 * gain, gain, 10 * gain])
    beat, beat_class = task_beat(cfg, args.beat)
    warm, rest, driven = colony_states(cfg, beat, args.beat, scales)
    shown = warm[len(warm) - min(max(args.warmup_steps, 0), len(warm)):]
    plt.style.use(STYLE)
    fig = draw(shown, rest, driven, encoded_input(cfg, beat), beat_class, gain)
    print(f"Beat {args.beat}: {beat_class}")
    paths = save_figure(fig, f"{cfg['name']}-dde-beat{args.beat}", topic="states")
    print("Saved:", ", ".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
