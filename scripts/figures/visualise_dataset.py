"""Draw the beats of a run config's task: for each class, its mean and the band that holds
the middle 80% of its beats at each time.

The beats are the ones a run of the config reads (:func:`training.run.load_task`): the same
pool, class balance and scaling, so the counts are those of the runs.

    python scripts/figures/visualise_dataset.py experiments/v0.2/designs/centre.yaml

It writes ``results/figures/dataset/<name>-beats.png`` (and ``.pdf``), where ``<name>`` is
the config's name, and prints the number of beats of each class.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from training.run import load_config, load_task, resolve
from utils.preprocessing import class_indices
from utils.visualisation import CLASS_COLOURS, SINGLE_COLUMN, STYLE, class_names, save_figure

SAMPLING_HZ = 125  # the beats of the prepared dataset are sampled at 125 Hz
BAND = (0.1, 0.9)


def class_bands(X: np.ndarray, Y: np.ndarray) -> dict[int, tuple]:
    """For each class: its mean beat, the 10% and 90% quantiles at each time, its count."""
    beats = np.asarray(X, dtype=float).reshape(len(X), -1)
    y = class_indices(Y)
    return {int(c): (beats[y == c].mean(axis=0), *np.quantile(beats[y == c], BAND, axis=0),
                     int(np.sum(y == c))) for c in np.unique(y)}


def draw(bands: dict[int, tuple], names: tuple[str, ...], ylabel: str) -> plt.Figure:
    """One row per class: the band in a light shade of the class colour, the mean over it."""
    fig, axes = plt.subplots(len(bands), 1, sharex=True, sharey=True, squeeze=False,
                             figsize=(SINGLE_COLUMN, 0.75 * len(bands) + 0.7))
    for ax, (c, (mean, low, high, count)) in zip(axes[:, 0], bands.items(), strict=True):
        colour = CLASS_COLOURS[names[c]]
        t = np.arange(len(mean)) * 1000 / SAMPLING_HZ
        ax.fill_between(t, low, high, color=colour, alpha=0.25, lw=0)
        ax.plot(t, mean, color=colour)
        ax.text(0.99, 0.95, f"{names[c]} (n = {count:,})", transform=ax.transAxes,
                ha="right", va="top")
    axes[-1, 0].set_xlabel("Time (ms)")
    fig.supylabel(ylabel, fontsize="medium")
    fig.legend(handles=[Line2D([], [], color="0.3", label="Mean"),
                        Patch(color="0.3", alpha=0.25, lw=0,
                              label=f"{BAND[0]:.0%}\u2013{BAND[1]:.0%} of beats")],
               loc="outside upper center", ncols=2)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the beats of a run config's task.")
    parser.add_argument("config", type=Path,
                        help="A run config, e.g. experiments/v0.2/designs/centre.yaml.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.pop("study", None)
    cfg = resolve(cfg)
    X, Y, _ = load_task(cfg["task"])
    names = class_names(2 if cfg["task"]["labels"] == "binary" else 5)
    scaler = cfg["task"]["scaler"]
    ylabel = ("ECG, z-score of each beat" if scaler == "sequence_zscore"
              else f"ECG ({scaler} scaling)")
    bands = class_bands(X, Y)

    plt.style.use(STYLE)
    paths = save_figure(draw(bands, names, ylabel), f"{cfg['name']}-beats", topic="dataset")
    for c, band in bands.items():
        print(f"{names[c]:<17} {band[-1]:>6,} beats")
    print("Saved:", ", ".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
