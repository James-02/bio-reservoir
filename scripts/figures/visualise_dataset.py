"""Draw the beats of a run config's task: the mean of each class with the band of its middle
80% (the design of Fig. 5a of ``v0.1``), one beat with additive noise (Fig. 5b), and one
beat under three scalings (Fig. 5c).

The beats are the ones a run of the config reads (:func:`training.run.load_task`): the same
pool, class balance and scaling, so the counts are those of the runs. The noise figure adds
Gaussian noise of SD ``--noise-std`` to beat ``--beat`` as the run scales it; the scaler
figure scales the same beat, raw, by its own minimum and maximum, by the mean and SD of the
pool at each time (``standard``) and by its own mean and SD (``sequence_zscore``), and marks
the config's scaling as used.

    python scripts/figures/visualise_dataset.py experiments/v0.2/designs/centre.yaml

It writes ``results/figures/dataset/<name>-beats.png``, ``<name>-noise-beat<N>.png`` and
``<name>-scalers-beat<N>.png`` (and ``.pdf``), and prints the number of beats of each class.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from training.run import load_config, load_task, resolve
from utils.preprocessing import class_indices, standardize_data
from utils.visualisation import CLASS_COLOURS, OKABE_ITO, STYLE, class_names, save_figure

SAMPLING_HZ = 125  # the beats of the prepared dataset are sampled at 125 Hz
BAND = (0.1, 0.9)
#: The scalings of the scaler figure: name, label, and line (colour, width, style, alpha).
SCALERS = (("sequence_minmax_pos", "Min-max (reference)", (OKABE_ITO[0], 1.6, "-", 0.28)),
           ("standard", "Z-score (standard)", (OKABE_ITO[1], 1.7, (0, (4, 2)), 1.0)),
           ("sequence_zscore", "Z-score (per-seq)", (OKABE_ITO[2], 2.1, "-", 1.0)))


def milliseconds(steps: int) -> np.ndarray:
    return np.arange(steps) * 1000 / SAMPLING_HZ


def class_bands(X: np.ndarray, Y: np.ndarray) -> dict[int, tuple]:
    """For each class: its mean beat, the 10% and 90% quantiles at each time, its count."""
    beats = np.asarray(X, dtype=float).reshape(len(X), -1)
    y = class_indices(Y)
    return {int(c): (beats[y == c].mean(axis=0), *np.quantile(beats[y == c], BAND, axis=0),
                     int(np.sum(y == c))) for c in np.unique(y)}


def draw(bands: dict[int, tuple], names: tuple[str, ...], ylabel: str) -> plt.Figure:
    """One row per class: the band in a light shade of the class colour, the mean over it."""
    fig, axes = plt.subplots(len(bands), 1, sharex=True, sharey=True, squeeze=False,
                             figsize=(7, 1.6 * len(bands) + 1.2), layout="constrained")
    for ax, (c, (mean, low, high, count)) in zip(axes[:, 0], bands.items(), strict=True):
        colour = CLASS_COLOURS[names[c]]
        t = milliseconds(len(mean))
        ax.fill_between(t, low, high, color=colour, alpha=0.25, lw=0)
        ax.plot(t, mean, color=colour, lw=2.0)
        ax.text(0.99, 0.95, f"{names[c]} (n = {count:,})", transform=ax.transAxes,
                ha="right", va="top")
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1, 0].set_xlabel("Time (ms)")
    fig.supylabel(ylabel, fontsize="medium")
    fig.legend(handles=[Line2D([], [], color="0.3", lw=2.0, label="Mean"),
                        Patch(color="0.3", alpha=0.25, lw=0,
                              label=f"{BAND[0]:.0%}\u2013{BAND[1]:.0%} of beats")],
               loc="outside upper center", ncols=2, frameon=False)
    return fig


def draw_noise(beat: np.ndarray, noise: np.ndarray, noise_std: float) -> plt.Figure:
    """A beat, the noise added to it and the sum."""
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    t = milliseconds(len(beat))
    ax.plot(t, beat, color="orange", label="Training Data")
    ax.plot(t, noise, color="green", ls="dotted", label=f"Noise ($\\sigma$={noise_std:g})")
    ax.plot(t, beat + noise, color="blue", ls="dashed", label="Augmented Data")
    ax.set(xlabel="Time (ms)", ylabel="Value")
    ax.legend()
    return fig


def draw_scalers(scaled: dict[str, np.ndarray], used: str) -> plt.Figure:
    """A beat under each scaling of :data:`SCALERS`; the one a run uses is marked."""
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    for name, label, (colour, width, style, alpha) in SCALERS:
        ax.plot(milliseconds(len(scaled[name])), scaled[name], color=colour, lw=width, ls=style,
                alpha=alpha, label=label + (" [used]" if name == used else ""))
    ax.grid(True, alpha=0.2)
    ax.set(xlabel="Time (ms)", ylabel="Value")
    ax.legend(loc="upper right", fontsize=9)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the beats of a run config's task.")
    parser.add_argument("config", type=Path,
                        help="A run config, e.g. experiments/v0.2/designs/centre.yaml.")
    parser.add_argument("--beat", type=int, default=0,
                        help="Index of the beat of the noise and scaler figures in the task's "
                             "pool (default: 0).")
    parser.add_argument("--noise-std", type=float, default=0.2862,
                        help="SD of the noise of the noise figure (default: 0.2862, as in "
                             "v0.1).")
    parser.add_argument("--seed", type=int, default=1337, help="Seed of the noise.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.pop("study", None)
    cfg = resolve(cfg)
    task = cfg["task"]
    X, Y, _ = load_task(task)
    names = class_names(2 if task["labels"] == "binary" else 5)
    scaler = task["scaler"]
    ylabel = ("ECG, z-score of each beat" if scaler == "sequence_zscore"
              else f"ECG ({scaler} scaling)")
    bands = class_bands(X, Y)
    beat = np.asarray(X[args.beat], dtype=float).reshape(-1)
    noise = np.random.default_rng(args.seed).normal(scale=args.noise_std, size=beat.shape)
    raw, _, _ = load_task({**task, "scaler": "none"})
    scaled = {name: np.asarray(standardize_data(raw, raw, scaler_type=name)[0][args.beat],
                               dtype=float).reshape(-1) for name, *_ in SCALERS}

    plt.style.use(STYLE)
    name = cfg["name"]
    paths = [*save_figure(draw(bands, names, ylabel), f"{name}-beats", topic="dataset"),
             *save_figure(draw_noise(beat, noise, args.noise_std),
                          f"{name}-noise-beat{args.beat}", topic="dataset"),
             *save_figure(draw_scalers(scaled, scaler), f"{name}-scalers-beat{args.beat}",
                          topic="dataset")]
    for c, band in bands.items():
        print(f"{names[c]:<17} {band[-1]:>6,} beats")
    print(f"Beat {args.beat}: {names[class_indices(Y)[args.beat]]}")
    print("Saved:", ", ".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
