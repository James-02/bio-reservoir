"""Draw the heterogeneity figure of the v0.1 paper (Supplementary Fig. S10) from its runs.

The runs are those of ``scripts/run_heterogeneity.py``: one run folder per kinetic CV and
seed, ``heterogeneity-cv<CV>-seed<S>`` (with ``-array<A>`` when the array has its own seed).
Each dot is one run, the mean over its folds of the macro F1; the line joins the means over
the runs at each CV, with a band of one population SD.

    python scripts/figures/visualise_heterogeneity.py --archive v0.1   # the archived runs
    python scripts/figures/visualise_heterogeneity.py                  # runs in results/runs

It writes ``results/figures/diagnostics/heterogeneity_robustness.png`` (and ``.pdf``) and
prints the mean, SD and number of runs at each CV. The figure is drawn in matplotlib's
default style, as published: from the archived runs, with matplotlib 3.8.3 (the version that
drew it), the PDF renders the same as the published one, pixel for pixel.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils import config
from utils.results import FoldArtifact
from utils.visualisation import plot_sweep, save_figure

RUN_NAME = re.compile(r"heterogeneity-cv(?P<cv>\d+(?:\.\d+)?)-seed\d+(?:-array\d+)?")


def run_scores(runs_dir: Path, metric: str = "f1_macro") -> dict[float, list[float]]:
    """The mean over folds of ``metric`` for each heterogeneity run, by kinetic CV."""
    scores: dict[float, list[float]] = defaultdict(list)
    for run in sorted(p for p in Path(runs_dir).iterdir() if p.is_dir()):
        match = RUN_NAME.fullmatch(run.name)
        if match:
            folds = FoldArtifact.load_trial(run.name, results_dir=str(runs_dir))
            scores[float(match["cv"])].append(float(np.mean([f.metrics[metric] for f in folds])))
    return dict(sorted(scores.items()))


def draw(scores: dict[float, list[float]]) -> plt.Figure:
    """The figure as published: runs, their mean and SD, the CV = 0 mean, a shaded CV range."""
    levels = sorted(scores)
    position = {cv: i for i, cv in enumerate(levels)}
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.8))

    # The caption's representative low-to-moderate heterogeneity, CV 0.1 to 0.3.
    low, high = position.get(0.1, 3) - 0.35, position.get(0.3, 6) + 0.35
    ax.axvspan(low, high, alpha=0.08, color="#2ca02c", zorder=0)
    ax.text((low + high) / 2, 0.898, "Biological\nrange", ha="center", va="bottom",
            fontsize=7.5, color="#2ca02c", alpha=0.7, style="italic")

    baseline = float(np.mean(scores[levels[0]]))
    plot_sweep(ax, scores, reference=baseline, reference_label=f"Baseline F$_1$ = {baseline:.3f}",
               seed_label="Individual seeds", mean_label="Mean F$_1$ (\u00b11\u03c3)", ddof=0)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_xlabel("Coefficient of Variation (CV)", fontsize=10)
    ax.set_ylabel("Macro F$_1$ Score", fontsize=10)
    ax.set_ylim(0.875, 0.905)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the v0.1 heterogeneity figure "
                                                 "(Supplementary Fig. S10) from its runs.")
    parser.add_argument("--archive", metavar="VERSION",
                        help="Read the runs of an archived version, e.g. v0.1.")
    parser.add_argument("--runs_dir", help="Folder of the run folders (default: results/runs).")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir) if args.runs_dir else config.runs_dir(args.archive)
    scores = run_scores(runs_dir)
    if not scores:
        raise SystemExit(f"No heterogeneity-cv<CV>-seed<S> runs in {runs_dir}")
    with plt.style.context("default"):
        paths = save_figure(draw(scores), "heterogeneity_robustness", topic="diagnostics")

    print("CV      mean F1   SD      runs")
    for cv, values in scores.items():
        print(f"{cv:<7g} {np.mean(values):.4f}    {np.std(values):.4f}  {len(values)}")
    print("Saved:", ", ".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
