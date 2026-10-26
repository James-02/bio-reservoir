"""Draw the macro F1 against the heterogeneity of the colonies: the design of Supplementary
Fig. S10 of ``v0.1``.

From run folders, the figure as published: the runs of ``scripts/run_heterogeneity.py``,
one per kinetic CV and seed (``heterogeneity-cv<CV>-seed<S>``, with ``-array<A>`` when the
array has its own seed); each dot is a run, the mean over its folds. From a study config, the
same figure for the setting the study sweeps (``--x``; default the first it searches besides
the seed): the kinetic, cell-density or dilution CV, the imperfect reset, and so on; each dot
is a finished trial, one per seed. A study that also searches another setting is drawn at one
value of it (``--fix PATH=VALUE``; default the config's own value). The line joins the means
at each level, with a band of one SD (the population SD for the runs, as published; the
sample SD over the seeds of a study), and the dashed line is the mean at the first level.

    python scripts/figures/visualise_heterogeneity.py --archive v0.1
    python scripts/figures/visualise_heterogeneity.py \\
        experiments/v0.2/checks/realism-options/d3-p-density-ridge.yaml

It writes ``results/figures/diagnostics/heterogeneity_robustness.png`` from runs, or
``results/figures/diagnostics/<study name>.png`` from a study, with ``-<setting><value>`` for
each fixed setting (``--formats pdf,svg`` adds a PDF and an SVG), and prints the mean, SD and
number of runs or seeds at each level.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from training.run import load_config, resolve
from training.study import name as study_name
from training.study import trials_table
from utils import config
from utils.results import FoldArtifact
from utils.visualisation import add_formats_option, plot_sweep, save_figure

RUN_NAME = re.compile(r"heterogeneity-cv(?P<cv>\d+(?:\.\d+)?)-seed\d+(?:-array\d+)?")
SEED_PATHS = ("seed", "reservoir.seed")
LABELS = {
    "reservoir.params.param_heterogeneity_cv": "Kinetic heterogeneity (CV)",
    "reservoir.params.heterogeneity_density_cv": "Cell-density heterogeneity (CV)",
    "reservoir.params.heterogeneity_dilution_cv": "Dilution heterogeneity (CV)",
    "reservoir.params.reset_noise": "Imperfect reset (CV)",
    "reservoir.params.input_levels": "Input levels",
}


def run_scores(runs_dir: Path, metric: str = "f1_macro") -> dict[float, list[float]]:
    """The mean over folds of ``metric`` for each heterogeneity run, by kinetic CV."""
    scores: dict[float, list[float]] = defaultdict(list)
    for run in sorted(p for p in Path(runs_dir).iterdir() if p.is_dir()):
        match = RUN_NAME.fullmatch(run.name)
        if match:
            folds = FoldArtifact.load_trial(run.name, results_dir=str(runs_dir))
            scores[float(match["cv"])].append(float(np.mean([f.metrics[metric] for f in folds])))
    return dict(sorted(scores.items()))


def study_scores(trials, x: str, fixed: dict[str, float] | None = None,
                 metric: str = "f1") -> dict[float, list[float]]:
    """The scores of the finished trials by their value of ``x``, one per seed, among the
    trials with the ``fixed`` values of the other settings."""
    done = trials[trials["state"] == "COMPLETE"]
    for path, value in (fixed or {}).items():
        done = done[np.isclose(done[path].astype(float), value)]
    return {float(level): rows[metric].tolist() for level, rows in done.groupby(x, sort=True)}


def draw(scores: dict[float, list[float]], xlabel: str = "Coefficient of Variation (CV)",
         band: tuple[float, float] | None = None, ddof: int = 1,
         ylim: tuple[float, float] | None = None) -> plt.Figure:
    """The design of S10: the dots, their mean and SD, the mean at the first level dashed,
    and the levels from ``band[0]`` to ``band[1]`` shaded as the biological range."""
    levels = sorted(scores)
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.8))
    if band is not None:
        low, high = levels.index(band[0]) - 0.35, levels.index(band[1]) + 0.35
        ax.axvspan(low, high, alpha=0.08, color="#2ca02c", zorder=0)
        ax.text((low + high) / 2, 0.767, "Biological\nrange", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=7.5, color="#2ca02c", alpha=0.7,
                style="italic")
    baseline = float(np.mean(scores[levels[0]]))
    plot_sweep(scores, ax, ddof=ddof, reference=baseline,
               reference_label=f"Baseline F$_1$ = {baseline:.3f}",
               seed_label="Individual seeds", label="Mean F$_1$ (\u00b11\u03c3)")
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Macro F$_1$ Score", fontsize=10)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def _value(cfg: dict, path: str):
    for key in path.split("."):
        cfg = cfg[key]
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the macro F1 against the heterogeneity "
                                                 "of the colonies (Supplementary Fig. S10).")
    parser.add_argument("config", type=Path, nargs="?",
                        help="A study config that sweeps a heterogeneity setting; without it, "
                             "the heterogeneity runs are read.")
    parser.add_argument("--archive", metavar="VERSION",
                        help="Read the runs of an archived version, e.g. v0.1.")
    parser.add_argument("--runs_dir", help="Folder of the run folders (default: results/runs).")
    parser.add_argument("--storage", help="Optuna storage URL (default: the study's database).")
    parser.add_argument("--x", help="The swept setting (default: the first the study searches "
                                    "besides the seed).")
    parser.add_argument("--fix", action="append", default=[], metavar="PATH=VALUE",
                        help="The value of another searched setting (default: the config's).")
    parser.add_argument("--range", metavar="LOW,HIGH",
                        help="Shade the levels LOW to HIGH as the biological range (runs: "
                             "0.1,0.3, as published).")
    parser.add_argument("--ylim", metavar="LOW,HIGH",
                        help="Limits of the F1 axis, to draw several figures on one scale "
                             "(runs: 0.875,0.905, as published).")
    add_formats_option(parser)
    args = parser.parse_args()

    band = tuple(float(v) for v in args.range.split(",")) if args.range else None
    ylim = tuple(float(v) for v in args.ylim.split(",")) if args.ylim else None
    if args.config is None:
        runs_dir = Path(args.runs_dir) if args.runs_dir else config.runs_dir(args.archive)
        scores = run_scores(runs_dir)
        if not scores:
            raise SystemExit(f"No heterogeneity-cv<CV>-seed<S> runs in {runs_dir}")
        name, unit = "heterogeneity_robustness", "runs"
        drawn = dict(band=band or (0.1, 0.3), ddof=0, ylim=ylim or (0.875, 0.905))
    else:
        cfg = load_config(args.config)
        searched = [p for p in cfg["study"]["search"] if p not in SEED_PATHS]
        x = args.x or searched[0]
        run_cfg = resolve({k: v for k, v in cfg.items() if k != "study"})
        fixed = {p: float(_value(run_cfg, p)) for p in searched if p != x}
        fixed |= {p: float(v) for p, v in (item.split("=", 1) for item in args.fix)}
        scores = study_scores(trials_table(cfg, args.storage), x, fixed,
                              str(cfg["study"].get("metric", "f1")))
        if not scores:
            raise SystemExit(f"No finished trials of {study_name(cfg)} at {fixed}")
        name = study_name(cfg) + "".join(f"-{p.rsplit('.', 1)[-1]}{v:g}"
                                         for p, v in fixed.items())
        unit = "seeds"
        drawn = dict(band=band, ddof=1, ylim=ylim, xlabel=LABELS.get(x, x.rsplit(".", 1)[-1]))
    with plt.style.context("default"):
        paths = save_figure(draw(scores, **drawn), name, topic="diagnostics",
                            formats=args.formats)

    print(f"Level   mean F1   SD      {unit}")
    for level, values in scores.items():
        sd = np.std(values, ddof=drawn["ddof"]) if len(values) > 1 else float("nan")
        print(f"{level:<7g} {np.mean(values):.4f}    {sd:.4f}  {len(values)}")
    print("Saved:", ", ".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
