"""Draw a study's score against a setting it searches, one dot per seed.

The study is the one a config defines, read from its Optuna database with
:func:`training.study.trials_table`. Each dot is a finished trial, one per seed; the line
joins the means over the seeds, with a band of one sample SD. The x axis is the last setting
the study searches besides the seed (``--x`` chooses another); a study that searches two
settings gets one line for each value of the other (``--by``).

    python scripts/figures/visualise_sweep.py experiments/v0.2/phase-1/heterogeneity.yaml \\
        --reference 0
    python scripts/figures/visualise_sweep.py experiments/v0.2/phase-1/images-noise.yaml

It writes ``results/figures/sweeps/<study name>.png`` (and ``.pdf``) and prints the mean, SD
and number of seeds at each point.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from training.run import load_config
from training.study import name as study_name
from training.study import trials_table
from utils.visualisation import OKABE_ITO, STYLE, plot_sweep, save_figure

SEED_PATHS = ("seed", "reservoir.seed")
LABELS = {
    "reservoir.params.param_heterogeneity_cv": "Kinetic heterogeneity (CV)",
    "reservoir.params.input_scaling": "Input gain",
    "reservoir.params.diffusion_rate": "Coupling rate",
    "reservoir.params.input_connectivity": "Input connectivity",
    "reservoir.params.interaction_diameters": "Interaction diameters",
    "reservoir.params.rk4_substeps": "RK4 substeps",
    "reservoir.params.units": "Units",
    "reservoir.params.warmup": "Warm-up (steps)",
    "features.aggregation": "Images",
    "features.measurement_noise": "Measurement noise",
    "features.calibration_error": "Calibration error",
}
MARKERS = "osD^v<>p"


def label(path: str) -> str:
    """The axis label of a config path."""
    return LABELS.get(path, path.rsplit(".", 1)[-1].replace("_", " ").capitalize())


def sweep_scores(trials, x: str, by: str | None = None, metric: str = "f1") -> dict:
    """The scores of the finished trials as ``{value of by: {value of x: [one per seed]}}``,
    with the one key ``None`` when there is no ``by``."""
    done = trials[trials["state"] == "COMPLETE"]
    groups = [(None, done)] if by is None else list(done.groupby(by, sort=True))
    return {key: {level: rows[metric].tolist() for level, rows in group.groupby(x, sort=True)}
            for key, group in groups}


def draw(scores: dict, x: str, by: str | None = None, ylabel: str = "Macro-F1",
         reference: float | None = None) -> plt.Figure:
    """One sweep per value of ``by`` on shared levels of ``x``; ``reference`` draws the mean
    of the first sweep at that value of ``x`` as a dashed line."""
    levels = sorted({level for series in scores.values() for level in series})
    seeds = max(len(values) for series in scores.values() for values in series.values())
    fig, ax = plt.subplots()
    for i, (key, series) in enumerate(scores.items()):
        line = None
        if i == 0 and reference is not None:
            line = float(np.mean(series[reference]))
        plot_sweep(series, ax, levels=levels, color=OKABE_ITO[i % len(OKABE_ITO)],
                   marker=MARKERS[i % len(MARKERS)],
                   label=str(key) if key is not None else
                   "Mean \u00b1 SD" if seeds > 1 else "Score",
                   seed_label="Seeds" if i == 0 else None, reference=line,
                   reference_label=None if line is None else f"Mean at {reference:g}")
    ax.set(xlabel=label(x), ylabel=ylabel)
    if by is None:
        ax.legend()
    else:
        ax.legend(title=label(by))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw a study's score against a setting "
                                                 "it searches, one dot per seed.")
    parser.add_argument("config", type=Path,
                        help="A study config, e.g. experiments/v0.2/phase-1/heterogeneity.yaml.")
    parser.add_argument("--x", help="The setting on the x axis (default: the last one the "
                                    "study searches besides the seed).")
    parser.add_argument("--by", help="Another searched setting: one line for each of its values "
                                     "(default: the other setting, when there is one).")
    parser.add_argument("--reference", type=float, metavar="VALUE",
                        help="Draw the mean at this value of x as a dashed line.")
    parser.add_argument("--storage", help="Optuna storage URL (default: the study's database).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    searched = [path for path in cfg["study"]["search"] if path not in SEED_PATHS]
    x = args.x or searched[-1]
    others = [path for path in searched if path != x]
    if args.by is None and len(others) > 1:
        raise SystemExit(f"The study searches {searched}; choose one of the others with --by.")
    by = args.by or (others[0] if others else None)
    metric = str(cfg["study"].get("metric", "f1"))
    average = str((cfg.get("cv") or {}).get("primary_average", "macro"))
    scores = sweep_scores(trials_table(cfg, args.storage), x, by, metric)

    plt.style.use(STYLE)
    fig = draw(scores, x, by, f"{average.capitalize()}-F1" if metric == "f1" else metric,
               args.reference)
    paths = save_figure(fig, study_name(cfg), topic="sweeps")
    print(f"{label(by) if by else '':<14}{label(x):<28}mean     SD       seeds")
    for key, series in scores.items():
        for level, values in series.items():
            sd = np.std(values, ddof=1) if len(values) > 1 else float("nan")
            print(f"{'' if key is None else str(key):<14}{level!s:<28}{np.mean(values):.4f}   "
                  f"{sd:.4f}   {len(values)}")
    print("Saved:", ", ".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
