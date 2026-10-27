"""Draw the scores of slow-sensing studies (experiments/v0.2/slow-sensing/) from their
databases, as ``study-table`` reads them.

- One study that sweeps a setting: macro F1 against the setting (``--x``; default the first
  the study searches besides the task and the seed), one line per task, with the mean and
  SD over the reservoir seeds.
- Several studies (``--compare``): macro F1 on each task, one marker per study with the mean
  and SD over the seeds, and each task's chance level.

``--fix PATH=VALUE`` keeps the trials with that value (for example task.condition=device);
a study that does not search PATH ignores it.

    python scripts/figures/visualise_slow_sensing.py \\
        experiments/v0.2/slow-sensing/centre-delay.yaml --fix task.condition=device
    python scripts/figures/visualise_slow_sensing.py --compare \\
        experiments/v0.2/slow-sensing/{centre,uncoupled-centre,esn,raw}.yaml \\
        --fix task.condition=device --fix reservoir.params.input_scaling=0.155113

It writes ``results/figures/slow-sensing/<study name>.png``, with ``-<setting><value>`` for each
``--fix`` (for example ``-measurement_noise0.03``), or ``compare-<condition>.png``
(``--formats pdf,svg`` adds a PDF and an SVG).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from training import inducer
from training.run import load_config
from training.study import name as study_name
from training.study import trials_table
from utils.visualisation import OKABE_ITO, STYLE, add_formats_option, save_figure

PATTERN, SEED = "task.pattern", "reservoir.seed"
LABELS = {"task.image_delay": "Image delay after the pattern (min)",
          "reservoir.params.units": "Traps", "reservoir.params.diffusion_rate": "Coupling rate",
          "reservoir.params.input_scaling": "Input scaling",
          "features.measurement_noise": "Measurement noise", "task.n_per_class": "Traces per class"}


def finished(cfg: dict, fixed: dict, storage: str | None = None):
    """The finished trials of the config's study with the ``fixed`` values, and its task."""
    trials = trials_table(cfg, storage)
    trials = trials[trials["state"] == "COMPLETE"]
    for path, value in fixed.items():
        if path in trials.columns:
            column = trials[path]
            keep = (np.isclose(column.astype(float), float(value))
                    if isinstance(value, (int, float)) else column == value)
            trials = trials[keep]
    if PATTERN not in trials.columns:
        trials = trials.assign(**{PATTERN: cfg["task"]["pattern"]})
    return trials


def sweep(trials, x: str, title: str, metric: str = "f1") -> plt.Figure:
    """Macro F1 against ``x``, one line per task: the mean and SD over the seeds."""
    levels = sorted(trials[x].unique())
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    for pattern in (p for p in inducer.PATTERNS if p in set(trials[PATTERN])):
        colour = OKABE_ITO[inducer.PATTERNS.index(pattern)]  # the same colour in every figure
        scores = trials[trials[PATTERN] == pattern].groupby(x)[metric]
        mean, sd = scores.mean().reindex(levels), scores.std().reindex(levels).fillna(0)
        position = np.arange(len(levels))
        ax.plot(position, mean, "o-", color=colour, label=pattern, ms=5)
        ax.fill_between(position, mean - sd, mean + sd, color=colour, alpha=0.15)
    ax.set_xticks(np.arange(len(levels)), [f"{v:g}" for v in levels])
    ax.set(xlabel=LABELS.get(x, x), ylabel="Macro F1", title=title, ylim=(0.0, 1.02))
    ax.legend(fontsize=10, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def compare(studies: dict[str, object], title: str, metric: str = "f1") -> plt.Figure:
    """Macro F1 on each task, one marker per study (mean and SD over the seeds), and each
    task's chance level."""
    patterns = [p for p in inducer.PATTERNS if any(p in set(t[PATTERN]) for t in studies.values())]
    fig, ax = plt.subplots(figsize=(11, 4.5), layout="constrained")
    tasks = np.arange(len(patterns))
    chance = [1 / len(inducer.CLASS_NAMES[p]) for p in patterns]
    ax.hlines(chance, tasks - 0.45, tasks + 0.45, color="0.5", ls="--", lw=1, label="Chance")
    width = 0.8 / len(studies)
    for i, (label, trials) in enumerate(studies.items()):
        scores = trials.groupby(PATTERN)[metric]
        mean, sd = scores.mean().reindex(patterns), scores.std().reindex(patterns).fillna(0)
        x = tasks - 0.4 + width * (i + 0.5)
        ax.errorbar(x, mean, yerr=sd, fmt="o", color=OKABE_ITO[i % len(OKABE_ITO)], ms=6,
                    capsize=3, label=label)
    ax.set_xticks(tasks, patterns)
    ax.set(ylabel="Macro F1", title=title, ylim=(0.0, 1.02))
    ax.legend(fontsize=9, ncol=min(len(studies) + 1, 5), loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def _short(cfg: dict) -> str:
    return study_name(cfg).removeprefix("slow-sensing-")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the scores of slow-sensing studies.")
    parser.add_argument("configs", type=Path, nargs="+", help="Slow-sensing study configs.")
    parser.add_argument("--compare", action="store_true",
                        help="Draw the studies side by side on each task.")
    parser.add_argument("--x", help="The swept setting (default: the first the study searches "
                                    "besides the task and the seed).")
    parser.add_argument("--fix", action="append", default=[], metavar="PATH=VALUE",
                        help="Keep the trials with this value of PATH.")
    parser.add_argument("--storage", help="Optuna storage URL (default: the study's database).")
    add_formats_option(parser)
    args = parser.parse_args()

    fixed = {path: yaml.safe_load(value) for path, value in
             (item.split("=", 1) for item in args.fix)}
    configs = [load_config(path) for path in args.configs]
    plt.style.use(STYLE)
    condition = fixed.get("task.condition", "")
    if args.compare:
        studies = {_short(cfg): finished(cfg, fixed, args.storage) for cfg in configs}
        fig = compare(studies, f"Slow sensing, {condition or 'all'} delivery")
        name = f"compare-{condition or 'all'}"
    else:
        if len(configs) != 1:
            parser.error("Give one study, or --compare.")
        cfg = configs[0]
        trials = finished(cfg, fixed, args.storage)
        x = args.x or next(p for p in cfg["study"]["search"]
                           if p not in (PATTERN, SEED, "task.condition"))
        fig = sweep(trials, x, f"{_short(cfg)}{f', {condition} delivery' if condition else ''}")
        name = study_name(cfg) + "".join(f"-{path.rsplit('.', 1)[-1]}{value}"
                                          for path, value in fixed.items())
    paths = save_figure(fig, name, topic="slow-sensing", formats=args.formats)
    print("Saved:", ", ".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
