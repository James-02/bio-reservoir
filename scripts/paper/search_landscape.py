"""The equal-budget searches at 576 traps or units (three SI figures): the macro F1 of every
trial, over the trials and against each tuned setting.

- ``si_search_array``: the array and isolated traps (the array without exchange).
- ``si_search_esn``: the echo state network, read from its clean states and like the array.
- ``si_search_controls``: random tanh features and the bank of leaky integrators.

Each model was tuned by two TPE studies of 150 trials (a: sampler seed 6337, started from the
best setting of a preliminary search; b: seed 6338, from scratch), scored on the search seed
(6337). A star marks the setting carried forward, the best mean on the selection seeds. The
input gain of the colony models is shown as g_u (the product of the code's input factors).

    python scripts/paper/search_landscape.py <out folder> --formats png,pdf

Reads the study databases through ``utils.config.optuna_db`` (set BIORESERVOIR_RESULTS_DIR to
the folder that holds ``optuna/``).
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402
from paper_panels import DESIGN, FIRST, LW, SMALL, STAR, panel, tidy  # noqa: E402

from utils import config  # noqa: E402
from utils.visualisation import STYLE, add_formats_option, save_figure  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.ERROR)

#: g_u per unit of the code's input_scaling (its dde_scaling times cell_coupling).
GAIN = 6.439480590749527e-4 * 12.74579065885262 * 1e3
#: The y axis starts this far below the best trial of the panel's models.
WINDOW = 0.1
#: Tuned settings: code name -> axis label, scale to the paper's units, logarithmic axis.
SETTINGS = {
    "input_scaling": ("Input gain $g_u$ ($10^{-3}$)", GAIN, True),
    "diffusion_rate": ("Exchange rate $\\kappa$ (min$^{-1}$)", 1.0, True),
    "washout_rate": ("Washout $\\mu$", 1.0, False),
    "cell_density": ("Cell density $d$", 1.0, False),
    "esn_leak_rate": ("Leak rate", 1.0, False),
    "esn_spectral_radius": ("Spectral radius", 1.0, True),
    "esn_input_scaling": ("Input gain", 1.0, True),
    "rc_connectivity": ("Recurrent connectivity", 1.0, False),
    "bias_scaling": ("Bias", 1.0, False),
    "rf_input_scaling": ("Input gain", 1.0, True),
    "units": ("Integrators", 1.0, False),
    "tau_max": ("Longest time constant (min)", 1.0, True),
    "band_pass": ("Band-pass differences", 1.0, False),
}
SECOND = {"color": "#CC79A7", "marker": "D"}


class Model:
    """A model's two search studies and the setting carried forward."""

    def __init__(self, label: str, prefix: str, style: dict, rename: dict | None = None):
        self.label, self.style, self.rename = label, style, rename or {}
        self.studies = {w: self._load(f"{prefix}-{w}") for w in "ab"}
        scores = defaultdict(list)
        for t in self._load(f"{prefix}-top"):
            scores[self._key(t)].append(t.value)
        self.chosen = max(scores, key=lambda k: np.mean(scores[k])) if scores else None

    @staticmethod
    def _load(name: str) -> list:
        study = optuna.load_study(study_name=name, storage=f"sqlite:///{config.optuna_db(name)}")
        return [t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]

    def _key(self, trial) -> tuple:
        return tuple(sorted((k.rsplit(".", 1)[-1], v) for k, v in trial.params.items()
                            if not k.endswith("seed")))

    def trials(self):
        return [t for trials in self.studies.values() for t in trials]

    def best(self) -> float:
        return max(t.value for t in self.trials())

    def x(self, trial, name: str) -> float:
        code = self.rename.get(name, name)
        raw = trial.params[f"reservoir.params.{code}"]
        return float(raw) * SETTINGS[name][1]

    def chosen_trial(self):
        return next((t for t in self.trials() if self._key(t) == self.chosen), None)


def trials_panel(ax, models: list[Model]):
    for m in models:
        for which, ls in (("a", "-"), ("b", "--")):
            trials = m.studies[which]
            x = [t.number + 1 for t in trials]
            ax.scatter(x, [t.value for t in trials], s=36, alpha=0.35, edgecolors="none",
                       color=m.style["color"], marker=m.style["marker"])
            ax.plot(x, np.maximum.accumulate([t.value for t in trials]), lw=LW, ls=ls,
                    color=m.style["color"])
    ax.set_xlabel("Trial")
    ax.set_xlim(0, 152)


def setting_panel(ax, models: list[Model], name: str, design: bool):
    label, _, log = SETTINGS[name]
    categorical = False
    for m in models:
        trials = m.trials()
        x = np.array([m.x(t, name) for t in trials])
        if isinstance(trials[0].params.get(f"reservoir.params.{m.rename.get(name, name)}"),
                      bool):
            categorical = True
            x = x + np.random.default_rng(0).uniform(-0.18, 0.18, len(x))
        ax.scatter(x, [t.value for t in trials], s=36, alpha=0.35, edgecolors="none",
                   color=m.style["color"], marker=m.style["marker"])
        best = m.chosen_trial()
        if best is not None:
            star = dict(STAR) if design and m is models[0] else {
                **STAR, "color": m.style["color"], "markeredgecolor": "black"}
            ax.plot(m.x(best, name), best.value, **star)
    if log:
        ax.set_xscale("log")
        low, high = ax.get_xlim()
        subs = (1.0,) if np.log10(high / low) > 2.5 else (1.0, 2.0, 5.0)
        ax.xaxis.set_major_locator(LogLocator(subs=subs))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
    if categorical:
        ax.set_xticks([0, 1], ["No", "Yes"])
        ax.set_xlim(-0.6, 1.6)
    ax.set_xlabel(label)


def finish(axes, models: list[Model], letters: str):
    best = max(m.best() for m in models)
    for ax, letter in zip(axes, letters, strict=False):
        ax.set_ylim(best - WINDOW, best + 0.012)
        tidy(ax)
        panel(ax, letter)


def legend(fig, models: list[Model], design: bool):
    handles = [Line2D([], [], ls="none", ms=11, label=m.label, color=m.style["color"],
                      marker=m.style["marker"]) for m in models]
    handles += [Line2D([], [], color="black", lw=LW, ls="-", label="Best so far, study a"),
                Line2D([], [], color="black", lw=LW, ls="--", label="Best so far, study b")]
    if design:
        handles.append(Line2D([], [], label="Chosen design", **STAR))
    handles.append(Line2D([], [], label="Carried forward" + (" (comparison)" if design else ""),
                          **{**STAR, "color": "white", "markeredgecolor": "black"}))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.93),
               ncol=len(handles), fontsize=SMALL + 1, frameon=False, columnspacing=1.6,
               handletextpad=0.4)


def figure_array() -> plt.Figure:
    models = [Model("Array (design)", "P2-576-lattice", DESIGN),
              Model("Isolated traps", "P2-576-uncoupled", FIRST)]
    fig = plt.figure(figsize=(21, 11))
    grid = fig.add_gridspec(2, 6, hspace=0.45, wspace=0.9)
    axes = [fig.add_subplot(grid[0, 2 * i:2 * i + 2]) for i in range(3)]
    axes += [fig.add_subplot(grid[1, 2 * i + 1:2 * i + 3]) for i in range(2)]
    trials_panel(axes[0], models)
    setting_panel(axes[1], models, "input_scaling", True)
    setting_panel(axes[2], models[:1], "diffusion_rate", True)
    setting_panel(axes[3], models, "washout_rate", True)
    setting_panel(axes[4], models, "cell_density", True)
    finish(axes, models, "ABCDE")
    for ax in (axes[0], axes[3]):
        ax.set_ylabel("Macro F1")
    legend(fig, models, True)
    return fig


def figure_esn() -> plt.Figure:
    rename = {"esn_input_scaling": "input_scaling"}
    models = [Model("ESN, clean states", "P2-576-esn", DESIGN, rename),
              Model("ESN, read like the array", "P2-576-esn-image", FIRST, rename)]
    fig, grid = plt.subplots(2, 3, figsize=(21, 11), gridspec_kw={"hspace": 0.45,
                                                                   "wspace": 0.3})
    axes = list(grid.ravel())
    trials_panel(axes[0], models)
    for ax, name in zip(axes[1:], ("esn_leak_rate", "esn_spectral_radius", "esn_input_scaling",
                                   "rc_connectivity", "bias_scaling"), strict=True):
        setting_panel(ax, models, name, False)
    finish(axes, models, "ABCDEF")
    for ax in (axes[0], axes[3]):
        ax.set_ylabel("Macro F1")
    legend(fig, models, False)
    return fig


def figure_controls() -> plt.Figure:
    features = Model("Random tanh features", "extra-576-random-features", DESIGN,
                     {"rf_input_scaling": "input_scaling"})
    bank = Model("Leaky integrators", "extra-576-leaky-bank", SECOND)
    fig = plt.figure(figsize=(21, 11))
    grid = fig.add_gridspec(2, 8, hspace=0.55, wspace=0.9)
    top = [fig.add_subplot(grid[0, 2 * i + 1:2 * i + 3]) for i in range(3)]
    bottom = [fig.add_subplot(grid[1, 2 * i:2 * i + 2]) for i in range(4)]
    trials_panel(top[0], [features])
    setting_panel(top[1], [features], "rf_input_scaling", False)
    setting_panel(top[2], [features], "bias_scaling", False)
    trials_panel(bottom[0], [bank])
    for ax, name in zip(bottom[1:], ("units", "tau_max", "band_pass"), strict=True):
        setting_panel(ax, [bank], name, False)
    finish(top, [features], "ABC")
    finish(bottom, [bank], "DEFG")
    for ax in (top[0], bottom[0]):
        ax.set_ylabel("Macro F1")
    legend(fig, [features, bank], False)
    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=Path, help="Output folder.")
    add_formats_option(parser)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    with plt.style.context(STYLE):
        for name, draw in (("si_search_array", figure_array), ("si_search_esn", figure_esn),
                           ("si_search_controls", figure_controls)):
            for path in save_figure(draw(), name, folder=args.out, formats=args.formats):
                print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
