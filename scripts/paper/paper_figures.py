"""Draw the figures and write the numbers of the revised paper from the final runs.

Reads only collected results: run folders (``results/runs/<name>/metrics.json``) and study
databases (``results/optuna/<name>.db``), by the names of the final-run configs (design
``p1024`` or ``p1600``; ``P3-p1024-noise``, ``P5-balanced-p1024-headline-s6348``, ...).
``--preview`` reads the names of the 28 Oct preview runs instead (``P3-noise``,
``P5-balanced-headline-s6348``, ...; ``p1600`` there is ``P5-refined-*``).

    python scripts/paper/paper_figures.py --design p1024 numbers
    python scripts/paper/paper_figures.py --design p1024 figures --formats png,pdf
    python scripts/paper/paper_figures.py --design p1024 tables
    python scripts/paper/paper_figures.py --design p1024 images     # simulates 5 beats

``numbers`` writes ``numbers.json`` and ``numbers.tex`` (one ``\\newcommand`` per number) and
prints them; ``figures`` writes Fig. 2-5 and the SI figures; ``tables`` writes the SI tables
as Markdown. Everything goes to ``results/figures/paper/<design>/`` (``--out`` to change it).
A missing run or study is reported and its part left empty, so drafts work on partial results.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import yaml  # noqa: E402

from utils import config  # noqa: E402
from utils.visualisation import (  # noqa: E402
    CLASS_NAMES,
    OKABE_ITO,
    STYLE,
    add_formats_option,
    save_figure,
)

optuna.logging.set_verbosity(optuna.logging.ERROR)

SEEDS = (6348, 6349, 6350, 6351, 6352)
HEADLINE, FOREST, ONE = "images_at:140,187", "images:4", "final"
UNITS = {"p1024": 1024, "p1600": 1600}
LW = 2.6  # line width on the figures as drawn (about 12 in wide, printed at 7 in)
COLOURS = {"array": OKABE_ITO[0], "raw": "#7f7f7f", "forest": OKABE_ITO[1],
           "p1024": OKABE_ITO[0], "p1600": OKABE_ITO[2], "esn": OKABE_ITO[3],
           "other": OKABE_ITO[4]}

# Final-run name -> 28 Oct preview name.
PREVIEW = [(r"^P5-balanced-p1600-(headline|forest)", r"P5-refined-\1"),
           (r"^P5-(\w+)-p1024-", r"P5-\1-"),
           (r"^P3-p1024-observation", "P3-frozen"),
           (r"^P3-(validation-\d)-p1024-", r"P3-\1-p-"),
           (r"^P(\d)-p1024-", r"P\1-"),
           (r"^P2-refine$", "P2-frozen-refine")]


class Results:
    """The collected runs and studies, by final-run name."""

    def __init__(self, preview: bool = False):
        self.preview = preview
        self.missing: set[str] = set()

    def name(self, name: str) -> str:
        if self.preview:
            for pattern, replacement in PREVIEW:
                name = re.sub(pattern, replacement, name)
        return name

    def metrics(self, run: str) -> dict | None:
        path = config.runs_dir() / self.name(run) / "metrics.json"
        if not path.is_file():
            self.missing.add(self.name(run))
            return None
        return json.loads(path.read_text())

    def f1(self, run: str) -> float | None:
        metrics = self.metrics(run)
        return None if metrics is None else float(metrics["summary"]["f1"])

    def seeds(self, prefix: str, seeds=SEEDS) -> list[float]:
        """Macro F1 of the runs ``<prefix>-s<seed>`` that exist, in seed order."""
        return [v for v in (self.f1(f"{prefix}-s{s}") for s in seeds) if v is not None]

    def per_class(self, run: str) -> np.ndarray | None:
        """F1 of each class over all folds (the pooled confusion matrix)."""
        metrics = self.metrics(run)
        if metrics is None:
            return None
        cm = np.asarray(json.loads(metrics["pooled"]["confusion_matrix"])
                        if isinstance(metrics["pooled"]["confusion_matrix"], str)
                        else metrics["pooled"]["confusion_matrix"], dtype=float)
        tp = np.diag(cm)
        return 2 * tp / (cm.sum(axis=0) + cm.sum(axis=1))

    def trials(self, study: str) -> list[dict]:
        """The finished trials of ``study``: their searched values, seed and macro F1."""
        name = self.name(study)
        path = config.optuna_db(name)
        if not Path(path).is_file():
            self.missing.add(name)
            return []
        rows = []
        for trial in optuna.load_study(study_name=name, storage=f"sqlite:///{path}").trials:
            if trial.state.name != "COMPLETE":
                continue
            row = dict(trial.params)
            if "reservoir.seed" not in row and "seed" not in row and "config" in trial.user_attrs:
                row["reservoir.seed"] = yaml.safe_load(trial.user_attrs["config"])["reservoir"]\
                    .get("seed")
            row["f1"] = float(trial.value)
            rows.append(row)
        return rows

    def sweep(self, study: str, x: str, **fixed) -> dict:
        """Macro F1 over seeds at each value of ``x`` (other searched values fixed)."""
        out = defaultdict(list)
        for row in self.trials(study):
            if all(row.get(k.replace("__", ".")) == v for k, v in fixed.items()):
                out[row.get(x)].append(row["f1"])
        return dict(out)


def mean_sd(values) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return float("nan"), float("nan")
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def ordered(points: dict) -> tuple[list, list, list]:
    keys = sorted(k for k in points if k is not None and not isinstance(k, str))
    return keys, [mean_sd(points[k])[0] for k in keys], [mean_sd(points[k])[1] for k in keys]


# ---------------------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------------------

def numbers(r: Results, d: str) -> dict:
    """The numbers the text quotes, for design ``d``."""
    out: dict = {}

    def put(key, values):
        m, s = mean_sd(values)
        out[key] = {"mean": round(m, 4), "sd": round(s, 4), "n": len(list(values))}

    put("headline", r.seeds(f"P5-balanced-{d}-headline"))
    put("forest", r.seeds(f"P5-balanced-{d}-forest"))
    put("full", r.seeds(f"P5-balanced-{d}-full"))
    put("full_forest", r.seeds(f"P5-balanced-{d}-full-forest"))
    for key in ("abstract", "uncoupled", "esn", "esn-image"):
        put(key.replace("-", "_"), r.seeds(f"P5-balanced-{key}"))
    for key in ("ridge", "rf", "histgb", "ridge-wide", "histgb-wide"):
        value = r.f1(f"P5-balanced-raw-{key}")
        out[f"raw_{key.replace('-', '_')}"] = None if value is None else round(value, 4)
    raw, ceiling = out.get("raw_ridge"), out.get("raw_histgb")
    if raw is not None and ceiling is not None:
        gains = [v - raw for v in r.seeds(f"P5-balanced-{d}-headline")]
        put("gain_over_raw_ridge", gains)
        put("gap_closed", [g / (ceiling - raw) for g in gains])
    if out.get("raw_rf") is not None:
        put("forest_minus_raw_rf", [v - out["raw_rf"] for v in r.seeds(f"P5-balanced-{d}-forest")])
    for task in ("unbalanced", "binary"):
        for row in (f"{d}-headline", f"{d}-forest", "abstract", "uncoupled", "esn", "esn-image"):
            value = r.f1(f"P5-{task}-{row}-s{SEEDS[0]}")
            out[f"{task}_{row.replace(f'{d}-', 'p_').replace('-', '_')}"] = \
                None if value is None else round(value, 4)
        for key in ("ridge", "rf", "histgb"):
            value = r.f1(f"P5-{task}-raw-{key}")
            out[f"{task}_raw_{key}"] = None if value is None else round(value, 4)
    for obs in ("c", "e"):
        for key in ("ridge", "lr", "svm", "knn", "rf", "histgb"):
            value = r.f1(f"P4-{d}-{obs}-{key}")
            out[f"readout_{obs}_{key}"] = None if value is None else round(value, 4)
    ablation = {}
    for stem, on, off in (("coupling", "diffusion", "none"), ("delay", 10, 0),
                          ("scramble", False, True)):
        key = {"coupling": "reservoir.params.coupling", "delay": "reservoir.params.delay",
               "scramble": "task.scramble"}[stem]
        for sfx in ("", "-forest"):
            points = r.sweep(f"P6-{d}-ablation-{stem}{sfx}", key)
            if points:
                ablation[f"{stem}{sfx or '-ridge'}"] = {"intact": round(mean_sd(points[on])[0], 4),
                                                        "ablated": round(mean_sd(points[off])[0], 4)}
    out["ablations"] = ablation
    put("surrogate_ridge", r.seeds(f"P6-{d}-surrogate-c", SEEDS[:3]))
    put("surrogate_forest", r.seeds(f"P6-{d}-surrogate-e", SEEDS[:3]))
    for split in ("validation-1", "validation-2"):
        put(f"{split.replace('-', '_')}_headline", r.seeds(f"P3-{split}-{d}-c", SEEDS[:3]))
        put(f"{split.replace('-', '_')}_forest", r.seeds(f"P3-{split}-{d}-e", SEEDS[:3]))
    return out


TEX_PREFIX = {"p1024": "Pten", "p1600": "Psixteen"}


def tex_macros(values: dict, d: str) -> str:
    """One ``\\newcommand`` per number, named after the design and the key: for p1024
    ``\\PtenHeadline`` (the mean) and ``\\PtenHeadlineSd`` (the SD over seeds)."""
    def camel(key):
        digits = ("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine")
        key = re.sub(r"\d", lambda m: f"_{digits[int(m[0])]}_", key)
        return "".join(part[:1].upper() + part[1:] for part in re.split(r"[_\-]+", key) if part)

    lines = [f"% Numbers of design {d}, from scripts/paper/paper_figures.py numbers"]

    def emit(key, value):
        lines.append(f"\\newcommand{{\\{TEX_PREFIX[d]}{camel(key)}}}{{{value}}}")

    for key, value in values.items():
        if isinstance(value, dict) and "mean" in value:
            emit(key, f"{value['mean']:.3f}")
            emit(f"{key}_sd", f"{value['sd']:.3f}")
        elif isinstance(value, float):
            emit(key, f"{value:.3f}")
        elif key == "ablations":
            for name, pair in value.items():
                emit(f"ablation_{name}", f"{pair['ablated']:.3f}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------

def dots(ax, y, values, colour, *, label=None, size=40, horizontal=True):
    """The seeds as dots and their mean as a bar, at position ``y``."""
    if not values:
        return
    m = float(np.mean(values))
    jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else [0.0]
    if horizontal:
        ax.barh(y, m, height=0.6, color=colour, alpha=0.35, label=label)
        ax.scatter(values, y + np.asarray(jitter), s=size, color=colour, zorder=3)
    else:
        ax.bar(y, m, width=0.6, color=colour, alpha=0.35, label=label)
        ax.scatter(y + np.asarray(jitter), values, s=size, color=colour, zorder=3)


def reference_line(ax, value, text, colour="#7f7f7f", style="--", vertical=True):
    """A reference score as a line across the axes, named in the axes' legend."""
    if value is None:
        return
    draw = ax.axvline if vertical else ax.axhline
    draw(value, color=colour, ls=style, lw=LW * 0.8, label=f"{text} ({value:.3f})")


def figure_readouts(r: Results, designs: list[str]):
    """Fig. 2: the array against the raw beat, per readout; the per-class F1 of ridge."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.2, 1]})
    groups = [("Ridge", "raw-ridge", "headline"), ("Forest", "raw-rf", "forest")]
    width = 0.8 / (1 + len(designs))
    for i, (_, raw, row) in enumerate(groups):
        value = r.f1(f"P5-balanced-{raw}")
        if value is not None:
            ax.bar(i - 0.4 + width / 2, value, width=width, color=COLOURS["raw"],
                   label="Raw beat" if i == 0 else None)
        for j, d in enumerate(designs):
            values = r.seeds(f"P5-balanced-{d}-{row}")
            x = i - 0.4 + width * (j + 1.5)
            if values:
                ax.bar(x, np.mean(values), width=width, color=COLOURS[d], alpha=0.6,
                       label=f"Array, {UNITS[d]:,} traps" if i == 0 else None)
                ax.scatter(np.full(len(values), x), values, s=25, color="k", zorder=3)
    svm = {d: r.f1(f"P4-{d}-c-svm") for d in designs}
    for j, d in enumerate(designs):
        if svm[d] is not None:
            ax.bar(2 - 0.4 + width * (j + 1.5), svm[d], width=width, color=COLOURS[d], alpha=0.6,
                   hatch="//")
    reference_line(ax, r.f1("P5-balanced-raw-histgb"), "raw beat, HistGB", vertical=False)
    ax.set_xticks(range(3), ["Ridge\n(linear)", "Random forest", "RBF SVM\n(2 images, 1 seed)"])
    ax.set_ylim(0.6, 0.93)
    ax.set_ylabel("Macro F1 (balanced, 5 folds)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=12,
              frameon=False)
    ax.set_title("a", loc="left", fontweight="bold")

    rows = [("Raw beat, ridge", "P5-balanced-raw-ridge", COLOURS["raw"]),
            *[(f"Array ({UNITS[d]:,}), ridge", f"P5-balanced-{d}-headline-s{SEEDS[0]}",
               COLOURS[d]) for d in designs],
            ("Raw beat, HistGB", "P5-balanced-raw-histgb", "k")]
    width = 0.8 / len(rows)
    for i, (label, run, colour) in enumerate(rows):
        f1 = r.per_class(run)
        if f1 is not None:
            bx.bar(np.arange(len(f1)) - 0.4 + width * (i + 0.5), f1, width=width, color=colour,
                   alpha=0.75 if colour != "k" else 0.9, label=label)
    bx.set_xticks(range(5), [c[:5] + "." if len(c) > 7 else c for c in CLASS_NAMES])
    bx.set_ylim(0.4, 1.0)
    bx.set_ylabel("F1 of the class (seed 6348)")
    bx.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=12,
              frameon=False)
    bx.set_title("b", loc="left", fontweight="bold")
    fig.tight_layout()
    return fig


def figure_ladder(r: Results, d: str):
    """Fig. 3: what each step toward a device costs, for ridge and the forest."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)
    noise = {sfx: r.sweep(f"P3-{d}-noise{sfx}", "features.measurement_noise",
                          features__aggregation=agg)
             for sfx, agg in (("", HEADLINE), ("-forest", FOREST))}
    noref = {sfx: r.sweep(f"P3-{d}-calibration-noref{sfx}", "features.calibration_error",
                          features__aggregation=agg)
             for sfx, agg in (("", HEADLINE), ("-forest", FOREST))}
    for ax, sfx, row, title in ((axes[0], "", "headline", "Ridge, 2 images"),
                                (axes[1], "-forest", "forest", "Random forest, 4 images")):
        steps = [
            ("All states, clean mean", r.seeds(f"P5-balanced-{d}-full{sfx}")),
            ("LuxI images, no noise", noise[sfx].get(0.0, [])),
            ("+ 1% noise (design)", r.seeds(f"P5-balanced-{d}-{row}")),
            ("+ 2% noise", noise[sfx].get(0.02, [])),
            ("+ 5% noise", noise[sfx].get(0.05, [])),
            ("1% noise, 2% calibration\nerror, no reference", noref[sfx].get(0.02, [])),
        ]
        for i, (_, values) in enumerate(steps):
            dots(ax, -i, values, COLOURS[d] if sfx == "" else COLOURS["forest"])
        if sfx == "":
            others = [("ESN, clean mean", r.seeds("P5-balanced-esn"), COLOURS["esn"]),
                      ("ESN, 2 images", r.seeds("P5-balanced-esn-image"), COLOURS["esn"]),
                      ("A′ (abstract), 2 images", r.seeds("P5-balanced-abstract"),
                       COLOURS["other"]),
                      ("Uncoupled array, 2 images", r.seeds("P5-balanced-uncoupled"),
                       COLOURS["other"])]
            for k, (_, values, colour) in enumerate(others):
                dots(ax, -(len(steps) + 0.5 + k), values, colour)
            labels = [s[0] for s in steps] + [o[0] for o in others]
            ax.set_yticks([-i for i in range(len(steps))] +
                          [-(len(steps) + 0.5 + k) for k in range(len(others))], labels)
        reference_line(ax, r.f1("P5-balanced-raw-ridge" if sfx == "" else "P5-balanced-raw-rf"),
                       "raw beat" + (", ridge" if sfx == "" else ", forest"))
        reference_line(ax, r.f1("P5-balanced-raw-histgb"), "raw beat, HistGB", colour="k",
                       style=":")
        ax.set_xlim(0.6, 0.93)
        ax.set_xlabel("Macro F1")
        ax.set_title(title)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=12, frameon=False)
    fig.tight_layout()
    return fig


def figure_controls(r: Results, d: str):
    """Fig. 4: ablations and controls, ridge and forest."""
    n = UNITS[d]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, sfx, obs, title in ((axes[0], "", "c", "Ridge, 2 images"),
                                (axes[1], "-forest", "e", "Random forest, 4 images")):
        def ablated(stem, key, value, sfx=sfx):
            return r.sweep(f"P6-{d}-ablation-{stem}{sfx}", key).get(value, [])
        rf_rows = defaultdict(list)
        for row in r.trials(f"P6-random-features{sfx}"):
            rf_rows[(row["reservoir.params.units"], row["reservoir.params.activation"])].append(
                row["f1"])
        bank = r.sweep(f"P6-leaky-bank{sfx}", "reservoir.params.expansion")
        rows = [
            ("Array (design)", ablated("coupling", "reservoir.params.coupling", "diffusion"),
             COLOURS[d]),
            ("No coupling", ablated("coupling", "reservoir.params.coupling", "none"),
             COLOURS["other"]),
            ("No delay (no oscillation)", ablated("delay", "reservoir.params.delay", 0),
             COLOURS["other"]),
            ("Scrambled time", ablated("scramble", "task.scramble", True), COLOURS["other"]),
            ("Linear surrogate of the array", r.seeds(f"P6-{d}-surrogate-{obs}", SEEDS[:3]),
             COLOURS["esn"]),
            (f"Random tanh features ({n:,})", rf_rows.get((n, "tanh"), []), COLOURS["esn"]),
            ("Random tanh features (8,000)", rf_rows.get((8000, "tanh"), []), COLOURS["esn"]),
            (f"Random linear features ({n:,})", rf_rows.get((n, "linear"), []),
             COLOURS["esn"]),
            ("Leaky-integrator bank", bank.get(None, []), COLOURS["esn"]),
        ]
        for i, (_, values, colour) in enumerate(rows):
            dots(ax, -i, values, colour)
        ax.set_yticks([-i for i in range(len(rows))], [row[0] for row in rows])
        reference_line(ax, r.f1("P5-balanced-raw-ridge" if sfx == "" else "P5-balanced-raw-rf"),
                       "raw beat" + (", ridge" if sfx == "" else ", forest"))
        ax.axvline(0.2, color="#bbbbbb", lw=LW * 0.6, label="chance (0.2)")
        ax.set_xlim(0.15, 0.95)
        ax.set_xlabel("Macro F1")
        ax.set_title(title)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=12, frameon=False)
    fig.tight_layout()
    return fig


def line(ax, points, colour, label, *, marker="o", style="-", band=True):
    x, m, s = ordered(points)
    if not x:
        return
    ax.plot(x, m, style, marker=marker, color=colour, lw=LW, ms=7, label=label)
    if band:
        ax.fill_between(x, np.subtract(m, s), np.add(m, s), color=colour, alpha=0.15, lw=0)


def figure_constraints(r: Results, designs: list[str], d: str, flat_share: Path | None = None):
    """Fig. 5: traps, measurement noise, calibration with and without the reference image,
    and colony variability (with the share of colonies that do not oscillate, from
    ``flat_share.py``'s CSV when given)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5))
    ax = axes[0, 0]
    for e in designs:
        line(ax, r.sweep(f"P6-{e}-units", "reservoir.params.units"), COLOURS[e],
             f"Ridge, 2 images ({e})")
        line(ax, r.sweep(f"P6-{e}-units-forest", "reservoir.params.units"), COLOURS[e],
             f"Forest, 4 images ({e})", marker="s", style="--")
    ax.set_xscale("log")
    ax.set_xlabel("Traps in the array")
    ax.set_ylabel("Macro F1")
    ax.legend(fontsize=11, frameon=False)
    ax.set_title("a", loc="left", fontweight="bold")

    ax = axes[0, 1]
    for agg, label, colour in ((ONE, "1 image", OKABE_ITO[5]), (HEADLINE, "2 images", COLOURS[d]),
                               (FOREST, "4 images", OKABE_ITO[4])):
        line(ax, r.sweep(f"P3-{d}-noise", "features.measurement_noise",
                         features__aggregation=agg), colour, f"Ridge, {label}")
    line(ax, r.sweep(f"P3-{d}-noise-forest", "features.measurement_noise",
                     features__aggregation=FOREST), COLOURS["forest"], "Forest, 4 images",
         marker="s", style="--")
    ax.set_xlabel("Measurement noise (SD, fraction of the signal)")
    ax.set_ylabel("Macro F1")
    ax.legend(fontsize=11, frameon=False)
    ax.set_title("b", loc="left", fontweight="bold")

    ax = axes[1, 0]
    for study, label, colour, style in (
            (f"P3-{d}-calibration", "Ridge, with reference", COLOURS[d], "-"),
            (f"P3-{d}-calibration-noref", "Ridge, no reference", COLOURS[d], ":"),
            (f"P3-{d}-calibration-forest", "Forest, with reference", COLOURS["forest"], "-"),
            (f"P3-{d}-calibration-noref-forest", "Forest, no reference", COLOURS["forest"], ":")):
        agg = FOREST if "forest" in study else HEADLINE
        line(ax, r.sweep(study, "features.calibration_error", features__aggregation=agg), colour,
             label, style=style, marker="s" if "forest" in study else "o")
    ax.set_xlabel("Calibration error (SD of the per-trap gain)")
    ax.set_ylabel("Macro F1")
    ax.legend(fontsize=11, frameon=False)
    ax.set_title("c", loc="left", fontweight="bold")

    ax = axes[1, 1]
    for sfx, label, colour, marker in (("", "Ridge, 2 images", COLOURS[d], "o"),
                                       ("-forest", "Forest, 4 images", COLOURS["forest"], "s")):
        points = r.sweep(f"P3-{d}-heterogeneity{sfx}", "reservoir.params.param_heterogeneity_cv")
        points.update(r.sweep(f"P3-{d}-heterogeneity-64{sfx}",
                              "reservoir.params.param_heterogeneity_cv"))
        line(ax, points, colour, label, marker=marker)
    ax.axvline(0.1, color="#bbbbbb", lw=LW * 0.6, ls="--")
    ax.set_xlabel("Variability of the colonies' kinetics (CV)")
    ax.set_ylabel("Macro F1")
    ax.legend(fontsize=11, frameon=False, loc="lower left")
    if flat_share is not None and flat_share.is_file():
        shares = defaultdict(list)
        for row in csv.DictReader(flat_share.open()):
            if row["design"] == d:
                shares[float(row["cv"])].append(100 * float(row["flat"]))
        x, m, _ = ordered(shares)
        twin = ax.twinx()
        twin.plot(x, m, ":", marker="^", color="#555555", lw=LW, ms=7)
        twin.set_ylabel("Colonies not oscillating (%, dotted)")
        twin.set_ylim(0, 100)
    ax.set_title("d", loc="left", fontweight="bold")
    fig.tight_layout()
    return fig


def figure_realism(r: Results, d: str):
    """SI: the other realism settings, ridge on 2 images (and the forest where it ran)."""
    panels = [("P3-{d}-density", "reservoir.params.heterogeneity_density_cv",
               "Cell-density variability (CV)"),
              ("P3-{d}-dilution", "reservoir.params.heterogeneity_dilution_cv",
               "Dilution variability (CV)"),
              ("P3-{d}-reset", "reservoir.params.reset_noise", "Imperfect reset (CV)"),
              ("P3-{d}-warmup", "reservoir.params.warmup", "Settling time before a beat (min)"),
              ("P3-{d}-image-time", "features.aggregation", "Time of the first image (min)"),
              ("P3-{d}-input-levels", "reservoir.params.input_levels", "Input levels")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (study, x, label) in zip(axes.flat, panels, strict=True):
        points = r.sweep(study.format(d=d), x)
        if x == "features.aggregation":
            points = {int(k.split(":")[1].split(",")[0]): v for k, v in points.items()
                      if k.startswith("images_at")}
        if x == "reservoir.params.input_levels":
            labels = sorted((k for k in points), key=lambda k: (isinstance(k, str), k))
            for i, k in enumerate(labels):
                dots(ax, i, points[k], COLOURS[d], horizontal=False)
            ax.set_xticks(range(len(labels)), [str(k) for k in labels])
        else:
            line(ax, points, COLOURS[d], "Ridge, 2 images")
            if x == "reservoir.params.warmup":
                line(ax, r.sweep(f"P3-{d}-warmup-forest", x), COLOURS["forest"],
                     "Forest, 4 images", marker="s", style="--")
                ax.legend(fontsize=11, frameon=False)
        ax.set_xlabel(label)
        ax.set_ylabel("Macro F1")
    fig.tight_layout()
    return fig


def figure_refinement(r: Results):
    """SI: the refinement grid, the gain of each variant over P1024 on each selection seed."""
    rows = r.trials("P2-refine")
    if not rows:
        return None
    keys = ("reservoir.params.input_scaling", "reservoir.params.diffusion_rate",
            "reservoir.params.units", "features.aggregation")
    base_key = (min(row[keys[0]] for row in rows), min(row[keys[1]] for row in rows), 1024,
                HEADLINE)
    by = defaultdict(dict)
    for row in rows:
        by[tuple(row[k] for k in keys)][row["reservoir.seed"]] = row["f1"]
    base = by[base_key]
    gains = {k: [by[k][s] - base[s] for s in sorted(base) if s in by[k]]
             for k in by if k != base_key}
    order = sorted(gains, key=lambda k: -np.mean(gains[k]))
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, k in enumerate(order):
        colour = COLOURS["p1600"] if k[2] == 1600 else COLOURS["p1024"]
        dots(ax, -i, gains[k], colour)
    ax.set_yticks([-i for i in range(len(order))],
                  [f"gain x{k[0] / base_key[0]:.1f}, coupling {k[1]:.2f}, {k[2]:,} traps, "
                   f"{1 if k[3] == ONE else 2 if k[3] == HEADLINE else 4} images" for k in order])
    ax.axvline(0.005, color="k", ls="--", lw=LW * 0.8)
    ax.axvline(0.0, color="#bbbbbb", lw=LW * 0.6)
    ax.set_xlabel("Gain in macro F1 over P1024 (each dot a selection seed)")
    fig.tight_layout()
    return fig


def figure_images(design: Path, seed: int = SEEDS[0]):
    """What the readout sees: for one typical beat of each class (the pool beat closest to its
    class's mean), the beat as the traps receive it and the array's LuxI images at 46, 93, 140
    and 187 min, each over the trap's image at the start of the beat (the reference). The
    array is stepped through each beat as a run steps it (simulates five beats)."""
    from training.figures import VARIABLES, array_states, build, encoded_input, run_config
    from training.run import load_task
    from utils.preprocessing import standardize_data

    cfg = run_config(design)
    cfg["reservoir"]["seed"] = seed
    X, Y, _ = load_task(cfg["task"])
    X, _ = standardize_data(X, X[:1], scaler_type=str(cfg["task"]["scaler"]))
    X = np.asarray(X, dtype=float).reshape(len(X), -1)
    labels = np.argmax(np.asarray(Y).reshape(len(Y), -1), axis=1)
    side = int(round(np.sqrt(cfg["reservoir"]["params"]["units"])))
    times = (46, 93, 140, 187)
    lux = VARIABLES.index("I")
    reference = build(cfg)
    # the noise-free state every beat starts from, as features.reference reads it
    start = np.asarray(reference.start_state(input_dim=1)).reshape(-1, 4)[:, lux]
    fig, axes = plt.subplots(5, 1 + len(times), figsize=(15, 15),
                             gridspec_kw={"width_ratios": [1.6] + [1] * len(times)})
    images = []
    for k, name in enumerate(CLASS_NAMES):
        members = np.flatnonzero(labels == k)
        index = int(members[np.argmin(((X[members] - X[members].mean(axis=0)) ** 2).sum(1))])
        beat = X[index].reshape(-1, 1)
        states, _ = array_states(cfg, beat, index)
        ax = axes[k, 0]
        ax.plot(np.arange(len(beat)), encoded_input(cfg, beat), color=OKABE_ITO[k], lw=LW)
        for t in times:
            ax.axvline(t, color="#999999", lw=LW * 0.5, ls=":")
        ax.set_ylabel(name, color=OKABE_ITO[k])
        ax.set_xlim(0, len(beat))
        if k == 4:
            ax.set_xlabel("Time in the beat (min)")
        for j, t in enumerate(times):
            image = np.log2(states[t - 1, :, lux] / start).reshape(side, side)
            images.append(axes[k, j + 1].imshow(image, cmap="RdBu_r", interpolation="nearest"))
            axes[k, j + 1].set_xticks([])
            axes[k, j + 1].set_yticks([])
            if k == 0:
                axes[k, j + 1].set_title(f"{t} min")
    values = np.concatenate([np.ravel(im.get_array()) for im in images])
    values = values[np.isfinite(values)]
    span = max(np.percentile(np.abs(values), 98), 1e-3) if len(values) else 1.0
    norm = matplotlib.colors.Normalize(-span, span)
    for im in images:
        im.set_norm(norm)
    fig.colorbar(images[0], ax=axes[:, 1:], shrink=0.6,
                 label="log$_2$ of LuxI over the reference image")
    return fig


# ---------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------

def fmt(values) -> str:
    if values is None or (isinstance(values, list) and not values):
        return "–"
    if isinstance(values, float):
        return f"{values:.3f}"
    m, s = mean_sd(values)
    return f"{m:.3f} ± {s:.3f}" if len(values) > 1 else f"{m:.3f}"


def tables(r: Results, d: str) -> str:
    out = [f"# SI tables, design {d}\n"]
    out.append("## Ladder on the three tasks (macro F1; balanced: mean ± SD over 5 seeds; "
               "unbalanced and binary: seed 6348)\n")
    out.append("| Row | Balanced | Unbalanced | Binary |\n|---|---|---|---|")
    for label, row in ((f"Array, ridge, 2 images ({d})", f"{d}-headline"),
                       (f"Array, forest, 4 images ({d})", f"{d}-forest"),
                       ("Array, clean mean, ridge", f"{d}-full"),
                       ("Array, clean mean, forest", f"{d}-full-forest"),
                       ("A′ (abstract), ridge", "abstract"), ("Uncoupled, ridge", "uncoupled"),
                       ("ESN, clean, ridge", "esn"), ("ESN, 2 images, ridge", "esn-image")):
        out.append(f"| {label} | {fmt(r.seeds(f'P5-balanced-{row}'))} | "
                   f"{fmt(r.f1(f'P5-unbalanced-{row}-s{SEEDS[0]}'))} | "
                   f"{fmt(r.f1(f'P5-binary-{row}-s{SEEDS[0]}'))} |")
    for label, key in (("Raw beat, ridge", "ridge"), ("Raw beat, forest", "rf"),
                       ("Raw beat, HistGB", "histgb"), ("Raw beat, ridge (wide grid)", "ridge-wide"),
                       ("Raw beat, HistGB (wide grid)", "histgb-wide")):
        out.append(f"| {label} | " + " | ".join(fmt(r.f1(f"P5-{t}-raw-{key}"))
                                                for t in ("balanced", "unbalanced", "binary"))
                   + " |")
    out.append("\n## Readouts refitted on the saved images (seed 6348)\n")
    out.append("| Readout | 2 images | 4 images |\n|---|---|---|")
    for key, label in (("ridge", "Ridge"), ("lr", "Logistic regression"), ("svm", "RBF SVM"),
                       ("knn", "kNN"), ("rf", "Random forest"), ("histgb", "HistGB")):
        out.append(f"| {label} | {fmt(r.f1(f'P4-{d}-c-{key}'))} | "
                   f"{fmt(r.f1(f'P4-{d}-e-{key}'))} |")
    out.append("\n## Held-out validation splits (4 classes: N, S, V, Q)\n")
    out.append("| Model | Validation 1 | Validation 2 |\n|---|---|---|")
    for label, key in ((f"Array, ridge, 2 images ({d})", f"{d}-c"),
                       (f"Array, ridge, 1 image ({d})", f"{d}-a"),
                       (f"Array, forest, 4 images ({d})", f"{d}-e"),
                       ("A′", "abstract"), ("Uncoupled", "uncoupled"), ("ESN, clean", "esn"),
                       ("ESN, 2 images", "esn-image")):
        out.append(f"| {label} | " + " | ".join(fmt(r.seeds(f"P3-{s}-{key}"))
                                                for s in ("validation-1", "validation-2")) + " |")
    out.append("\n## Per-class F1 (seed 6348, all folds)\n")
    out.append("| Run | " + " | ".join(CLASS_NAMES) + " |\n|---|" + "---|" * len(CLASS_NAMES))
    for label, run in (("Raw beat, ridge", "P5-balanced-raw-ridge"),
                       (f"Array, ridge ({d})", f"P5-balanced-{d}-headline-s{SEEDS[0]}"),
                       (f"Array, forest ({d})", f"P5-balanced-{d}-forest-s{SEEDS[0]}"),
                       ("Raw beat, HistGB", "P5-balanced-raw-histgb")):
        f1 = r.per_class(run)
        out.append(f"| {label} | " + (" | ".join(f"{v:.3f}" for v in f1) if f1 is not None
                                      else " | ".join("–" for _ in CLASS_NAMES)) + " |")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("numbers", "figures", "tables", "images", "all"),
                        help="images simulates five beats of the design; the others only read "
                             "results.")
    parser.add_argument("--design", choices=sorted(UNITS), default="p1024",
                        help="The design the figures and numbers are for (default p1024).")
    parser.add_argument("--compare", action="store_true",
                        help="Draw both designs where a figure can show both.")
    parser.add_argument("--preview", action="store_true",
                        help="Read the names of the 28 Oct preview runs.")
    parser.add_argument("--out", type=Path, help="Output folder (default "
                        "results/figures/paper/<design>).")
    parser.add_argument("--flat-share", type=Path,
                        help="CSV of scripts/paper/flat_share.py for Fig. 5d.")
    add_formats_option(parser)
    args = parser.parse_args(argv)
    r = Results(preview=args.preview)
    d = args.design
    out = args.out or config.figures_dir(f"paper/{d}")
    out.mkdir(parents=True, exist_ok=True)
    designs = sorted(UNITS) if args.compare else [d]
    if args.command in ("numbers", "all"):
        values = numbers(r, d)
        (out / "numbers.json").write_text(json.dumps(values, indent=1))
        (out / "numbers.tex").write_text(tex_macros(values, d))
        print(json.dumps(values, indent=1))
    if args.command in ("tables", "all"):
        text = tables(r, d)
        (out / "si_tables.md").write_text(text)
        print(text)
    if args.command in ("figures", "all"):
        with plt.style.context(STYLE):
            drawn = {"fig2_readouts": figure_readouts(r, designs),
                     "fig3_ladder": figure_ladder(r, d),
                     "fig4_controls": figure_controls(r, d),
                     "fig5_constraints": figure_constraints(r, designs, d, args.flat_share),
                     "si_realism": figure_realism(r, d),
                     "si_refinement": figure_refinement(r)}
            for name, fig in drawn.items():
                if fig is not None:
                    for path in save_figure(fig, name, folder=out, formats=args.formats):
                        print(path)
    if args.command == "images":
        with plt.style.context(STYLE):
            fig = figure_images(Path("experiments/v0.2/designs") / f"{d}.yaml")
            for path in save_figure(fig, "fig1_images", folder=out, formats=args.formats):
                print(path)
    if r.missing:
        print(f"\nNot found ({len(r.missing)}): " + ", ".join(sorted(r.missing)[:40])
              + (" ..." if len(r.missing) > 40 else ""), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
