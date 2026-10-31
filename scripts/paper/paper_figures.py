"""Draw the figures and tables and write the numbers of the revised paper, for design p576.

Reads only collected results: run folders (``runs/<name>/metrics.json``) and study databases
(``optuna/optuna-<name>.db``) in the results folder (``BIORESERVOIR_RESULTS_DIR``), by the
names of the final-run configs: ``P5-balanced-p576-headline-s6348``, ``P3-p576-noise``,
``extra-576-connectivity``, ... (``--design p576r``: the alternative design's runs).

    python scripts/paper/paper_figures.py numbers --flat-share flat_share.csv
    python scripts/paper/paper_figures.py tables
    python scripts/paper/paper_figures.py figures --oscillation oscillation_map.csv \\
        --flat-share flat_share.csv --formats png,pdf
    python scripts/paper/paper_figures.py images      # simulates five beats: Fig. 1d
    python scripts/paper/paper_figures.py dynamics    # simulates the design: Fig. 2a

``numbers`` writes ``numbers.json`` and ``numbers.tex`` (one ``\\newcommand`` per number) and
prints them; ``tables`` writes Tables 2-3 and the SI tables as Markdown; ``figures`` draws
Fig. 2-6 and the SI figures. Fig. 2a needs ``dynamics.npz`` in the output folder (written by
``dynamics``), Fig. 2b the CSV of ``oscillation_map.py``, and Fig. 6b's share of flat colonies
the CSV of ``flat_share.py``; without them those panels stay empty. Everything goes to
``results/figures/paper/<design>/`` (``--out`` to change it). A missing run or study is
reported and its part left empty, so drafts work on partial results. ``images`` and
``dynamics`` read the design file (``experiments/v0.2/designs/<design>.yaml``) and run from
the repository root.
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
from matplotlib.ticker import FuncFormatter  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

from utils import config  # noqa: E402
from utils.visualisation import (  # noqa: E402
    CLASS_COLOURS,
    CLASS_NAMES,
    OKABE_ITO,
    STYLE,
    add_formats_option,
    save_figure,
)

optuna.logging.set_verbosity(optuna.logging.ERROR)

SEEDS = (6348, 6349, 6350, 6351, 6352)
HEADLINE, FOREST, ONE = "images_at:140,187", "images:4", "final"
#: Each design and the prefix of its LaTeX macros.
DESIGNS = {"p576": "P", "p576r": "Pr"}
#: The flows of the flow x density map (studies extra-576-map-flow-<flow>).
FLOWS = (0.3, 0.4, 0.6, 0.9, 1.2)
FLOW, DENSITY = "reservoir.params.washout_rate", "reservoir.params.cell_density"
#: Trap arrays built by Prindle et al. (2012): 416 and 500 traps; the largest 12,224.
BUILT, LARGEST = (416, 500), 12224
#: Periods measured by Danino et al. (2010) over their flow rates (min).
DANINO_PERIODS = (52, 90)
LW = 3.2  # line width on the figures as drawn (up to 18 in wide, printed at 7 in)
THIN = 2.7  # reference lines and single colonies: 1 pt once printed
SMALL = 13  # legends and labels inside the axes (about 5.5 pt printed)
COLOURS = {"array": OKABE_ITO[0], "forest": OKABE_ITO[1], "svm": OKABE_ITO[2],
           "raw": "#7f7f7f", "digital": OKABE_ITO[3], "isolated": OKABE_ITO[4],
           "alternative": "#8c564b", "ideal": "#000000"}
READOUTS = (("ridge", "Ridge"), ("lr", "Logistic\nregression"), ("knn", "kNN"),
            ("rf", "Random\nforest"), ("histgb", "HistGB"), ("svm", "RBF SVM"))
#: The raw-beat run of each readout, with the readout's settings on the array.
RAW = {"ridge": "P5-balanced-raw-ridge", "lr": "extra-576-raw-lr",
       "knn": "extra-576-raw-knn", "rf": "P5-balanced-raw-rf",
       "histgb": "P5-balanced-raw-histgb", "svm": "extra-576-raw-svm"}
#: Comparison models tuned at 576 units with the array's budget, each read by ridge.
TUNED = {"uncoupled-576": "Isolated traps (no AHL exchange)",
         "esn-image-576": "ESN, read like the array",
         "esn-576": "ESN, clean states",
         "abstract-576": "Submitted model (corrected)"}
#: Controls tuned like the models (two TPE studies), their best point on the reporting seeds.
CONTROLS = {"random_features": "extra-576-random-features-report",
            "fading_memories": "extra-576-leaky-bank-report"}


class Results:
    """The collected runs and studies, by final-run name."""

    def __init__(self):
        self.missing: set[str] = set()

    def metrics(self, run: str) -> dict | None:
        path = config.runs_dir() / run / "metrics.json"
        if not path.is_file():
            self.missing.add(run)
            return None
        return json.loads(path.read_text())

    def settings(self, run: str) -> dict | None:
        """The config a run ran with (its ``config.yaml``)."""
        path = config.runs_dir() / run / "config.yaml"
        return yaml.safe_load(path.read_text()) if path.is_file() else None

    def f1(self, run: str) -> float | None:
        metrics = self.metrics(run)
        return None if metrics is None else float(metrics["summary"]["f1"])

    def by_seed(self, prefix: str, seeds=SEEDS) -> dict[int, float]:
        """Macro F1 of the runs ``<prefix>-s<seed>`` that exist, by seed."""
        values = {s: self.f1(f"{prefix}-s{s}") for s in seeds}
        return {s: v for s, v in values.items() if v is not None}

    def seeds(self, prefix: str, seeds=SEEDS) -> list[float]:
        """Macro F1 of the runs ``<prefix>-s<seed>`` that exist, in seed order."""
        return list(self.by_seed(prefix, seeds).values())

    def per_class(self, run: str) -> np.ndarray | None:
        """F1 of each class over all folds (the pooled confusion matrix)."""
        metrics = self.metrics(run)
        if metrics is None:
            return None
        cm = metrics["pooled"]["confusion_matrix"]
        cm = np.asarray(json.loads(cm) if isinstance(cm, str) else cm, dtype=float)
        tp = np.diag(cm)
        return 2 * tp / (cm.sum(axis=0) + cm.sum(axis=1))

    def trials(self, study: str) -> list[dict]:
        """The finished trials of ``study``: their searched values, seed and macro F1."""
        path = config.optuna_db(study)
        if not Path(path).is_file():
            self.missing.add(study)
            return []
        rows = []
        for trial in optuna.load_study(study_name=study, storage=f"sqlite:///{path}").trials:
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
        """Macro F1 over seeds at each value of ``x`` (other searched values fixed; write
        ``features.aggregation`` as ``features__aggregation``)."""
        out = defaultdict(list)
        for row in self.trials(study):
            if all(row.get(k.replace("__", ".")) == v for k, v in fixed.items()):
                out[row.get(x)].append(row["f1"])
        return dict(out)

    def grid(self, study: str, *keys: str) -> dict[tuple, dict[int, float]]:
        """Macro F1 by the values of ``keys``, then by seed."""
        out = defaultdict(dict)
        for row in self.trials(study):
            seed = row.get("reservoir.seed", row.get("seed"))
            out[tuple(row.get(k) for k in keys)][seed] = row["f1"]
        return dict(out)

    def report(self, study: str) -> dict[int, float]:
        """Macro F1 by seed of a study that reruns one point on the reporting seeds."""
        return {seed: values[seed] for values in self.grid(study).values() for seed in values}


def mean_sd(values) -> tuple[float, float]:
    values = np.asarray(list(values), dtype=float)
    if not len(values):
        return float("nan"), float("nan")
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def ordered(points: dict) -> tuple[list, list, list]:
    keys = sorted(k for k in points if k is not None and not isinstance(k, str))
    return keys, [mean_sd(points[k])[0] for k in keys], [mean_sd(points[k])[1] for k in keys]


def paired(a: dict, b: dict | float) -> list[float]:
    """``a - b`` seed by seed (``b`` a single score: the same for every seed)."""
    if isinstance(b, dict):
        return [a[s] - b[s] for s in sorted(a) if s in b]
    return [a[s] - b for s in sorted(a)]


def scaled(points: dict, factor: float = 100.0) -> dict:
    """The points with their x values multiplied by ``factor`` (fractions to percent)."""
    return {k * factor: v for k, v in points.items() if k is not None and not isinstance(k, str)}


# ---------------------------------------------------------------------------------------
# What the figures, tables and numbers read
# ---------------------------------------------------------------------------------------

def readout(r: Results, d: str, obs: str, key: str) -> dict[int, float]:
    """Macro F1 by seed of readout ``key`` refitted on the saved features ``obs`` (``c``: two
    images, ``e``: four) of each reporting seed: Phase 4 ran seed 6348, the extra runs of
    p576 the other seeds."""
    values = {}
    first = r.f1(f"P4-{d}-{obs}-{key}")
    if first is not None:
        values[SEEDS[0]] = first
    if d == "p576":
        values.update(r.by_seed(f"extra-576-readout-{obs}-{key}", SEEDS[1:]))
    return values


def ablation(r: Results, d: str, stem: str, value, forest: bool = False) -> list[float]:
    key = {"coupling": "reservoir.params.coupling", "delay": "reservoir.params.delay",
           "scramble": "task.scramble"}[stem]
    return r.sweep(f"P6-{d}-ablation-{stem}{'-forest' if forest else ''}", key).get(value, [])


def untuned_control(r: Results, name: str, forest: bool = False) -> list[float]:
    """The controls at their standard settings (Phase 6): 576 random tanh features, or the
    bank of leaky integrators of 24 timescales."""
    sfx = "-forest" if forest else ""
    if name == "random_features":
        return [row["f1"] for row in r.trials(f"P6-random-features{sfx}")
                if row["reservoir.params.units"] == 576
                and row["reservoir.params.activation"] == "tanh"]
    return r.sweep(f"P6-leaky-bank{sfx}", "reservoir.params.expansion").get(None, [])


def traps(r: Results, d: str, forest: bool = False) -> dict[int, list[float]]:
    """Macro F1 at each array size: Phase 6 (100-4,096 traps) and, for ridge on p576, the
    trap-ceiling studies above 4,096 traps."""
    points = r.sweep(f"P6-{d}-units{'-forest' if forest else ''}", "reservoir.params.units")
    if d == "p576" and not forest:
        studies = [f"trap-ceiling-576-units-{n}" for n in (6400, 9216, 12544)]
        studies += [f"trap-ceiling-576-units-{n}-s{s}" for n in (16384, 25600, 36864)
                    for s in SEEDS]
        for study in studies:
            for size, values in r.sweep(study, "reservoir.params.units").items():
                points.setdefault(size, []).extend(values)
    return points


def ideal_traps(r: Results, d: str) -> dict[int, list[float]]:
    """Ridge on the clean mean of all states (no imaging) at 576 traps and, for p576, at
    4,096 and 16,384."""
    points = {576: r.seeds(f"P5-balanced-{d}-full")}
    if d == "p576":
        studies = ["trap-ceiling-576-ideal-4096"]
        studies += [f"trap-ceiling-576-ideal-16384-s{s}" for s in SEEDS]
        for study in studies:
            for size, values in r.sweep(study, "reservoir.params.units").items():
                points.setdefault(size, []).extend(values)
    return {k: v for k, v in points.items() if v}


def f1_map(r: Results) -> tuple[list[float], list[float], np.ndarray]:
    """Mean macro F1 over the reporting seeds on the flow x density grid (rows: densities)."""
    cells = defaultdict(list)
    for flow in FLOWS:
        for row in r.trials(f"extra-576-map-flow-{flow:g}"):
            cells[(row[FLOW], row[DENSITY])].extend([row["f1"]])
    if not cells:
        return [], [], np.zeros((0, 0))
    flows, densities = sorted({k[0] for k in cells}), sorted({k[1] for k in cells})
    grid = np.array([[np.mean(cells[(f, x)]) if cells.get((f, x)) else np.nan for f in flows]
                     for x in densities])
    return flows, densities, grid


def oscillation_map(path: Path | None, d: str) -> tuple[list, list, np.ndarray, np.ndarray]:
    """Share of oscillating colonies (%) and median period (min) on the flow x density grid,
    from the CSV of ``oscillation_map.py``."""
    if path is None or not path.is_file():
        return [], [], np.zeros((0, 0)), np.zeros((0, 0))
    rows = [row for row in csv.DictReader(path.open()) if row["design"] == d]
    flows = sorted({float(row["flow"]) for row in rows})
    densities = sorted({float(row["density"]) for row in rows})
    share = np.full((len(densities), len(flows)), np.nan)
    period = share.copy()
    for i, x in enumerate(densities):
        for j, f in enumerate(flows):
            part = [row for row in rows if float(row["flow"]) == f
                    and float(row["density"]) == x]
            share[i, j] = 100 * np.mean([float(row["oscillating"]) for row in part])
            periods = [float(row["period"]) for row in part if row["period"] not in ("", "nan")]
            period[i, j] = np.median(periods) if periods else np.nan
    return flows, densities, share, period


def flat_shares(path: Path | None, d: str) -> dict[float, list[float]]:
    """Colonies that do not oscillate (%) at each kinetic CV, from ``flat_share.py``'s CSV."""
    shares = defaultdict(list)
    if path is not None and path.is_file():
        for row in csv.DictReader(path.open()):
            if row["design"] == d:
                shares[float(row["cv"])].append(100 * float(row["flat"]))
    return dict(shares)


def design_point(r: Results, d: str) -> tuple[float, float] | None:
    """The design's flow and cell density, from the config of its headline run."""
    cfg = r.settings(f"P5-balanced-{d}-headline-s{SEEDS[0]}") or {}
    params = cfg.get("reservoir", {}).get("params", {})
    if "washout_rate" not in params or "cell_density" not in params:
        return None
    return float(params["washout_rate"]), float(params["cell_density"])


def image_times(r: Results, d: str) -> tuple[dict[int, list[float]], list[float]]:
    """Ridge on two images by the time of the earlier one (the later at 187 min), and ridge
    on the final image alone."""
    points = r.sweep(f"P3-{d}-image-time", "features.aggregation")
    times = {int(k.split(":")[1].split(",")[0]): v for k, v in points.items()
             if str(k).startswith("images_at")}
    return times, points.get(ONE, [])


# ---------------------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------------------

def numbers(r: Results, d: str, flat_share: Path | None = None,
            dynamics: Path | None = None) -> dict:
    """The numbers the text quotes, for design ``d``."""
    out: dict = {}

    def put(key, values):
        values = list(values)
        if values:
            m, s = mean_sd(values)
            out[key] = {"mean": round(m, 4), "sd": round(s, 4), "n": len(values)}

    def one(key, value):
        if value is not None and np.isfinite(value):
            out[key] = round(float(value), 4)

    headline = r.by_seed(f"P5-balanced-{d}-headline")
    put("headline", headline.values())
    put("forest", r.seeds(f"P5-balanced-{d}-forest"))
    put("ideal", r.seeds(f"P5-balanced-{d}-full"))
    put("ideal_forest", r.seeds(f"P5-balanced-{d}-full-forest"))
    for obs, images in (("c", "two"), ("e", "four")):
        for key, _ in READOUTS:
            put(f"readout_{images}_{key}", readout(r, d, obs, key).values())
    raw = {key: r.f1(run) for key, run in RAW.items()}
    for key, value in raw.items():
        one(f"raw_{key}", value)
    for key in ("ridge-wide", "histgb-wide"):
        one(f"raw_{key.replace('-', '_')}", r.f1(f"P5-balanced-raw-{key}"))
    for key in TUNED:
        other = r.by_seed(f"P5-balanced-{key}")
        put(key.replace("-", "_"), other.values())
        put(f"headline_minus_{key.replace('-', '_')}", paired(headline, other))
    for name, study in CONTROLS.items():
        tuned = r.report(study)
        put(name, tuned.values())
        put(f"headline_minus_{name}", paired(headline, tuned))
        put(f"{name}_untuned", untuned_control(r, name))
    if headline and raw["ridge"] is not None:
        put("gain_over_raw_ridge", paired(headline, raw["ridge"]))
        put("ideal_gain_over_raw_ridge", [v - raw["ridge"]
                                          for v in r.seeds(f"P5-balanced-{d}-full")])
        if raw["histgb"] is not None:
            put("gap_closed", [100 * g / (raw["histgb"] - raw["ridge"])
                               for g in paired(headline, raw["ridge"])])
    forest = r.seeds(f"P5-balanced-{d}-forest")
    if forest and raw["rf"] is not None:
        put("forest_minus_raw_rf", [v - raw["rf"] for v in forest])
        one("forest_minus_raw_rf_worst", min(v - raw["rf"] for v in forest))
    svm = readout(r, d, "c", "svm")
    if svm and raw["histgb"] is not None:
        put("svm_minus_raw_histgb", paired(svm, raw["histgb"]))
        put("svm_minus_headline", paired(svm, headline))
    for task in ("unbalanced", "binary"):
        for row in ("headline", "forest"):
            one(f"{task}_{row}", r.f1(f"P5-{task}-{d}-{row}-s{SEEDS[0]}"))
        for key in ("ridge", "rf", "histgb"):
            one(f"{task}_raw_{key}", r.f1(f"P5-{task}-raw-{key}"))
        for key in TUNED:
            one(f"{task}_{key.replace('-', '_')}", r.f1(f"P5-{task}-{key}-s{SEEDS[0]}"))
    for split in ("validation-1", "validation-2"):
        tag = split.replace("-", "_")
        put(f"{tag}_headline", r.seeds(f"P3-{split}-{d}-c", SEEDS[:3]))
        put(f"{tag}_forest", r.seeds(f"P3-{split}-{d}-e", SEEDS[:3]))
        put(f"{tag}_one_image", r.seeds(f"P3-{split}-{d}-a", SEEDS[:3]))
    for stem, on, off in (("coupling", "diffusion", "none"), ("delay", 10, 0),
                          ("scramble", False, True)):
        for forest_ in (False, True):
            tag = f"ablation_{stem}{'_forest' if forest_ else ''}"
            put(f"{tag}_intact", ablation(r, d, stem, on, forest_))
            put(tag, ablation(r, d, stem, off, forest_))
    put("imitation", r.seeds(f"P6-{d}-surrogate-c", SEEDS[:3]))
    put("imitation_forest", r.seeds(f"P6-{d}-surrogate-e", SEEDS[:3]))
    connectivity = r.grid("extra-576-connectivity", "reservoir.params.input_connectivity",
                          "reservoir.params.coupling") if d == "p576" else {}
    for share in sorted({k[0] for k in connectivity}):
        on, off = connectivity.get((share, "diffusion"), {}), connectivity.get((share, "none"), {})
        tag = f"induced_{round(100 * share)}"
        put(f"{tag}_exchange", on.values())
        put(f"{tag}_isolated", off.values())
        put(f"{tag}_exchange_adds", paired(on, off))
    curve, forest_curve = traps(r, d), traps(r, d, forest=True)
    for size in (100, 256, 576, 1024, 4096):
        put(f"traps_{size}", curve.get(size, []))
        put(f"traps_forest_{size}", forest_curve.get(size, []))
    put("traps_plateau", [v for size, values in curve.items() if size >= 9216 for v in values])
    for size, values in ideal_traps(r, d).items():
        put(f"traps_ideal_{size}", values)
    for agg, tag in ((ONE, "one_image"), (HEADLINE, "two_images"), (FOREST, "four_images")):
        points = r.sweep(f"P3-{d}-noise", "features.measurement_noise", features__aggregation=agg)
        for level in (0.0, 0.01, 0.02, 0.05, 0.1):
            put(f"noise_{tag}_{round(100 * level)}", points.get(level, []))
    times, final = image_times(r, d)
    if times:
        means = [mean_sd(v)[0] for v in times.values()]
        one("image_time_lowest", min(means))
        one("image_time_highest", max(means))
        one("image_time_first", min(times))
        one("image_time_last", max(times))
    put("image_final_only", final)
    for study, tag in ((f"P3-{d}-calibration", "reference"),
                       (f"P3-{d}-calibration-noref", "no_reference")):
        points = r.sweep(study, "features.calibration_error", features__aggregation=HEADLINE)
        for level in (0.02, 0.05):
            put(f"calibration_{tag}_{round(100 * level)}", points.get(level, []))
    kinetic = r.sweep(f"P3-{d}-heterogeneity", "reservoir.params.param_heterogeneity_cv")
    kinetic.update(r.sweep(f"P3-{d}-heterogeneity-64", "reservoir.params.param_heterogeneity_cv"))
    for cv in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
        put(f"kinetic_cv_{round(100 * cv)}", kinetic.get(cv, []))
    for cv, values in flat_shares(flat_share, d).items():
        if cv in (0.1, 0.2, 0.3, 0.5):
            put(f"flat_cv_{round(100 * cv)}", values)
    for study, key, tag in ((f"P3-{d}-density", "density", "density_spread"),
                            (f"P3-{d}-dilution", "dilution", "washout_spread")):
        points = r.sweep(study, f"reservoir.params.heterogeneity_{key}_cv")
        for cv in (0.1, 0.2, 0.3):
            put(f"{tag}_{round(100 * cv)}", points.get(cv, []))
    if d == "p576":
        put("alternative_headline", r.seeds("P5-balanced-p576r-headline"))
        points = r.sweep("P3-p576r-density", "reservoir.params.heterogeneity_density_cv")
        for cv in (0.1, 0.2, 0.3):
            put(f"alternative_density_spread_{round(100 * cv)}", points.get(cv, []))
        put("alternative_kinetic_cv_0", r.sweep("P3-p576r-heterogeneity",
                                                "reservoir.params.param_heterogeneity_cv")
            .get(0.0, []))
    put("warmup_at_rest", r.sweep(f"P3-{d}-warmup-input", "reservoir.params.warmup_input")
        .get("rest", []))
    layout = r.sweep(f"P6-{d}-gaussian", "reservoir.params.interaction_diameters")
    if layout:
        means = [mean_sd(v)[0] for v in layout.values()]
        one("gaussian_lowest", min(means))
        one("gaussian_highest", max(means))
    flows, densities, grid = f1_map(r) if d == "p576" else ([], [], np.zeros((0, 0)))
    if grid.size:
        one("map_best", np.nanmax(grid))
        one("map_fastest_flow_best", np.nanmax(grid[:, flows.index(max(flows))]))
    point = design_point(r, d)
    if point is not None:
        one("flow", point[0])
        one("density", point[1])
    if dynamics is not None and dynamics.is_file():
        one("period", float(np.load(dynamics)["period"]))
    return out


def tex_value(key: str, value: float) -> str:
    """Shares in percent with one decimal (flat colonies) or none (gap closed); scores and
    settings with three decimals."""
    if key.startswith("flat_"):
        return f"{value:.1f}"
    if key.startswith(("gap_closed", "period")) or key.startswith("image_time_first") \
            or key.startswith("image_time_last"):
        return f"{value:.0f}"
    return f"{value:.3f}"


def tex_macros(values: dict, d: str) -> str:
    """One ``\\newcommand`` per number, named after the design and the key: for p576
    ``\\PHeadline`` (the mean) and ``\\PHeadlineSd`` (the SD over seeds)."""
    def camel(key):
        digits = ("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine")
        key = re.sub(r"\d", lambda m: f"_{digits[int(m[0])]}_", key)
        return "".join(part[:1].upper() + part[1:] for part in re.split(r"[_\-]+", key) if part)

    lines = [f"% Numbers of design {d}, from scripts/paper/paper_figures.py numbers"]
    for key, value in values.items():
        name = f"\\{DESIGNS[d]}{camel(key)}"
        if isinstance(value, dict):
            lines.append(f"\\newcommand{{{name}}}{{{tex_value(key, value['mean'])}}}")
            lines.append(f"\\newcommand{{{name}Sd}}{{{tex_value(key, value['sd'])}}}")
        else:
            lines.append(f"\\newcommand{{{name}}}{{{tex_value(key, value)}}}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------------------

def seed_dots(ax, y, values, colour, *, marker="o", label=None, size=50, half=0.3,
              horizontal=True):
    """The seeds as dots around position ``y`` and their mean as a short bar across them (one
    value: a diamond); ``horizontal``: the scores along x."""
    values = list(values)
    if not values:
        return
    m = float(np.mean(values))
    if len(values) == 1:
        ax.scatter(*(([m], [y]) if horizontal else ([y], [m])), s=size * 1.6, color=colour,
                   marker="D", zorder=3, label=label)
        return
    across = [y - half, y + half]
    ax.plot(*(([m, m], across) if horizontal else (across, [m, m])), color=colour,
            lw=LW * 1.4, solid_capstyle="butt", label=label)
    jitter = y + np.linspace(-half * 0.55, half * 0.55, len(values))
    ax.scatter(*((values, jitter) if horizontal else (jitter, values)), s=size, color=colour,
               marker=marker, zorder=3, edgecolor="white", linewidth=0.6)


def line(ax, points, colour, label, *, marker="o", style="-", band=True):
    """Mean over seeds at each x, with a band of one SD."""
    x, m, s = ordered(points)
    if not x:
        return
    ax.plot(x, m, style, marker=marker, color=colour, lw=LW, ms=7, label=label)
    if band:
        ax.fill_between(x, np.subtract(m, s), np.add(m, s), color=colour, alpha=0.15, lw=0)


def panel(ax, letter: str):
    ax.set_title(letter, loc="left", fontweight="bold")


def plain_log_ticks(axis, ticks=None):
    """Numbers instead of powers of ten on a log axis (``ticks``: where)."""
    if ticks is not None:
        axis.set_ticks(ticks)
        axis.set_minor_locator(matplotlib.ticker.NullLocator())
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))


def legend(ax, **kwargs):
    kwargs.setdefault("fontsize", SMALL)
    kwargs.setdefault("frameon", False)
    ax.legend(**kwargs)


def mark_design(ax, flows, densities, point):
    """A star at the design's flow and density, on a grid drawn with ``imshow``."""
    if point is None or not flows:
        return
    x = np.interp(point[0], flows, range(len(flows)))
    y = np.interp(point[1], densities, range(len(densities)))
    ax.plot(x, y, marker="*", ms=24, color="#d62728", markeredgecolor="white", mew=1.2,
            ls="none", label="Design")


def heatmap(ax, values, flows, densities, cmap, *, vmin=None, vmax=None, text=None):
    image = ax.imshow(values, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if text is not None and np.isfinite(values[i, j]):
                light = image.norm(values[i, j]) > 0.55 if cmap == "Blues" else \
                    image.norm(values[i, j]) < 0.55
                ax.text(j, i, text(i, j), ha="center", va="center", fontsize=SMALL - 1,
                        color="white" if light else "black")
    ax.set_xticks(range(len(flows)), [f"{f:g}" for f in flows])
    ax.set_yticks(range(len(densities)), [f"{x:g}" for x in densities])
    ax.set_xlabel("AHL washout rate (flow)")
    ax.set_ylabel("Cell density")
    return image


# ---------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------

def figure_dynamics(r: Results, d: str, dynamics: Path | None, oscillation: Path | None):
    """Fig. 2: the colonies before and during a beat; where they oscillate over flow and cell
    density; macro F1 over the same grid."""
    fig = plt.figure(figsize=(16, 12.5))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.1], hspace=0.3)
    top = outer[0].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
    bottom = outer[1].subgridspec(1, 2, wspace=0.3)
    ax, inducer = fig.add_subplot(top[0]), fig.add_subplot(top[1])
    panel(ax, "a")
    if dynamics is not None and dynamics.is_file():
        data = np.load(dynamics)
        colours = [OKABE_ITO[i] for i in (0, 2, 3, 5)]
        for k in range(data["history"].shape[1]):
            colour = colours[k % len(colours)]
            ax.plot(data["t_history"], data["history"][:, k], color=colour, lw=THIN,
                    alpha=0.8)
            ax.plot(data["t_beat"], data["driven"][:, k], color=colour, lw=THIN, alpha=0.8,
                    label="Single colonies" if k == 0 else None)
        ax.plot(data["t_history"], data["history_mean"], color="k", lw=LW,
                label="Mean of the 576 colonies")
        ax.plot(data["t_beat"], data["driven_mean"], color="k", lw=LW)
        ax.plot(data["t_beat"], data["rest_mean"], color="k", lw=THIN, ls="--",
                label="Mean, without the beat")
        for t, text in ((0, "reference\nimage"), (140, "image"), (187, "image")):
            ax.axvline(t, color="#999999", ls=":", lw=THIN)
            ax.text(t, 1.0, text, transform=blended_transform_factory(ax.transData,
                                                                     ax.transAxes),
                    ha="center", va="bottom", fontsize=SMALL - 1, color="#555555")
        ax.text(0.01, 0.95, f"Period at rest {float(data['period']):.0f} min (Danino et al.: "
                f"{DANINO_PERIODS[0]}–{DANINO_PERIODS[1]} min)", transform=ax.transAxes,
                fontsize=SMALL, va="top")
        name = CLASS_NAMES[int(data["beat_class"])]
        inducer.plot(np.concatenate([data["t_history"], data["t_beat"]]),
                     np.concatenate([np.zeros(len(data["t_history"])), data["inducer"]]),
                     color=CLASS_COLOURS[name], lw=LW, label=f"Inducer: one {name.lower()} beat")
        legend(inducer, loc="upper left")
        legend(ax, loc="upper left", bbox_to_anchor=(0.0, 0.88), ncol=3)
    ax.set_ylabel("LuxI (model units)")
    ax.tick_params(labelbottom=False)
    inducer.sharex(ax)
    inducer.set_ylabel("Inducer")
    inducer.set_xlabel("Time from the start of the beat (min)")

    bx, cx = fig.add_subplot(bottom[0]), fig.add_subplot(bottom[1])
    point = design_point(r, d)
    flows, densities, share, period = oscillation_map(oscillation, d)
    if flows:
        image = heatmap(bx, share, flows, densities, "Blues", vmin=0, vmax=100,
                        text=lambda i, j: f"{share[i, j]:.0f}%" + (
                            f"\n{period[i, j]:.0f} min" if np.isfinite(period[i, j]) else ""))
        fig.colorbar(image, ax=bx, label="Colonies oscillating (%)")
        mark_design(bx, flows, densities, point)
    panel(bx, "b")
    flows, densities, grid = f1_map(r)
    if flows:
        image = heatmap(cx, grid, flows, densities, "viridis",
                        vmin=max(0.75, float(np.nanmin(grid))), vmax=float(np.nanmax(grid)),
                        text=lambda i, j: f"{grid[i, j]:.3f}".lstrip("0"))
        fig.colorbar(image, ax=cx, label="Macro F1 (ridge, 2 images)")
        mark_design(cx, flows, densities, point)
    panel(cx, "c")
    return fig


def figure_performance(r: Results, d: str):
    """Fig. 3: the array against the raw beat and the comparison models; each readout on the
    array's images against the same readout on the raw beat; per-class F1."""
    fig = plt.figure(figsize=(17, 11))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.05, 1], hspace=0.5, wspace=0.3)
    ax, bx, cx = (fig.add_subplot(grid[:, 0]), fig.add_subplot(grid[0, 1]),
                  fig.add_subplot(grid[1, 1]))
    raw_rows = []
    for key, label in READOUTS:
        value = r.f1(RAW[key]) if key in ("ridge", "rf", "svm", "histgb") else None
        if value is not None:
            raw_rows.append((label.replace("\n", " "), [value], COLOURS["raw"]))
    groups = [
        ("Array, 576 traps", [
            ("Ridge, 2 images (design)", r.seeds(f"P5-balanced-{d}-headline"), COLOURS["array"]),
            ("RBF SVM, 2 images", readout(r, d, "c", "svm").values(), COLOURS["svm"]),
            ("Random forest, 4 images", r.seeds(f"P5-balanced-{d}-forest"),
             COLOURS["forest"])]),
        ("Tuned at 576, read like the array (ridge)", [
            (TUNED["uncoupled-576"], r.seeds("P5-balanced-uncoupled-576"),
             COLOURS["isolated"]),
            (TUNED["esn-image-576"], r.seeds("P5-balanced-esn-image-576"), COLOURS["digital"]),
            ("Random tanh features", r.report(CONTROLS["random_features"]).values(),
             COLOURS["digital"])]),
        ("Ideal reading: clean states (ridge)", [
            ("Array", r.seeds(f"P5-balanced-{d}-full"), COLOURS["ideal"]),
            ("ESN, 576 units", r.seeds("P5-balanced-esn-576"), COLOURS["ideal"])]),
        ("Raw beat", raw_rows),
    ]
    y, ticks, labels = 0.0, [], []
    transform = blended_transform_factory(ax.transAxes, ax.transData)
    for title, rows in groups:
        ax.text(0.0, y + 0.1, title, transform=transform, fontsize=SMALL, fontstyle="italic",
                color="#444444", va="bottom")
        y -= 0.9
        for label, values, colour in rows:
            seed_dots(ax, y, values, colour)
            ticks.append(y)
            labels.append(label)
            y -= 1.0
        y -= 0.35
    headline = r.seeds(f"P5-balanced-{d}-headline")
    if headline:
        ax.axvline(np.mean(headline), color=COLOURS["array"], ls=":", lw=THIN)
    ax.set_yticks(ticks, labels)
    ax.set_ylim(y + 0.6, 0.9)
    ax.set_xlim(0.70, 0.92)
    ax.set_xticks(np.arange(0.70, 0.91, 0.05))
    ax.set_xlabel("Macro F1 (5 classes, 5 folds)")
    panel(ax, "a")

    for i, (key, _) in enumerate(READOUTS):
        raw, values = r.f1(RAW[key]), list(readout(r, d, "c", key).values())
        if raw is not None and values:
            bx.annotate("", xy=(i, np.mean(values)), xytext=(i, raw),
                        arrowprops={"arrowstyle": "-|>", "color": "#aaaaaa", "lw": THIN,
                                    "shrinkA": 6, "shrinkB": 6})
        if raw is not None:
            bx.scatter([i], [raw], s=80, marker="D", color=COLOURS["raw"], zorder=3,
                       label="Raw beat" if i == 0 else None)
        if values:
            jitter = np.linspace(-0.12, 0.12, len(values))
            bx.scatter(i + jitter, values, s=30, color=COLOURS["array"], zorder=3,
                       label="Array, 2 images (5 seeds)" if i == 0 else None)
    bx.set_xticks(range(len(READOUTS)), [label for _, label in READOUTS], fontsize=SMALL)
    bx.set_ylabel("Macro F1")
    bx.set_ylim(0.70, 0.92)
    legend(bx, loc="lower right")
    panel(bx, "b")

    rows = [("Raw beat, ridge", [RAW["ridge"]], COLOURS["raw"], "D"),
            ("Array, ridge", [f"P5-balanced-{d}-headline-s{s}" for s in SEEDS],
             COLOURS["array"], "o"),
            ("Array, RBF SVM", [f"P4-{d}-c-svm"] + ([f"extra-576-readout-c-svm-s{s}"
                                                    for s in SEEDS[1:]] if d == "p576" else []),
             COLOURS["svm"], "o"),
            ("Raw beat, HistGB", [RAW["histgb"]], "k", "D")]
    width = 0.8 / len(rows)
    for k, (label, runs, colour, marker) in enumerate(rows):
        f1 = [v for v in (r.per_class(run) for run in runs) if v is not None]
        if not f1:
            continue
        x = np.arange(len(CLASS_NAMES)) - 0.4 + width * (k + 0.5)
        m = np.mean(f1, axis=0)
        s = np.std(f1, axis=0, ddof=1) if len(f1) > 1 else np.zeros_like(m)
        cx.errorbar(x, m, yerr=s, fmt=marker, color=colour, ms=8, lw=THIN, capsize=0,
                    label=label)
    cx.set_xticks(range(len(CLASS_NAMES)), [c[:5] + "." if len(c) > 7 else c
                                             for c in CLASS_NAMES], fontsize=SMALL)
    cx.set_ylabel("F1 of the class")
    legend(cx, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    panel(cx, "c")
    return fig


def figure_mechanism(r: Results, d: str):
    """Fig. 4: ablations and controls, ridge and forest; the share of traps induced with and
    without AHL exchange."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(17, 7), gridspec_kw={"width_ratios": [1.45, 1],
                                                                    "wspace": 0.45})
    rows = [
        ("Array (design)", ablation(r, d, "coupling", "diffusion"),
         ablation(r, d, "coupling", "diffusion", True)),
        ("No AHL exchange", ablation(r, d, "coupling", "none"),
         ablation(r, d, "coupling", "none", True)),
        ("No feedback delay\n(colonies do not oscillate)", ablation(r, d, "delay", 0),
         ablation(r, d, "delay", 0, True)),
        ("Beat samples shuffled", ablation(r, d, "scramble", True),
         ablation(r, d, "scramble", True, True)),
        ("Linear imitation of the array", r.seeds(f"P6-{d}-surrogate-c", SEEDS[:3]),
         r.seeds(f"P6-{d}-surrogate-e", SEEDS[:3])),
        ("Random tanh features", r.report(CONTROLS["random_features"]).values(),
         untuned_control(r, "random_features", True)),
        ("Fading memories\n(leaky integrators)", r.report(CONTROLS["fading_memories"]).values(),
         untuned_control(r, "fading_memories", True)),
    ]
    for i, (_, ridge, forest) in enumerate(rows):
        seed_dots(ax, -i + 0.18, ridge, COLOURS["array"], half=0.15,
                  label="Ridge, 2 images" if i == 0 else None)
        seed_dots(ax, -i - 0.18, forest, COLOURS["forest"], marker="s", half=0.15,
                  label="Random forest, 4 images" if i == 0 else None)
    ax.set_yticks([-i for i in range(len(rows))], [row[0] for row in rows])
    for key, colour, text in (("ridge", COLOURS["array"], "Raw beat, ridge"),
                              ("rf", COLOURS["forest"], "Raw beat, forest")):
        value = r.f1(RAW[key])
        if value is not None:
            ax.axvline(value, color=colour, ls="--", lw=THIN, label=f"{text} ({value:.3f})")
    ax.axvline(0.2, color="#bbbbbb", lw=THIN, label="Chance (0.2)")
    ax.set_xlim(0.15, 0.95)
    ax.set_xlabel("Macro F1")
    legend(ax, loc="upper center", bbox_to_anchor=(0.45, -0.12), ncol=2)
    panel(ax, "a")

    table = r.grid("extra-576-connectivity", "reservoir.params.input_connectivity",
                   "reservoir.params.coupling") if d == "p576" else {}
    shares = sorted({k[0] for k in table})
    for coupling, label, colour, style, marker in (
            ("diffusion", "With AHL exchange", COLOURS["array"], "-", "o"),
            ("none", "Isolated traps", COLOURS["isolated"], "--", "s")):
        line(bx, {100 * s: list(table.get((s, coupling), {}).values()) for s in shares}, colour,
             label, style=style, marker=marker)
    for s in shares:
        on, off = table.get((s, "diffusion"), {}), table.get((s, "none"), {})
        gain = paired(on, off)
        if gain:
            middle = (np.mean(list(on.values())) + np.mean(list(off.values()))) / 2
            bx.text(100 * s, middle, f"+{np.mean(gain):.3f}", ha="center", va="center",
                    fontsize=SMALL, color=COLOURS["array"],
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    bx.set_xscale("log")
    bx.set_xlim(8, 125)
    bx.set_xticks([100 * s for s in shares], [f"{100 * s:g}" for s in shares])
    bx.minorticks_off()
    bx.set_xlabel("Traps that receive the inducer (%)")
    bx.set_ylabel("Macro F1 (ridge, 2 images)")
    legend(bx, loc="lower right")
    panel(bx, "b")
    return fig


def figure_scaling(r: Results, d: str):
    """Fig. 5: macro F1 against the number of traps; measurement noise and the number of
    images; the time of the earlier image."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={"width_ratios": [1.4, 1, 1],
                                                                "wspace": 0.32})
    ax = axes[0]
    ax.axvspan(*BUILT, color="#dddddd", lw=0, label="Built arrays (416, 500 traps)")
    ax.axvline(LARGEST, color="#999999", ls=":", lw=THIN, label="Largest built (12,224)")
    line(ax, traps(r, d), COLOURS["array"], "Ridge, 2 images")
    line(ax, traps(r, d, forest=True), COLOURS["forest"], "Forest, 4 images", marker="s",
         style="--")
    line(ax, ideal_traps(r, d), COLOURS["ideal"], "Ridge, clean states", marker="^", style=":",
         band=False)
    headline = r.seeds(f"P5-balanced-{d}-headline")
    if headline:
        ax.plot(576, np.mean(headline), marker="*", ms=24, color="#d62728",
                markeredgecolor="white", mew=1.2, ls="none", label="Design (576)", zorder=4)
    ax.set_xscale("log")
    plain_log_ticks(ax.xaxis)
    ax.set_xlabel("Traps in the array")
    ax.set_ylabel("Macro F1")
    legend(ax, loc="lower right")
    panel(ax, "a")

    bx = axes[1]
    for agg, label, colour, marker in ((ONE, "Ridge, 1 image", OKABE_ITO[5], "v"),
                                       (HEADLINE, "Ridge, 2 images", COLOURS["array"], "o"),
                                       (FOREST, "Ridge, 4 images", OKABE_ITO[3], "D")):
        line(bx, scaled(r.sweep(f"P3-{d}-noise", "features.measurement_noise",
                                features__aggregation=agg)), colour, label, marker=marker)
    line(bx, scaled(r.sweep(f"P3-{d}-noise-forest", "features.measurement_noise",
                            features__aggregation=FOREST)), COLOURS["forest"],
         "Forest, 4 images", marker="s", style="--")
    bx.axvline(1, color="#bbbbbb", ls=":", lw=THIN)
    bx.set_xlabel("Measurement noise (% of the signal)")
    bx.set_ylabel("Macro F1")
    legend(bx, loc="lower left")
    panel(bx, "b")

    cx = axes[2]
    times, final = image_times(r, d)
    line(cx, times, COLOURS["array"], "Ridge, 2 images (the later at 187 min)")
    if final:
        cx.axhline(np.mean(final), color=OKABE_ITO[5], ls="--", lw=THIN,
                   label=f"Ridge, the 187-min image alone ({np.mean(final):.3f})")
    cx.axvline(140, color="#bbbbbb", ls=":", lw=THIN)
    if times:
        cx.set_xticks(sorted(times)[::2])
    cx.set_xlabel("Time of the earlier image (min)")
    cx.set_ylabel("Macro F1")
    means = ([np.mean(final)] if final else []) + (ordered(times)[1] if times else [])
    if means:
        cx.set_ylim(min(means) - 0.01, max(means) + 0.015)
    legend(cx, loc="upper left")
    panel(cx, "c")
    return fig


def figure_robustness(r: Results, d: str, flat_share: Path | None):
    """Fig. 6: calibration error with and without the reference image; variability of the
    colonies' kinetics, with the share that do not oscillate; spread of cell density and
    washout between traps (and the alternative design b32)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={"wspace": 0.58})
    below = {"loc": "upper center", "bbox_to_anchor": (0.5, -0.2)}
    ax = axes[0]
    for study, label, colour, style, agg, marker in (
            (f"P3-{d}-calibration", "Ridge, with reference", COLOURS["array"], "-", HEADLINE,
             "o"),
            (f"P3-{d}-calibration-noref", "Ridge, no reference", COLOURS["array"], ":",
             HEADLINE, "o"),
            (f"P3-{d}-calibration-forest", "Forest, with reference", COLOURS["forest"], "-",
             FOREST, "s"),
            (f"P3-{d}-calibration-noref-forest", "Forest, no reference", COLOURS["forest"], ":",
             FOREST, "s")):
        line(ax, scaled(r.sweep(study, "features.calibration_error", features__aggregation=agg)),
             colour, label, style=style, marker=marker)
    ax.set_xlabel("Calibration error (% SD of a trap's gain)")
    ax.set_ylabel("Macro F1")
    legend(ax, **below)
    panel(ax, "a")

    bx = axes[1]
    for sfx, label, colour, marker in (("", "Ridge, 2 images", COLOURS["array"], "o"),
                                       ("-forest", "Forest, 4 images", COLOURS["forest"], "s")):
        points = r.sweep(f"P3-{d}-heterogeneity{sfx}", "reservoir.params.param_heterogeneity_cv")
        points.update(r.sweep(f"P3-{d}-heterogeneity-64{sfx}",
                              "reservoir.params.param_heterogeneity_cv"))
        line(bx, scaled(points), colour, label, marker=marker)
    bx.axvline(10, color="#bbbbbb", ls=":", lw=THIN)
    bx.set_xlabel("Variability of the colonies' kinetics (CV, %)")
    bx.set_ylabel("Macro F1")
    shares = flat_shares(flat_share, d)
    if shares:
        x, m, _ = ordered(scaled(shares))
        twin = bx.twinx()
        twin.plot(x, m, ":", marker="^", color="#555555", lw=LW, ms=7,
                  label="Colonies not oscillating")
        twin.set_ylabel("Not oscillating (%)", color="#555555")
        twin.set_ylim(0, 100)
        handles = bx.get_legend_handles_labels()
        extra = twin.get_legend_handles_labels()
        bx.legend(handles[0] + extra[0], handles[1] + extra[1], fontsize=SMALL, frameon=False,
                  **below)
    else:
        legend(bx, **below)
    panel(bx, "b")

    cx = axes[2]
    for study, key, label, colour, style, marker in (
            (f"P3-{d}-density", "density", "Cell density, design", COLOURS["array"], "-", "o"),
            (f"P3-{d}-dilution", "dilution", "Washout, design", COLOURS["array"], ":", "D"),
            ("P3-p576r-density" if d == "p576" else None, "density",
             "Cell density, alternative b32", COLOURS["alternative"], "-", "s")):
        if study is not None:
            line(cx, scaled(r.sweep(study, f"reservoir.params.heterogeneity_{key}_cv")), colour,
                 label, style=style, marker=marker)
    cx.set_xlabel("Spread between traps (CV, %)")
    cx.set_ylabel("Macro F1")
    legend(cx, **below)
    panel(cx, "c")
    return fig


def figure_settings(r: Results, d: str):
    """SI: the other settings of the array, ridge on 2 images unless stated."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), gridspec_kw={"hspace": 0.42,
                                                                 "wspace": 0.32})

    def categories(ax, points, order, labels, colour=COLOURS["array"]):
        for i, key in enumerate(order):
            seed_dots(ax, i, points.get(key, []), colour, horizontal=False)
        ax.set_xticks(range(len(order)), labels)
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.set_ylabel("Macro F1")

    reset = r.sweep(f"P3-{d}-reset", "reservoir.params.reset_noise")
    order = sorted(k for k in reset if k is not None)
    categories(axes[0, 0], reset, order, [f"{100 * k:g}" for k in order])
    axes[0, 0].set_xlabel("Imperfect reset (CV of the start state, %)")
    panel(axes[0, 0], "a")

    ax = axes[0, 1]
    line(ax, r.sweep(f"P3-{d}-warmup", "reservoir.params.warmup"), COLOURS["array"],
         "Ridge, 2 images")
    line(ax, r.sweep(f"P3-{d}-warmup-forest", "reservoir.params.warmup"), COLOURS["forest"],
         "Forest, 4 images", marker="s", style="--")
    ax.set_xlabel("Settling time before each beat (min)")
    ax.set_ylabel("Macro F1")
    legend(ax, loc="lower right")
    panel(ax, "b")

    rest = r.sweep(f"P3-{d}-warmup-input", "reservoir.params.warmup_input")
    categories(axes[0, 2], rest, ["zero", "rest"], ["None (design)", "Resting level"])
    axes[0, 2].set_xlabel("Inducer while settling")
    panel(axes[0, 2], "c")

    levels = r.sweep(f"P3-{d}-input-levels", "reservoir.params.input_levels")
    order = sorted((k for k in levels), key=lambda k: (isinstance(k, str), k))
    categories(axes[1, 0], levels, order, [str(k) for k in order])
    axes[1, 0].set_xlabel("Inducer levels")
    panel(axes[1, 0], "d")

    ax = axes[1, 1]
    line(ax, r.sweep(f"P6-{d}-gaussian", "reservoir.params.interaction_diameters"),
         COLOURS["array"], "Gaussian layout")
    headline = r.seeds(f"P5-balanced-{d}-headline")
    if headline:
        ax.axhline(np.mean(headline), color="#999999", ls="--", lw=THIN,
                   label=f"Lattice (design, {np.mean(headline):.3f})")
    ax.set_xlabel("Reach of the coupling (colony spacings)")
    ax.set_ylabel("Macro F1")
    legend(ax, loc="lower right")
    panel(ax, "e")

    ax = axes[1, 2]
    for agg, label, colour, marker in ((ONE, "1 image", OKABE_ITO[5], "v"),
                                       (HEADLINE, "2 images", COLOURS["array"], "o"),
                                       (FOREST, "4 images", OKABE_ITO[3], "D")):
        line(ax, r.sweep(f"P6-{d}-units-images", "reservoir.params.units",
                         features__aggregation=agg), colour, f"Ridge, {label}", marker=marker)
    sizes = sorted({row["reservoir.params.units"] for row in r.trials(f"P6-{d}-units-images")})
    ax.set_xscale("log")
    plain_log_ticks(ax.xaxis, sizes or None)
    ax.set_xlabel("Traps in the array")
    ax.set_ylabel("Macro F1 (3 seeds)")
    legend(ax, loc="lower right")
    panel(ax, "f")
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
            ax.axvline(t, color="#999999", lw=THIN, ls=":")
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


def simulate_dynamics(design: Path, seed: int = SEEDS[0], colonies: int = 4, before: int = 300,
                      beat_class: int = 2) -> dict:
    """The design's colonies before and during one beat (Fig. 2a): LuxI of a few colonies and
    the mean of all over the last ``before`` minutes of the warm-up (stepped from the start
    state as the node steps it), then during a typical beat of class ``beat_class`` (the pool
    beat closest to its class's mean) as a run steps it, and over the same time without the
    beat. The period at rest is the median over the colonies of their median LuxI peak
    spacing in the last ``before`` minutes of the warm-up."""
    from scipy.signal import find_peaks

    from training.figures import VARIABLES, array_states, build, encoded_input, run_config
    from training.run import load_task
    from utils.preprocessing import standardize_data

    cfg = run_config(design)
    cfg["reservoir"]["seed"] = seed
    lux = VARIABLES.index("I")
    warmup = int(cfg["reservoir"]["params"]["warmup"])
    X, Y, _ = load_task(cfg["task"])
    X, _ = standardize_data(X, X[:1], scaler_type=str(cfg["task"]["scaler"]))
    X = np.asarray(X, dtype=float).reshape(len(X), -1)
    labels = np.argmax(np.asarray(Y).reshape(len(Y), -1), axis=1)
    members = np.flatnonzero(labels == beat_class)
    index = int(members[np.argmin(((X[members] - X[members].mean(axis=0)) ** 2).sum(1))])
    beat = X[index].reshape(-1, 1)

    free = build(cfg, warmup=0)
    free.initialize(np.zeros((1, 1)))
    history = np.array([free._step(free._warmup_drive)[:, lux] for _ in range(warmup)])
    node = build(cfg)
    node.initialize(np.zeros((1, 1)))
    work = node.copy()
    work.states = work._warmup_states.copy()
    if getattr(work, "_warmup_ring", None) is not None:
        work._ring, work._ring_pos = work._warmup_ring[0].copy(), work._warmup_ring[1]
    start = np.asarray(node.start_state(input_dim=1)).reshape(-1, 4)[:, lux]
    rest = np.array([work._step(work._warmup_drive)[:, lux] for _ in range(len(beat))])
    driven = array_states(cfg, beat, index)[0][:, :, lux]
    periods = []
    for col in range(history.shape[1]):
        x = history[-before:, col]
        peaks, _ = find_peaks(x, prominence=0.3 * (x.max() - x.min()))
        if len(peaks) >= 2:
            periods.append(np.median(np.diff(peaks)))
    pick = np.random.default_rng(seed).choice(history.shape[1], colonies, replace=False)
    return {"t_history": np.arange(-before, 0), "history": history[-before:, pick],
            "history_mean": history[-before:].mean(axis=1), "t_beat": np.arange(len(beat)),
            "driven": driven[:, pick], "driven_mean": driven.mean(axis=1),
            "rest": rest[:, pick], "rest_mean": rest.mean(axis=1),
            "inducer": encoded_input(cfg, beat),
            "period": float(np.median(periods)) if periods else float("nan"),
            "beat_class": beat_class, "beat_index": index,
            # how far the stepped warm-up ends from the node's start state (should be ~0)
            "start_gap": float(np.max(np.abs(history[-1] - start)) / np.mean(np.abs(start)))}


# ---------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------

def fmt(values) -> str:
    values = list(values) if values is not None and not isinstance(values, float) else values
    if values is None or (isinstance(values, list) and not values):
        return "–"
    if isinstance(values, float):
        return f"{values:.3f}"
    m, s = mean_sd(values)
    return f"{m:.3f} ± {s:.3f}" if len(values) > 1 else f"{m:.3f}"


def fmt_diff(values) -> str:
    values = list(values)
    if not values:
        return "–"
    m, s = mean_sd(values)
    return f"{m:+.3f} ± {s:.3f}" if len(values) > 1 else f"{m:+.3f}"


def tables(r: Results, d: str) -> str:
    headline = r.by_seed(f"P5-balanced-{d}-headline")
    search = "2 × 150 TPE trials, 5 best on 5 selection seeds"
    out = [f"# Tables, design {d}\n",
           "## Table 2. Equal-budget comparison\n",
           "Balanced 5-class task, macro F1 over 5 folds, mean ± SD over the reporting seeds "
           "6348-6352 (raw beat: one run, the folds fixed). Last column: the array's ridge "
           "minus the row, seed by seed (positive: the array is better).\n",
           "| Model | Read by | Tuned | Macro F1 | Array minus row |", "|---|---|---|---|---|"]
    rows = [
        ("Array, 576 traps (design)", "Ridge, 2 images",
         f"{search} (gain, coupling rate, flow, density)", headline),
        ("Array, 576 traps", "RBF SVM, 2 images", "readout grid (C, gamma; 3 inner folds)",
         readout(r, d, "c", "svm")),
        ("Array, 576 traps", "Random forest, 4 images", "readout grid",
         r.by_seed(f"P5-balanced-{d}-forest")),
        ("Isolated traps (no AHL exchange)", "Ridge, 2 images",
         f"{search} (gain, flow, density)", r.by_seed("P5-balanced-uncoupled-576")),
        ("ESN, 576 units", "Ridge, 2 images", f"{search} (Phase 2 space)",
         r.by_seed("P5-balanced-esn-image-576")),
        ("ESN, 576 units", "Ridge, clean states", f"{search} (Phase 2 space)",
         r.by_seed("P5-balanced-esn-576")),
        ("Random tanh features, 576", "Ridge, 2 images", f"{search} (input scaling, bias)",
         r.report(CONTROLS["random_features"])),
        ("Fading memories (leaky integrators)", "Ridge, 2 images",
         f"{search} (number, longest time constant, band-pass)",
         r.report(CONTROLS["fading_memories"])),
    ]
    for model, read, tuned, values in rows:
        out.append(f"| {model} | {read} | {tuned} | {fmt(values.values())} | "
                   f"{'(reference)' if values is headline else fmt_diff(paired(headline, values))} |")
    for key, label in READOUTS:
        value = r.f1(RAW[key])
        out.append(f"| Raw beat | {label.replace(chr(10), ' ')} | readout grid | {fmt(value)} | "
                   f"{fmt_diff(paired(headline, value)) if value is not None else '–'} |")

    out += ["\n## Table 3. Other tasks and held-out beats\n",
            "Macro F1. Balanced: mean ± SD over 5 seeds; unbalanced and binary: seed 6348; "
            "validation splits (803 beats of each of N, S, V, Q, never used for tuning): mean ± "
            "SD over seeds 6348-6350 (comparison models: seed 6348).\n",
            "| Model | Balanced (5 classes) | Unbalanced (5 classes) | Binary | Validation 1 | "
            "Validation 2 |", "|---|---|---|---|---|---|"]
    for label, row, val, seeds in (
            ("Array, ridge, 2 images (design)", f"{d}-headline", f"{d}-c", SEEDS[:3]),
            ("Array, forest, 4 images", f"{d}-forest", f"{d}-e", SEEDS[:3]),
            (TUNED["uncoupled-576"], "uncoupled-576", "uncoupled-576", SEEDS[:1]),
            (TUNED["esn-image-576"], "esn-image-576", "esn-image-576", SEEDS[:1]),
            (TUNED["esn-576"], "esn-576", "esn-576", SEEDS[:1])):
        out.append(f"| {label} | {fmt(r.seeds(f'P5-balanced-{row}'))} | "
                   f"{fmt(r.f1(f'P5-unbalanced-{row}-s{SEEDS[0]}'))} | "
                   f"{fmt(r.f1(f'P5-binary-{row}-s{SEEDS[0]}'))} | "
                   + " | ".join(fmt(r.seeds(f"P3-{s}-{val}", seeds))
                                for s in ("validation-1", "validation-2")) + " |")
    for key, label in (("ridge", "ridge"), ("rf", "random forest"), ("histgb", "HistGB")):
        out.append(f"| Raw beat, {label} | " + " | ".join(
            fmt(r.f1(f"P5-{t}-raw-{key}")) for t in ("balanced", "unbalanced", "binary"))
            + " | – | – |")

    out += ["\n## SI. Readouts refitted on the array's saved images (5 reporting seeds)\n",
            "| Readout | Array, 2 images | Array, 4 images | Raw beat |", "|---|---|---|---|"]
    for key, label in READOUTS:
        out.append(f"| {label.replace(chr(10), ' ')} | {fmt(readout(r, d, 'c', key).values())} "
                   f"| {fmt(readout(r, d, 'e', key).values())} | {fmt(r.f1(RAW[key]))} |")

    out += ["\n## SI. Per-class F1, balanced task (pooled over the folds; mean ± SD over "
            "seeds)\n", "| Run | " + " | ".join(CLASS_NAMES) + " |",
            "|---|" + "---|" * len(CLASS_NAMES)]
    for label, runs in (
            ("Raw beat, ridge", [RAW["ridge"]]),
            ("Array, ridge (design)", [f"P5-balanced-{d}-headline-s{s}" for s in SEEDS]),
            ("Array, RBF SVM", [f"P4-{d}-c-svm"] + [f"extra-576-readout-c-svm-s{s}"
                                                    for s in SEEDS[1:]]),
            ("Array, forest, 4 images", [f"P5-balanced-{d}-forest-s{s}" for s in SEEDS]),
            ("Raw beat, HistGB", [RAW["histgb"]])):
        f1 = [v for v in (r.per_class(run) for run in runs) if v is not None]
        out.append(f"| {label} | " + (" | ".join(fmt(list(np.asarray(f1)[:, c]))
                                                 for c in range(len(CLASS_NAMES)))
                                      if f1 else " | ".join("–" for _ in CLASS_NAMES)) + " |")

    out += ["\n## SI. Comparison models: tuned at 576 and at their earlier sizes (ridge)\n",
            "| Model | At 576, tuned at 576 | Earlier settings and size (28 Oct) |",
            "|---|---|---|"]
    for key, label in TUNED.items():
        earlier = key.removesuffix("-576")
        out.append(f"| {label} | {fmt(r.seeds(f'P5-balanced-{key}'))} | "
                   f"{fmt(r.seeds(f'P5-balanced-{earlier}'))} |")
    for name, study in CONTROLS.items():
        out.append(f"| {name.replace('_', ' ').capitalize()} | "
                   f"{fmt(r.report(study).values())} (tuned) | "
                   f"{fmt(untuned_control(r, name))} (standard settings) |")

    out.append("\n## SI. Robustness and settings (ridge on 2 images unless stated; mean ± SD "
               "over the seeds)\n")
    sweeps = [
        (f"P3-{d}-noise", "features.measurement_noise", "Measurement noise",
         {"features__aggregation": HEADLINE}),
        (f"P3-{d}-noise", "features.measurement_noise", "Measurement noise, 1 image",
         {"features__aggregation": ONE}),
        (f"P3-{d}-noise", "features.measurement_noise", "Measurement noise, 4 images",
         {"features__aggregation": FOREST}),
        (f"P3-{d}-calibration", "features.calibration_error", "Calibration error, reference",
         {"features__aggregation": HEADLINE}),
        (f"P3-{d}-calibration-noref", "features.calibration_error",
         "Calibration error, no reference", {"features__aggregation": HEADLINE}),
        (f"P3-{d}-heterogeneity", "reservoir.params.param_heterogeneity_cv",
         "Kinetic variability (CV)", {}),
        (f"P3-{d}-heterogeneity-64", "reservoir.params.param_heterogeneity_cv",
         "Kinetic variability (CV), 64 substeps", {}),
        (f"P3-{d}-density", "reservoir.params.heterogeneity_density_cv", "Cell-density spread",
         {}),
        (f"P3-{d}-dilution", "reservoir.params.heterogeneity_dilution_cv", "Washout spread", {}),
        (f"P3-{d}-reset", "reservoir.params.reset_noise", "Imperfect reset", {}),
        (f"P3-{d}-warmup", "reservoir.params.warmup", "Settling time (min)", {}),
        (f"P3-{d}-warmup-input", "reservoir.params.warmup_input", "Inducer while settling", {}),
        (f"P3-{d}-input-levels", "reservoir.params.input_levels", "Inducer levels", {}),
        (f"P3-{d}-image-time", "features.aggregation", "Images", {}),
        (f"P6-{d}-gaussian", "reservoir.params.interaction_diameters",
         "Gaussian layout (reach)", {}),
    ]
    for study, key, label, fixed in sweeps:
        points = r.sweep(study, key, **fixed)
        if points:
            order = sorted(points, key=lambda k: (isinstance(k, str), str(k) if
                                                  isinstance(k, str) else k))
            out.append(f"- **{label}** (`{study}`): " + "; ".join(
                f"{k}: {fmt(points[k])}" for k in order))
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("numbers", "tables", "figures", "images",
                                            "dynamics", "all"),
                        help="images and dynamics simulate the design; the others only read "
                             "results; all runs numbers, tables and figures.")
    parser.add_argument("--design", choices=sorted(DESIGNS), default="p576",
                        help="The design the figures and numbers are for (default p576).")
    parser.add_argument("--out", type=Path, help="Output folder (default "
                        "results/figures/paper/<design>).")
    parser.add_argument("--flat-share", type=Path,
                        help="CSV of scripts/paper/flat_share.py (Fig. 6b and the numbers).")
    parser.add_argument("--oscillation", type=Path,
                        help="CSV of scripts/paper/oscillation_map.py (Fig. 2b).")
    add_formats_option(parser)
    args = parser.parse_args(argv)
    r = Results()
    d = args.design
    out = args.out or config.figures_dir(f"paper/{d}")
    out.mkdir(parents=True, exist_ok=True)
    design = Path("experiments/v0.2/designs") / f"{d}.yaml"
    dynamics = out / "dynamics.npz"
    if args.command == "dynamics":
        data = simulate_dynamics(design)
        np.savez(dynamics, **data)
        print(f"{dynamics}: period at rest {data['period']:.1f} min; the stepped warm-up ends "
              f"{data['start_gap']:.2e} (relative) from the start state")
    if args.command in ("numbers", "all"):
        values = numbers(r, d, args.flat_share, dynamics)
        (out / "numbers.json").write_text(json.dumps(values, indent=1))
        (out / "numbers.tex").write_text(tex_macros(values, d))
        print(json.dumps(values, indent=1))
    if args.command in ("tables", "all"):
        text = tables(r, d)
        (out / "tables.md").write_text(text)
        print(text)
    if args.command in ("figures", "all"):
        with plt.style.context(STYLE):
            drawn = {"fig2_dynamics": figure_dynamics(r, d, dynamics, args.oscillation),
                     "fig3_performance": figure_performance(r, d),
                     "fig4_mechanism": figure_mechanism(r, d),
                     "fig5_scaling": figure_scaling(r, d),
                     "fig6_robustness": figure_robustness(r, d, args.flat_share),
                     "si_settings": figure_settings(r, d)}
            for name, fig in drawn.items():
                for path in save_figure(fig, name, folder=out, formats=args.formats):
                    print(path)
    if args.command == "images":
        with plt.style.context(STYLE):
            fig = figure_images(design)
            for path in save_figure(fig, "fig1_images", folder=out, formats=args.formats):
                print(path)
    if r.missing:
        print(f"\nNot found ({len(r.missing)}): " + ", ".join(sorted(r.missing)[:40])
              + (" ..." if len(r.missing) > 40 else ""), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
