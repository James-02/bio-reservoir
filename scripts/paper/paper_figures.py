"""Write the numbers and tables of the revised paper, for design p576.

Reads only collected results: run folders (``runs/<name>/metrics.json``) and study databases
(``optuna/optuna-<name>.db``) in the results folder (``BIORESERVOIR_RESULTS_DIR``), by the
names of the final-run configs: ``P5-balanced-p576-headline-s6348``, ``P3-p576-noise``,
``extra-576-connectivity``, ... (``--design p576r``: the alternative design's runs). The
figures are drawn by ``paper_panels.py``, which reads the results through this module.

    python scripts/paper/paper_figures.py numbers --flat-share flat_share.csv \\
        --simulated simulated.npz
    python scripts/paper/paper_figures.py tables

``numbers`` writes ``numbers.json`` and ``numbers.tex`` (one ``\\newcommand`` per number) and
prints them; ``tables`` writes the main and SI tables as Markdown (``tables.md``). Both go to
``results/figures/paper/<design>/`` (``--out`` to change it). A missing run or study is
reported and its part left empty, so drafts work on partial results.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import optuna
import yaml

from utils import config
from utils.visualisation import CLASS_NAMES

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
#: The runs on the sealed test split (experiments/v0.2-final-test, run once), by key.
FINAL_TEST = {"headline": "p576-headline", "forest": "p576-forest", "ideal": "p576-full",
              **{key: key for key in TUNED},
              **{f"raw-{key}": f"raw-{key}" for key in ("ridge", "rf", "histgb")}}


def test_split(r: Results, key: str) -> tuple[float, float, float] | None:
    """Macro F1 on the test split and its bootstrap 95% CI."""
    metrics = r.metrics(f"final-test-{FINAL_TEST[key]}")
    if metrics is None:
        return None
    s = metrics["summary"]
    return float(s["f1"]), float(s["f1_macro_ci95_low"]), float(s["f1_macro_ci95_high"])


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

    def confusion(self, run: str) -> np.ndarray | None:
        """The run's confusion matrix pooled over its folds (counts)."""
        metrics = self.metrics(run)
        if metrics is None:
            return None
        cm = metrics["pooled"]["confusion_matrix"]
        return np.asarray(json.loads(cm) if isinstance(cm, str) else cm, dtype=float)

    def per_class(self, run: str) -> np.ndarray | None:
        """F1 of each class over all folds (the pooled confusion matrix)."""
        cm = self.confusion(run)
        if cm is None:
            return None
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
            cells[(row[FLOW], row[DENSITY])].append(row["f1"])
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
            simulated: Path | None = None) -> dict:
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
        put(f"{task}_headline", r.seeds(f"P5-{task}-{d}-headline"))
        one(f"{task}_forest", r.f1(f"P5-{task}-{d}-forest-s{SEEDS[0]}"))
        for key in ("ridge", "rf", "histgb"):
            one(f"{task}_raw_{key}", r.f1(f"P5-{task}-raw-{key}"))
        for key in TUNED:
            one(f"{task}_{key.replace('-', '_')}", r.f1(f"P5-{task}-{key}-s{SEEDS[0]}"))
    for split in ("validation-1", "validation-2"):
        tag = split.replace("-", "_")
        put(f"{tag}_headline", r.seeds(f"P3-{split}-{d}-c", SEEDS[:3]))
        put(f"{tag}_forest", r.seeds(f"P3-{split}-{d}-e", SEEDS[:3]))
        put(f"{tag}_one_image", r.seeds(f"P3-{split}-{d}-a", SEEDS[:3]))
    for key in FINAL_TEST if d == "p576" else ():
        test = test_split(r, key)
        if test is not None:
            tag = f"test_{key.replace('-', '_')}"
            one(tag, test[0])
            one(f"{tag}_low", test[1])
            one(f"{tag}_high", test[2])
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
    if simulated is not None and simulated.is_file():
        one("period", float(np.load(simulated)["period"]))
    return out


def tex_value(key: str, value: float) -> str:
    """Shares in percent with one decimal (flat colonies) or none (gap closed); scores and
    settings with three decimals."""
    if key.startswith("flat_"):
        return f"{value:.1f}"
    if key.startswith(("gap_closed", "period", "image_time_first", "image_time_last")):
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


def table1(r: Results, d: str) -> list[str]:
    """Table 1: the colony kinetics and the design's settings, with their basis. Values come
    from the code (``DEFAULT_DDE_PARAMS``) and the config of the design's headline run."""
    from bioreservoir.dde import DEFAULT_DDE_PARAMS

    cfg = r.settings(f"P5-balanced-{d}-headline-s{SEEDS[0]}") or {}
    params, features = cfg.get("reservoir", {}).get("params", {}), cfg.get("features", {})
    if not params:
        return []
    roles = {"CA": ("C_A", "maximal production of AiiA"),
             "CI": ("C_I", "maximal production of LuxI"),
             "del_": ("δ", "leakiness of the lux promoter"),
             "alpha": ("α", "activation of the lux promoter by AHL"),
             "k": ("k", "saturation of AHL synthesis by LuxI"),
             "k1": ("k₁", "saturation of promoter activation"),
             "b": ("b", "AHL synthesis by LuxI"),
             "gammaA": ("γ_A", "enzymatic degradation of AiiA"),
             "gammaI": ("γ_I", "enzymatic degradation of LuxI"),
             "gammaH": ("γ_H", "degradation of AHL by AiiA"),
             "f": ("f", "saturation of protein degradation"),
             "g": ("g", "saturation of AHL degradation"),
             "d0": ("d₀", "cell density at which protein synthesis stops"),
             "D": ("D", "diffusion of AHL through the cell membrane"),
             "delay": ("τ", "delay of protein production (min)")}
    out = ["## Table 1. Parameters of the design and their basis (draft)\n",
           "| Parameter | Value | Basis |", "|---|---|---|"]
    for key, value in DEFAULT_DDE_PARAMS.items():
        if key in ("mu", "d"):
            continue
        symbol, role = roles.get(key, (key, ""))
        out.append(f"| {symbol}, {role} | {value:g} | Danino et al. 2010 (SI), scaled "
                   f"parameters, unchanged |")
    images = features.get("aggregation", "").split(":", 1)[-1].replace(",", " and ")
    side = int(round(np.sqrt(params["units"])))
    out += [
        f"| Traps | {params['units']:,} ({side} × {side}) | Of the order of built trap arrays "
        f"(416 and 500 traps; Prindle et al. 2012) |",
        f"| µ, dilution of external AHL by the flow | {params['washout_rate']:.3f} | Tuned; "
        f"Danino et al. varied it with the flow rate (our reference value "
        f"{DEFAULT_DDE_PARAMS['mu']:g}) |",
        f"| d, cell density | {params['cell_density']:.3f} | Tuned below d₀ = "
        f"{DEFAULT_DDE_PARAMS['d0']:g}; Danino et al. varied it (our reference value "
        f"{DEFAULT_DDE_PARAMS['d']:g}) |",
        f"| AHL exchange between neighbouring traps (rate per min) | "
        f"{params['diffusion_rate']:.2f} | Tuned; the discrete diffusion of external AHL "
        f"between neighbours of Danino et al.'s spatial model, on a 2D lattice (in Prindle et "
        f"al. 2012, AHL did not couple traps at their high flow) |",
        f"| Input gain | {params['input_scaling']:.3f} | Tuned |",
        f"| Kinetic variability between colonies (CV) | "
        f"{100 * params['param_heterogeneity_cv']:g}% | Assumed; as the 10% parameter spread "
        f"Danino et al. simulated, below the single-cell 15% of Mondragón-Palomino et al. "
        f"2011 |",
        "| Inducer | the same level to every trap, one per minute | Computational abstraction; "
        "periodic forcing has been shown (Bennett et al. 2008; Mondragón-Palomino et al. "
        "2011) |",
        f"| Warm-up before each beat | {params['warmup']} min, no inducer | Design decision |",
        f"| Images | reference at 0 min; {images} min | Assumed; Danino et al. imaged every "
        f"2–4.5 min, less often at high magnification to avoid photobleaching |",
        f"| Measurement noise | {100 * features.get('measurement_noise', 0):g}% per image | "
        f"Assumed |",
        f"| Solver | RK4, {params['rk4_substeps']} substeps per min | Numerical (convergence "
        f"check in the SI) |",
    ]
    return out + [""]


def tables(r: Results, d: str) -> str:
    headline = r.by_seed(f"P5-balanced-{d}-headline")
    search = "2 × 150 TPE trials, 5 best on 5 selection seeds"
    out = [f"# Tables, design {d}\n", *table1(r, d),
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

    out += ["\n## Table 3. Other tasks and held-out beats (ridge on two images)\n",
            "Macro F1. Balanced: mean ± SD over 5 seeds; unbalanced and binary: array over 5 "
            "seeds, comparison models seed 6348; validation splits (803 beats of each of N, S, "
            "V, Q, never used for tuning): mean ± SD over seeds 6348-6350 (comparison models: "
            "seed 6348); test split (370 beats of each of N, S, V, Q, evaluated once after the "
            "design was frozen; seed 6348, fitted on the balanced pool): F1 (bootstrap 95% CI).\n",
            "| Model | Balanced (5 classes) | Unbalanced (5 classes) | Binary | Validation 1 | "
            "Validation 2 | Test |", "|---|---|---|---|---|---|---|"]

    def fmt_test(key):
        test = test_split(r, key) if d == "p576" else None
        return "–" if test is None else f"{test[0]:.3f} ({test[1]:.3f}–{test[2]:.3f})"

    for label, row, val, seeds, test in (
            ("Array (design)", f"{d}-headline", f"{d}-c", SEEDS[:3], "headline"),
            (TUNED["uncoupled-576"], "uncoupled-576", "uncoupled-576", SEEDS[:1],
             "uncoupled-576"),
            (TUNED["esn-image-576"], "esn-image-576", "esn-image-576", SEEDS[:1],
             "esn-image-576"),
            (TUNED["esn-576"], "esn-576", "esn-576", SEEDS[:1], "esn-576")):
        out.append(f"| {label} | {fmt(r.seeds(f'P5-balanced-{row}'))} | "
                   + " | ".join(fmt(r.seeds(f"P5-{t}-{row}")) for t in ("unbalanced", "binary"))
                   + " | " + " | ".join(fmt(r.seeds(f"P3-{s}-{val}", seeds))
                                        for s in ("validation-1", "validation-2"))
                   + f" | {fmt_test(test)} |")
    for key, label in (("ridge", "ridge"), ("histgb", "HistGB")):
        out.append(f"| Raw beat, {label} | " + " | ".join(
            fmt(r.f1(f"P5-{t}-raw-{key}")) for t in ("balanced", "unbalanced", "binary"))
            + f" | – | – | {fmt_test(f'raw-{key}')} |")

    out += ["\n## Table 4. Ablations and controls (balanced task, mean ± SD over seeds)\n",
            "Ridge on two images; the last column reads the same features with a random "
            "forest on four images, to show what a nonlinear readout recovers. Controls: "
            "ridge rows tuned (2 × 150 TPE trials), forest rows at standard settings.\n",
            "| Array or control | Ridge | Random forest |", "|---|---|---|"]
    for label, ridge, forest in (
            ("Array (design)", ablation(r, d, "coupling", "diffusion"),
             ablation(r, d, "coupling", "diffusion", True)),
            ("No AHL exchange", ablation(r, d, "coupling", "none"),
             ablation(r, d, "coupling", "none", True)),
            ("No feedback delay (colonies do not oscillate)", ablation(r, d, "delay", 0),
             ablation(r, d, "delay", 0, True)),
            ("Beat samples shuffled", ablation(r, d, "scramble", True),
             ablation(r, d, "scramble", True, True)),
            ("Linear imitation of the array", r.seeds(f"P6-{d}-surrogate-c", SEEDS[:3]),
             r.seeds(f"P6-{d}-surrogate-e", SEEDS[:3])),
            ("Random tanh features, 576", r.report(CONTROLS["random_features"]).values(),
             untuned_control(r, "random_features", True)),
            ("Fading memories (leaky integrators)",
             r.report(CONTROLS["fading_memories"]).values(),
             untuned_control(r, "fading_memories", True))):
        out.append(f"| {label} | {fmt(ridge)} | {fmt(forest)} |")
    out.append(f"| Raw beat | {fmt(r.f1(RAW['ridge']))} | {fmt(r.f1(RAW['rf']))} |")

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
        (f"P3-{d}-warmup", "reservoir.params.warmup", "Warm-up (min)", {}),
        (f"P3-{d}-warmup-input", "reservoir.params.warmup_input", "Inducer during warm-up", {}),
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
    parser.add_argument("command", choices=("numbers", "tables", "all"),
                        help="all writes the numbers and the tables.")
    parser.add_argument("--design", choices=sorted(DESIGNS), default="p576",
                        help="The design the numbers and tables are for (default p576).")
    parser.add_argument("--out", type=Path, help="Output folder (default "
                        "results/figures/paper/<design>).")
    parser.add_argument("--flat-share", type=Path,
                        help="CSV of scripts/paper/flat_share.py (shares of flat colonies).")
    parser.add_argument("--simulated", type=Path,
                        help="simulated.npz of paper_panels.py simulate (the period at rest).")
    args = parser.parse_args(argv)
    r = Results()
    d = args.design
    out = args.out or config.figures_dir(f"paper/{d}")
    out.mkdir(parents=True, exist_ok=True)
    if args.command in ("numbers", "all"):
        values = numbers(r, d, args.flat_share, args.simulated)
        (out / "numbers.json").write_text(json.dumps(values, indent=1))
        (out / "numbers.tex").write_text(tex_macros(values, d))
        print(json.dumps(values, indent=1))
    if args.command in ("tables", "all"):
        text = tables(r, d)
        (out / "tables.md").write_text(text)
        print(text)
    if r.missing:
        print(f"\nNot found ({len(r.missing)}): " + ", ".join(sorted(r.missing)[:40])
              + (" ..." if len(r.missing) > 40 else ""), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
