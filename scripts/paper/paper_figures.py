"""Write the numbers the paper quotes.

Reads only collected results: run folders (``runs/<name>/metrics.json``) and study databases
(``optuna/optuna-<name>.db``) in the results folder (``BIORESERVOIR_RESULTS_DIR``), by the
names of the configs in ``experiments/paper``: ``balanced-array-s6348``,
``robustness-noise``, ``induced-share``, ... The figures are drawn by ``paper_panels.py`` and
the tables written by ``article_tables.py`` and ``si_tables.py``, which read the results
through this module.

    python scripts/paper/paper_figures.py --flat-share flat_share.csv --simulated simulated.npz

Writes ``numbers.json`` and ``numbers.tex`` (one ``\\newcommand`` per number) to
``results/figures/paper/`` (``--out`` to change it) and prints them. A missing run or study
is reported and its part left empty.
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

optuna.logging.set_verbosity(optuna.logging.ERROR)

SEEDS = (6348, 6349, 6350, 6351, 6352)
HEADLINE, FOREST, ONE = "images_at:140,187", "images:4", "final"
#: The prefix of the LaTeX macros.
PREFIX = "P"
#: The washout rates of the washout x cell-density map (studies washout-map-<washout>).
FLOWS = (0.3, 0.4, 0.6, 0.9, 1.2)
FLOW, DENSITY = "reservoir.params.washout_rate", "reservoir.params.cell_density"
#: Trap arrays built by Prindle et al. (2012): 416 and 500 traps; the largest 12,224.
BUILT, LARGEST = (416, 500), 12224
#: Periods measured by Danino et al. (2010) over their flow rates (min).
DANINO_PERIODS = (52, 90)
READOUTS = (("ridge", "Ridge"), ("lr", "Logistic\nregression"), ("knn", "kNN"),
            ("rf", "Random\nforest"), ("histgb", "HistGB"), ("svm", "RBF SVM"))
#: The raw-beat run of each readout, with the readout's settings on the array.
RAW = {key: f"balanced-raw-{key}" for key, _ in READOUTS}
#: Comparison models tuned at 576 units with the array's budget, each read by ridge.
TUNED = {"isolated-traps": "Isolated traps (no AHL exchange)",
         "esn-image": "ESN, read like the array",
         "esn-mean": "ESN, clean states"}
#: Controls tuned like the models (two TPE studies), their best point on the reporting seeds.
CONTROLS = {"random_features": "random-features", "fading_memories": "leaky-integrators"}
#: The readings of the array's saved features that the readouts are refitted on.
READINGS = {"c": "two-images", "e": "four-images"}
#: The ablation studies (with the intact array at the same seeds), by the setting they change.
ABLATIONS = {"coupling": "no-exchange", "delay": "no-delay", "scramble": "scrambled"}
#: The runs on the test split (experiments/paper/test.yaml, run once), by key.
FINAL_TEST = {"headline": "array", "forest": "array-forest", "ideal": "array-ideal",
              **{key: key for key in TUNED},
              **{f"raw-{key}": f"raw-{key}" for key in ("ridge", "rf", "histgb")}}


def test_split(r: Results, key: str) -> tuple[float, float, float] | None:
    """Macro F1 on the test split and its bootstrap 95% CI."""
    metrics = r.metrics(f"test-{FINAL_TEST[key]}")
    if metrics is None:
        return None
    s = metrics["summary"]
    return float(s["f1"]), float(s["f1_macro_ci95_low"]), float(s["f1_macro_ci95_high"])


class Results:
    """The collected runs and studies, by name."""

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

def readout(r: Results, obs: str, key: str) -> dict[int, float]:
    """Macro F1 by seed of readout ``key`` refitted on the saved features ``obs`` (``c``: two
    images, ``e``: four) of each reporting seed."""
    return r.by_seed(f"readout-{READINGS[obs]}-{key}")


def ablation(r: Results, stem: str, value, forest: bool = False) -> list[float]:
    key = {"coupling": "reservoir.params.coupling", "delay": "reservoir.params.delay",
           "scramble": "task.scramble"}[stem]
    study = f"ablation-{ABLATIONS[stem]}{'-forest' if forest else ''}"
    return r.sweep(study, key).get(value, [])


def untuned_control(r: Results, name: str, forest: bool = False) -> list[float]:
    """The controls at their standard settings: 576 random tanh features, or the bank of
    leaky integrators of 24 timescales."""
    sfx = "-forest" if forest else ""
    if name == "random_features":
        return [row["f1"] for row in r.trials(f"random-features-standard{sfx}")
                if row["reservoir.params.units"] == 576
                and row["reservoir.params.activation"] == "tanh"]
    return r.sweep(f"leaky-integrators-standard{sfx}", "reservoir.params.expansion").get(None, [])


def traps(r: Results, forest: bool = False) -> dict[int, list[float]]:
    """Macro F1 at each array size: 100-4,096 traps and, for ridge, the larger arrays."""
    points = r.sweep(f"array-size{'-forest' if forest else ''}", "reservoir.params.units")
    if not forest:
        studies = [f"array-size-{n}" for n in (6400, 9216, 12544)]
        studies += [f"array-size-{n}-s{s}" for n in (16384, 25600, 36864) for s in SEEDS]
        for study in studies:
            for size, values in r.sweep(study, "reservoir.params.units").items():
                points.setdefault(size, []).extend(values)
    return points


def ideal_traps(r: Results) -> dict[int, list[float]]:
    """Ridge on the clean mean of all states (no imaging) at 576, 4,096 and 16,384 traps."""
    points = {576: r.seeds("balanced-array-ideal")}
    studies = ["array-size-ideal-4096"] + [f"array-size-ideal-16384-s{s}" for s in SEEDS]
    for study in studies:
        for size, values in r.sweep(study, "reservoir.params.units").items():
            points.setdefault(size, []).extend(values)
    return {k: v for k, v in points.items() if v}


def f1_map(r: Results) -> tuple[list[float], list[float], np.ndarray]:
    """Mean macro F1 over the reporting seeds on the washout x density grid (rows: densities)."""
    cells = defaultdict(list)
    for flow in FLOWS:
        for row in r.trials(f"washout-map-{flow:g}"):
            cells[(row[FLOW], row[DENSITY])].append(row["f1"])
    if not cells:
        return [], [], np.zeros((0, 0))
    flows, densities = sorted({k[0] for k in cells}), sorted({k[1] for k in cells})
    grid = np.array([[np.mean(cells[(f, x)]) if cells.get((f, x)) else np.nan for f in flows]
                     for x in densities])
    return flows, densities, grid


def oscillation_map(path: Path | None) -> tuple[list, list, np.ndarray, np.ndarray]:
    """Share of oscillating colonies (%) and median period (min) on the flow x density grid,
    from the CSV of ``oscillation_map.py``."""
    if path is None or not path.is_file():
        return [], [], np.zeros((0, 0)), np.zeros((0, 0))
    rows = list(csv.DictReader(path.open()))
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


def flat_shares(path: Path | None) -> dict[float, list[float]]:
    """Colonies that do not oscillate (%) at each kinetic CV, from ``flat_share.py``'s CSV."""
    shares = defaultdict(list)
    if path is not None and path.is_file():
        for row in csv.DictReader(path.open()):
            shares[float(row["cv"])].append(100 * float(row["flat"]))
    return dict(shares)


def design_point(r: Results) -> tuple[float, float] | None:
    """The design's flow and cell density, from the config of its headline run."""
    cfg = r.settings(f"balanced-array-s{SEEDS[0]}") or {}
    params = cfg.get("reservoir", {}).get("params", {})
    if "washout_rate" not in params or "cell_density" not in params:
        return None
    return float(params["washout_rate"]), float(params["cell_density"])


def image_times(r: Results) -> tuple[dict[int, list[float]], list[float]]:
    """Ridge on two images by the time of the earlier one (the later at 187 min), and ridge
    on the final image alone."""
    points = r.sweep("robustness-image-time", "features.aggregation")
    times = {int(k.split(":")[1].split(",")[0]): v for k, v in points.items()
             if str(k).startswith("images_at")}
    return times, points.get(ONE, [])


# ---------------------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------------------

def numbers(r: Results, flat_share: Path | None = None,
            simulated: Path | None = None) -> dict:
    """The numbers the text quotes."""
    out: dict = {}

    def put(key, values):
        values = list(values)
        if values:
            m, s = mean_sd(values)
            out[key] = {"mean": round(m, 8), "sd": round(s, 8), "n": len(values)}

    def one(key, value):
        if value is not None and np.isfinite(value):
            out[key] = round(float(value), 8)

    headline = r.by_seed("balanced-array")
    put("headline", headline.values())
    put("forest", r.seeds("balanced-array-forest"))
    put("ideal", r.seeds("balanced-array-ideal"))
    put("ideal_forest", r.seeds("balanced-array-ideal-forest"))
    for obs, images in (("c", "two"), ("e", "four")):
        for key, _ in READOUTS:
            put(f"readout_{images}_{key}", readout(r, obs, key).values())
    raw = {key: r.f1(run) for key, run in RAW.items()}
    for key, value in raw.items():
        one(f"raw_{key}", value)
    for key in TUNED:
        other = r.by_seed(f"balanced-{key}")
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
                                          for v in r.seeds("balanced-array-ideal")])
        if raw["histgb"] is not None:
            put("gap_closed", [100 * g / (raw["histgb"] - raw["ridge"])
                               for g in paired(headline, raw["ridge"])])
    forest = r.seeds("balanced-array-forest")
    if forest and raw["rf"] is not None:
        put("forest_minus_raw_rf", [v - raw["rf"] for v in forest])
        one("forest_minus_raw_rf_worst", min(v - raw["rf"] for v in forest))
    svm = readout(r, "c", "svm")
    if svm and raw["histgb"] is not None:
        put("svm_minus_raw_histgb", paired(svm, raw["histgb"]))
        put("svm_minus_headline", paired(svm, headline))
    for task in ("unbalanced", "binary"):
        put(f"{task}_headline", r.seeds(f"{task}-array"))
        one(f"{task}_forest", r.f1(f"{task}-array-forest-s{SEEDS[0]}"))
        for key in ("ridge", "rf", "histgb"):
            one(f"{task}_raw_{key}", r.f1(f"{task}-raw-{key}"))
        for key in TUNED:
            put(f"{task}_{key.replace('-', '_')}", r.seeds(f"{task}-{key}"))
    for split in ("validation-1", "validation-2"):
        tag = split.replace("-", "_")
        put(f"{tag}_headline", r.seeds(f"{split}-array"))
        put(f"{tag}_forest", r.seeds(f"{split}-array-forest"))
        put(f"{tag}_one_image", r.seeds(f"{split}-array-one-image"))
        for key in TUNED:
            put(f"{tag}_{key.replace('-', '_')}", r.seeds(f"{split}-{key}"))
    for key in FINAL_TEST:
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
            put(f"{tag}_intact", ablation(r, stem, on, forest_))
            put(tag, ablation(r, stem, off, forest_))
    put("imitation", r.seeds("surrogate-two-images"))
    put("imitation_forest", r.seeds("surrogate-four-images"))
    connectivity = r.grid("induced-share", "reservoir.params.input_connectivity",
                          "reservoir.params.coupling")
    for share in sorted({k[0] for k in connectivity}):
        on, off = connectivity.get((share, "diffusion"), {}), connectivity.get((share, "none"), {})
        tag = f"induced_{round(100 * share)}"
        put(f"{tag}_exchange", on.values())
        put(f"{tag}_isolated", off.values())
        put(f"{tag}_exchange_adds", paired(on, off))
    curve, forest_curve = traps(r), traps(r, forest=True)
    for size in (100, 256, 576, 1024, 4096):
        put(f"traps_{size}", curve.get(size, []))
        put(f"traps_forest_{size}", forest_curve.get(size, []))
    put("traps_plateau", [v for size, values in curve.items() if size >= 9216 for v in values])
    for size, values in ideal_traps(r).items():
        put(f"traps_ideal_{size}", values)
    for agg, tag in ((ONE, "one_image"), (HEADLINE, "two_images"), (FOREST, "four_images")):
        points = r.sweep("robustness-noise", "features.measurement_noise", features__aggregation=agg)
        for level in (0.0, 0.01, 0.02, 0.05, 0.1):
            put(f"noise_{tag}_{round(100 * level)}", points.get(level, []))
    times, final = image_times(r)
    if times:
        means = [mean_sd(v)[0] for v in times.values()]
        one("image_time_lowest", min(means))
        one("image_time_highest", max(means))
        one("image_time_first", min(times))
        one("image_time_last", max(times))
    put("image_final_only", final)
    for study, tag in (("robustness-calibration", "reference"),
                       ("robustness-calibration-no-reference", "no_reference")):
        points = r.sweep(study, "features.calibration_error", features__aggregation=HEADLINE)
        for level in (0.02, 0.05):
            put(f"calibration_{tag}_{round(100 * level)}", points.get(level, []))
    kinetic = r.sweep("heterogeneity-kinetic", "reservoir.params.param_heterogeneity_cv")
    kinetic.update(r.sweep("heterogeneity-kinetic-64-substeps", "reservoir.params.param_heterogeneity_cv"))
    for cv in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
        put(f"kinetic_cv_{round(100 * cv)}", kinetic.get(cv, []))
    for cv, values in flat_shares(flat_share).items():
        if cv in (0.1, 0.2, 0.3, 0.5):
            put(f"flat_cv_{round(100 * cv)}", values)
    for study, key, tag in (("heterogeneity-density", "density", "density_spread"),
                            ("heterogeneity-washout", "dilution", "washout_spread")):
        points = r.sweep(study, f"reservoir.params.heterogeneity_{key}_cv")
        for cv in (0.1, 0.2, 0.3):
            put(f"{tag}_{round(100 * cv)}", points.get(cv, []))
    put("warmup_at_rest", r.sweep("robustness-warmup-input", "reservoir.params.warmup_input")
        .get("rest", []))
    layout = r.sweep("layout-gaussian", "reservoir.params.interaction_diameters")
    if layout:
        means = [mean_sd(v)[0] for v in layout.values()]
        one("gaussian_lowest", min(means))
        one("gaussian_highest", max(means))
    flows, densities, grid = f1_map(r)
    if grid.size:
        one("map_best", np.nanmax(grid))
        one("map_fastest_flow_best", np.nanmax(grid[:, flows.index(max(flows))]))
    point = design_point(r)
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


def tex_macros(values: dict) -> str:
    """One ``\\newcommand`` per number, named after the key: ``\\PHeadline`` (the mean) and
    ``\\PHeadlineSd`` (the SD over seeds)."""
    def camel(key):
        digits = ("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine")
        key = re.sub(r"\d", lambda m: f"_{digits[int(m[0])]}_", key)
        return "".join(part[:1].upper() + part[1:] for part in re.split(r"[_\-]+", key) if part)

    lines = ["% Numbers from scripts/paper/paper_figures.py"]
    for key, value in values.items():
        name = f"\\{PREFIX}{camel(key)}"
        if isinstance(value, dict):
            lines.append(f"\\newcommand{{{name}}}{{{tex_value(key, value['mean'])}}}")
            lines.append(f"\\newcommand{{{name}Sd}}{{{tex_value(key, value['sd'])}}}")
        else:
            lines.append(f"\\newcommand{{{name}}}{{{tex_value(key, value)}}}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path,
                        help="Output folder (default results/figures/paper).")
    parser.add_argument("--flat-share", type=Path,
                        help="CSV of scripts/paper/flat_share.py (shares of flat colonies).")
    parser.add_argument("--simulated", type=Path,
                        help="simulated.npz of paper_panels.py simulate (the period at rest).")
    args = parser.parse_args(argv)
    r = Results()
    out = args.out or config.figures_dir("paper")
    out.mkdir(parents=True, exist_ok=True)
    values = numbers(r, args.flat_share, args.simulated)
    (out / "numbers.json").write_text(json.dumps(values, indent=1))
    (out / "numbers.tex").write_text(tex_macros(values))
    print(json.dumps(values, indent=1))
    if r.missing:
        print(f"\nNot found ({len(r.missing)}): " + ", ".join(sorted(r.missing)[:40])
              + (" ..." if len(r.missing) > 40 else ""), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
