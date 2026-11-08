"""The additional results of the Supporting Information as LaTeX tables, from the runs.

Writes one file per subsection of the SI's Additional Results, following the stages of the
design studies (one ``table`` environment per table, labels ``tab:si-*``):
``si_evaluation.tex`` (every metric, seeds and folds, per-class F1), ``si_readouts.tex``,
``si_comparisons.tex`` (comparisons by seed, ablations and controls), ``si_array.tex``
(array size, imaging, start and inducer, layout) and ``si_heterogeneity.tex``.

    python scripts/paper/si_tables.py --out <folder> --flat-share <flat_share.csv>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from paper_figures import (
    CONTROLS,
    FOREST,
    HEADLINE,
    ONE,
    RAW,
    READOUTS,
    SEEDS,
    Results,
    flat_shares,
    ideal_traps,
    image_times,
    mean_sd,
    paired,
    readout,
    traps,
    untuned_control,
)

from utils import config

TASKS = {"balanced": "Balanced", "unbalanced": "Unbalanced", "binary": "Binary"}
CLASSES = {"balanced": ["N", "S", "V", "F", "Q"], "unbalanced": ["N", "S", "V", "F", "Q"],
           "binary": ["Normal", "Arrhythmia"]}
METRICS = (("f1", "Macro F1"), ("accuracy", "Accuracy"), ("precision", "Precision"),
           ("recall", "Recall"), ("mcc", "MCC"), ("kappa", r"$\kappa$"))


def pm(values) -> str:
    """``0.880 \\pm 0.005`` for several values, ``0.880`` for one, a dash for none."""
    values = [v for v in np.ravel(values) if v is not None and np.isfinite(v)] \
        if values is not None else []
    if not values:
        return "--"
    m, s = mean_sd(values)
    return f"${m:.3f} \\pm {s:.3f}$" if len(values) > 1 else f"{m:.3f}"


def tabular(spec: str, header: list[str], rows: list[list[str]]) -> list[str]:
    lines = [f"\\begin{{tabular}}{{{spec}}}", r"    \toprule",
             "    " + " & ".join(header) + r" \\", r"    \midrule"]
    for row in rows:
        lines.append(r"    \midrule" if row == ["MIDRULE"] else "    " + " & ".join(row) + r" \\")
    return lines + [r"    \bottomrule", r"\end{tabular}"]


def table(label: str, caption: str, spec: str, header: list[str], rows: list[list[str]],
          size: str = "small", colsep: str | None = None) -> str:
    lines = [r"\begin{table}[H]", r"    \centering", f"    \\{size}"]
    if colsep:
        lines.append(f"    \\setlength{{\\tabcolsep}}{{{colsep}}}")
    lines += [f"    \\caption{{{caption}}}", f"    \\label{{{label}}}"]
    lines += ["    " + line for line in tabular(spec, header, rows)]
    return "\n".join(lines + [r"\end{table}", ""])


def numeric(points: dict) -> list:
    return sorted(k for k in points if k is not None and not isinstance(k, str))


def metrics_table(r: Results, d: str) -> str:
    rows = []
    for task, name in TASKS.items():
        if rows:
            rows.append(["MIDRULE"])
        rows.append([rf"\multicolumn{{7}}{{@{{}}l}}{{\textit{{{name} task}}}}"])
        runs = [("Array (design)", [f"P5-{task}-{d}-headline-s{s}" for s in SEEDS])]
        runs += [(f"Raw beat, {label}", [f"P5-{task}-raw-{key}"])
                 for key, label in (("ridge", "ridge"), ("rf", "forest"),
                                    ("histgb", "HistGB"))]
        for label, names in runs:
            found = [m for m in (r.metrics(n) for n in names) if m is not None]
            rows.append([label] + [pm([m["mean"][key] for m in found]) for key, _ in METRICS])
    return table("tab:si-metrics",
                 "Every metric on the three tasks: the mean over the five folds; for the array, "
                 "the mean $\\pm$ SD over the five reporting seeds (the raw-beat readouts have "
                 "no reservoir seed). MCC: Matthews correlation coefficient~\\cite{Matthews1975}; "
                 "$\\kappa$: Cohen's kappa~\\cite{Cohen1960}.",
                 "@{}lcccccc@{}", ["Readout"] + [m for _, m in METRICS], rows, "footnotesize",
                 colsep="3pt")


def folds_table(r: Results, d: str) -> str:
    rows = []
    for seed in SEEDS:
        m = r.metrics(f"P5-balanced-{d}-headline-s{seed}")
        if m is not None:
            f = [fold["f1"] for fold in m["folds"]]
            rows.append([str(seed)] + [f"{v:.3f}" for v in f] + [f"{np.mean(f):.3f}"])
    raw = r.metrics(RAW["ridge"])
    if raw is not None:
        f = [fold["f1"] for fold in raw["folds"]]
        rows += [["MIDRULE"], ["Raw beat, ridge"] + [f"{v:.3f}" for v in f]
                 + [f"{np.mean(f):.3f}"]]
    return table("tab:si-folds",
                 "Macro F1 of the design (ridge on two images) on the balanced task, by "
                 "reservoir seed and fold; the folds are the same for every seed and for the "
                 "raw beat.",
                 "@{}lcccccc@{}", ["Seed", "Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5",
                                   "Mean"], rows)


def per_class_table(r: Results, d: str) -> str:
    def row(label, names):
        found = [v for v in (r.per_class(n) for n in names) if v is not None]
        if not found:
            return None
        stacked = np.array(found)
        return [label] + [pm(stacked[:, k]) for k in range(stacked.shape[1])]

    rows = []
    for task, name in TASKS.items():
        classes = CLASSES[task]
        if rows:
            rows.append(["MIDRULE"])
        rows.append([rf"\multicolumn{{6}}{{@{{}}l}}{{\textit{{{name} task}}}}"])
        rows.append([""] + classes + [""] * (5 - len(classes)))
        candidates = [("Array, ridge", [f"P5-{task}-{d}-headline-s{s}" for s in SEEDS])]
        if task == "balanced":
            candidates.append(("Array, RBF SVM", [f"P4-{d}-c-svm"]
                               + [f"extra-576-readout-c-svm-s{s}" for s in SEEDS[1:]]))
        candidates += [("Raw beat, ridge", [f"P5-{task}-raw-ridge"]),
                       ("Raw beat, HistGB", [f"P5-{task}-raw-histgb"])]
        for label, names in candidates:
            line = row(label, names)
            if line is not None:
                rows.append(line + [""] * (6 - len(line)))
    return table("tab:si-per-class",
                 "F1-score of each class, from the confusion matrix pooled over the folds; for "
                 "the array, the mean $\\pm$ SD over the reporting seeds. N: Normal; S: "
                 "Supraventricular ectopic; V: Ventricular ectopic; F: Fusion; Q: Unknown. "
                 "Figure~\\ref{fig:confusion-tasks} shows the confusion matrices of the "
                 "design.",
                 "@{}lccccc@{}", ["Readout", "", "", "", "", ""], rows, "footnotesize")


def readouts_table(r: Results, d: str) -> str:
    rows = []
    for key, label in READOUTS:
        rows.append([label.replace("\n", " "), pm(list(readout(r, d, "c", key).values())),
                     pm(list(readout(r, d, "e", key).values())), pm(r.f1(RAW[key]))])
    return table("tab:si-readouts",
                 "Every readout on the array's two and four images and on the raw beat "
                 "(balanced task, macro F1; array: mean $\\pm$ SD over the reporting seeds). "
                 "Settings in Table~\\ref{tab:readouts}.",
                 "@{}lccc@{}", ["Readout", "Array, 2 images", "Array, 4 images", "Raw beat"],
                 rows)


SWEEP = ("balanced task, ridge on two images; macro F1, mean $\\pm$ SD over the reporting "
         "seeds")


def imaging_tables(r: Results, d: str) -> str:
    noise = [r.sweep(f"P3-{d}-noise", "features.measurement_noise", features__aggregation=agg)
             for agg in (ONE, HEADLINE, FOREST)]
    levels = sorted(set().union(*(numeric(p) for p in noise)))
    noise_rows = [[f"{100 * v:g}"] + [pm(p.get(v)) for p in noise] for v in levels]
    times, final = image_times(r, d)
    time_rows = [[str(t), pm(times[t])] for t in sorted(times)]
    time_rows += [["MIDRULE"], ["187 only", pm(final)]]
    calibration = [r.sweep(f"P3-{d}-calibration{sfx}", "features.calibration_error",
                           features__aggregation=HEADLINE) for sfx in ("", "-noref")]
    errors = sorted(set().union(*(numeric(p) for p in calibration)))
    calibration_rows = [[f"{100 * v:g}"] + [pm(p.get(v)) for p in calibration] for v in errors]
    return "\n".join([
        table("tab:si-noise",
              "Measurement noise of each image, read by ridge from one image (187\\,min), "
              "two (the design) or four (balanced task; macro F1, mean $\\pm$ SD over the "
              "reporting seeds).",
              "@{}cccc@{}", ["Noise (\\%)", "One image", "Two images", "Four images"],
              noise_rows),
        table("tab:si-image-time",
              f"Time of the earlier image, with the later at 187\\,min, and the image at "
              f"187\\,min alone ({SWEEP}; Figure~\\ref{{fig:settings}}G).",
              "@{}lc@{}", ["Earlier image (min)", "Macro F1"], time_rows),
        table("tab:si-calibration",
              "Calibration error: each trap's gain in the test beats multiplied by "
              "$1 + c\\,\\varepsilon$, unknown to the readout, with and without the "
              f"reference image ({SWEEP}).",
              "@{}ccc@{}", ["$c$ (\\%)", "Reference", "No reference"], calibration_rows),
    ])


def start_tables(r: Results, d: str) -> str:
    reset = r.sweep(f"P3-{d}-reset", "reservoir.params.reset_noise")
    warmup = r.sweep(f"P3-{d}-warmup", "reservoir.params.warmup")
    rest = r.sweep(f"P3-{d}-warmup-input", "reservoir.params.warmup_input").get("rest")
    warmup_rows = [[f"{v:,}", pm(warmup[v])] for v in numeric(warmup)]
    warmup_rows += [["MIDRULE"], ["900, inducer at rest", pm(rest)]]
    levels = r.sweep(f"P3-{d}-input-levels", "reservoir.params.input_levels")
    level_rows = [["1 (equal)" if v == 1 else str(v), pm(levels[v])] for v in numeric(levels)]
    level_rows += [["Continuous", pm(levels.get("continuous"))]]
    induced = r.grid("extra-576-connectivity", "reservoir.params.input_connectivity",
                     "reservoir.params.coupling") if d == "p576" else {}
    shares = sorted({k[0] for k in induced})
    induced_rows = [[f"{100 * s:g}"] + [pm(list(induced.get((s, c), {}).values()))
                                        for c in ("diffusion", "none")] for s in shares]
    layout = r.sweep(f"P6-{d}-gaussian", "reservoir.params.interaction_diameters")
    layout_rows = [[f"{v:g}", pm(layout[v])] for v in numeric(layout)]
    layout_rows += [["MIDRULE"], ["Grid (design)", pm(r.seeds(f"P5-balanced-{d}-headline"))]]
    return "\n".join([
        table("tab:si-reset",
              "Reset error: the starting state of every beat perturbed with this coefficient "
              f"of variation ({SWEEP}; Figure~\\ref{{fig:settings}}A).",
              "@{}cc@{}", ["Reset error (CV)", "Macro F1"],
              [[f"{v:g}", pm(reset[v])] for v in numeric(reset)]),
        table("tab:si-warmup",
              "Length of the warm-up, without inducer, and the design's warm-up at the "
              f"resting inducer level ({SWEEP}; Figure~\\ref{{fig:settings}}B,C).",
              "@{}lc@{}", ["Warm-up (min)", "Macro F1"], warmup_rows),
        table("tab:si-levels",
              "Inducer strengths between traps: each trap's input gain scaled to one of 2, 3 "
              f"or 5 levels, or to a continuous range ({SWEEP}; "
              "Figure~\\ref{fig:settings}D).",
              "@{}lc@{}", ["Levels", "Macro F1"], level_rows),
        table("tab:si-induced",
              "Share of traps that receive the inducer, with the exchange of AHL and in "
              f"isolated traps at the design's settings ({SWEEP}).",
              "@{}ccc@{}", ["Traps induced (\\%)", "Exchange", "Isolated traps"],
              induced_rows),
        table("tab:si-layout",
              "Random layout of the traps, with an exchange that decays with distance, against "
              f"its range in trap spacings, and the grid of the design ({SWEEP}; "
              "Figure~\\ref{fig:settings}E).",
              "@{}lc@{}", ["Exchange range", "Macro F1"], layout_rows),
    ])


def heterogeneity_tables(r: Results, d: str, flat_path: Path | None) -> str:
    kinetic = r.sweep(f"P3-{d}-heterogeneity", "reservoir.params.param_heterogeneity_cv")
    kinetic.update(r.sweep(f"P3-{d}-heterogeneity-64", "reservoir.params.param_heterogeneity_cv"))
    flat = {round(cv, 4): v for cv, v in flat_shares(flat_path, d).items()}

    def colonies(cv):
        counts = [round(576 * share / 100) for share in flat.get(round(cv, 4), [])]
        if not counts:
            return "--"
        if not any(counts):
            return "0"
        mean = np.mean(counts)
        return f"{mean:.1f} ({min(counts)}--{max(counts)})" if mean < 10 else \
            f"{mean:.0f} ({min(counts)}--{max(counts)})"

    kinetic_rows = [[f"{v:.2g}", pm(kinetic[v]), colonies(v)] for v in numeric(kinetic)]
    spreads = [r.sweep(f"P3-{d}-{key}", f"reservoir.params.heterogeneity_{key}_cv")
               for key in ("density", "dilution")]
    cvs = sorted(set().union(*(numeric(p) for p in spreads)))
    spread_rows = [[f"{v:g}"] + [pm(p.get(v)) for p in spreads] for v in cvs]
    return "\n".join([
        table("tab:si-kinetic",
              "Kinetic variation between colonies: macro F1 (balanced task, ridge on two "
              "images; mean $\\pm$ SD over the reporting seeds) and the colonies, of 576, that "
              "no longer oscillate after the warm-up without inducer (mean and range over the "
              "seeds; a colony is flat when its LuxI varies by less than a factor of 1.5 over "
              "200\\,min). From a CV of 0.3, the array is integrated with 64 steps per minute.",
              "@{}ccc@{}", ["Kinetic variation (CV)", "Macro F1", "Flat colonies (of 576)"],
              kinetic_rows),
        table("tab:si-trap-variation",
              "Variation of the cell density and of the washout from trap to trap, each "
              f"drawn once per array ({SWEEP}).",
              "@{}ccc@{}", ["Variation between traps (CV)", "Cell density", "Washout"],
              spread_rows),
    ])


def traps_table(r: Results, d: str) -> str:
    ridge, forest, ideal = traps(r, d), traps(r, d, forest=True), ideal_traps(r, d)
    rows = [[f"{n:,}", pm(ridge.get(n)), pm(forest.get(n)), pm(ideal.get(n))]
            for n in sorted(set(ridge) | set(forest) | set(ideal))]
    return table("tab:si-traps",
                 "Macro F1 against the number of traps (balanced task; mean $\\pm$ SD over the "
                 "reporting seeds): ridge on two images (the design), the random forest on "
                 "four images, and ridge on the ideal reading (the mean of the full observed "
                 "state over the beat, without noise). Figure~\\ref{fig:settings}F compares "
                 "one, two and four images up to 2,500 traps; Table~\\ref{tab:runtime} gives "
                 "the cost of the larger arrays.",
                 "@{}rccc@{}", ["Traps $N$", "Ridge, 2 images", "Forest, 4 images",
                                "Ideal reading"], rows)


def comparisons_table(r: Results, d: str) -> str:
    headline = r.by_seed(f"P5-balanced-{d}-headline")
    rows = [["Array (design)"] + [f"{headline[s]:.3f}" for s in SEEDS] + ["--"], ["MIDRULE"]]
    for label, values in (
            ("Isolated traps", r.by_seed("P5-balanced-uncoupled-576")),
            ("ESN, read like the array", r.by_seed("P5-balanced-esn-image-576")),
            ("ESN, clean states", r.by_seed("P5-balanced-esn-576")),
            ("Random tanh features", r.report(CONTROLS["random_features"])),
            ("Leaky integrators", r.report(CONTROLS["fading_memories"]))):
        rows.append([label] + [f"{values[s]:.3f}" if s in values else "--" for s in SEEDS]
                    + [pm(paired(headline, values))])
    return table("tab:si-comparisons",
                 "The design and the comparisons by reservoir seed, each tuned with the same "
                 "budget (Table~\\ref{tab:selection}) and read by ridge (balanced task, macro "
                 "F1). The last column is the design minus the comparison, seed by seed (mean "
                 "$\\pm$ SD; positive: the design is better). The leaky integrators have no "
                 "random parts, so their score does not change with the seed.",
                 "@{}lcccccc@{}", ["Model"] + [str(s) for s in SEEDS] + ["Design minus"],
                 rows, "footnotesize")


def ablations_table(r: Results, d: str) -> str:
    def grid(stem, forest):
        key = {"coupling": "reservoir.params.coupling", "delay": "reservoir.params.delay",
               "scramble": "task.scramble"}[stem]
        return {k[0]: v for k, v in
                r.grid(f"P6-{d}-ablation-{stem}{'-forest' if forest else ''}", key).items()}

    def cells(intact, changed):
        return [pm(list(changed.values())), pm(paired(intact, changed))]

    ridge, forest = r.by_seed(f"P5-balanced-{d}-headline"), r.by_seed(f"P5-balanced-{d}-forest")
    rows = [["Array (design)", pm(list(ridge.values())), "--", pm(list(forest.values())), "--"],
            ["MIDRULE"]]
    for label, stem, on, off in (("No AHL exchange", "coupling", "diffusion", "none"),
                                 ("No feedback delay", "delay", 10, 0),
                                 ("Beat samples shuffled", "scramble", False, True)):
        a, b = grid(stem, False), grid(stem, True)
        rows.append([label] + cells(a.get(on, {}), a.get(off, {}))
                    + cells(b.get(on, {}), b.get(off, {})))
    rows.append(["Linear imitation$^{a}$"] + cells(ridge, r.by_seed(f"P6-{d}-surrogate-c"))
                + cells(forest, r.by_seed(f"P6-{d}-surrogate-e")))
    rows.append(["MIDRULE"])
    for label, name in (("Random tanh features$^{b}$", "random_features"),
                        ("Leaky integrators$^{b}$", "fading_memories")):
        rows.append([label, pm(untuned_control(r, name)), "--",
                     pm(untuned_control(r, name, True)), "--"])
    rows.append(["Raw beat", pm(r.f1(RAW["ridge"])), "--", pm(r.f1(RAW["rf"])), "--"])
    return table("tab:si-ablations",
                 "Ablations and controls, read by ridge on two images and by the random forest "
                 "on four (balanced task, macro F1, mean $\\pm$ SD over the reporting seeds). "
                 "Each ablation changes the design without retuning it; the loss is the "
                 "design minus the ablation, seed by seed. $^{a}$A ridge regression from the "
                 "beat to the array's features, on three of the seeds. $^{b}$At standard "
                 "settings (576 units; 24 time constants), untuned, so that both readouts "
                 "read the same features.",
                 "@{}lcccc@{}", ["Array or control", "Ridge", "Ridge loss", "Forest",
                                 "Forest loss"], rows, "footnotesize")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--design", default="p576", help="The design (default p576).")
    parser.add_argument("--out", type=Path, help="Output folder (default "
                        "results/figures/paper/<design>).")
    parser.add_argument("--flat-share", type=Path, help="CSV of scripts/paper/flat_share.py.")
    args = parser.parse_args(argv)
    r, d = Results(), args.design
    out = args.out or config.figures_dir(f"paper/{d}")
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "si_evaluation.tex": [metrics_table(r, d), folds_table(r, d), per_class_table(r, d)],
        "si_readouts.tex": [readouts_table(r, d)],
        "si_comparisons.tex": [comparisons_table(r, d), ablations_table(r, d)],
        "si_array.tex": [traps_table(r, d), imaging_tables(r, d), start_tables(r, d)],
        "si_heterogeneity.tex": [heterogeneity_tables(r, d, args.flat_share)],
    }
    for name, tables in files.items():
        (out / name).write_text("\n".join(["% From scripts/paper/si_tables.py", *tables]))
        print(out / name)
    if r.missing:
        print(f"Not found ({len(r.missing)}): " + ", ".join(sorted(r.missing)[:30]),
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
