"""The additional results of the Supporting Information as LaTeX tables, from the runs.

Writes ``si_tables.tex`` (one ``table`` environment per table, labels ``tab:si-*``) for the
SI to ``\\input``: every metric on the three tasks, the headline by seed and fold, per-class
F1, every readout on two and four images, the robustness and heterogeneity studies, the
array size and the share of colonies that do not oscillate.

    python scripts/paper/si_tables.py --out <folder> --flat-share <flat_share.csv>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from paper_figures import (
    HEADLINE,
    RAW,
    READOUTS,
    SEEDS,
    Results,
    flat_shares,
    ideal_traps,
    mean_sd,
    readout,
    traps,
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


def table(label: str, caption: str, spec: str, header: list[str], rows: list[list[str]],
          size: str = "small", colsep: str | None = None) -> str:
    lines = [r"\begin{table}[H]", r"    \centering", f"    \\{size}"]
    if colsep:
        lines.append(f"    \\setlength{{\\tabcolsep}}{{{colsep}}}")
    lines += [f"    \\caption{{{caption}}}", f"    \\label{{{label}}}",
              f"    \\begin{{tabular}}{{{spec}}}", r"        \toprule",
              "        " + " & ".join(header) + r" \\", r"        \midrule"]
    for row in rows:
        lines.append(r"        \midrule" if row == ["MIDRULE"] else
                     "        " + " & ".join(row) + r" \\")
    lines += [r"        \bottomrule", r"    \end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


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
                 "Supraventricular ectopic; V: Ventricular ectopic; F: Fusion; Q: Unknown.",
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


def sweeps(r: Results, d: str) -> dict[str, list[tuple[str, dict, callable]]]:
    """The robustness and heterogeneity studies in three groups: title, scores by value,
    value label."""
    def pct(v):
        return f"{100 * v:g}"

    def image(v):
        return "187 only" if v == "final" else str(v).split(":")[1].split(",")[0]

    imaging = []
    for agg, name in (("final", "one image"), (HEADLINE, "two images"),
                      ("images:4", "four images")):
        imaging.append((f"Measurement noise (\\%), {name}",
                        r.sweep(f"P3-{d}-noise", "features.measurement_noise",
                                features__aggregation=agg), pct))
    imaging += [
        ("Calibration error (\\%), with the reference",
         r.sweep(f"P3-{d}-calibration", "features.calibration_error",
                 features__aggregation=HEADLINE), pct),
        ("Calibration error (\\%), without the reference",
         r.sweep(f"P3-{d}-calibration-noref", "features.calibration_error",
                 features__aggregation=HEADLINE), pct),
        ("Earlier image (min; later at 187\\,min)",
         r.sweep(f"P3-{d}-image-time", "features.aggregation"), image),
    ]
    start = [
        ("Reset error (CV)", r.sweep(f"P3-{d}-reset", "reservoir.params.reset_noise"),
         lambda v: f"{v:g}"),
        ("Warm-up (min)", r.sweep(f"P3-{d}-warmup", "reservoir.params.warmup"), str),
        ("Inducer during warm-up", r.sweep(f"P3-{d}-warmup-input",
                                           "reservoir.params.warmup_input"),
         lambda v: {"zero": "None", "rest": "Rest"}.get(v, v)),
        ("Inducer strengths between traps (levels)",
         r.sweep(f"P3-{d}-input-levels", "reservoir.params.input_levels"),
         lambda v: "Cont." if v == "continuous" else str(v)),
        ("Random layout, exchange range (spacings)",
         r.sweep(f"P6-{d}-gaussian", "reservoir.params.interaction_diameters"),
         lambda v: f"{v:g}"),
    ]
    kinetic = r.sweep(f"P3-{d}-heterogeneity", "reservoir.params.param_heterogeneity_cv")
    kinetic.update(r.sweep(f"P3-{d}-heterogeneity-64", "reservoir.params.param_heterogeneity_cv"))
    heterogeneity = [
        ("Kinetic variation (CV)", kinetic, lambda v: f"{v:.2g}"),
        ("Cell density between traps (CV)",
         r.sweep(f"P3-{d}-density", "reservoir.params.heterogeneity_density_cv"),
         lambda v: f"{v:g}"),
        ("Washout between traps (CV)",
         r.sweep(f"P3-{d}-dilution", "reservoir.params.heterogeneity_dilution_cv"),
         lambda v: f"{v:g}"),
    ]
    return {"imaging": imaging, "start": start, "heterogeneity": heterogeneity}


def sweep_block(title: str, points: dict, label) -> list[str]:
    keys = sorted((k for k in points if k is not None),
                  key=lambda k: (isinstance(k, str) and k != "final", str(k) if
                                 isinstance(k, str) else k))
    if not keys:
        return []
    means = [mean_sd(points[k]) for k in keys]
    n = len(keys)
    return [
        r"\begin{tabular}{@{}l" + "c" * n + "@{}}",
        rf"\multicolumn{{{n + 1}}}{{@{{}}l}}{{\textit{{{title}}}}} \\",
        "Value & " + " & ".join(label(k) for k in keys) + r" \\",
        "Macro F1 & " + " & ".join(f"{m:.3f}" for m, _ in means) + r" \\",
        "SD & " + " & ".join(f"{s:.3f}" for _, s in means) + r" \\",
        r"\end{tabular}", r"\par\medskip", ""]


def robustness_tables(r: Results, d: str) -> str:
    captions = {
        "imaging": "Macro F1 against how the array is imaged (balanced task, ridge on two "
                   "images unless stated; mean and SD over the reporting seeds).",
        "start": "Macro F1 against how each beat starts, the inducer strengths and the "
                 "layout of the traps (balanced task, ridge on two images; mean and SD over "
                 "the reporting seeds).",
        "heterogeneity": "Macro F1 against the variation between colonies (balanced task, "
                         "ridge on two images; mean and SD over the reporting seeds)."}
    out = []
    for group, studies in sweeps(r, d).items():
        blocks = [line for title, points, label in studies
                  for line in sweep_block(title, points, label)]
        out += [r"\begin{table}[H]", r"    \raggedright", r"    \footnotesize",
                f"    \\caption{{{captions[group]}}}",
                f"    \\label{{tab:si-robustness-{group}}}", *blocks, r"\end{table}", ""]
    return "\n".join(out)


def traps_table(r: Results, d: str) -> str:
    ridge, forest, ideal = traps(r, d), traps(r, d, forest=True), ideal_traps(r, d)
    rows = [[f"{n:,}", pm(ridge.get(n)), pm(forest.get(n)), pm(ideal.get(n))]
            for n in sorted(set(ridge) | set(forest) | set(ideal))]
    return table("tab:si-traps",
                 "Macro F1 against the number of traps (balanced task; mean $\\pm$ SD over the "
                 "reporting seeds): ridge on two images (the design), the random forest on "
                 "four images, and ridge on the ideal reading (the mean of the full observed "
                 "state over the beat, without noise).",
                 "@{}rccc@{}", ["Traps $N$", "Ridge, 2 images", "Forest, 4 images",
                                "Ideal reading"], rows)


def flat_table(path: Path | None, d: str) -> str:
    rows = [[f"{cv:g}", f"${np.mean(v):.1f} \\pm {np.std(v, ddof=1):.1f}$"]
            for cv, v in sorted(flat_shares(path, d).items())]
    return table("tab:si-flat",
                 "Share of colonies that do not oscillate (\\%) against the kinetic variation, "
                 "after the warm-up and without inducer (mean $\\pm$ SD over the reporting "
                 "seeds; a colony is flat when its LuxI varies by less than a factor of 1.5 "
                 "over 200\\,min).",
                 "@{}cc@{}", ["Kinetic variation (CV)", "Flat colonies (\\%)"], rows)


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
    text = "\n".join(["% Additional results, from scripts/paper/si_tables.py",
                      metrics_table(r, d), folds_table(r, d), per_class_table(r, d),
                      readouts_table(r, d), traps_table(r, d), robustness_tables(r, d),
                      flat_table(args.flat_share, d)])
    (out / "si_tables.tex").write_text(text)
    print(out / "si_tables.tex")
    if r.missing:
        print(f"Not found ({len(r.missing)}): " + ", ".join(sorted(r.missing)[:30]),
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
