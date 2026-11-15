"""Tables 2-4 of the main text as LaTeX, from the runs.

Writes ``table2_readouts.tex`` (each readout on the array's two images and on the raw beat),
``table3_tasks.tex`` (the three tasks and the held-out splits) and ``table4_mechanism.tex``
(the equal-budget comparisons and the ablations, read by ridge and by the random forest).

    python scripts/paper/article_tables.py --out <report>/tables/article
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from paper_figures import ABLATIONS, CONTROLS, RAW, READOUTS, Results, readout, test_split
from si_tables import pm, table

from utils import config


def readouts_table(r: Results) -> str:
    rows = [[label.replace("\n", " "), pm(list(readout(r, "c", key).values())),
             pm(r.f1(RAW[key]))] for key, label in READOUTS]
    return table("tab:readouts",
                 "Each readout on the array's two images and on the 187 samples of the raw "
                 "beat (balanced task; macro F1, mean $\\pm$ SD over the reporting seeds; the "
                 "raw beat has no reservoir seed). Ridge is the design's readout. kNN, "
                 "$k$-nearest neighbors; HistGB, histogram-based gradient boosting; RBF SVM, "
                 "support vector machine with a radial basis function kernel. Settings in "
                 "Table~\\ref{S-tab:readouts}; four images in Table~\\ref{S-tab:si-readouts}.",
                 "@{}lcc@{}", ["Readout", "Array, 2 images", "Raw beat"], rows)


def tasks_table(r: Results) -> str:
    def test(key):
        found = test_split(r, key)
        return "--" if found is None else f"{found[0]:.3f}"

    rows = []
    for label, run, key in (
            ("Array (design)", "array", "headline"),
            ("Isolated traps", "isolated-traps", "isolated-traps"),
            ("ESN, read like the array", "esn-image", "esn-image"),
            ("ESN, clean states", "esn-mean", "esn-mean")):
        rows.append([label, pm(r.seeds(f"balanced-{run}"))]
                    + [pm(r.seeds(f"{task}-{run}")) for task in ("unbalanced", "binary")]
                    + [pm(r.seeds(f"{s}-{run}")) for s in ("validation-1", "validation-2")]
                    + [test(key)])
    rows.append(["MIDRULE"])
    for key, label in (("ridge", "Raw beat, ridge"), ("histgb", "Raw beat, HistGB")):
        rows.append([label] + [pm(r.f1(f"{task}-raw-{key}"))
                               for task in ("balanced", "unbalanced", "binary")]
                    + ["--", "--", test(f"raw-{key}")])
    return table("tab:tasks",
                 "Scores on the three tasks and on the held-out splits (macro F1 of the ridge "
                 "readout; mean $\\pm$ SD over the five reservoir seeds; the raw beat has no "
                 "seed). On the held-out splits the readout is trained on the whole balanced "
                 "pool and tested on new beats of four classes (N, S, V and Q; no F beats are "
                 "left outside the pool); the test split was scored once, on one seed (95\\% "
                 "confidence intervals in Table~\\ref{S-tab:si-heldout}). Isolated traps: the "
                 "array without AHL exchange; ESN: an echo state network of 576 units "
                 "(Table~\\ref{tab:mechanism}). --, not run.",
                 "@{}lcccccc@{}", ["Model", "Balanced", "Unbalanced", "Binary", "Validation 1",
                                   "Validation 2", "Test"], rows, "footnotesize", colsep="3pt")


def mechanism_table(r: Results) -> str:
    def ablation(stem, value, forest):
        key = {"coupling": "reservoir.params.coupling", "delay": "reservoir.params.delay",
               "scramble": "task.scramble"}[stem]
        study = r.grid(f"ablation-{ABLATIONS[stem]}{'-forest' if forest else ''}", key)
        return pm(list(study.get((value,), {}).values()))

    rows = [["Array (design)", pm(r.seeds("balanced-array")),
             pm(r.seeds("balanced-array-forest"))],
            ["Array, ideal reading$^{a}$", pm(r.seeds("balanced-array-ideal")),
             pm(r.seeds("balanced-array-ideal-forest"))], ["MIDRULE"],
            [r"\multicolumn{3}{@{}l}{\textit{Tuned with the same budget}}"]]
    forest = {name: f"{study}-forest" for name, study in CONTROLS.items()}
    for label, values, forest_values in (
            ("Isolated traps", r.seeds("balanced-isolated-traps"),
             r.seeds("balanced-isolated-traps-forest")),
            ("ESN, read like the array", r.seeds("balanced-esn-image"),
             r.seeds("balanced-esn-image-forest")),
            ("ESN, clean states$^{b}$", r.seeds("balanced-esn-mean"),
             r.seeds("balanced-esn-mean-forest")),
            ("Random tanh features", list(r.report(CONTROLS["random_features"]).values()),
             list(r.report(forest["random_features"]).values())),
            ("Leaky integrators", list(r.report(CONTROLS["fading_memories"]).values()),
             list(r.report(forest["fading_memories"]).values()))):
        rows.append([label, pm(values), pm(forest_values)])
    rows += [["MIDRULE"], [r"\multicolumn{3}{@{}l}{\textit{Ablations of the design, not "
                           r"retuned}}"]]
    for label, stem, value in (("No AHL exchange", "coupling", "none"),
                               ("No feedback delay", "delay", 0),
                               ("Beat samples shuffled", "scramble", True)):
        rows.append([label, ablation(stem, value, False), ablation(stem, value, True)])
    rows.append(["Linear imitation$^{c}$", pm(r.seeds("surrogate-two-images")),
                 pm(r.seeds("surrogate-four-images"))])
    rows += [["MIDRULE"], ["Raw beat", pm(r.f1(RAW["ridge"])), pm(r.f1(RAW["rf"]))]]
    return table("tab:mechanism",
                 "What the colonies contribute (balanced task; macro F1, mean $\\pm$ SD over "
                 "the five reservoir seeds), read by ridge on two images and by a random "
                 "forest on four. Each comparison was tuned for ridge with the same budget "
                 "(Table~\\ref{S-tab:selection}) and read by the forest at the same settings; "
                 "the ablations change the design without retuning it. The comparisons by seed "
                 "are in Table~\\ref{S-tab:si-comparisons}, and the ablations in "
                 "Table~\\ref{S-tab:si-ablations}. "
                 "$^{a}$The noise-free mean of each colony's LuxI and external AHL over the "
                 "beat, a reference that a device could not image. $^{b}$The mean of the "
                 "network's noise-free states over the "
                 "beat. $^{c}$A ridge regression from the beat to the array's features.",
                 "@{}lcc@{}", ["Model", "Ridge", "Random forest"], rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path,
                        help="Output folder (default results/figures/paper).")
    args = parser.parse_args(argv)
    r = Results()
    out = args.out or config.figures_dir("paper")
    out.mkdir(parents=True, exist_ok=True)
    for name, text in (("table2_readouts.tex", readouts_table(r)),
                       ("table3_tasks.tex", tasks_table(r)),
                       ("table4_mechanism.tex", mechanism_table(r))):
        (out / name).write_text("% From scripts/paper/article_tables.py\n" + text)
        print(out / name)
    if r.missing:
        print(f"Not found ({len(r.missing)}): " + ", ".join(sorted(r.missing)[:30]),
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
