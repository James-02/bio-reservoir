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

from paper_figures import CONTROLS, RAW, READOUTS, SEEDS, Results, readout, test_split
from si_tables import pm, table

from utils import config


def readouts_table(r: Results, d: str) -> str:
    rows = [[label.replace("\n", " "), pm(list(readout(r, d, "c", key).values())),
             pm(r.f1(RAW[key]))] for key, label in READOUTS]
    return table("tab:readouts",
                 "Each readout on the array's two images and on the 187 samples of the raw "
                 "beat (balanced task; macro F1, mean $\\pm$ SD over the reporting seeds; the "
                 "raw beat has no reservoir seed). Ridge is the design's readout. kNN, "
                 "$k$-nearest neighbors; HistGB, histogram-based gradient boosting; RBF SVM, "
                 "support vector machine with a radial basis function kernel. Settings in "
                 "Table~\\ref{S-tab:readouts}; four images in Table~\\ref{S-tab:si-readouts}.",
                 "@{}lcc@{}", ["Readout", "Array, 2 images", "Raw beat"], rows)


def tasks_table(r: Results, d: str) -> str:
    def test(key):
        found = test_split(r, key)
        return "--" if found is None else f"{found[0]:.3f}"

    rows = []
    for label, run, split, seeds, key in (
            ("Array (design)", f"{d}-headline", f"{d}-c", SEEDS[:3], "headline"),
            ("Isolated traps", "uncoupled-576", "uncoupled-576", SEEDS[:1], "uncoupled-576"),
            ("ESN, read like the array", "esn-image-576", "esn-image-576", SEEDS[:1],
             "esn-image-576"),
            ("ESN, clean states", "esn-576", "esn-576", SEEDS[:1], "esn-576")):
        rows.append([label, pm(r.seeds(f"P5-balanced-{run}"))]
                    + [pm(r.seeds(f"P5-{task}-{run}")) for task in ("unbalanced", "binary")]
                    + [pm(r.seeds(f"P3-{s}-{split}", seeds))
                       for s in ("validation-1", "validation-2")] + [test(key)])
    rows.append(["MIDRULE"])
    for key, label in (("ridge", "Raw beat, ridge"), ("histgb", "Raw beat, HistGB")):
        rows.append([label] + [pm(r.f1(f"P5-{task}-raw-{key}"))
                               for task in ("balanced", "unbalanced", "binary")]
                    + ["--", "--", test(f"raw-{key}")])
    return table("tab:tasks",
                 "The three tasks and the held-out splits, read by ridge (macro F1). Balanced "
                 "task: mean $\\pm$ SD over the five reporting seeds for every model. "
                 "Unbalanced and binary tasks: five seeds for the array, one (6348) for the "
                 "comparisons. Validation splits (803 beats of each of four classes, N, S, V "
                 "and Q): three seeds for the array, one for the comparisons. Test split (370 "
                 "beats of each of the same four classes): one seed, scored once; the bootstrap "
                 "95\\% confidence intervals, about $\\pm$0.015, are in "
                 "Table~\\ref{S-tab:si-heldout}. The raw beat has no reservoir seed, so its "
                 "scores have no SD. On the held-out splits the readout "
                 "is trained on the whole balanced pool, the macro F1 averages the four "
                 "classes present, and a beat predicted as F counts as an error "
                 "(Table~\\ref{S-tab:dataset}). Isolated traps: the array without AHL "
                 "exchange, tuned with the same budget; ESN: an echo state network of 576 units "
                 "read like the array or from its clean states (Table~\\ref{tab:mechanism}). "
                 "--, not run.",
                 "@{}lcccccc@{}", ["Model", "Balanced", "Unbalanced", "Binary", "Validation 1",
                                   "Validation 2", "Test"], rows, "footnotesize")


def mechanism_table(r: Results, d: str) -> str:
    def ablation(stem, value, forest):
        key = {"coupling": "reservoir.params.coupling", "delay": "reservoir.params.delay",
               "scramble": "task.scramble"}[stem]
        study = r.grid(f"P6-{d}-ablation-{stem}{'-forest' if forest else ''}", key)
        return pm(list(study.get((value,), {}).values()))

    rows = [["Array (design)", pm(r.seeds(f"P5-balanced-{d}-headline")),
             pm(r.seeds(f"P5-balanced-{d}-forest"))],
            ["Array, ideal reading$^{a}$", pm(r.seeds(f"P5-balanced-{d}-full")),
             pm(r.seeds(f"P5-balanced-{d}-full-forest"))], ["MIDRULE"],
            [r"\multicolumn{3}{@{}l}{\textit{Tuned with the same budget}}"]]
    for label, values in (
            ("Isolated traps", r.seeds("P5-balanced-uncoupled-576")),
            ("ESN, read like the array", r.seeds("P5-balanced-esn-image-576")),
            ("ESN, clean states$^{b}$", r.seeds("P5-balanced-esn-576")),
            ("Random tanh features", list(r.report(CONTROLS["random_features"]).values())),
            ("Leaky integrators", list(r.report(CONTROLS["fading_memories"]).values()))):
        rows.append([label, pm(values), "--"])
    rows += [["MIDRULE"], [r"\multicolumn{3}{@{}l}{\textit{Ablations of the design, not "
                           r"retuned}}"]]
    for label, stem, value in (("No AHL exchange", "coupling", "none"),
                               ("No feedback delay", "delay", 0),
                               ("Beat samples shuffled", "scramble", True)):
        rows.append([label, ablation(stem, value, False), ablation(stem, value, True)])
    rows.append(["Linear imitation$^{c}$", pm(r.seeds(f"P6-{d}-surrogate-c", SEEDS[:3])),
                 pm(r.seeds(f"P6-{d}-surrogate-e", SEEDS[:3]))])
    rows += [["MIDRULE"], ["Raw beat$^{d}$", pm(r.f1(RAW["ridge"])), pm(r.f1(RAW["rf"]))]]
    return table("tab:mechanism",
                 "What the colonies contribute (balanced task; macro F1, mean $\\pm$ SD over "
                 "the reporting seeds), read by ridge on two images and by the random forest "
                 "on four, unless noted. The comparisons were tuned for ridge "
                 "(Table~\\ref{S-tab:selection}); by seed in Table~\\ref{S-tab:si-comparisons}, "
                 "and the controls under both readouts in Table~\\ref{S-tab:si-ablations}. "
                 "$^{a}$The mean of each colony's LuxI and external AHL over the beat, "
                 "without noise. $^{b}$The mean of the network's noise-free states over the "
                 "beat. $^{c}$A ridge regression from the beat to the array's features, on "
                 "three of the seeds. $^{d}$The 187 samples of the beat. --, not run.",
                 "@{}lcc@{}", ["Model", "Ridge", "Random forest"], rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--design", default="p576", help="The design (default p576).")
    parser.add_argument("--out", type=Path, help="Output folder (default "
                        "results/figures/paper/<design>).")
    args = parser.parse_args(argv)
    r, d = Results(), args.design
    out = args.out or config.figures_dir(f"paper/{d}")
    out.mkdir(parents=True, exist_ok=True)
    for name, text in (("table2_readouts.tex", readouts_table(r, d)),
                       ("table3_tasks.tex", tasks_table(r, d)),
                       ("table4_mechanism.tex", mechanism_table(r, d))):
        (out / name).write_text("% From scripts/paper/article_tables.py\n" + text)
        print(out / name)
    if r.missing:
        print(f"Not found ({len(r.missing)}): " + ", ".join(sorted(r.missing)[:30]),
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
