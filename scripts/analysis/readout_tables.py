#!/usr/bin/env python3
"""Tables S13-S15: the best trial of each classifier family in the Phase 4 readout studies.

Reads the archived study databases and prints, for each study, one row per classifier
family (best macro F1, trial number and key hyperparameters) as LaTeX table rows, or
writes them as JSON with ``--json``. The v0.1 tables reported these F1 values but took
most of the other families' hyperparameters from earlier databases, and left out the
MLP and perceptron families.

Usage
-----
    python scripts/analysis/readout_tables.py
    python scripts/analysis/readout_tables.py --archive v0.1 --json readout_tables.json
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

import optuna

from utils import config

STUDIES = ("P4-readout-balanced", "P4-readout-unbalanced", "P4-readout-binary")
FAMILIES = {"RF": "Random Forest", "GB": "Gradient Boosting", "KNN": "KNN",
            "Ridge": "Ridge Classifier", "LR": "Logistic Regression", "SVM": "SVM",
            "DT": "Decision Tree", "MLP": "MLP", "Bayes": "Naive Bayes",
            "Perceptron": "Perceptron"}
# Trial parameters shown per family, with their LaTeX labels (@ marks the value).
KEY_PARAMS = {
    "RF": (("rf_n_estimators", r"$n_{\mathrm{est}} = @$"), ("rf_criterion", "criterion = @"),
           ("rf_max_depth", r"depth $= @$")),
    "GB": (("gb_n_estimators", r"$n_{\mathrm{est}} = @$"), ("gb_learning_rate", r"lr $= @$"),
           ("gb_loss", "loss = @")),
    "KNN": (("knn_n_neighbors", "$k = @$"), ("knn_weights", "weights = @"),
            ("knn_p", "$p = @$")),
    "Ridge": (("ridge_alpha", r"$\alpha = @$"), ("ridge_solver", "solver = @")),
    "LR": (("lr_C", "$C = @$"), ("lr_solver", "solver = @")),
    "SVM": (("svc_C", "$C = @$"), ("svc_kernel", "kernel = @"), ("svc_degree", r"degree $= @$")),
    "DT": (("dt_criterion", "criterion = @"), ("min_samples_leaf", r"leaf $= @$")),
    "MLP": (("mlp_hidden_layer_sizes", "hidden $= @$"), ("mlp_activation", "activation = @"),
            ("mlp_solver", "solver = @")),
    "Perceptron": (("perceptron_penalty", "penalty = @"),
                   ("perceptron_alpha", r"$\alpha = @$")),
    "Bayes": (("bayes_var_smoothing", "smoothing $= @$"),),
}


def _number(value: Any) -> str:
    """Whole numbers from 100 up, three significant figures below, powers of ten outside
    [0.01, 10^4)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value).replace("_", r"\_")
    if isinstance(value, int):
        return str(value)
    if value != 0 and not 1e-2 <= abs(value) < 1e4:
        exponent = math.floor(math.log10(abs(value)))
        mantissa = f"{value / 10 ** exponent:.2g}"
        return (f"10^{{{exponent}}}" if mantissa == "1"
                else rf"{mantissa} \times 10^{{{exponent}}}")
    return f"{value:.0f}" if abs(value) >= 100 else f"{value:.3g}"


def best_per_family(study: optuna.Study) -> list[dict[str, Any]]:
    """The best complete trial of each classifier family, best first."""
    best: dict[str, optuna.trial.FrozenTrial] = {}
    for trial in sorted(study.trials, key=lambda t: t.number):
        if (trial.state != optuna.trial.TrialState.COMPLETE or trial.value is None
                or "classifier" not in trial.params):
            continue
        name = trial.params["classifier"]
        if name not in best or trial.value > best[name].value:
            best[name] = trial
    rows = []
    for name, trial in sorted(best.items(), key=lambda item: -item[1].value):
        params = {key: trial.params[key] for key, _ in KEY_PARAMS.get(name, ())
                  if key in trial.params}
        rows.append({"classifier": name, "label": FAMILIES.get(name, name),
                     "f1": float(trial.value), "trial": trial.number, "params": params})
    return rows


def summary(archive: str = "v0.1") -> dict[str, Any]:
    """Per study: trial counts and the best trial of each classifier family."""
    out = {}
    for name in STUDIES:
        path = config.optuna_db(name, archive)
        study = optuna.load_study(study_name=name, storage=f"sqlite:///{path}")
        complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        out[name] = {"trials": len(study.trials), "complete": len(complete),
                     "rows": best_per_family(study)}
    return out


def latex_rows(rows: list[dict[str, Any]]) -> list[str]:
    lines = []
    for i, row in enumerate(rows):
        f1 = f"{row['f1']:.3f}"
        f1 = rf"\textbf{{{f1}}}" if i == 0 else f1
        labels = dict(KEY_PARAMS.get(row["classifier"], ()))
        params = ", ".join(labels[key].replace("@", _number(value))
                           for key, value in row["params"].items())
        lines.append(f"        {row['label']} & {f1} & {params} \\\\")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archive", default="v0.1", help="Archive version to read.")
    parser.add_argument("--json", default=None, help="Write the rows to this JSON file.")
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tables = summary(args.archive)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(tables, f, indent=1)
            f.write("\n")
    for name, table in tables.items():
        print(f"% {name}: {table['trials']} trials ({table['complete']} complete)")
        print("\n".join(latex_rows(table["rows"])))
        print()


if __name__ == "__main__":
    main()
