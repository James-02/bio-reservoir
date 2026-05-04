"""Analyse per-fold classification run artefacts and generate paper metrics.

Loads fold NPZ files produced by this repo's classification pipeline, prints
comprehensive per-fold and aggregate metrics tables, and optionally saves
standard diagnostic figures.

Usage
-----
Minimal (metrics only, no plots)::

    python scripts/analysis/analyse_fold_results.py --trial_name r1a-refined-best

All plots saved::

    python scripts/analysis/analyse_fold_results.py --trial_name r1a-refined-best --plots

Select specific plot types::

    python scripts/analysis/analyse_fold_results.py --trial_name r1a-refined-best \\
        --plots confusion,tsne,class

Via glob pattern::

    python scripts/analysis/analyse_fold_results.py \\
        --glob 'results/runs/final-test-classification-fold-*.npz' --plots all

Notes
-----
- All figures are saved under ``results/metrics/<trial_name>/``.
- ``--plots`` alone saves all plot types (equivalent to ``--plots all``).
- ``--show`` displays each figure interactively in addition to saving.
- t-SNE on reservoir states is skipped if the fold files were saved without
  ``save_states=True``.

Available plot types
--------------------
confusion
    Aggregate normalised confusion matrix + per-fold confusion matrix grid.
class
    Per-class metric bars and metrics-across-folds bar chart.
tsne
    t-SNE of concatenated test predictions coloured by true/predicted class.
states
    t-SNE of reservoir states coloured by true class (requires save_states).
all
    All of the above.
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef

from utils.analysis import evaluate_performance
from utils.results import FoldArtifact
from utils.visualisation import (
    plot_class_metrics,
    plot_confusion_matrix,
    plot_metrics_across_folds,
    plot_tsne_clustering,
    plot_tsne_states,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _argmax_labels(Y: np.ndarray) -> np.ndarray:
    """Reduce one-hot (N, T, C) / (N, 1, C) / (N, C) to class indices (N,)."""
    Y = np.asarray(Y)
    return np.argmax(Y, axis=-1).flatten()


def _stack_preds(artifacts: List[FoldArtifact]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_pred) as 1-D class-index arrays across all folds."""
    y_true = np.concatenate([_argmax_labels(a.Y_test) for a in artifacts])
    y_pred = np.concatenate([_argmax_labels(a.Y_pred) for a in artifacts])
    return y_true, y_pred


def _fmt(v: Any, decimals: int = 4) -> str:
    return f"{v:.{decimals}f}" if isinstance(v, (float, np.floating)) else str(v)


# ---------------------------------------------------------------------------
# Console tables
# ---------------------------------------------------------------------------

_SCALAR_KEYS = ["f1", "accuracy", "precision", "recall", "rmse", "r2_score", "runtime"]
_SCALAR_HEADERS = ["fold", "F1", "Accuracy", "Precision", "Recall", "RMSE", "R²", "Runtime(s)"]
_SCALAR_COL_W = [5] + [10] * len(_SCALAR_KEYS)


def _print_fold_table(artifacts: List[FoldArtifact]) -> None:
    """Print per-fold scalar metrics with a mean ± std footer."""
    row_fmt = "  ".join(f"{{:<{w}}}" for w in _SCALAR_COL_W)
    sep = "  ".join("-" * w for w in _SCALAR_COL_W)

    print("Per-fold metrics:")
    print(row_fmt.format(*_SCALAR_HEADERS))
    print(sep)

    rows: List[List[float]] = []
    for a in artifacts:
        m = a.metrics
        vals = [float(m[k]) if k in m else 0.0 for k in _SCALAR_KEYS]
        rows.append(vals)
        print(row_fmt.format(str(a.fold_index), *[_fmt(v) for v in vals]))

    arr = np.array(rows)
    print(sep)
    print(row_fmt.format("mean", *[_fmt(v) for v in arr.mean(axis=0)]))
    print(row_fmt.format("std",  *[_fmt(v) for v in arr.std(axis=0)]))
    print()


def _print_class_table(artifacts: List[FoldArtifact]) -> None:
    """Print per-class precision / recall / F1 / support averaged across folds."""
    # Gather class keys from fold 0; order: numeric classes first, then avg rows
    cm0 = artifacts[0].metrics.class_metrics
    agg_keys = ("macro avg", "weighted avg")
    class_keys = [k for k in cm0 if k not in agg_keys and k != "accuracy"]

    metric_names = ["precision", "recall", "f1-score", "support"]
    col_w = [16, 16, 16, 16, 10]
    row_fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
    sep = "  ".join("-" * w for w in col_w)

    print("Per-class metrics (mean ± std across folds):")
    print(row_fmt.format("class", "precision", "recall", "F1", "support"))
    print(sep)

    for key in list(class_keys) + list(agg_keys):
        fold_vals = []
        for a in artifacts:
            cm = a.metrics.class_metrics
            if key in cm:
                fold_vals.append([cm[key].get(mn, 0.0) for mn in metric_names])
        if not fold_vals:
            continue
        arr = np.array(fold_vals)
        means = arr.mean(axis=0)
        stds = arr.std(axis=0)
        # Display support as integer mean (no std — it's constant across folds)
        cells = [f"{means[i]:.4f}±{stds[i]:.4f}" for i in range(3)]
        cells.append(f"{means[3]:.0f}")
        label = key if key in agg_keys else f"class {key}"
        print(row_fmt.format(label, *cells))
    print()


def _print_extra_metrics(artifacts: List[FoldArtifact]) -> None:
    """Print MCC and Cohen's Kappa on concatenated predictions."""
    y_true, y_pred = _stack_preds(artifacts)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    print("Aggregate metrics (concatenated predictions across all folds):")
    print(f"  MCC:          {mcc:.4f}")
    print(f"  Cohen's Kappa:{kappa:.4f}")
    print()


def _print_model_info(artifacts: List[FoldArtifact]) -> None:
    """Print the classifier class and hypers from fold 0."""
    mh = artifacts[0].model_hypers or {}
    clf_cls = mh.get("model", None)
    clf_hypers = mh.get("model_hypers", {})
    clf_name = clf_cls.__name__ if (clf_cls and hasattr(clf_cls, "__name__")) else str(clf_cls) if clf_cls else "unknown"
    print(f"Classifier : {clf_name}")
    if clf_hypers:
        for k, v in sorted(clf_hypers.items()):
            print(f"  {k}: {v}")
    print()


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _fig_out(trial_name: str, suffix: str) -> str:
    """Return filename passed to visualisation helpers.

    All plot_* functions in utils.visualisation already prepend ``metrics/``
    themselves before calling ``_save_figure``, which in turn prepends
    ``results/``.  So we only need to provide ``<trial_name>/<suffix>``.
    """
    return os.path.join(trial_name, suffix)


def _run_plots(
    artifacts: List[FoldArtifact],
    trial_name: str,
    wanted: set,
    show: bool,
) -> None:
    Y_test_all = np.concatenate([a.Y_test for a in artifacts])
    Y_pred_all = np.concatenate([a.Y_pred for a in artifacts])
    agg = evaluate_performance(Y_test_all, Y_pred_all)

    if "confusion" in wanted:
        plot_confusion_matrix(
            agg.confusion_matrix,
            show=show,
            filename=_fig_out(trial_name, "confusion-matrix.png"),
        )

    if "class" in wanted:
        plot_class_metrics(
            agg.class_metrics,
            show=show,
            filename=_fig_out(trial_name, "class-metrics.png"),
        )
        plot_metrics_across_folds(
            [a.metrics for a in artifacts],
            metric_names=["f1", "accuracy", "precision", "recall", "rmse"],
            show=show,
            filename=_fig_out(trial_name, "metrics-across-folds.png"),
        )

    if "tsne" in wanted:
        plot_tsne_clustering(
            Y_pred_all,
            Y_test_all,
            show=show,
            filename=_fig_out(trial_name, "tsne-predictions.png"),
        )

    if "states" in wanted:
        states = [a.test_states for a in artifacts if a.test_states is not None]
        labels = [a.Y_test for a in artifacts if a.test_states is not None]
        if states:
            plot_tsne_states(
                np.concatenate(states, axis=0),
                np.concatenate(labels, axis=0),
                show=show,
                filename=_fig_out(trial_name, "tsne-states.png"),
            )
        else:
            print("  No test_states in fold files — skipping state t-SNE.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyse saved fold NPZ artefacts and report headline metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
plot types:
  confusion   aggregate + per-fold normalised confusion matrices
  class       per-class metric bars and metrics-across-folds chart
  tsne        t-SNE of test predictions coloured by true/predicted class
  states      t-SNE of reservoir states coloured by true class
  all         all of the above (also the default when --plots is given alone)
""",
    )
    parser.add_argument(
        "--trial_name",
        type=str,
        default=None,
        help="Trial name to load via FoldArtifact.load_trial().",
    )
    parser.add_argument(
        "--glob",
        dest="glob_pattern",
        default=None,
        help="Glob pattern for fold NPZ files (alternative to --trial_name).",
    )
    parser.add_argument(
        "--plots",
        nargs="?",
        const="all",
        default="",
        metavar="TYPE[,TYPE...]",
        help="Comma-separated plot types to generate (default: none; bare flag = all).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively (also saves them).",
    )

    args = parser.parse_args()

    # --- Load artifacts ---
    if args.trial_name:
        artifacts = FoldArtifact.load_trial(args.trial_name)
        trial_name = args.trial_name
    elif args.glob_pattern:
        paths = sorted(glob.glob(args.glob_pattern))
        if not paths:
            raise SystemExit(f"No files matched: {args.glob_pattern!r}")
        artifacts = [FoldArtifact.load(p) for p in paths]
        trial_name = os.path.basename(os.path.dirname(paths[0])) or "analysis"
    else:
        raise SystemExit("Provide --trial_name or --glob.")

    print(f"Trial  : {trial_name}")
    print(f"Folds  : {len(artifacts)}")
    print()

    # --- Console output ---
    _print_model_info(artifacts)
    _print_fold_table(artifacts)
    _print_class_table(artifacts)
    _print_extra_metrics(artifacts)

    # --- Plots ---
    plots_arg = (args.plots or "").strip()
    wanted = {p.strip().lower() for p in plots_arg.split(",") if p.strip()}
    if "all" in wanted:
        wanted = {"confusion", "class", "tsne", "states"}

    if wanted:
        _run_plots(artifacts, trial_name, wanted, args.show)
        print(f"Figures saved to results/metrics/{trial_name}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
