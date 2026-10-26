"""Analyse per-fold classification run artefacts and generate paper metrics.

Loads fold NPZ files produced by this repo's classification pipeline, prints
comprehensive per-fold and aggregate metrics tables, and optionally saves
standard diagnostic figures.

Usage
-----
Minimal (metrics only, no plots)::

    python scripts/analysis/analyse_fold_results.py --trial_name P4-readout-balanced-best

All plots saved::

    python scripts/analysis/analyse_fold_results.py --trial_name P4-readout-balanced-best --plots

Select specific plot types::

    python scripts/analysis/analyse_fold_results.py --trial_name P4-readout-balanced-best \\
        --plots confusion

Via glob pattern::

    python scripts/analysis/analyse_fold_results.py \\
        --glob 'results/runs/<run>/fold-*.npz' --plots all

Notes
-----
- All figures are saved under ``results/figures/metrics/<trial_name>/``.
- ``--plots`` alone saves all plot types (equivalent to ``--plots all``).
- The PCA of the states is skipped if the fold files were saved without
  ``save_states=True``.

Available plot types
--------------------
confusion
    The confusion matrix of all folds together, each row divided by its sum.
states
    The test states of all folds on their first two principal components (each feature
    scaled to unit variance), one colour per true class.
all
    All of the above.
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils.analysis import evaluate_performance
from utils.preprocessing import class_indices
from utils.results import FoldArtifact
from utils.visualisation import (
    _save_figure,
    binary_classes,
    classes,
    plot_confusion_matrix,
    use_v0_1_style,
)

PLOT_TYPES = ("confusion", "states")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _stack_preds(artifacts: list[FoldArtifact]) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_pred) as 1-D class-index arrays across all folds."""
    y_true = np.concatenate([class_indices(a.Y_test) for a in artifacts])
    y_pred = np.concatenate([class_indices(a.Y_pred) for a in artifacts])
    return y_true, y_pred


def _fmt(v: Any, decimals: int = 4) -> str:
    return f"{v:.{decimals}f}" if isinstance(v, (float, np.floating)) else str(v)


# ---------------------------------------------------------------------------
# Console tables
# ---------------------------------------------------------------------------

_SCALAR_KEYS = ["f1", "accuracy", "precision", "recall", "mcc", "kappa", "runtime"]
_SCALAR_HEADERS = ["fold", "F1", "Accuracy", "Precision", "Recall", "MCC", "Kappa", "Runtime(s)"]
_SCALAR_COL_W = [5] + [10] * len(_SCALAR_KEYS)


def _print_fold_table(artifacts: list[FoldArtifact]) -> None:
    """Print per-fold scalar metrics with a mean ± std footer."""
    row_fmt = "  ".join(f"{{:<{w}}}" for w in _SCALAR_COL_W)
    sep = "  ".join("-" * w for w in _SCALAR_COL_W)

    print("Per-fold metrics:")
    print(row_fmt.format(*_SCALAR_HEADERS))
    print(sep)

    rows: list[list[float]] = []
    for a in artifacts:
        m = a.metrics
        vals = [float(m[k]) if k in m else 0.0 for k in _SCALAR_KEYS]
        rows.append(vals)
        print(row_fmt.format(str(a.fold_index), *[_fmt(v) for v in vals]))

    arr = np.array(rows)
    print(sep)
    print(row_fmt.format("mean", *[_fmt(v) for v in arr.mean(axis=0)]))
    print(row_fmt.format("std",  *[_fmt(v) for v in arr.std(axis=0, ddof=1)]))
    print()


def _print_class_table(artifacts: list[FoldArtifact]) -> None:
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
        stds = arr.std(axis=0, ddof=1)
        # Display support as integer mean (no std — it's constant across folds)
        cells = [f"{means[i]:.4f}±{stds[i]:.4f}" for i in range(3)]
        cells.append(f"{means[3]:.0f}")
        label = key if key in agg_keys else f"class {key}"
        print(row_fmt.format(label, *cells))
    print()


def _print_extra_metrics(artifacts: list[FoldArtifact]) -> None:
    """Print MCC and Cohen's Kappa on concatenated predictions."""
    y_true, y_pred = _stack_preds(artifacts)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    print("Aggregate metrics (concatenated predictions across all folds):")
    print(f"  MCC:          {mcc:.4f}")
    print(f"  Cohen's Kappa:{kappa:.4f}")
    print()


def _print_model_info(artifacts: list[FoldArtifact]) -> None:
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

def plot_state_pca(states: np.ndarray, labels: np.ndarray, filename: str) -> None:
    """Draw the states on their first two principal components, one colour per true class.

    Each feature is scaled to unit variance first, so that no state variable dominates by
    its units. ``labels`` are the true labels of the states, in any shape that
    :func:`utils.preprocessing.class_indices` reads.
    """
    pca = make_pipeline(StandardScaler(), PCA(n_components=2))
    points = pca.fit_transform(states.reshape(len(states), -1))
    explained = pca[-1].explained_variance_ratio_
    y = class_indices(labels)
    names = binary_classes if np.asarray(labels).shape[-1] == 2 else classes
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in np.unique(y):
        ax.scatter(*points[y == c].T, s=6, alpha=0.6, label=names[c])
    ax.set_xlabel(f"PC 1 ({explained[0]:.0%} of the variance)")
    ax.set_ylabel(f"PC 2 ({explained[1]:.0%} of the variance)")
    ax.legend(markerscale=2)
    _save_figure(filename=os.path.join("metrics", filename))
    plt.close(fig)


def _run_plots(artifacts: list[FoldArtifact], trial_name: str, wanted: set) -> None:
    if "confusion" in wanted:
        Y_test = np.concatenate([a.Y_test for a in artifacts])
        Y_pred = np.concatenate([a.Y_pred for a in artifacts])
        plot_confusion_matrix(evaluate_performance(Y_test, Y_pred).confusion_matrix,
                              show=False,
                              filename=os.path.join(trial_name, "confusion-matrix.png"))
        plt.close("all")

    if "states" in wanted:
        with_states = [a for a in artifacts if a.test_states is not None]
        if with_states:
            plot_state_pca(np.concatenate([a.test_states for a in with_states]),
                           np.concatenate([a.Y_test for a in with_states]),
                           os.path.join(trial_name, "states-pca.png"))
        else:
            print("  No test_states in fold files: no PCA of the states.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    use_v0_1_style()
    parser = argparse.ArgumentParser(
        description="Analyse saved fold NPZ artefacts and report headline metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
plot types:
  confusion   the confusion matrix of all folds, each row divided by its sum
  states      PCA of the test states, one colour per true class
  all         both (also the default when --plots is given alone)
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
        wanted = set(PLOT_TYPES)
    unknown = wanted - set(PLOT_TYPES)
    if unknown:
        raise SystemExit(f"Unknown plot types {sorted(unknown)}; choose from {PLOT_TYPES} "
                         "or all.")

    if wanted:
        _run_plots(artifacts, trial_name, wanted)
        print(f"Figures saved to results/figures/metrics/{trial_name}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
