"""Plot cross-validation fold results saved as NPZ files.

This script is intentionally small: point it at a folder (or file glob) containing
per-fold `.npz` artifacts and it will:

- Load fold metrics and predictions
- Plot metrics across folds
- Plot an aggregated confusion matrix across folds
- Optionally plot t-SNE clustering using saved per-sample readout vectors

It reuses the plotting helpers in `utils/visualisation.py`.

Expected NPZ contents (best-effort, keys are optional):
- metrics: dict-like OR JSON string of metrics
- confusion_matrix: (C, C) array
- Y_true / y_true: (N, C) one-hot true labels
- Y_pred / y_pred: (N, C) probabilities/logits/one-hot predictions

Optional for t-SNE:
- readout / readout_outputs / y_scores: (N, C) model outputs per sample

The loader is forgiving: it will proceed with whatever is available.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils.visualisation import (
    plot_class_metrics,
    plot_confusion_matrix,
    plot_metrics_across_folds,
    plot_tsne_clustering,
)
from utils.results import FoldArtifact


@dataclass
class FoldResult:
    path: str
    metrics: Optional[Dict[str, Any]] = None
    confusion_matrix: Optional[np.ndarray] = None
    y_true: Optional[np.ndarray] = None
    y_pred: Optional[np.ndarray] = None
    readout: Optional[np.ndarray] = None


_METRICS_KEYS = ("metrics", "metric", "results")
_CM_KEYS = ("confusion_matrix", "cm")
_YTRUE_KEYS = ("Y_true", "y_true", "test_y", "Y_test", "y_test")
_YPRED_KEYS = ("Y_pred", "y_pred", "pred_y", "yhat", "Y_hat")
_READOUT_KEYS = ("readout", "readout_outputs", "y_scores", "scores", "logits", "probas")


def _maybe_load_dict(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    # np.savez stores python objects as 0-d object arrays when allow_pickle=True
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
        if isinstance(value, dict):
            return value
    if isinstance(value, (str, bytes)):
        try:
            return json.loads(value)
        except Exception:
            return None
    return None


def _first_key(npz: np.lib.npyio.NpzFile, keys: Sequence[str]) -> Optional[str]:
    for k in keys:
        if k in npz.files:
            return k
    return None


def _load_fold(path: str) -> FoldResult:
    with np.load(path, allow_pickle=True) as npz:
        metrics_k = _first_key(npz, _METRICS_KEYS)
        cm_k = _first_key(npz, _CM_KEYS)
        ytrue_k = _first_key(npz, _YTRUE_KEYS)
        ypred_k = _first_key(npz, _YPRED_KEYS)
        readout_k = _first_key(npz, _READOUT_KEYS)

        metrics = _maybe_load_dict(npz[metrics_k]) if metrics_k else None
        cm = npz[cm_k] if cm_k else None
        y_true = npz[ytrue_k] if ytrue_k else None
        y_pred = npz[ypred_k] if ypred_k else None
        readout = npz[readout_k] if readout_k else None

        # Sometimes confusion_matrix is inside metrics.
        if cm is None and metrics is not None and "confusion_matrix" in metrics:
            cm = np.asarray(metrics["confusion_matrix"])

        return FoldResult(
            path=path,
            metrics=metrics,
            confusion_matrix=cm,
            y_true=y_true,
            y_pred=y_pred,
            readout=readout,
        )


def _coerce_metrics_list(folds: List[FoldResult]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for fr in folds:
        if fr.metrics is None:
            continue
        out.append(fr.metrics)
    return out


def _sum_confusion_matrices(folds: List[FoldResult]) -> Optional[np.ndarray]:
    cms = [fr.confusion_matrix for fr in folds if fr.confusion_matrix is not None]
    if not cms:
        return None
    cm_sum = np.zeros_like(cms[0], dtype=float)
    for cm in cms:
        cm_sum = cm_sum + cm
    return cm_sum


def _stack_one_hot_arrays(folds: List[FoldResult]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    y_true = [fr.y_true for fr in folds if fr.y_true is not None]
    y_pred = [fr.y_pred for fr in folds if fr.y_pred is not None]
    if y_true and y_pred and len(y_true) == len(y_pred):
        return np.concatenate(y_true, axis=0), np.concatenate(y_pred, axis=0)
    return None, None


def _stack_readouts(folds: List[FoldResult]) -> Optional[np.ndarray]:
    arrs = [fr.readout for fr in folds if fr.readout is not None]
    if not arrs:
        return None
    return np.concatenate(arrs, axis=0)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Plot CV fold results from saved npz artifacts")
    parser.add_argument(
        "paths",
        nargs="*",
        help="NPZ file paths and/or globs (e.g. results/folds/*.npz). Omit when using --trial_name.",
    )
    parser.add_argument(
        "--trial_name",
        type=str,
        default=None,
        help="Trial name to load via FoldArtifact.load_trial() (alternative to positional paths).",
    )
    parser.add_argument(
        "--out-prefix",
        default="cv_",
        help="Prefix applied to saved figure filenames (saved under results/metrics/)",
    )
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    parser.add_argument("--tsne", action="store_true", help="Also plot t-SNE using readout vectors")

    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.trial_name and not args.paths:
        raise SystemExit("Provide either positional paths/globs or --trial_name.")

    if args.trial_name:
        artifacts = FoldArtifact.load_trial(args.trial_name)
        folds = [_load_fold(a.path) for a in artifacts]
    else:
        expanded: List[str] = []
        for p in args.paths:
            matches = glob.glob(p)
            expanded.extend(matches if matches else [p])

        expanded = [p for p in expanded if os.path.isfile(p) and p.endswith(".npz")]
        expanded.sort()

        if not expanded:
            raise SystemExit("No .npz files matched the given paths/globs")

        folds = [_load_fold(p) for p in expanded]

    metrics_list = _coerce_metrics_list(folds)
    if metrics_list:
        plot_metrics_across_folds(
            metrics_list,
            show=args.show,
            filename=f"{args.out_prefix}metrics_across_folds.png",
        )

        # If class metrics exist, show from the first fold (common structure).
        cm0 = metrics_list[0].get("class_metrics") if isinstance(metrics_list[0], dict) else None
        if isinstance(cm0, dict):
            plot_class_metrics(
                cm0,
                show=args.show,
                filename=f"{args.out_prefix}class_metrics.png",
            )

    cm_sum = _sum_confusion_matrices(folds)
    if cm_sum is not None:
        plot_confusion_matrix(
            cm_sum,
            show=args.show,
            filename=f"{args.out_prefix}confusion_matrix_sum.png",
        )

    y_true, y_pred = _stack_one_hot_arrays(folds)
    if args.tsne:
        readouts = _stack_readouts(folds)
        if readouts is None:
            # Fall back to using y_pred as embedding input if no explicit readout saved.
            readouts = y_pred
        if readouts is not None and y_true is not None:
            plot_tsne_clustering(
                readouts,
                y_true,
                show=args.show,
                filename=f"{args.out_prefix}tsne.png",
            )

    print(f"Loaded {len(folds)} fold file(s).")
    print("Saved figures under results/metrics/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
