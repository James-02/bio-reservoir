"""Diagnostics + richer plots for fold-based classification runs.

This script is similar to `analyse_fold_results.py` but focuses on *diagnostic*
visualisations that help explain *why* certain folds/classes perform poorly,
especially for reservoir-computing models.

It expects fold artifacts saved as NPZ files containing at least:
- `Y_test`: true labels (typically one-hot, shape (N,1,C))
- `Y_pred`: predicted probabilities/logits (shape (N,1,C) or (N,C))
- `metrics`: dict containing `confusion_matrix` and `class_metrics` (optional)

Optionally, if present, it will also use:
- `test_states`: reservoir state representation per sample

Outputs are saved under `results/diagnostics/`.

Usage:
    python3 analyse_fold_diagnostics.py --glob 'results/runs/final-test-classification-fold-*.npz' --plots
    python3 analyse_fold_diagnostics.py --trial_name my-study-0 --plots

Notes:
- For state-space plots, we do PCA first (fast + denoise), then (optionally)
  t-SNE if you request it.
- For prediction-quality plots, we avoid t-SNE on 5D probabilities (often
  uninformative) and instead plot confidence/entropy/calibration.
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from utils.analysis import evaluate_performance
from utils.results import FoldArtifact
from utils.visualisation import classes as categorical_classes, binary_classes


RESULTS_DIR = os.path.join("results", "diagnostics")


def _class_names(n_classes: int) -> List[str]:
    """Return human-readable class names consistent with utils.visualisation."""
    if int(n_classes) == 2:
        return list(binary_classes)
    if int(n_classes) <= len(categorical_classes):
        return list(categorical_classes)[: int(n_classes)]
    return [str(i) for i in range(int(n_classes))]


@dataclass(frozen=True)
class FoldData:
    path: str
    y_pred: np.ndarray
    y_true: np.ndarray
    metrics: Dict[str, Any] | None
    test_states: Any | None


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _savefig(name: str) -> None:
    _ensure_dir(RESULTS_DIR)
    plt.savefig(os.path.join(RESULTS_DIR, name), dpi=250, bbox_inches="tight")


def _to_1d_labels(Y: np.ndarray) -> np.ndarray:
    Y = np.asarray(Y)
    if Y.ndim == 3 and Y.shape[1] == 1:
        return np.argmax(Y[:, 0, :], axis=-1)
    if Y.ndim == 2:
        return np.argmax(Y, axis=-1)
    if Y.ndim == 1:
        return Y.astype(int)
    raise ValueError(f"Unsupported Y shape: {Y.shape}")


def _to_prob_matrix(Y_pred: np.ndarray) -> np.ndarray:
    Y_pred = np.asarray(Y_pred)
    if Y_pred.ndim == 3 and Y_pred.shape[1] == 1:
        M = Y_pred[:, 0, :]
    elif Y_pred.ndim == 2:
        M = Y_pred
    else:
        M = Y_pred.reshape(len(Y_pred), -1)

    # If it looks like logits (not summing to 1), softmax it.
    row_sums = M.sum(axis=1)
    if not (np.all(np.isfinite(row_sums)) and np.allclose(row_sums.mean(), 1.0, atol=1e-2)):
        M = M - M.max(axis=1, keepdims=True)
        exp = np.exp(M)
        M = exp / exp.sum(axis=1, keepdims=True)
    return M


def _entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def _load_fold(path: str) -> FoldData:
    d = np.load(path, allow_pickle=True)
    y_pred = d["Y_pred"]
    y_true = d["Y_test"]
    metrics = None
    if "metrics" in d.files:
        m = d["metrics"]
        if isinstance(m, np.ndarray) and m.shape == ():
            metrics = m.item()
        elif isinstance(m, dict):
            metrics = m

    test_states = d["test_states"] if "test_states" in d.files else None
    return FoldData(path=path, y_pred=y_pred, y_true=y_true, metrics=metrics, test_states=test_states)


def _stack_folds(folds: List[FoldData]) -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.concatenate([f.y_true for f in folds], axis=0)
    y_pred = np.concatenate([f.y_pred for f in folds], axis=0)
    return y_true, y_pred


def plot_confidence_hist(prob: np.ndarray, y_true: np.ndarray, y_pred_cls: np.ndarray, *, name: str) -> None:
    correct = (y_pred_cls == y_true)
    conf = prob.max(axis=1)

    plt.figure(figsize=(8, 4.8))
    plt.hist(conf[correct], bins=25, alpha=0.7, label=f"Correct (n={correct.sum()})")
    plt.hist(conf[~correct], bins=25, alpha=0.7, label=f"Incorrect (n={(~correct).sum()})")
    plt.xlabel("Max predicted probability")
    plt.ylabel("Count")
    plt.title("Prediction confidence distribution")
    plt.legend()
    _savefig(name)
    plt.close()


def plot_entropy_by_class(prob: np.ndarray, y_true: np.ndarray, *, name: str) -> None:
    ent = _entropy(prob)
    classes = np.unique(y_true)
    labels = _class_names(int(prob.shape[1]))

    plt.figure(figsize=(9, 4.8))
    data = [ent[y_true == c] for c in classes]
    tick_labels = [labels[int(c)] if int(c) < len(labels) else str(int(c)) for c in classes]
    plt.boxplot(data, labels=tick_labels, showfliers=False)
    plt.xlabel("True class")
    plt.ylabel("Prediction entropy")
    plt.title("Uncertainty (entropy) by class")
    _savefig(name)
    plt.close()


def plot_per_class_recall_from_cm(cm: np.ndarray, *, name: str) -> None:
    # recall per class = diag / row_sum
    cm = cm.astype(float)
    rec = np.diag(cm) / np.maximum(cm.sum(axis=1), 1.0)
    labels = _class_names(int(len(rec)))
    plt.figure(figsize=(8, 4.8))
    plt.bar(np.arange(len(rec)), rec)
    plt.ylim(0, 1)
    plt.xticks(np.arange(len(rec)), labels, rotation=20, ha="right")
    plt.xlabel("Class")
    plt.ylabel("Recall")
    plt.title("Per-class recall")
    _savefig(name)
    plt.close()


def plot_pca_explained_variance(states: np.ndarray, *, name: str, max_components: int = 50) -> None:
    """Plot cumulative explained variance ratio for PCA components.

    Computed via SVD on the centered state matrix.
    """
    X = states - states.mean(axis=0, keepdims=True)
    # Singular values S relate to variance along PCs: var_i ∝ S_i^2
    _, S, _ = np.linalg.svd(X, full_matrices=False)
    var = S**2
    total = float(var.sum()) if var.size else 1.0
    evr = var / total
    k = min(int(max_components), evr.size)
    cum = np.cumsum(evr[:k])

    plt.figure(figsize=(8, 4.8))
    plt.plot(np.arange(1, k + 1), cum, marker="o")
    plt.ylim(0, 1.01)
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA explained variance (reservoir states)")
    plt.grid(True, alpha=0.25)
    _savefig(name)
    plt.close()


def _flatten_states(test_states: Any) -> np.ndarray:
    """Convert saved `test_states` to a 2D matrix (N, D)."""
    arr = np.asarray(test_states)

    # Common saved format earlier in the repo was a python list of arrays.
    if arr.dtype == object:
        arr = np.asarray(list(test_states))

    # Often: (N, 1, D) or (N, D)
    if arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr[:, 0, :]
    if arr.ndim == 2:
        return arr

    # If still higher rank: flatten remaining dims
    return arr.reshape(arr.shape[0], -1)


def plot_state_pca(states: np.ndarray, y_true: np.ndarray, *, name: str, max_points: int = 3000) -> None:
    # Deterministic subsample for speed
    if len(states) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(states), size=max_points, replace=False)
        states = states[idx]
        y_true = y_true[idx]

    # Center
    X = states - states.mean(axis=0, keepdims=True)

    # PCA via SVD (no sklearn dependency)
    # Take first 2 components for plotting
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    emb = U[:, :2] * S[:2]

    labels = _class_names(int(np.max(y_true) + 1))

    plt.figure(figsize=(8.5, 6.5))
    sc = plt.scatter(emb[:, 0], emb[:, 1], c=y_true.astype(int), cmap="tab10", s=10, alpha=0.75)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Reservoir state representation (PCA)")
    cbar = plt.colorbar(sc)
    # Use discrete tick labels for classes (works well for 2 or 5 classes)
    ticks = np.unique(y_true.astype(int))
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([labels[t] if t < len(labels) else str(int(t)) for t in ticks])
    cbar.set_label("True class")
    _savefig(name)
    plt.close()


def _artifact_to_folddata(a: FoldArtifact) -> FoldData:
    """Convert a FoldArtifact to the local FoldData view used by this script."""
    return FoldData(
        path=a.path,
        y_pred=a.Y_pred,
        y_true=a.Y_test,
        metrics=a.metrics,
        test_states=a.test_states,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Extra diagnostics for fold NPZ results")
    parser.add_argument(
        "--glob",
        dest="glob_pattern",
        default="results/runs/final-test-classification-fold-*.npz",
        help="Glob pattern for fold result files.",
    )
    parser.add_argument(
        "--trial_name",
        type=str,
        default=None,
        help="Trial name to load via FoldArtifact.load_trial() (alternative to --glob).",
    )
    parser.add_argument("--plots", action="store_true", help="Generate plots under results/diagnostics/")
    parser.add_argument("--max-points", type=int, default=3000, help="Max points for scatter plots")

    args = parser.parse_args()

    if args.trial_name:
        artifacts = FoldArtifact.load_trial(args.trial_name)
        folds = [_artifact_to_folddata(a) for a in artifacts]
    else:
        paths = sorted(glob.glob(args.glob_pattern))
        if not paths:
            raise SystemExit(f"No fold files matched: {args.glob_pattern!r}")
        folds = [_load_fold(p) for p in paths]

    y_true_raw, y_pred_raw = _stack_folds(folds)
    y_true = _to_1d_labels(y_true_raw)
    prob = _to_prob_matrix(y_pred_raw)
    y_pred_cls = np.argmax(prob, axis=1)

    # Aggregate metrics (same as analyse_fold_results.py, but always recomputed from saved arrays)
    agg = evaluate_performance(y_true_raw, y_pred_raw)
    print("Aggregate (recomputed):")
    for k in ["accuracy", "f1", "precision", "recall", "rmse"]:
        print(f"  {k}: {agg[k]:.4f}")

    if args.plots:
        plot_confidence_hist(prob, y_true, y_pred_cls, name="confidence_correct_vs_incorrect.png")
        plot_entropy_by_class(prob, y_true, name="entropy_by_class.png")

        # Confusion-derived recall plot
        if "confusion_matrix" in agg:
            plot_per_class_recall_from_cm(agg["confusion_matrix"], name="per_class_recall.png")

        # State representation plots (if available)
        states_all = []
        y_true_states = []
        for f in folds:
            if f.test_states is None:
                continue
            st = _flatten_states(f.test_states)
            yt = _to_1d_labels(f.y_true)
            states_all.append(st)
            y_true_states.append(yt)

        if states_all:
            states = np.concatenate(states_all, axis=0)
            yts = np.concatenate(y_true_states, axis=0)
            plot_pca_explained_variance(states, name="states_pca_explained_variance.png", max_components=50)
            plot_state_pca(states, yts, name="states_pca.png", max_points=int(args.max_points))
        else:
            print("No test_states found in fold files; skipping state-space plots.")

        print(f"Saved plots to: {RESULTS_DIR}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
