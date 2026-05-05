#!/usr/bin/env python3
"""Compute per-class per-fold metrics, MCC, kappa, and confusion matrices
for the Phase 4 readout-optimized best trials.

Loads frozen reservoir states from NPZ fold artifacts, fits the winning
classifier on each fold, computes per-class metrics per fold (for std),
and aggregates MCC / Cohen's kappa from the confusion matrix.

Usage:
    python scripts/compute_headline_metrics.py
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    cohen_kappa_score,
    f1_score,
    accuracy_score,
)
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed

from utils.results import FoldArtifact

SEED = 6337

# ── Task definitions ──────────────────────────────────────────────────────
TASKS = [
    {
        "name": "ro1-readout-fixed (balanced 5-class)",
        "trial_name": "r1a-refined-fixed-best",
        "clf_cls": RandomForestClassifier,
        "hypers": {
            "n_estimators": 326, "criterion": "log_loss", "oob_score": True,
            "max_features": "sqrt", "max_depth": 23, "min_samples_split": 2,
            "min_samples_leaf": 1, "class_weight": "balanced", "random_state": SEED,
        },
    },
    {
        "name": "ro1-readout-binary-fixed",
        "trial_name": "r1a-refined-binary-fixed-best-full",
        "clf_cls": RandomForestClassifier,
        "hypers": {
            "n_estimators": 279, "criterion": "log_loss", "oob_score": True,
            "max_features": "sqrt", "max_depth": 27, "min_samples_split": 6,
            "min_samples_leaf": 1, "class_weight": "balanced", "random_state": SEED,
        },
    },
    {
        "name": "ro2-readout-unbalanced-fixed (5-class)",
        "trial_name": "f2-unbalanced-fixed-best",
        "clf_cls": RandomForestClassifier,
        "hypers": {
            "n_estimators": 321, "criterion": "entropy", "oob_score": False,
            "max_features": "sqrt", "max_depth": 23, "min_samples_split": 5,
            "min_samples_leaf": 2, "class_weight": "balanced", "random_state": SEED,
        },
    },
]


def _fit_fold(clf_cls, hypers, artifact):
    """Fit classifier on one fold, return per-class report + predictions."""
    X_train = artifact.train_states.reshape(artifact.train_states.shape[0], -1)
    Y_train = artifact.Y_train.squeeze()
    X_test = artifact.test_states.reshape(artifact.test_states.shape[0], -1)

    # Convert one-hot to integer labels
    y_train = np.argmax(Y_train, axis=-1) if Y_train.ndim > 1 and Y_train.shape[-1] > 1 else Y_train.ravel()

    clf = clf_cls(**hypers)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    y_true = np.argmax(artifact.Y_test.squeeze(), axis=-1)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "fold": artifact.fold_index,
        "y_true": y_true,
        "y_pred": y_pred,
        "report": report,
    }


def run_task(task):
    """Run all folds for one task, print results."""
    print(f"\n{'=' * 72}")
    print(f"  {task['name']}")
    print(f"  trial_name={task['trial_name']}  clf={task['clf_cls'].__name__}")
    print(f"{'=' * 72}")

    artifacts = FoldArtifact.load_trial(task["trial_name"])
    n_folds = len(artifacts)
    print(f"  Loaded {n_folds} folds from results/runs/{task['trial_name']}/")

    # Fit all folds in parallel
    results = Parallel(n_jobs=n_folds, backend="loky")(
        delayed(_fit_fold)(task["clf_cls"], task["hypers"], a) for a in artifacts
    )
    results.sort(key=lambda r: r["fold"])

    # ── Aggregate confusion matrix across folds ──
    all_y_true = np.concatenate([r["y_true"] for r in results])
    all_y_pred = np.concatenate([r["y_pred"] for r in results])
    classes = sorted(set(all_y_true) | set(all_y_pred))

    cm = confusion_matrix(all_y_true, all_y_pred, labels=classes)
    mcc = matthews_corrcoef(all_y_true, all_y_pred)
    kappa = cohen_kappa_score(all_y_true, all_y_pred)
    macro_f1 = f1_score(all_y_true, all_y_pred, average="macro")
    accuracy = accuracy_score(all_y_true, all_y_pred)

    print(f"\n  Aggregated (all folds pooled):")
    print(f"    Macro F1  = {macro_f1:.4f}")
    print(f"    Accuracy  = {accuracy:.4f}")
    print(f"    MCC       = {mcc:.4f}")
    print(f"    Kappa     = {kappa:.4f}")

    # ── Per-fold aggregate metrics ──
    fold_f1s = [f1_score(r["y_true"], r["y_pred"], average="macro") for r in results]
    fold_accs = [accuracy_score(r["y_true"], r["y_pred"]) for r in results]
    print(f"\n  Per-fold F1:  mean={np.mean(fold_f1s):.4f}  std={np.std(fold_f1s):.4f}")
    print(f"  Per-fold Acc: mean={np.mean(fold_accs):.4f}  std={np.std(fold_accs):.4f}")
    for i, f1 in enumerate(fold_f1s):
        print(f"    Fold {i}: F1={f1:.4f}  Acc={fold_accs[i]:.4f}")

    # ── Per-class per-fold breakdown ──
    class_labels = sorted(results[0]["report"].keys())
    class_labels = [c for c in class_labels if c.isdigit()]

    print(f"\n  Per-class metrics (mean ± std across {n_folds} folds):")
    print(f"  {'Class':<8} {'Precision':>18} {'Recall':>18} {'F1':>18} {'Support':>10}")
    print(f"  {'-'*76}")

    class_data = {}
    for cls in class_labels:
        precs = [r["report"][cls]["precision"] for r in results]
        recs = [r["report"][cls]["recall"] for r in results]
        f1s = [r["report"][cls]["f1-score"] for r in results]
        sups = [r["report"][cls]["support"] for r in results]

        class_data[cls] = {
            "precision_mean": np.mean(precs), "precision_std": np.std(precs),
            "recall_mean": np.mean(recs), "recall_std": np.std(recs),
            "f1_mean": np.mean(f1s), "f1_std": np.std(f1s),
            "support": int(np.mean(sups)),
        }

        print(f"  {cls:<8} "
              f"{np.mean(precs):.3f} ± {np.std(precs):.3f}    "
              f"{np.mean(recs):.3f} ± {np.std(recs):.3f}    "
              f"{np.mean(f1s):.3f} ± {np.std(f1s):.3f}    "
              f"{int(np.mean(sups)):>6}")

    # ── LaTeX table rows ──
    CLASS_NAMES_5 = {
        "0": "Normal (N)", "1": "Supraventricular (S)",
        "2": "Ventricular (V)", "3": "Fusion (F)", "4": "Unknown (Q)",
    }
    CLASS_NAMES_2 = {"0": "Normal", "1": "Arrhythmia"}
    names = CLASS_NAMES_2 if len(class_labels) == 2 else CLASS_NAMES_5

    print(f"\n  LaTeX rows:")
    for cls in class_labels:
        d = class_data[cls]
        name = names.get(cls, f"Class {cls}")
        print(f"  {name:<25} & "
              f"${d['precision_mean']:.3f} \\pm {d['precision_std']:.3f}$ & "
              f"${d['recall_mean']:.3f} \\pm {d['recall_std']:.3f}$ & "
              f"${d['f1_mean']:.3f} \\pm {d['f1_std']:.3f}$ & "
              f"{d['support']} \\\\")

    print(f"\n  Confusion matrix:")
    print(f"  {cm}")

    return {
        "name": task["name"],
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "mcc": mcc,
        "kappa": kappa,
        "fold_f1_mean": np.mean(fold_f1s),
        "fold_f1_std": np.std(fold_f1s),
        "class_data": class_data,
    }


if __name__ == "__main__":
    all_results = []
    for task in TASKS:
        all_results.append(run_task(task))

    print(f"\n\n{'=' * 72}")
    print("  SUMMARY")
    print(f"{'=' * 72}")
    for r in all_results:
        print(f"  {r['name']:<40}  "
              f"F1={r['fold_f1_mean']:.4f}±{r['fold_f1_std']:.4f}  "
              f"Acc={r['accuracy']:.4f}  "
              f"MCC={r['mcc']:.4f}  "
              f"κ={r['kappa']:.4f}")
