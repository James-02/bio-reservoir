#!/usr/bin/env python
"""Re-run the b3-baseline-binary-v4 best trial (KNN) on the full binary dataset.

Evaluates the KNN hyperparameters from trial 356 of b3-baseline-binary-v4
with 5-fold stratified CV on the full balanced binary dataset (~37,714
instances), matching the headline binary task size.

Usage:
    python scripts/rerun_baseline_binary_full.py
    python scripts/rerun_baseline_binary_full.py --rows 10000 --folds 3 --workers 3
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, accuracy_score, matthews_corrcoef, cohen_kappa_score,
    classification_report,
)

from utils.preprocessing import load_ecg_data, standardize_data

# ── Best trial 356 hyperparameters (from b3-baseline-binary-v4) ─────────────
KNN_HYPERS = {
    "n_neighbors": 1,
    "weights": "uniform",
    "p": 1.1136890171790428,
}

SEED = 6337
N_FOLDS = 5


def _run_fold(args):
    """Run a single fold (designed for parallel execution)."""
    fold_i, train_idx, val_idx, X, Y = args
    X_tr, Y_tr = X[train_idx], Y[train_idx]
    X_va, Y_va = X[val_idx], Y[val_idx]

    # No augmentation (v4)
    X_tr, X_va = standardize_data(X_tr, X_va, scaler_type="sequence_zscore")

    # Flatten to 2-D for sklearn
    X_tr = X_tr.reshape(X_tr.shape[0], -1)
    X_va = X_va.reshape(X_va.shape[0], -1)

    # Convert labels to 1-D
    y_tr = np.argmax(Y_tr, axis=-1) if Y_tr.ndim > 1 else Y_tr
    y_va = np.argmax(Y_va, axis=-1) if Y_va.ndim > 1 else Y_va

    clf = KNeighborsClassifier(**KNN_HYPERS)

    fold_t0 = time.perf_counter()
    fit_t0 = time.perf_counter()
    clf.fit(X_tr, y_tr)
    fit_s = time.perf_counter() - fit_t0

    pred_t0 = time.perf_counter()
    y_pred = clf.predict(X_va)
    pred_s = time.perf_counter() - pred_t0
    fold_s = time.perf_counter() - fold_t0

    f1 = f1_score(y_va, y_pred, average="macro")
    acc = accuracy_score(y_va, y_pred)
    mcc = matthews_corrcoef(y_va, y_pred)
    kappa = cohen_kappa_score(y_va, y_pred)
    report = classification_report(y_va, y_pred, output_dict=True)

    return {
        "fold_i": fold_i,
        "f1": f1, "acc": acc, "mcc": mcc, "kappa": kappa,
        "fit_s": fit_s, "pred_s": pred_s, "fold_s": fold_s,
        "report": report,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Re-run baseline binary KNN trial with stratified CV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--rows", type=int, default=50000,
                        help="Max rows to load from the ECG dataset.")
    parser.add_argument("--folds", type=int, default=N_FOLDS,
                        help="Number of stratified CV folds.")
    parser.add_argument("--workers", type=int, default=5,
                        help="Parallel fold workers.")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()

    print("Loading full binary dataset...")
    X_train, Y_train, X_test, Y_test = load_ecg_data(
        rows=args.rows,
        binary=True,
        balance_classes=True,
        max_per_class=None,
        scaler_type="sequence_zscore",
        seed=args.seed,
        noise_rate=0.0,
        noise_ratio=0.0,
    )

    # Combine train+test for CV pool
    X = np.concatenate((X_train, X_test), axis=0)
    Y = np.concatenate((Y_train, Y_test), axis=0)

    if Y.ndim > 1:
        y_1d = np.argmax(Y, axis=-1)
    else:
        y_1d = Y.copy()

    print(f"CV pool: {len(X)} instances, {len(np.unique(y_1d))} classes")
    print(f"Class distribution: {dict(zip(*np.unique(y_1d, return_counts=True)))}")
    print(f"KNN hypers: {KNN_HYPERS}")
    print()

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_args = [
        (fold_i, train_idx, val_idx, X, Y)
        for fold_i, (train_idx, val_idx) in enumerate(skf.split(X, y_1d))
    ]

    overall_t0 = time.perf_counter()

    print(f"Running {args.folds} folds in parallel ({args.workers} workers)...")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_run_fold, fold_args))

    results.sort(key=lambda r: r["fold_i"])

    f1s, accs, mccs, kappas = [], [], [], []

    for r in results:
        f1s.append(r["f1"])
        accs.append(r["acc"])
        mccs.append(r["mcc"])
        kappas.append(r["kappa"])
        print(f"Fold {r['fold_i']}: F1={r['f1']:.4f}  Acc={r['acc']:.4f}  "
              f"MCC={r['mcc']:.4f}  Kappa={r['kappa']:.4f}  "
              f"fit={r['fit_s']:.1f}s  predict={r['pred_s']:.1f}s  "
              f"total={r['fold_s']:.1f}s")

    overall_s = time.perf_counter() - overall_t0

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"F1:       {np.mean(f1s):.4f} +/- {np.std(f1s, ddof=1):.4f}")
    print(f"Accuracy: {np.mean(accs):.4f} +/- {np.std(accs, ddof=1):.4f}")
    print(f"MCC:      {np.mean(mccs):.4f} +/- {np.std(mccs, ddof=1):.4f}")
    print(f"Kappa:    {np.mean(kappas):.4f} +/- {np.std(kappas, ddof=1):.4f}")
    print(f"Wall time: {overall_s:.1f}s")


if __name__ == "__main__":
    main()
