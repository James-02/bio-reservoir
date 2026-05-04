"""Compute per-class per-fold std for the winning readout classifiers."""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, matthews_corrcoef
import sys


def run_task(task_dir, clf_factory, n_folds=5):
    all_reports = []
    all_mcc = []
    for fi in range(n_folds):
        f = np.load(f"results/runs/{task_dir}/fold-{fi}.npz", allow_pickle=True)
        Xtr = f["train_states"].reshape(f["train_states"].shape[0], -1)
        Xte = f["test_states"].reshape(f["test_states"].shape[0], -1)
        Ytr = np.argmax(f["Y_train"].reshape(f["Y_train"].shape[0], -1), axis=1)
        Yte = np.argmax(f["Y_test"].reshape(f["Y_test"].shape[0], -1), axis=1)
        clf = clf_factory()
        clf.fit(Xtr, Ytr)
        Yp = clf.predict(Xte)
        r = classification_report(Yte, Yp, output_dict=True)
        mcc = matthews_corrcoef(Yte, Yp)
        all_reports.append(r)
        all_mcc.append(mcc)
        print(f"  Fold {fi}: F1={r['macro avg']['f1-score']:.4f}  MCC={mcc:.4f}")
        sys.stdout.flush()

    classes = sorted(
        k for k in all_reports[0] if k not in ("accuracy", "macro avg", "weighted avg")
    )
    for c in classes:
        ps = [r[c]["precision"] for r in all_reports]
        rs = [r[c]["recall"] for r in all_reports]
        fs = [r[c]["f1-score"] for r in all_reports]
        ss = [r[c]["support"] for r in all_reports]
        print(
            f"  Class {c}: prec={np.mean(ps):.3f}+/-{np.std(ps):.3f}"
            f"  rec={np.mean(rs):.3f}+/-{np.std(rs):.3f}"
            f"  f1={np.mean(fs):.3f}+/-{np.std(fs):.3f}"
            f"  sup={np.mean(ss):.0f}"
        )
    mf = [r["macro avg"]["f1-score"] for r in all_reports]
    ac = [r["accuracy"] for r in all_reports]
    print(
        f"  Macro F1: {np.mean(mf):.4f}+/-{np.std(mf):.4f}"
        f"  Acc: {np.mean(ac):.4f}+/-{np.std(ac):.4f}"
        f"  MCC: {np.mean(all_mcc):.4f}+/-{np.std(all_mcc):.4f}"
    )
    sys.stdout.flush()


print("=" * 70)
print("BINARY (r1a-refined-best-binary, RF n=285 entropy)")
print("=" * 70)
run_task(
    "r1a-refined-best-binary",
    lambda: RandomForestClassifier(
        n_estimators=285, criterion="entropy", max_depth=47,
        max_features="sqrt", min_samples_split=3, min_samples_leaf=1,
        oob_score=True, class_weight="balanced", random_state=6337,
        n_jobs=-1,
    ),
)

print()
print("=" * 70)
print("UNBALANCED (f2-unbalanced-best, KNN k=1 uniform p=1.61)")
print("=" * 70)
run_task(
    "f2-unbalanced-best",
    lambda: KNeighborsClassifier(n_neighbors=1, weights="uniform", p=1.6055, n_jobs=-1),
)

print()
print("=" * 70)
print("BALANCED (r1a-refined-best, KNN k=4 distance p=1.03)")
print("=" * 70)
run_task(
    "r1a-refined-best",
    lambda: KNeighborsClassifier(n_neighbors=4, weights="distance", p=1.031, n_jobs=-1),
)
