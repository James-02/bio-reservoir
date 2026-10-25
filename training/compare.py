"""Compare two runs: per-fold metrics, prediction agreement and a paired bootstrap CI.

A run is given as a run folder, a registry run id, a folder name under ``results/runs``,
or ``archive:<version>/<run>`` for a frozen run, e.g.
``archive:v0.1/P4-readout-balanced-best``. Metrics are recomputed from the stored labels
and predictions, so runs from any code version compare on the same basis. Predictions
are paired beat by beat only when both runs have the same folds (same test labels in the
same order in every fold). Paired runs also get the lower one-sided 95% bound of B - A,
tested against ``margin`` for non-inferiority, and, with a third run C on the same beats
(``ceiling``), the share of the gap from A to C that B closes: (B - A) / (C - A).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from utils import config as settings
from utils import provenance

METRICS = ("f1_macro", "f1_weighted", "accuracy", "mcc")


def run_path(ref: str | Path) -> Path:
    ref = str(ref)
    if ref.startswith("archive:"):
        version, _, name = ref[len("archive:"):].partition("/")
        path = settings.runs_dir(version) / name
    elif Path(ref).is_dir():
        path = Path(ref)
    elif ref in (runs := provenance.load_registry(provenance.REGISTRY_FILE)):
        # The registry stores paths relative to the repository, or only the folder name
        # when the results folder is elsewhere.
        path = settings.PROJECT_ROOT / runs[ref]["path"]
        if not path.is_dir():
            path = settings.runs_dir() / Path(runs[ref]["path"]).name
    else:
        path = settings.runs_dir() / ref
    if not sorted(path.glob("fold-*.npz")):
        raise FileNotFoundError(f"No fold files for run {ref!r} in {path}")
    return path


def load_folds(ref: str | Path, states: bool = False) -> list:
    """Fold artefacts of a run, sorted by fold; states only when asked for."""
    from utils.results import FoldArtifact

    paths = sorted(run_path(ref).glob("fold-*.npz"))
    if states:
        folds = [FoldArtifact.load(str(p)) for p in paths]
    else:
        folds = []
        for p in paths:
            with np.load(p, allow_pickle=True) as data:
                index = data["test_index"] if "test_index" in data.files else None
                folds.append(_Labels(int(data["fold_index"]), data["Y_test"], data["Y_pred"],
                                     index))
    return sorted(folds, key=lambda f: f.fold_index)


class _Labels:
    def __init__(self, fold_index: int, Y_test: np.ndarray, Y_pred: np.ndarray,
                 test_index: np.ndarray | None = None):
        self.fold_index, self.Y_test, self.Y_pred = fold_index, Y_test, Y_pred
        self.test_index = test_index


def _classes(Y: np.ndarray) -> np.ndarray:
    return np.argmax(np.asarray(Y).reshape(len(Y), -1), axis=1)


def _scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Metrics over the classes of the test labels (other predicted classes are errors)."""
    labels = np.unique(y_true)
    return {"f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro")),
            "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted")),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "mcc": float(matthews_corrcoef(y_true, y_pred))}


def _f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return f1_score(y_true, y_pred, labels=np.unique(y_true), average="macro")


def _paired_draws(y_true: np.ndarray, preds: list[np.ndarray], n_boot: int = 2000,
                  seed: int = 0) -> np.ndarray:
    """Macro-F1 of each prediction set on ``n_boot`` resamples of the beats, one row per draw;
    every set is scored on the same beats."""
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, len(preds)))
    for i in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        draws[i] = [_f1_macro(y_true[idx], pred[idx]) for pred in preds]
    return draws


def bootstrap_delta(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                    n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Macro-F1 of B minus A, with a 95% percentile CI from resampling beats in pairs."""
    delta = _f1_macro(y_true, pred_b) - _f1_macro(y_true, pred_a)
    draws = np.diff(_paired_draws(y_true, [pred_a, pred_b], n_boot, seed), axis=1)[:, 0]
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(delta), float(low), float(high)


def gap_closed(f1_a, f1_b, f1_ceiling):
    """(B - A) / (C - A): the share of the gap from A to the ceiling C that B closes; NaN
    where C does not exceed A."""
    span = np.asarray(f1_ceiling, dtype=float) - f1_a
    return np.where(span > 0, (np.asarray(f1_b, dtype=float) - f1_a)
                    / np.where(span > 0, span, 1.0), np.nan)


def _same_beats(folds_a: list, folds_b: list) -> bool:
    if [f.fold_index for f in folds_a] != [f.fold_index for f in folds_b]:
        return False
    return all(np.array_equal(_classes(a.Y_test), _classes(b.Y_test))
               and (a.test_index is None or b.test_index is None
                    or np.array_equal(a.test_index, b.test_index))
               for a, b in zip(folds_a, folds_b, strict=True))


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 2000,
                 seed: int = 0) -> tuple[float, float]:
    """95% percentile CI of one run's macro-F1 from resampling the test beats."""
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        draws[i] = _f1_macro(y_true[idx], y_pred[idx])
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def compare(ref_a: str | Path, ref_b: str | Path, *, out: str | Path | None = None,
            n_boot: int = 2000, seed: int = 0, margin: float | None = None,
            ceiling: str | Path | None = None) -> dict[str, Any]:
    folds_a, folds_b = load_folds(ref_a), load_folds(ref_b)
    paths = [run_path(ref_a), run_path(ref_b)]
    names = [p.name for p in paths]
    same_folds = [f.fold_index for f in folds_a] == [f.fold_index for f in folds_b]
    rows = []
    paired = same_folds
    by_index = True
    for fa, fb in zip(folds_a, folds_b, strict=False):
        ta, tb = _classes(fa.Y_test), _classes(fb.Y_test)
        pa, pb = _classes(fa.Y_pred), _classes(fb.Y_pred)
        same = len(ta) == len(tb) and bool(np.array_equal(ta, tb))
        if fa.test_index is not None and fb.test_index is not None:
            same = same and bool(np.array_equal(fa.test_index, fb.test_index))
        else:
            by_index = False
        paired = paired and same
        row = {"fold": fa.fold_index, "n_test": [len(ta), len(tb)], "same_labels": same,
               "a": _scores(ta, pa), "b": _scores(tb, pb)}
        row["delta_f1_macro"] = row["b"]["f1_macro"] - row["a"]["f1_macro"]
        row["agreement"] = float(np.mean(pa == pb)) if same else None
        rows.append(row)

    def stats(side: str) -> dict[str, dict[str, float]]:
        values = {k: np.array([r[side][k] for r in rows]) for k in METRICS}
        return {"mean": {k: float(v.mean()) for k, v in values.items()},
                "sd": {k: float(v.std(ddof=1)) if len(v) > 1 else 0.0
                       for k, v in values.items()}}

    result: dict[str, Any] = {"a": names[0], "b": names[1], "paired": paired,
                              "paired_by": "beat position" if by_index else "test labels",
                              "folds": rows, "summary_a": stats("a"), "summary_b": stats("b")}
    if (ceiling is not None or margin is not None) and not paired:
        raise ValueError("A margin or a ceiling needs runs A and B on the same folds and beats.")
    if paired:
        y_true = np.concatenate([_classes(f.Y_test) for f in folds_a])
        preds = [np.concatenate([_classes(f.Y_pred) for f in folds]) for folds in (folds_a, folds_b)]
        if ceiling is not None:
            folds_c = load_folds(ceiling)
            if not _same_beats(folds_a, folds_c):
                raise ValueError(f"{run_path(ceiling).name} does not share the folds and beats "
                                 f"of {names[0]}.")
            preds.append(np.concatenate([_classes(f.Y_pred) for f in folds_c]))
        f1 = [_f1_macro(y_true, pred) for pred in preds]
        draws = _paired_draws(y_true, preds, n_boot, seed)
        low, high, bound = np.percentile(draws[:, 1] - draws[:, 0], [2.5, 97.5, 5.0])
        pooled = {"delta": float(f1[1] - f1[0]), "ci95": [float(low), float(high)],
                  "lower_95_one_sided": float(bound), "n_beats": int(len(y_true)),
                  "n_boot": n_boot}
        if margin is not None:
            pooled.update(margin=float(margin), non_inferior=bool(bound > margin))
        result["pooled_delta_f1_macro"] = pooled
        result["agreement"] = float(np.mean(preds[0] == preds[1]))
        if ceiling is not None:
            gaps = gap_closed(draws[:, 0], draws[:, 1], draws[:, 2])
            defined = gaps[np.isfinite(gaps)]
            low, high = np.percentile(defined, [2.5, 97.5]) if len(defined) else (np.nan,) * 2
            result["gap_closed"] = {"ceiling": run_path(ceiling).name, "f1_ceiling": float(f1[2]),
                                    "gap": float(gap_closed(*f1)), "ci95": [float(low), float(high)],
                                    "undefined_draws": int(len(gaps) - len(defined))}
    if out:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "compare.json").write_text(json.dumps(result, indent=1) + "\n")
        plot_comparison(out, result, folds_a, folds_b, paths)
    return result


def plot_comparison(out: Path, result: dict[str, Any], folds_a: list, folds_b: list,
                    paths: list[Path]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    rows = result["folds"]
    fig, ax = plt.subplots(figsize=(5, 3))
    for side, marker in (("a", "o"), ("b", "s")):
        ax.plot([r["fold"] for r in rows], [r[side]["f1_macro"] for r in rows], marker=marker,
                label=result[side])
    ax.set(xlabel="Fold", ylabel="Macro-F1", title="Per-fold macro-F1")
    ax.legend(fontsize=7)
    fig.savefig(out / "fold_f1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, folds, name in ((axes[0], folds_a, result["a"]), (axes[1], folds_b, result["b"])):
        y_true = np.concatenate([_classes(f.Y_test) for f in folds])
        y_pred = np.concatenate([_classes(f.Y_pred) for f in folds])
        cm = confusion_matrix(y_true, y_pred).astype(float)
        cm /= np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
        ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
        for (i, j), value in np.ndenumerate(cm):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if value > 0.5 else "black")
        ax.set(title=name, xlabel="Predicted class", ylabel="True class")
    fig.savefig(out / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    states = []
    for path in paths:
        with np.load(sorted(path.glob("fold-*.npz"))[0], allow_pickle=True) as data:
            states.append(data["test_states"][:3].reshape(3, -1)
                          if "test_states" in data.files else None)
    if all(s is not None for s in states) and states[0].shape == states[1].shape:
        fig, axes = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
        for k, ax in enumerate(axes):
            ax.plot(states[0][k], lw=0.6, label=result["a"])
            ax.plot(states[1][k], lw=0.6, alpha=0.7, label=result["b"])
            ax.set_ylabel(f"beat {k}")
        axes[0].legend(fontsize=7)
        axes[-1].set_xlabel("Feature (unit x observed variable)")
        fig.suptitle("State features of the first test beats of the first fold")
        fig.savefig(out / "state_features.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def format_comparison(result: dict[str, Any]) -> str:
    lines = [f"A: {result['a']}", f"B: {result['b']}", "",
             "fold  macro-F1 A  macro-F1 B   delta  agreement  same labels"]
    for r in result["folds"]:
        agreement = f"{r['agreement']:.4f}" if r["agreement"] is not None else "   -  "
        lines.append(f"{r['fold']:>4}  {r['a']['f1_macro']:10.4f}  {r['b']['f1_macro']:10.4f}"
                     f"  {r['delta_f1_macro']:+.4f}  {agreement:>9}  {r['same_labels']}")
    sa, sb = result["summary_a"], result["summary_b"]
    lines.append("")
    for key in METRICS:
        lines.append(f"{key:12s} A {sa['mean'][key]:.4f} +/- {sa['sd'][key]:.4f}   "
                     f"B {sb['mean'][key]:.4f} +/- {sb['sd'][key]:.4f}  (mean +/- sample SD)")
    if result["paired"]:
        pooled = result["pooled_delta_f1_macro"]
        lines.append(f"pooled macro-F1 B - A {pooled['delta']:+.4f} "
                     f"[{pooled['ci95'][0]:+.4f}, {pooled['ci95'][1]:+.4f}] "
                     f"(paired bootstrap 95% CI, {pooled['n_beats']} beats); "
                     f"prediction agreement {result['agreement']:.4f} "
                     f"(paired by {result['paired_by']})")
        line = f"lower one-sided 95% bound {pooled['lower_95_one_sided']:+.4f}"
        if "margin" in pooled:
            verdict = "non-inferior" if pooled["non_inferior"] else "not shown non-inferior"
            line += f"; margin {pooled['margin']:+.3f}: {verdict}"
        lines.append(line)
        if "gap_closed" in result:
            gap = result["gap_closed"]
            lines.append(f"gap closed toward {gap['ceiling']} (macro-F1 {gap['f1_ceiling']:.4f}): "
                         f"{gap['gap']:.3f} [{gap['ci95'][0]:.3f}, {gap['ci95'][1]:.3f}]")
    else:
        lines.append("Folds differ: predictions are not paired.")
    return "\n".join(lines)
