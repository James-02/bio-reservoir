"""No-reservoir baseline: classifiers on raw preprocessed ECG features.

Provides a direct comparison to the reservoir readout studies (ro1-readout
etc.) by running the same classifier families and hyperparameter search
on the raw z-scored 187-dimensional ECG feature vectors — bypassing the
reservoir transformation entirely.

This addresses the limitation acknowledged in the paper:

    "we did not include a direct non-reservoir baseline trained on the
     raw z-scored ECG segments"

Architecture
------------
The objective mirrors ``optimization.readout.objective`` as closely as
possible for comparability:

* Same classifier families and hyperparameter ranges (via ``choose_classifier``).
* Same 5-fold stratified CV protocol with fold-level augmentation and
  standardisation (matching ``training.cross_validate._prepare_fold_data``).
* Same macro-F1 optimisation target and per-fold / per-class metrics storage.
* Same Optuna infrastructure (TPE sampler, SQLite storage, ``research()``).

The only difference is that classifiers receive ``(N, 187)`` raw ECG vectors
instead of ``(N, D_reservoir)`` reservoir state vectors.

CLI
---
    python -m optimization.optimize research --type baseline \\
        --study_name b1-baseline --processes 32 --trials 500

    python -m optimization.optimize evaluate --type baseline \\
        --study_name b1-baseline --top_n 10
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import time as _time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd  # type: ignore

from optuna.visualization.matplotlib import (
    plot_slice,
    plot_param_importances,
    plot_edf,
)

from sklearn.model_selection import StratifiedKFold

from optimization.base import (
    DEFAULT_SEED,
    research, plot_results, print_topn_trials,
    parse_show_sections, SHOW_DEFAULT,
    get_default_storage, build_study,
)
from optimization.environment import get_environment
from optimization.readout import choose_classifier, _important_params
from training.profiling import ResourceMonitor
from utils.analysis import (
    compute_class_dicts,
    compute_mean_dicts,
    evaluate_performance,
)
from utils.preprocessing import (
    load_ecg_data,
    augment_data,
    standardize_data,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset loading (mirrors optimization.classification._get_dataset)
# ---------------------------------------------------------------------------

_DATA_CACHE: Dict[tuple, Tuple[np.ndarray, ...]] = {}


def _get_dataset(
    *,
    instances: int,
    binary: bool,
    balance_classes: bool = True,
    max_per_class: Optional[int] = None,
    scaler_type: str = "sequence_zscore",
    seed: int = DEFAULT_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and cache the ECG dataset.

    Unlike the classification module, noise augmentation is NOT applied here
    because it will be applied per-fold inside the objective (matching the
    CV protocol used for reservoir training).
    """
    key = (
        int(instances), bool(binary), bool(balance_classes),
        max_per_class, scaler_type, int(seed),
    )
    if key not in _DATA_CACHE:
        _DATA_CACHE[key] = load_ecg_data(
            rows=instances,
            balance_classes=balance_classes,
            max_per_class=max_per_class,
            binary=binary,
            noise_rate=0.0,
            noise_ratio=0.0,
            scaler_type=scaler_type,
            seed=seed,
            preserve_split=False,
        )
    return _DATA_CACHE[key]


# ---------------------------------------------------------------------------
# Fold helpers (mirrors training.cross_validate._iter_folds / _prepare_fold_data)
# ---------------------------------------------------------------------------

def _labels_to_1d(Y: np.ndarray) -> np.ndarray:
    """Convert labels to a 1-D class-index vector for stratified splitting."""
    Y = np.asarray(Y)
    if Y.ndim == 1:
        return Y.astype(int)
    if Y.ndim == 3:  # (N, 1, C) one-hot
        return np.argmax(Y[:, 0, :], axis=-1).astype(int)
    if Y.ndim == 2:  # (N, C) one-hot
        return np.argmax(Y, axis=-1).astype(int)
    raise ValueError(f"Unsupported Y shape for stratification: {Y.shape}")


def _flatten_X(X: np.ndarray) -> np.ndarray:
    """Flatten ECG time-series to 2-D feature matrix for sklearn classifiers.

    Input may be (N, T, 1) from the preprocessing pipeline; output is (N, T).
    """
    X = np.asarray(X)
    if X.ndim == 3 and X.shape[-1] == 1:
        return X.reshape(X.shape[0], -1)
    if X.ndim == 2:
        return X
    raise ValueError(f"Unexpected X shape: {X.shape}")


def _prepare_fold(
    X: np.ndarray,
    Y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    fold_index: int,
    *,
    noise_rate: float,
    noise_ratio: float,
    seed: int,
    scaler_type: str = "sequence_zscore",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare one fold's data with augmentation and standardisation.

    Mirrors ``training.cross_validate._prepare_fold_data`` exactly so the
    baseline uses identical per-fold preprocessing.
    """
    X_train, X_val = X[train_idx], X[val_idx]
    Y_train, Y_val = Y[train_idx], Y[val_idx]

    fold_rng = np.random.RandomState(int(seed) + int(fold_index))
    X_train, Y_train = augment_data(
        X_train, Y_train, noise_rate, noise_ratio, rng=fold_rng
    )
    X_train, X_val = standardize_data(X_train, X_val, scaler_type=scaler_type)

    # Flatten to 2-D for sklearn classifiers.
    X_train = _flatten_X(X_train)
    X_val = _flatten_X(X_val)

    return X_train, Y_train, X_val, Y_val


# ---------------------------------------------------------------------------
# Summary table (reused from readout.py pattern)
# ---------------------------------------------------------------------------

def _compute_group_stats(df_group: pd.DataFrame, top_n: int) -> Dict[str, Any]:
    """Aggregate stats over top-N and all trials in *df_group*."""
    vals = df_group["value"].dropna().sort_values(ascending=False)
    top = vals.head(int(top_n))
    return {
        "n_trials":  len(vals),
        "top_mean":  float(top.mean()) if len(top) else float("nan"),
        "top_std":   float(top.std(ddof=0)) if len(top) else float("nan"),
        "top_min":   float(top.min()) if len(top) else float("nan"),
        "top_max":   float(top.max()) if len(top) else float("nan"),
        "all_mean":  float(vals.mean()) if len(vals) else float("nan"),
        "all_std":   float(vals.std(ddof=0)) if len(vals) else float("nan"),
        "all_min":   float(vals.min()) if len(vals) else float("nan"),
        "all_max":   float(vals.max()) if len(vals) else float("nan"),
    }


def _print_classifier_summary_table(df: pd.DataFrame, top_n: int = 10) -> None:
    """Print per-classifier F1 summary table."""
    rows: list = []
    if "params_classifier" in df.columns:
        for clf in sorted(c for c in df["params_classifier"].dropna().unique()):
            df_clf = df[df["params_classifier"] == clf]
            stats = _compute_group_stats(df_clf, top_n)
            stats["classifier"] = str(clf)
            rows.append(stats)
    if not rows:
        return
    tab = pd.DataFrame(rows)
    col_order = [
        "classifier", "n_trials",
        "top_mean", "top_std", "top_min", "top_max",
        "all_mean", "all_std", "all_min", "all_max",
    ]
    tab = tab[col_order]
    print(f"\n=== Summary table (F1 over top {int(top_n)} and all trials) ===")
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def objective(trial: optuna.Trial, **kwargs) -> float:
    """Optuna objective: classifier on raw preprocessed ECG features.

    Loads the same preprocessed dataset and fold protocol as the reservoir
    pipeline, but fits classifiers directly on the 187-d ECG vectors.
    """
    trial.set_user_attr(
        "trial_start_utc",
        datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )
    monitor = ResourceMonitor(interval=0.5).start()
    _trial_t0 = _time.perf_counter()

    # ── Classifier selection ────────────────────────────────────────────────
    classifier_name = trial.suggest_categorical("classifier", kwargs["classifiers"])
    objective_average = str(kwargs.get("objective_average", "macro"))
    seed = int(kwargs.get("seed", DEFAULT_SEED))
    binary = bool(kwargs.get("binary", False))
    clf_cls, hypers = choose_classifier(trial, classifier_name, seed=seed, binary=binary)

    # ── Dataset ─────────────────────────────────────────────────────────────
    instances = int(kwargs.get("instances", 4015))
    balance_classes = bool(kwargs.get("balance_classes", True))
    max_per_class = kwargs.get("max_per_class")
    scaler_type = str(kwargs.get("scaler_type", "sequence_zscore"))
    noise_rate = float(kwargs.get("noise_rate", 0.0))
    noise_ratio = float(kwargs.get("noise_ratio", 0.0))
    folds = int(kwargs.get("folds", 5))

    X_train, Y_train, X_test, Y_test = _get_dataset(
        instances=instances,
        binary=binary,
        balance_classes=balance_classes,
        max_per_class=max_per_class,
        scaler_type=scaler_type,
        seed=seed,
    )

    X = np.concatenate((X_train, X_test), axis=0)
    Y = np.concatenate((Y_train, Y_test), axis=0)

    # Cap pool size to instances (matches classification.py behaviour).
    if len(X) > instances:
        rng_cap = np.random.default_rng(seed)
        idx = rng_cap.permutation(len(X))[:instances]
        idx.sort()
        X, Y = X[idx], Y[idx]

    # ── Environment metadata ────────────────────────────────────────────────
    for k, v in get_environment().items():
        trial.set_user_attr(f"env_{k}", v)

    trial.set_user_attr("study_name",     kwargs.get("study_name", ""))
    trial.set_user_attr("job_id",         kwargs.get("job_id", 0))
    trial.set_user_attr("baseline_mode",  "no_reservoir")
    trial.set_user_attr("classifier",     classifier_name)
    trial.set_user_attr("classifier_hypers", str(hypers))
    trial.set_user_attr("dataset_instances",     instances)
    trial.set_user_attr("dataset_binary",        binary)
    trial.set_user_attr("dataset_scaler",        scaler_type)
    trial.set_user_attr("dataset_noise_rate",    noise_rate)
    trial.set_user_attr("dataset_noise_ratio",   noise_ratio)
    trial.set_user_attr("dataset_balance_classes", balance_classes)
    trial.set_user_attr("dataset_max_per_class", str(max_per_class))
    trial.set_user_attr("dataset_cv_pool_size",  len(X))
    trial.set_user_attr("metric_primary_average", objective_average)
    trial.set_user_attr("metric_n_folds",        folds)
    trial.set_user_attr("feature_dim",           int(np.prod(X.shape[1:])))

    # ── Stratified k-fold CV ────────────────────────────────────────────────
    y_1d = _labels_to_1d(Y)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    f1s: List[float] = []
    accuracies: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []
    mccs: List[float] = []
    kappas: List[float] = []
    fold_runtimes: List[float] = []
    fold_fit_times: List[float] = []
    fold_predict_times: List[float] = []
    fold_class_metrics: List[dict] = []
    fold_confusion_matrices: List[np.ndarray] = []

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X, y_1d)):
        X_tr, Y_tr, X_va, Y_va = _prepare_fold(
            X, Y, train_idx, val_idx, fold_i,
            noise_rate=noise_rate,
            noise_ratio=noise_ratio,
            seed=seed,
            scaler_type=scaler_type,
        )

        # Y must be 1-D class indices for sklearn.
        y_tr = _labels_to_1d(Y_tr)
        y_va = _labels_to_1d(Y_va)

        clf = clf_cls(**hypers)

        _fold_t0 = _time.perf_counter()
        _fit_t0 = _time.perf_counter()
        clf.fit(X_tr, y_tr)
        _fit_s = _time.perf_counter() - _fit_t0

        _pred_t0 = _time.perf_counter()
        y_pred = clf.predict(X_va)
        _pred_s = _time.perf_counter() - _pred_t0
        _fold_runtime = round(_time.perf_counter() - _fold_t0, 4)

        # Wrap predictions into the shape evaluate_performance expects.
        # evaluate_performance handles both 1-D and 3-D; pass 3-D for
        # consistency with the reservoir pipeline.
        Y_va_3d = Y_va.reshape(Y_va.shape[0], 1, -1) if Y_va.ndim >= 2 else Y_va
        y_pred_oh = np.zeros_like(Y_va) if Y_va.ndim >= 2 else y_pred
        if Y_va.ndim >= 2:
            for idx, cls_idx in enumerate(y_pred):
                y_pred_oh[idx, ..., int(cls_idx)] = 1.0
            y_pred_3d = y_pred_oh.reshape(Y_va_3d.shape)
        else:
            Y_va_3d = Y_va
            y_pred_3d = y_pred

        metrics = evaluate_performance(
            Y_va_3d, y_pred_3d, time=_fold_runtime,
            primary_average=objective_average,
        )

        f1s.append(float(metrics.f1))
        accuracies.append(float(metrics.accuracy))
        precisions.append(float(metrics.precision))
        recalls.append(float(metrics.recall))
        mccs.append(float(metrics.mcc))
        kappas.append(float(metrics.kappa))
        fold_runtimes.append(_fold_runtime)
        fold_fit_times.append(_fit_s)
        fold_predict_times.append(_pred_s)
        fold_class_metrics.append(metrics.class_metrics)
        fold_confusion_matrices.append(metrics.confusion_matrix)

    mean_f1 = float(np.mean(f1s))
    trial_runtime = round(_time.perf_counter() - _trial_t0, 4)

    # ── Store per-fold metrics ──────────────────────────────────────────────
    for i, (f1, acc, prec, rec, mcc, kap, rt, fit_t, pred_t) in enumerate(
        zip(f1s, accuracies, precisions, recalls, mccs, kappas,
            fold_runtimes, fold_fit_times, fold_predict_times)
    ):
        trial.set_user_attr(f"fold_{i}_f1",        f1)
        trial.set_user_attr(f"fold_{i}_accuracy",  acc)
        trial.set_user_attr(f"fold_{i}_precision", prec)
        trial.set_user_attr(f"fold_{i}_recall",    rec)
        trial.set_user_attr(f"fold_{i}_mcc",       mcc)
        trial.set_user_attr(f"fold_{i}_kappa",     kap)
        trial.set_user_attr(f"fold_{i}_runtime_s", rt)
        trial.set_user_attr(f"fold_{i}_fit_s",     round(fit_t, 6))
        trial.set_user_attr(f"fold_{i}_predict_s", round(pred_t, 6))

    # ── Store aggregate metrics ─────────────────────────────────────────────
    trial.set_user_attr("metric_f1",        mean_f1)
    trial.set_user_attr("metric_f1_std",    float(np.std(f1s, ddof=0)))
    trial.set_user_attr("metric_f1_macro",
                        float(np.mean([m.get("macro avg", {}).get("f1-score", float("nan"))
                                       for m in fold_class_metrics])))
    trial.set_user_attr("metric_f1_weighted",
                        float(np.mean([m.get("weighted avg", {}).get("f1-score", float("nan"))
                                       for m in fold_class_metrics])))
    trial.set_user_attr("metric_accuracy",  float(np.mean(accuracies)))
    trial.set_user_attr("metric_precision", float(np.mean(precisions)))
    trial.set_user_attr("metric_recall",    float(np.mean(recalls)))
    trial.set_user_attr("metric_precision_macro",
                        float(np.mean([m.get("macro avg", {}).get("precision", float("nan"))
                                       for m in fold_class_metrics])))
    trial.set_user_attr("metric_precision_weighted",
                        float(np.mean([m.get("weighted avg", {}).get("precision", float("nan"))
                                       for m in fold_class_metrics])))
    trial.set_user_attr("metric_recall_macro",
                        float(np.mean([m.get("macro avg", {}).get("recall", float("nan"))
                                       for m in fold_class_metrics])))
    trial.set_user_attr("metric_recall_weighted",
                        float(np.mean([m.get("weighted avg", {}).get("recall", float("nan"))
                                       for m in fold_class_metrics])))
    trial.set_user_attr("metric_mcc",       float(np.mean(mccs)))
    trial.set_user_attr("metric_kappa",     float(np.mean(kappas)))
    trial.set_user_attr("metric_runtime",   trial_runtime)

    # Per-stage timing aggregates
    trial.set_user_attr("profile_fit_mean_s",
                        round(float(np.mean(fold_fit_times)), 6))
    trial.set_user_attr("profile_fit_std_s",
                        round(float(np.std(fold_fit_times, ddof=0)), 6))
    trial.set_user_attr("profile_fit_total_s",
                        round(float(np.sum(fold_fit_times)), 6))
    trial.set_user_attr("profile_predict_mean_s",
                        round(float(np.mean(fold_predict_times)), 6))
    trial.set_user_attr("profile_predict_std_s",
                        round(float(np.std(fold_predict_times, ddof=0)), 6))
    trial.set_user_attr("profile_predict_total_s",
                        round(float(np.sum(fold_predict_times)), 6))

    # Per-class metrics: mean AND std across folds
    _CLASS_KEY_MAP = {"f1-score": "f1", "precision": "precision",
                      "recall": "recall", "support": "support"}
    _class_stats = compute_class_dicts(fold_class_metrics)
    for cls_key, cls_vals in _class_stats.items():
        for mk, mv in cls_vals.items():
            raw_name, stat = mk.rsplit("_", 1)
            safe_mk = _CLASS_KEY_MAP.get(raw_name, raw_name.replace("-", "_"))
            trial.set_user_attr(f"metric_class_{cls_key}_{safe_mk}_{stat}", mv)

    # Macro/weighted averages
    _class_metrics_avg = compute_mean_dicts(fold_class_metrics)
    for cls_key, cls_vals in _class_metrics_avg.items():
        if not isinstance(cls_vals, dict):
            continue
        if cls_key in ("macro avg", "weighted avg"):
            tag = cls_key.split()[0]
            for mk, mv in cls_vals.items():
                safe_mk = _CLASS_KEY_MAP.get(mk, mk.replace("-", "_"))
                trial.set_user_attr(f"metric_{tag}_{safe_mk}", mv)

    # Aggregated confusion matrix
    agg_cm = sum(fold_confusion_matrices)
    trial.set_user_attr("metric_confusion_matrix", np.asarray(agg_cm).tolist())

    # Resource monitoring
    for k, v in monitor.stop().items():
        trial.set_user_attr(k, v)

    trial.set_user_attr(
        "trial_end_utc",
        datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )

    return mean_f1


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_research(args: argparse.Namespace) -> None:
    """Run the no-reservoir baseline Optuna study."""
    study_name = args.study_name or getattr(args, "study", None)
    os.makedirs("logs", exist_ok=True)
    storage = args.storage or get_default_storage(study_name)
    study = build_study(study_name=study_name, storage=storage, seed=int(args.seed))

    # ── Load study config from registry (if available) as defaults ──────────
    # CLI args override study-config values; study-config overrides hardcoded
    # defaults.  This is needed because launch_studies() only passes --study,
    # not every individual flag (--instances, --binary, --max_per_class, etc.).
    scfg: Dict[str, Any] = {}
    sfixed: Dict[str, Any] = {}
    try:
        from optimization.studies import STUDY_REGISTRY
        scfg = STUDY_REGISTRY.get(study_name, {})
        sfixed = scfg.get("fixed", {})
    except ImportError:
        pass

    def _cli_or(attr: str, study_key: str, fixed_key: str, default):
        """Resolve: CLI arg > study-level key > fixed key > default."""
        cli_val = getattr(args, attr, None)
        if cli_val is not None:
            return cli_val
        if study_key and study_key in scfg:
            return scfg[study_key]
        if fixed_key and fixed_key in sfixed:
            return sfixed[fixed_key]
        return default

    _mpc_raw = _cli_or("max_per_class", None, "max_per_class", None)
    max_per_class = int(_mpc_raw) if _mpc_raw is not None else None

    # binary: --binary is store_true so False means "not provided" — fall
    # through to the study config.
    binary = bool(getattr(args, "binary", False) or scfg.get("binary", False))

    params: Dict[str, Any] = {
        "classifiers": list(args.classifiers),
        "study_name": study_name,
        "job_id": int(args.job_id),
        "seed": int(args.seed),
        # Dataset parameters — CLI > study config > hardcoded defaults.
        "instances": int(_cli_or("instances", "instances", None, 4015)),
        "binary": binary,
        "balance_classes": bool(sfixed.get("balance_classes", True)),
        "max_per_class": max_per_class,
        "scaler_type": str(sfixed.get("scaler_type", "sequence_zscore")),
        "noise_rate": float(_cli_or("noise_rate", None, "noise_rate", 0.2999)),
        "noise_ratio": float(_cli_or("noise_ratio", None, "noise_ratio", 0.2862)),
        "folds": int(_cli_or("folds", "folds", None, 5)),
        "objective_average": str(
            _cli_or("primary_average", None, "objective_average", "macro")
        ),
    }

    research(
        study,
        int(getattr(args, "trials", None) or 500),
        objective,
        processes=int(args.processes),
        **params,
    )


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Analyse an existing no-reservoir baseline study."""
    study_name = args.study_name or getattr(args, "study", None)
    if not study_name:
        raise SystemExit("--study_name (or --study) is required for evaluate.")
    storage = args.storage or get_default_storage(study_name, must_exist=True)
    study = optuna.load_study(study_name=study_name, storage=storage)
    df = study.trials_dataframe()
    sections = parse_show_sections(getattr(args, "show", None) or SHOW_DEFAULT)
    classifiers_filter = getattr(args, "classifiers", None)

    # ── Environment ─────────────────────────────────────────────────────────
    if "env" in sections:
        best_trial = study.best_trial
        env_attrs = {k: v for k, v in best_trial.user_attrs.items()
                     if k.startswith("env_")}
        if env_attrs:
            sep = "=" * 72
            print(f"\n{sep}")
            print(f"  Environment — trial {best_trial.number}")
            print(sep)
            maxw = max(len(k.replace('env_', '')) for k in env_attrs)
            for k in sorted(env_attrs):
                label = k.replace('env_', '')
                print(f"    {label:<{maxw}} : {env_attrs[k]}")
        sections = sections - {"env"}

    # Global top-N table
    print_topn_trials(
        df, int(args.top_n),
        title=f"ALL classifiers — {study_name}",
        group_col="params_classifier",
        sections=sections,
        best_only_detail=True,
    )

    # Per-classifier top-N tables
    per_clf_sections = sections - {"env"}
    if "params_classifier" in df.columns:
        all_clfs = sorted({c for c in df["params_classifier"].dropna().unique()})
        if classifiers_filter:
            filter_set = {c.upper() for c in classifiers_filter}
            all_clfs = [c for c in all_clfs if c.upper() in filter_set]
        for clf in all_clfs:
            df_clf = df[df["params_classifier"] == clf]
            if df_clf.empty:
                continue
            print_topn_trials(
                df_clf, int(args.top_n),
                title=f"classifier={clf}",
                sections=per_clf_sections,
                strip_prefix=clf,
                best_only_detail=True,
            )

    _print_classifier_summary_table(df, int(args.top_n))

    # ── Plots ───────────────────────────────────────────────────────────────
    overview_arg = args.plots if args.plots is not None else "slice"
    wanted = {p.strip().lower() for p in overview_arg.split(",") if p.strip()}
    wanted.discard("none")
    classifier_plots_arg = getattr(args, "classifier_plots", None) or ""
    classifier_wanted = {
        p.strip().lower() for p in classifier_plots_arg.split(",") if p.strip()
    }
    classifier_wanted.discard("none")
    if not wanted and not classifier_wanted:
        return

    max_params = int(getattr(args, "max_params", 8))
    top_params = _important_params(study, max_n=max_params)

    if "slice" in wanted:
        params = top_params if top_params else None
        plot_results(
            study, plot_slice, params=params,
            filename=f"{study_name}-slice.png",
        )
    if "importance" in wanted:
        params = top_params if top_params else None
        plot_results(
            study, plot_param_importances, params=params,
            filename=f"{study_name}-importance.png",
        )
    if "edf" in wanted:
        plot_results(
            study, plot_edf,
            filename=f"{study_name}-edf.png",
        )

    # Per-classifier plots
    classifier_values = {
        t.params.get("classifier")
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and "classifier" in t.params
    }
    classifier_values.discard(None)

    for clf_name in sorted(classifier_values):
        label = str(clf_name)
        safe_label = label.replace(" ", "_")
        clf_param_names = set()
        for t in study.trials:
            if t.state != optuna.trial.TrialState.COMPLETE:
                continue
            if t.params.get("classifier") != clf_name:
                continue
            for pname in t.params.keys():
                if pname != "classifier":
                    clf_param_names.add(pname)
        if not clf_param_names:
            continue
        params_list = sorted(clf_param_names)
        if "slice" in classifier_wanted:
            plot_results(
                study, plot_slice, params=params_list,
                filename=f"{study_name}-{safe_label}-slice.png",
            )
        if "importance" in classifier_wanted:
            plot_results(
                study, plot_param_importances, params=params_list,
                filename=f"{study_name}-{safe_label}-importance.png",
            )
