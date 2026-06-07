"""Hyperparameter optimisation for the BioReservoir ECG classification pipeline.

Usage
-----
# List all named studies:
python -m optimization.optimize list-studies

# Run a named study (recommended):
python -m optimization.optimize research --type training --study g1-scaling --processes 64

# Evaluate an existing study database:
python -m optimization.optimize evaluate --type training --study g1-scaling

# Override study defaults from the CLI:
python -m optimization.optimize research --type training --study g1-scaling \\
    --instances 4015 --folds 1 --trials 150 --processes 64

Named studies are defined in optimization/studies.py.  Each study specifies
which parameters to optimise (with Optuna suggest specs) and which to hold
fixed.  Parameters not listed in the study are taken from BASELINE in studies.py.
"""

from __future__ import annotations

import argparse
import os
import logging
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import optuna
from reservoirpy.nodes import ScikitLearnNode
from optuna.visualization.matplotlib import (
    plot_param_importances,
    plot_slice,
    plot_edf,
    plot_contour,
    plot_intermediate_values,
    plot_rank,
    plot_optimization_history,
)
from sklearn.neighbors import KNeighborsClassifier


from utils.preprocessing import load_ecg_data
from utils.analysis import compute_class_dicts
from training import classify, cross_validate
from training.profiling import ResourceMonitor
from optimization.base import (
    DEFAULT_SEED,
    research,
    plot_results,
    print_topn_trials,
    parse_show_sections,
    SHOW_DEFAULT,
    get_default_storage,
    build_study,
    add_common_args,
)
from optimization.environment import get_environment
from optimization.reservoirs import build_reservoir
from optimization.studies import BASELINE, get_study, list_studies

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KNN readout hyperparameters (fixed for all reservoir optimisation studies).
# ro1-readout best (trial 658-664, F1=0.8900): n_neighbors=1, weights=uniform, p≈1.02
# ro1-readout-binary best:                     (see readout.py evaluate output)
#
# NOTE: _KNN_HYPERS below are updated to ro1-readout best values.
# r1b-refined was run with the old values (n_neighbors=4, weights=distance, p=1.42)
# for its 480 completed trials — those results are internally consistent for
# the r1a vs r1b topology comparison since both studies used the same hypers.
# f2-unbalanced must use these updated values for headline reporting.
# ---------------------------------------------------------------------------
_KNN_HYPERS: Dict[str, Any] = {
    "n_neighbors": 1,
    "weights":     "uniform",
    "algorithm":   "auto",
    "leaf_size":   32,
    "p":           1.02,
}


# ---------------------------------------------------------------------------
# Dataset cache  (keyed on all dataset parameters to avoid stale hits)
# ---------------------------------------------------------------------------
_DATA_CACHE: Dict[Tuple, Tuple[np.ndarray, ...]] = {}


def _get_dataset(
    *,
    instances: int,
    binary: bool,
    balance_classes: bool = True,
    max_per_class: Optional[int] = None,
    noise_rate: float = 0.0,
    noise_ratio: float = 0.0,
    scaler_type: str = "sequence_zscore",
    seed: int = 6337,
    preserve_split: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and cache the ECG dataset for the given configuration.

    The cache key includes all dataset parameters so that studies varying
    noise augmentation or class-balance settings get fresh, separate copies.
    """
    key = (
        int(instances), bool(binary), bool(balance_classes),
        max_per_class, float(noise_rate), float(noise_ratio),
        scaler_type, int(seed), bool(preserve_split),
    )
    if key not in _DATA_CACHE:
        _DATA_CACHE[key] = load_ecg_data(
            rows=instances,
            balance_classes=balance_classes,
            max_per_class=max_per_class,
            binary=binary,
            noise_rate=noise_rate,
            noise_ratio=noise_ratio,
            scaler_type=scaler_type,
            seed=seed,
            preserve_split=preserve_split,
        )
    return _DATA_CACHE[key]


# ---------------------------------------------------------------------------
# Study config parameter helpers
# ---------------------------------------------------------------------------

def _suggest(
    trial: optuna.Trial,
    name: str,
    spec: Dict[str, Any],
    current_params: Dict[str, Any],
) -> Any:
    """Dispatch an Optuna suggest call from a parameter spec dict.

    Handles conditional parameters: when the spec contains a
    ``conditional_on`` key, the suggestion is only made when
    ``current_params[conditional_on] == condition_value``.  Otherwise the
    BASELINE value is returned (and recorded as a fixed categorical so it
    still appears in the trial database).
    """
    if "conditional_on" in spec:
        dep_name  = spec["conditional_on"]
        dep_value = spec["condition_value"]
        if current_params.get(dep_name) != dep_value:
            fallback = current_params.get(name, BASELINE.get(name))
            # Log the fixed value so it appears in the trials_dataframe.
            return trial.suggest_categorical(f"fixed_{name}", [fallback])

    t = spec["type"]
    if t == "float":
        return trial.suggest_float(
            name, spec["low"], spec["high"],
            log=spec.get("log", False),
            step=spec.get("step"),
        )
    if t == "int":
        return trial.suggest_int(
            name, spec["low"], spec["high"],
            step=spec.get("step", 1),
        )
    if t == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"Unknown suggest type {t!r} for parameter {name!r}.")


def _build_params(
    trial: optuna.Trial,
    study_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Resolve all parameter values for this trial.

    Merge order: BASELINE <- study fixed overrides <- Optuna-suggested values.
    Conditional parameters see the already-resolved upstream values via
    ``current_params``, so upstream params must be listed first in the
    study's ``optimize`` dict.
    """
    params: Dict[str, Any] = dict(BASELINE)
    params.update(study_config.get("fixed", {}))
    for name, spec in study_config.get("optimize", {}).items():
        params[name] = _suggest(trial, name, spec, params)
    return params


def _build_topology(params: Dict[str, Any], seed: int):
    """Deprecated helper kept only to avoid breaking old imports."""
    raise RuntimeError(
        "_build_topology is deprecated. Use optimization.reservoirs.build_reservoir instead."
    )



# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def objective(trial: optuna.Trial, **kwargs) -> float:
    """Evaluate one trial: build reservoir from study config and run CV.

    All parameter values come from merging BASELINE with the active study
    config (fixed overrides + Optuna-suggested values).  Nothing is
    hard-coded here; edit optimization/studies.py to change what is
    optimised or fixed.
    """
    import datetime

    seed: int         = int(kwargs.get("seed", 6337))
    study_config: Dict[str, Any] = kwargs.get("study_config", {})
    profile: bool     = True  # always-on; negligible overhead

    trial.set_user_attr(
        "trial_start_utc",
        datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )
    monitor = ResourceMonitor(interval=0.5).start()

    # =========================================================================
    # 1. Resolve all parameters
    # =========================================================================
    params = _build_params(trial, study_config)

    # CLI-level overrides (--instances, --folds, --binary) take precedence
    # over study config defaults, enabling quick adjustments without editing
    # studies.py (useful for smoke tests: --instances 20).
    instances: int  = int(kwargs.get("instances")  or params.get("instances", 4015))
    binary:    bool = bool(kwargs.get("binary")     or params.get("binary",    False))
    folds:     int  = int(kwargs.get("folds")       or study_config.get("folds", 1))

    # scaler_type may be an optimisable param (g0-scaler) or fixed at BASELINE.
    scaler_type  = str(params.get("scaler_type", "sequence_zscore"))
    objective_average = str(kwargs.get("primary_average") or params.get("objective_average", "macro"))
    preserve_split = bool(kwargs.get("preserve_split") or params.get("preserve_split", False))
    noise_rate   = float(params["noise_rate"])
    noise_ratio  = float(params["noise_ratio"])

    if preserve_split and folds != 1:
        raise ValueError("preserve_split=True requires folds=1 so the original train/test split remains untouched.")

    # =========================================================================
    # 2. Dataset
    # =========================================================================
    X_train, Y_train, X_test, Y_test = _get_dataset(
        instances=instances,
        binary=binary,
        balance_classes=bool(params.get("balance_classes", True)),
        max_per_class=params.get("max_per_class"),
        noise_rate=noise_rate if preserve_split else 0.0,
        noise_ratio=noise_ratio if preserve_split else 0.0,
        scaler_type=scaler_type,
        seed=seed,
        preserve_split=preserve_split,
    )

    # =========================================================================
    # 3. Reservoir
    # =========================================================================
    reservoir, reservoir_attrs = build_reservoir(params, seed=seed)

    # =========================================================================
    # 4. Readout  (KNN, fixed — validated in ro1-readout study)
    # =========================================================================
    readout = ScikitLearnNode(KNeighborsClassifier, model_hypers=_KNN_HYPERS)

    # =========================================================================
    # 5. Store ALL parameters as user_attrs for complete reproducibility.
    #    The .db file alone is sufficient to reproduce any trial.
    # =========================================================================
    for k, v in get_environment().items():
        trial.set_user_attr(f"env_{k}", v)

    trial.set_user_attr("dataset_instances",        instances)
    trial.set_user_attr("dataset_binary",           binary)
    trial.set_user_attr("dataset_scaler",           scaler_type)
    trial.set_user_attr("dataset_noise_rate",       noise_rate)
    trial.set_user_attr("dataset_noise_ratio",      noise_ratio)
    trial.set_user_attr("dataset_preserve_split",   preserve_split)
    trial.set_user_attr("dataset_balance_classes",  params.get("balance_classes", True))
    trial.set_user_attr("dataset_max_per_class",     str(params.get("max_per_class")))
    trial.set_user_attr("dataset_seed",             seed)
    trial.set_user_attr("metric_primary_average",   objective_average)

    for key, value in reservoir_attrs.items():
        trial.set_user_attr(key, value)

    trial.set_user_attr("readout_type",         "KNN")
    trial.set_user_attr("readout_knn_hypers",   str(_KNN_HYPERS))
    # Resolve readout mode: aggregation comes from fixed/optimize/BASELINE;
    # window is only meaningful for "last_N" (g5b-readout study).
    _ra = str(params.get("readout_aggregation", "final"))
    _rw = int(params["readout_window"]) if _ra == "last_N" else 1
    # Guard: last_N with window=1 is identical to "final" but will crash the
    # pipeline which expects "last_N:K" with K>1.  Fall back to signal_mean.
    if _ra == "last_N" and _rw <= 1:
        _ra = "signal_mean"
        _rw = 1
    trial.set_user_attr("readout_aggregation",  _ra)
    trial.set_user_attr("readout_window",       _rw)

    trial.set_user_attr("study_name",  kwargs.get("study_name", ""))
    trial.set_user_attr("study_phase", study_config.get("description", "")[:80])
    trial.set_user_attr("job_id",      kwargs.get("job_id", 0))

    # =========================================================================
    # 6. Cross-validation
    # =========================================================================
    trial_name: str      = f"{kwargs['study_name']}-{trial.number}"
    save_threshold: float = float(kwargs.get("save_threshold", 0.0))
    save_states: bool    = bool(kwargs.get("save_states", False))

    readout_aggregation: str = _ra
    readout_window: int      = _rw

    _fw = int(kwargs.get("fold_workers", 1))
    if preserve_split:
        trial.set_user_attr("dataset_cv_pool_size", len(X_train) + len(X_test))
        metrics = classify(
            reservoir,
            readout,
            X_train,
            Y_train,
            X_test,
            Y_test,
            folds=1,
            trial_name=trial_name,
            save_threshold=save_threshold,
            save_states=save_states,
            noise_rate=0.0,
            noise_ratio=0.0,
            readout_aggregation=readout_aggregation,
            readout_window=readout_window,
            primary_average=objective_average,
        )
    else:
        X = np.concatenate((X_train, X_test), axis=0)
        Y = np.concatenate((Y_train, Y_test), axis=0)

        n_before = len(X)
        if n_before > instances:
            rng_cap = np.random.default_rng(seed)
            idx = rng_cap.permutation(n_before)[:instances]
            idx.sort()
            X, Y = X[idx], Y[idx]
            logger.debug("CV pool capped: %d → %d samples.", n_before, len(X))

        trial.set_user_attr("dataset_cv_pool_size", len(X))

        metrics = cross_validate(
            reservoir,
            readout,
            X,
            Y,
            folds=folds,
            trial_name=trial_name,
            save_threshold=save_threshold,
            save_states=save_states,
            noise_rate=noise_rate,
            noise_ratio=noise_ratio,
            fold_backend="loky" if _fw > 1 else "sequential",
            fold_workers=_fw if _fw > 1 else None,
            seed=seed,
            scaler_type=scaler_type,
            readout_aggregation=readout_aggregation,
            readout_window=readout_window,
            primary_average=objective_average,
            profile=profile,
        )

    # =========================================================================
    # 7. Store output metrics
    # =========================================================================
    _folds_data = metrics.get("folds", [])
    _fold_f1s = [fd.get("f1", float("nan")) for fd in _folds_data]

    trial.set_user_attr("metric_f1",        metrics.get("f1",        float("nan")))
    trial.set_user_attr("metric_f1_std",    float(np.std(_fold_f1s, ddof=0)) if _fold_f1s else float("nan"))
    trial.set_user_attr("metric_f1_macro",  metrics.get("f1_macro",  float("nan")))
    trial.set_user_attr("metric_f1_weighted", metrics.get("f1_weighted", float("nan")))
    trial.set_user_attr("metric_accuracy",  metrics.get("accuracy",  float("nan")))
    trial.set_user_attr("metric_precision", metrics.get("precision", float("nan")))
    trial.set_user_attr("metric_recall",    metrics.get("recall",    float("nan")))
    trial.set_user_attr("metric_precision_macro", metrics.get("precision_macro", float("nan")))
    trial.set_user_attr("metric_precision_weighted", metrics.get("precision_weighted", float("nan")))
    trial.set_user_attr("metric_recall_macro", metrics.get("recall_macro", float("nan")))
    trial.set_user_attr("metric_recall_weighted", metrics.get("recall_weighted", float("nan")))
    trial.set_user_attr("metric_mcc",       metrics.get("mcc",       float("nan")))
    trial.set_user_attr("metric_kappa",     metrics.get("kappa",     float("nan")))
    trial.set_user_attr("metric_runtime",   metrics.get("runtime",   float("nan")))
    trial.set_user_attr("metric_n_folds",   len(_folds_data))

    # Per-fold breakdown (scalars + MCC/kappa)
    for fold_dict in _folds_data:
        fi = fold_dict.get("fold_index", 0)
        trial.set_user_attr(f"fold_{fi}_f1",        fold_dict.get("f1",        float("nan")))
        trial.set_user_attr(f"fold_{fi}_f1_macro",  fold_dict.get("f1_macro",  float("nan")))
        trial.set_user_attr(f"fold_{fi}_f1_weighted", fold_dict.get("f1_weighted", float("nan")))
        trial.set_user_attr(f"fold_{fi}_accuracy",  fold_dict.get("accuracy",  float("nan")))
        trial.set_user_attr(f"fold_{fi}_precision", fold_dict.get("precision", float("nan")))
        trial.set_user_attr(f"fold_{fi}_recall",    fold_dict.get("recall",    float("nan")))
        trial.set_user_attr(f"fold_{fi}_precision_macro", fold_dict.get("precision_macro", float("nan")))
        trial.set_user_attr(f"fold_{fi}_precision_weighted", fold_dict.get("precision_weighted", float("nan")))
        trial.set_user_attr(f"fold_{fi}_recall_macro", fold_dict.get("recall_macro", float("nan")))
        trial.set_user_attr(f"fold_{fi}_recall_weighted", fold_dict.get("recall_weighted", float("nan")))
        trial.set_user_attr(f"fold_{fi}_mcc",       fold_dict.get("mcc",       float("nan")))
        trial.set_user_attr(f"fold_{fi}_kappa",     fold_dict.get("kappa",     float("nan")))
        trial.set_user_attr(f"fold_{fi}_runtime_s", fold_dict.get("runtime",   float("nan")))

    # Per-class metrics: mean AND std across folds
    _CLASS_KEY_MAP = {"f1-score": "f1", "precision": "precision",
                      "recall": "recall", "support": "support"}
    fold_class_dicts = [fd.get("class_metrics", {}) for fd in _folds_data]
    if fold_class_dicts:
        _class_stats = compute_class_dicts(fold_class_dicts)
        for cls_key, cls_vals in _class_stats.items():
            for mk, mv in cls_vals.items():
                raw_name, stat = mk.rsplit("_", 1)  # e.g. "f1-score_mean"
                safe_mk = _CLASS_KEY_MAP.get(raw_name, raw_name.replace("-", "_"))
                trial.set_user_attr(f"metric_class_{cls_key}_{safe_mk}_{stat}", mv)

    # Macro/weighted averages (from mean class_metrics)
    _class_metrics_avg = metrics.get("class_metrics", {})
    for cls_key, cls_vals in _class_metrics_avg.items():
        if not isinstance(cls_vals, dict):
            continue
        if cls_key in ("macro avg", "weighted avg"):
            tag = cls_key.split()[0]
            for mk, mv in cls_vals.items():
                safe_mk = _CLASS_KEY_MAP.get(mk, mk.replace("-", "_"))
                trial.set_user_attr(f"metric_{tag}_{safe_mk}", mv)

    # Aggregated confusion matrix (flattened for Optuna storage)
    _cm = metrics.get("confusion_matrix")
    if _cm is not None:
        trial.set_user_attr("metric_confusion_matrix", np.asarray(_cm).tolist())

    if profile:
        for section, stats in metrics.get("profiling", {}).items():
            safe = section.replace("/", "_")
            for k, v in stats.items():
                trial.set_user_attr(f"profile_{safe}_{k}", v)
        for k, v in metrics.get("fold_runtimes", {}).items():
            if k == "per_fold":
                for i, rt in enumerate(v):
                    trial.set_user_attr(f"fold_runtime_fold_{i}_s", rt)
            else:
                trial.set_user_attr(f"fold_runtime_{k}", v)

    for k, v in monitor.stop().items():
        trial.set_user_attr(k, v)

    trial.set_user_attr(
        "trial_end_utc",
        datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )
    return float(metrics["f1"])


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_list_studies(_args) -> None:
    list_studies()


def cmd_research(args: argparse.Namespace) -> None:
    study_config = get_study(args.study)
    study_name   = args.study_name or args.study

    os.makedirs("logs", exist_ok=True)
    storage = args.storage or get_default_storage(study_name)

    n_trials  = args.trials    or study_config["n_trials"]
    folds     = args.folds     or study_config["folds"]
    instances = args.instances or study_config["instances"]
    binary    = args.binary    or study_config.get("binary", False)

    study = build_study(
        study_name=study_name,
        storage=storage,
        study_config=study_config,
        sampler_override=args.sampler,
    )

    kwargs: Dict[str, Any] = {
        "study_name":     study_name,
        "study_config":   study_config,
        "instances":      instances,
        "binary":         binary,
        "folds":          folds,
        "save_threshold": float(args.save_threshold),
        "save_states":    bool(args.save_states),
        "fold_workers":   int(args.fold_workers),
        "profile":        bool(args.profile),
        "seed":           int(args.seed),
        "preserve_split": bool(args.preserve_split),
        "primary_average": args.primary_average,
        "log_level":      getattr(logging, args.log_level),
        "verbosity":      args.verbosity,
        "job_id":         int(args.job_id),
    }

    research(study, int(n_trials), objective, processes=int(args.processes), **kwargs)


# ---------------------------------------------------------------------------
# Rerun-best command
# ---------------------------------------------------------------------------

def cmd_rerun_best(args: argparse.Namespace) -> None:
    """Re-run the best Optuna trial of a completed study with save_states=True.

    This is the Phase 2 → Phase 3 bridge.  It loads the best trial from a
    completed Phase 2 study (e.g. ``r1a-refined``), resolves the full parameter
    set (BASELINE + study fixed + Optuna-suggested values), and re-executes
    cross-validation with ``save_states=True`` so that
    :mod:`optimization.readout` can optimise the readout classifier on the
    frozen state matrices without re-running the expensive reservoir.

    Fold artefacts are saved under ``results/runs/{trial_name}/`` where
    *trial_name* defaults to ``{study_name}-best``.

    Typical usage (run on the same host as the Phase 2 study)::

        # 1. Save reservoir states for best r1a-refined trial
        python -m optimization.optimize rerun-best \\
            --study r1a-refined --folds 5

        # 2. Optimise the readout on those frozen states
        python -m optimization.optimize research --type readout \\
            --study_name ro1-readout \\
            --trial_name r1a-refined-best \\
            --classifiers KNN RF Ridge LR \\
            --trials 500 --processes 32
    """
    from optimization.studies import get_study as _get_study_cfg

    study_name = args.study_name or getattr(args, "study", None)
    if not study_name:
        raise SystemExit("--study_name is required for rerun-best.")
    storage = args.storage or get_default_storage(study_name)
    study   = optuna.load_study(study_name=study_name, storage=storage)

    try:
        best = study.best_trial
    except ValueError:
        raise SystemExit(f"No completed trials in study '{study_name}'.")

    study_config = _get_study_cfg(study_name)
    params = _resolve_params_from_trial(best, study_config)

    seed        = int(getattr(args, "seed", 6337))
    instances   = int(getattr(args, "instances", None) or study_config.get("instances", 4015))
    binary      = bool(getattr(args, "binary", False) or study_config.get("binary", False))
    folds       = int(getattr(args, "folds", None) or study_config.get("folds", 1))
    scaler_type = str(params.get("scaler_type", "sequence_zscore"))
    trial_name  = getattr(args, "trial_name", None) or f"{study_name}-best"
    preserve_split = bool(getattr(args, "preserve_split", False) or params.get("preserve_split", False))
    objective_average = str(getattr(args, "primary_average", None) or params.get("objective_average", "macro"))
    noise_rate  = float(params["noise_rate"])
    noise_ratio = float(params["noise_ratio"])
    # Allow CLI overrides for re-validated augmentation values.
    if getattr(args, "noise_rate", None) is not None:
        noise_rate = float(args.noise_rate)
        logger.info("Overriding noise_rate → %.4f (from CLI)", noise_rate)
    if getattr(args, "noise_ratio", None) is not None:
        noise_ratio = float(args.noise_ratio)
        logger.info("Overriding noise_ratio → %.4f (from CLI)", noise_ratio)

    if preserve_split and folds != 1:
        raise SystemExit("--preserve_split requires --folds 1 so the original train/test split remains untouched.")

    logger.info(
        "Rerunning best trial %d of '%s' (F1=%.4f) → trial_name='%s', folds=%d, binary=%s, preserve_split=%s, average=%s",
        best.number, study_name, best.value, trial_name, folds, binary, preserve_split, objective_average,
    )

    X_train, Y_train, X_test, Y_test = _get_dataset(
        instances=instances,
        binary=binary,
        balance_classes=bool(params.get("balance_classes", True)),
        max_per_class=params.get("max_per_class"),
        noise_rate=noise_rate if preserve_split else 0.0,
        noise_ratio=noise_ratio if preserve_split else 0.0,
        scaler_type=scaler_type,
        seed=seed,
        preserve_split=preserve_split,
    )
    reservoir, _reservoir_attrs = build_reservoir(params, seed=seed)

    readout_study_name = getattr(args, "readout_study", None)
    if readout_study_name:
        from optimization.readout import choose_classifier
        ro_storage = get_default_storage(readout_study_name)
        ro_study = optuna.load_study(study_name=readout_study_name, storage=ro_storage)
        readout_trial_number = getattr(args, "readout_trial", None)
        if readout_trial_number is not None:
            ro_best = ro_study.trials[readout_trial_number]
        else:
            ro_best = ro_study.best_trial
        clf_name = ro_best.params["classifier"]
        clf_class, clf_hypers = choose_classifier(ro_best, clf_name, seed=seed)
        logger.info(
            "Using readout from study '%s' trial %d: %s %s",
            readout_study_name, ro_best.number, clf_name, clf_hypers,
        )
        readout = ScikitLearnNode(clf_class, model_hypers=clf_hypers)
    else:
        readout = ScikitLearnNode(KNeighborsClassifier, model_hypers=_KNN_HYPERS)

    _ra = str(params.get("readout_aggregation", "final"))
    _rw = int(params.get("readout_window", 1)) if _ra == "last_N" else 1

    processes = int(getattr(args, "processes", 1))
    fold_backend = "loky" if processes > 1 else "sequential"
    fold_workers = processes if processes > 1 else None
    profile = bool(getattr(args, "profile", False))

    if preserve_split:
        metrics = classify(
            reservoir,
            readout,
            X_train,
            Y_train,
            X_test,
            Y_test,
            trial_name=trial_name,
            save_states=True,
            readout_aggregation=_ra,
            readout_window=_rw,
            primary_average=objective_average,
        )
    else:
        X = np.concatenate((X_train, X_test), axis=0)
        Y = np.concatenate((Y_train, Y_test), axis=0)
        if len(X) > instances:
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.permutation(len(X))[:instances])
            X, Y = X[idx], Y[idx]

        metrics = cross_validate(
            reservoir, readout, X, Y,
            folds=folds,
            trial_name=trial_name,
            save_threshold=0.0,
            save_states=True,
            scaler_type=scaler_type,
            readout_aggregation=_ra,
            readout_window=_rw,
            noise_rate=noise_rate,
            noise_ratio=noise_ratio,
            fold_backend=fold_backend,
            fold_workers=fold_workers,
            seed=seed,
            profile=profile,
            primary_average=objective_average,
        )

    logger.info(
        "Saved %d fold artefact(s) with states under results/runs/%s/",
        folds, trial_name,
    )
    logger.info(
        "Rerun result — F1=%.4f  accuracy=%.4f  MCC=%.4f  kappa=%.4f",
        float(metrics.get("f1", 0.0)),
        float(metrics.get("accuracy", 0.0)),
        float(metrics.get("mcc", 0.0)),
        float(metrics.get("kappa", 0.0)),
    )
    if profile and "profiling" in metrics:
        logger.info("Pipeline timings:")
        for section, stats in metrics["profiling"].items():
            logger.info("  %s: %s", section, stats)
    logger.info(
        "Next: python -m optimization.optimize research --type readout "
        "--study_name ro1-readout --trial_name %s "
        "--classifiers KNN RF Ridge LR --trials 500 --processes <N>",
        trial_name,
    )


# ---------------------------------------------------------------------------
# Evaluate helpers
# ---------------------------------------------------------------------------

def _important_params(study: optuna.Study, max_n: int = 8) -> List[str]:
    """Return up to *max_n* non-fixed parameter names ranked by Optuna importance.

    ``fixed_*`` parameters are single-value conditional placeholders (e.g.
    ``fixed_readout_window`` when ``readout_aggregation != "last_N"``); they
    carry no information and should not appear in plots.

    Falls back to sorted alphabetical order when importance cannot be computed
    (e.g. fewer than 2 completed trials).
    """
    try:
        from optuna.importance import get_param_importances
        imps = get_param_importances(study)
        return [p for p in imps if not p.startswith("fixed_")][:max_n]
    except Exception:
        seen: Dict[str, bool] = {}
        for t in study.trials:
            if t.state.name == "COMPLETE":
                for p in t.params:
                    if not p.startswith("fixed_"):
                        seen[p] = True
        return sorted(seen)[:max_n]


def _resolve_params_from_trial(
    trial: optuna.Trial,
    study_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge BASELINE + study fixed overrides + trial params.

    Handles ``fixed_*`` prefix keys produced by :func:`_suggest` for
    conditional parameters whose condition was not met.
    """
    params: Dict[str, Any] = dict(BASELINE)
    params.update(study_config.get("fixed", {}))
    for k, v in trial.params.items():
        if k.startswith("fixed_"):
            real_k = k[len("fixed_"):]
            # Don't overwrite when the real param is also present as a direct suggest.
            if real_k not in trial.params:
                params[real_k] = v
        else:
            params[k] = v
    return params


def _plot_confusion_matrix(
    study: optuna.Study,
    study_name: str,
    study_config: Dict[str, Any],
    *,
    filename: Optional[str] = None,
) -> None:
    """Re-run the best trial and plot an aggregate confusion matrix.

    Cross-validation is re-executed with the best trial's parameters;
    ``metrics["confusion_matrix"]`` (the count matrix summed across all folds,
    produced by :func:`utils.analysis.compute_mean_metrics`) is passed
    directly to :func:`utils.visualisation.plot_confusion_matrix`, which
    normalises by row sums and renders a seaborn heat-map.

    The figure is saved to ``results/metrics/{filename}``.
    """
    from utils.visualisation import plot_confusion_matrix as _plot_cm

    try:
        best = study.best_trial
    except ValueError:
        logger.warning("[%s] No completed trials — skipping confusion matrix.", study_name)
        return

    params      = _resolve_params_from_trial(best, study_config)
    instances   = int(study_config.get("instances", 4015))
    binary      = bool(study_config.get("binary", False))
    folds       = int(study_config.get("folds", 1))
    scaler_type = str(params.get("scaler_type", "sequence_zscore"))
    seed        = 6337
    preserve_split = bool(params.get("preserve_split", False))
    objective_average = str(params.get("objective_average", "macro"))
    noise_rate  = float(params["noise_rate"])
    noise_ratio = float(params["noise_ratio"])

    logger.info(
        "[%s] Re-running best trial %d (F1=%.4f) to compute confusion matrix...",
        study_name, best.number, best.value,
    )

    X_train, Y_train, X_test, Y_test = _get_dataset(
        instances=instances,
        binary=binary,
        balance_classes=bool(params.get("balance_classes", True)),
        max_per_class=params.get("max_per_class"),
        noise_rate=noise_rate if preserve_split else 0.0,
        noise_ratio=noise_ratio if preserve_split else 0.0,
        scaler_type=scaler_type,
        seed=seed,
        preserve_split=preserve_split,
    )
    reservoir, _reservoir_attrs = build_reservoir(params, seed=seed)
    readout = ScikitLearnNode(KNeighborsClassifier, model_hypers=_KNN_HYPERS)

    _ra = str(params.get("readout_aggregation", "final"))
    _rw = int(params.get("readout_window", 1)) if _ra == "last_N" else 1

    if preserve_split:
        metrics = classify(
            reservoir,
            readout,
            X_train,
            Y_train,
            X_test,
            Y_test,
            readout_aggregation=_ra,
            readout_window=_rw,
            primary_average=objective_average,
        )
    else:
        X = np.concatenate((X_train, X_test), axis=0)
        Y = np.concatenate((Y_train, Y_test), axis=0)
        if len(X) > instances:
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.permutation(len(X))[:instances])
            X, Y = X[idx], Y[idx]

        metrics = cross_validate(
            reservoir, readout, X, Y,
            folds=folds,
            scaler_type=scaler_type,
            readout_aggregation=_ra,
            readout_window=_rw,
            noise_rate=noise_rate,
            noise_ratio=noise_ratio,
            seed=seed,
            primary_average=objective_average,
        )

    cm = metrics.get("confusion_matrix")
    if cm is None:
        logger.warning("[%s] No confusion matrix returned by cross_validate.", study_name)
        return

    fname = filename or f"{study_name}-confusion.png"
    _plot_cm(cm, filename=fname, show=False)
    logger.info("[%s] Confusion matrix saved → results/metrics/%s", study_name, fname)


# ---------------------------------------------------------------------------
# Evaluate command
# ---------------------------------------------------------------------------

def cmd_evaluate(args: argparse.Namespace) -> None:
    study_name = args.study_name or getattr(args, "study", None)
    if not study_name:
        raise SystemExit("--study_name (or --study) is required for evaluate.")
    storage = args.storage or get_default_storage(study_name, must_exist=True)
    study   = optuna.load_study(study_name=study_name, storage=storage)
    df      = study.trials_dataframe()
    sections = parse_show_sections(getattr(args, "show", None) or SHOW_DEFAULT)

    print_topn_trials(df, int(args.top_n), title=f"study — {study_name}", sections=sections)

    plots_arg = args.plots if args.plots is not None else "slice,importance"
    wanted = {p.strip().lower() for p in plots_arg.split(",") if p.strip()}
    if not wanted:
        return

    # Compute top-N most important non-fixed params for param-aware plots.
    max_params = int(getattr(args, "max_params", 8))
    top_params = _important_params(study, max_n=max_params)

    # Param-aware plots (slice, importance, contour) receive the filtered list.
    # Other plots (history, edf, rank, intermediate) don't consume a params arg.
    _params_plots = {"slice", "importance", "contour"}

    plot_map = {
        "slice":        (plot_slice,                f"{study_name}-slice.png"),
        "importance":   (plot_param_importances,    f"{study_name}-importance.png"),
        "rank":         (plot_rank,                 f"{study_name}-rank.png"),
        "edf":          (plot_edf,                  f"{study_name}-edf.png"),
        "intermediate": (plot_intermediate_values,  f"{study_name}-intermediate.png"),
        "contour":      (plot_contour,              f"{study_name}-contour.png"),
        "history":      (plot_optimization_history, f"{study_name}-history.png"),
    }
    for key, (fn, fname) in plot_map.items():
        if key in wanted:
            p = top_params if (key in _params_plots and top_params) else None
            plot_results(study, fn, params=p, filename=fname)

    if "confusion" in wanted:
        try:
            study_config = get_study(study_name)
        except Exception:
            study_config = {}
        _plot_confusion_matrix(
            study, study_name, study_config,
            filename=f"{study_name}-confusion.png",
        )
