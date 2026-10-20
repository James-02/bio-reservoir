"""training.cross_validate — k-fold cross-validation runner.

This module orchestrates the full CV pipeline: splitting data into stratified
folds, running each fold (reservoir training + readout fitting + prediction),
averaging metrics across folds, and optionally collecting pipeline profiling.

Public functions
----------------
cross_validate
    Full k-fold CV with configurable backend (sequential / loky).
run_fold
    Execute a single fold; can be called directly for debugging.
classify
    High-level entry point: train on X_train, evaluate on X_test *or* run CV
    when ``folds > 1``.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import reservoirpy as rpy
from reservoirpy.node import Node
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from training.pipeline import (
    _stack_runs,
    fit_readout,
    predict,
    selected_readout_params,
    train,
)
from training.profiling import Profiler, _merge_profiles
from utils import config
from utils.analysis import Metrics, compute_mean_metrics, evaluate_performance
from utils.logger import setup_logging
from utils.preprocessing import augment_data, standardize_data
from utils.results import FoldArtifact

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thread-pool guard (important when using numba parallel inside CV workers)
# ---------------------------------------------------------------------------

def _limit_native_threads() -> None:
    """Avoid nested native threadpools inside CV workers.

    When combining per-fold multiprocessing (separate process per fold) with
    Numba/OpenMP parallel regions inside the DDE solver, you can easily
    oversubscribe cores or hit unsafe fork/OpenMP issues.
    Only applies when the user hasn't already configured an explicit limit.
    """
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(var, "1")


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def _labels_to_1d(Y: np.ndarray) -> np.ndarray:
    """Convert labels to a 1-D class-index vector for stratified splitting."""
    Y = np.asarray(Y)
    if Y.ndim == 1:
        return Y.astype(int)
    if Y.ndim == 3:      # (N, 1, C) one-hot
        return np.argmax(Y[:, 0, :], axis=-1).astype(int)
    if Y.ndim == 2:      # (N, C) one-hot
        return np.argmax(Y, axis=-1).astype(int)
    raise ValueError(f"Unsupported Y shape for stratification: {Y.shape}")


# ---------------------------------------------------------------------------
# Fold dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Fold:
    train_idx: np.ndarray
    val_idx: np.ndarray
    fold_index: int


# ---------------------------------------------------------------------------
# Fold data helpers
# ---------------------------------------------------------------------------

def _iter_folds(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    folds: int,
    seed: int = config.SEED,
) -> list[Fold]:
    """Return train/val index pairs for cross-validation.

    When *folds* == 1 a single stratified 80/20 hold-out split is used
    (``StratifiedShuffleSplit``) because ``StratifiedKFold`` requires at
    least 2 splits.  For *folds* >= 2 standard k-fold is used.
    """
    y_1d = _labels_to_1d(Y)
    if folds == 1:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=int(seed))
    else:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(seed))
    return [
        Fold(train_idx=train_idx, val_idx=val_idx, fold_index=i)
        for i, (train_idx, val_idx) in enumerate(splitter.split(X, y_1d))
    ]


def _prepare_fold_data(
    X: np.ndarray,
    Y: np.ndarray,
    fold: Fold,
    *,
    noise_rate: float,
    noise_ratio: float,
    seed: int,
    scaler_type: str = "sequence_zscore",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train_fold, X_val = X[fold.train_idx], X[fold.val_idx]
    Y_train_fold, Y_val = Y[fold.train_idx], Y[fold.val_idx]

    # Augmentation is applied *only* to the training fold (never validation).
    # A fold-specific RNG ensures reproducibility across re-runs.
    fold_rng = np.random.RandomState(int(seed) + int(fold.fold_index))
    X_train_fold, Y_train_fold = augment_data(
        X_train_fold, Y_train_fold, noise_rate, noise_ratio, rng=fold_rng
    )
    X_train_fold, X_val = standardize_data(X_train_fold, X_val, scaler_type=scaler_type)
    return X_train_fold, Y_train_fold, X_val, Y_val


def _average_fold_artifacts(
    fold_artifacts: list[FoldArtifact],
    profile: bool = False,
) -> dict[str, Any]:
    """Average metrics across fold artefacts and optionally merge profiling.

    Returns a plain ``dict`` with the following structure:

    * Top-level scalar keys (``"f1"``, ``"accuracy"``, etc.) — mean across folds,
      same as before (backward compatible).
    * ``"confusion_matrix"`` — aggregate (summed) confusion matrix across all
      folds; normalisation is left to the caller / plotting function.
    * ``"class_metrics"`` — per-class precision/recall/F1/support, averaged
      across folds via :func:`~utils.analysis.compute_mean_dicts`.
    * ``"folds"`` — list of per-fold dicts, each containing the full
      :class:`~utils.analysis.Metrics` breakdown for that fold (all scalars,
      individual confusion matrix, per-class metrics).
    * ``"profiling"`` / ``"fold_runtimes"`` — present only when
      ``profile=True``.
    """
    import math
    mean: Metrics = compute_mean_metrics([fa.metrics for fa in fold_artifacts])
    result: dict[str, Any] = mean.to_dict()
    # Per-fold breakdown: each entry is the complete Metrics dict for that fold.
    result["folds"] = [
        {"fold_index": fa.fold_index, **fa.metrics.to_dict()}
        for fa in fold_artifacts
    ]
    if profile:
        result["profiling"] = _merge_profiles(
            [fa.profiling for fa in fold_artifacts]
        )
        runtimes = [fa.metrics.runtime for fa in fold_artifacts]
        n = len(runtimes)
        mean_rt = sum(runtimes) / n
        std_rt  = math.sqrt(sum((t - mean_rt) ** 2 for t in runtimes) / (n - 1)) if n > 1 else 0.0
        result["fold_runtimes"] = {
            "mean_s":   round(mean_rt, 4),
            "std_s":    round(std_rt,  4),
            "min_s":    round(min(runtimes), 4),
            "max_s":    round(max(runtimes), 4),
            "per_fold": [round(t, 4) for t in runtimes],
        }
    return result


# ---------------------------------------------------------------------------
# Single fold
# ---------------------------------------------------------------------------

def run_fold(
    reservoir: Node,
    readout: Node,
    X_train_fold: np.ndarray,
    Y_train_fold: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    *,
    fold_index: int,
    trial_name: str = "",
    save_states: bool = False,
    readout_aggregation: str = "final",
    readout_window: int = 1,
    primary_average: str = "weighted",
    profile: bool = False,
    labels: np.ndarray | None = None,
) -> FoldArtifact:
    """Execute a single CV fold: train → fit → validate → predict → metrics.

    Parameters
    ----------
    reservoir, readout : Node
        The run's template nodes. The fold works on copies, so every fold uses the same
        reservoir (weights, warm-up state, noise generator state) and an unfitted readout.
    X_train_fold, Y_train_fold : np.ndarray
        Augmented + scaled training data for this fold.
    X_val, Y_val : np.ndarray
        Scaled validation data.
    fold_index : int
        Zero-based fold index (used for logging and the returned artefact).
    trial_name : str
        Stored as metadata in the returned :class:`~utils.results.FoldArtifact`.
        Saving to disk is handled by :func:`cross_validate` after all folds
        complete and the trial-level threshold is evaluated.
    save_states : bool, default False
        Whether to retain ``train_states`` and ``test_states`` in the returned
        artefact.  Required when the artefact will later be saved with
        ``save_states=True`` (e.g. for readout re-optimisation or state
        visualisations).  When ``False`` the state arrays are freed after the
        fold completes, keeping per-fold memory usage low.
    readout_aggregation : str
        See :func:`training.pipeline._aggregate_states`.
    readout_window : int
        See :func:`training.pipeline.train`.
    profile : bool
        Enable per-phase timing.  Results are stored in the returned
        artefact's ``profiling`` field (not persisted to disk).
    labels : np.ndarray, optional
        Classes the averaged metrics run over (see
        :func:`~utils.analysis.evaluate_performance`); every class by default.

    Returns
    -------
    FoldArtifact
        Contains metrics, predictions and labels.  States included only when
        ``save_states=True``.  Saving to disk is deferred to the caller.
    """
    rpy.verbosity(0)
    setup_logging(level=logging.DEBUG)
    logger.info("Cross-validation fold %s", str(fold_index))
    _limit_native_threads()

    reservoir = reservoir.copy()
    readout = readout.copy()

    profiler = Profiler(enabled=profile)
    start_time = time.perf_counter()

    with profiler.section("train_reservoir"):
        train_states = train(
            reservoir, X_train_fold,
            readout_aggregation=readout_aggregation,
            readout_window=readout_window,
            profiler=profiler,
        )
    with profiler.section("fit_readout"):
        fit_readout(readout, train_states, Y_train_fold)
    with profiler.section("val_reservoir"):
        test_states = train(
            reservoir, X_val,
            readout_aggregation=readout_aggregation,
            readout_window=readout_window,
            profiler=profiler,
        )
    with profiler.section("predict"):
        Y_pred_list = predict(readout, test_states)

    Y_pred = _stack_runs(Y_pred_list)

    if len(Y_pred) != len(Y_val):
        raise ValueError(
            f"Prediction/label length mismatch in fold {fold_index}: "
            f"len(Y_pred)={len(Y_pred)} len(Y_val)={len(Y_val)}"
        )

    runtime = round(time.perf_counter() - start_time, 4)
    fold_metrics: Metrics = evaluate_performance(
        Y_val, Y_pred, runtime, primary_average=primary_average, labels=labels
    )
    selected = selected_readout_params(readout)

    return FoldArtifact(
        trial_name   = trial_name,
        fold_index   = fold_index,
        Y_train      = Y_train_fold,
        Y_test       = Y_val,
        Y_pred       = Y_pred,
        metrics      = fold_metrics,
        train_states = train_states if save_states else None,
        test_states  = test_states  if save_states else None,
        model_hypers = {**reservoir.hypers, **readout.hypers,
                        **({"selected_readout_params": selected} if selected else {})},
        profiling    = profiler.summary() if profiler.enabled else None,
    )


# ---------------------------------------------------------------------------
# Full CV
# ---------------------------------------------------------------------------

def cross_validate(
    reservoir: Node,
    readout: Node,
    X: np.ndarray,
    Y: np.ndarray,
    folds: int = 5,
    trial_name: str | None = None,
    save_threshold: float | None = None,
    save_states: bool = False,
    noise_rate: float = 0,
    noise_ratio: float = 0,
    fold_workers: int | None = None,
    fold_backend: Literal["sequential", "loky"] = "sequential",
    seed: int = config.SEED,
    scaler_type: str = "sequence_zscore",
    readout_aggregation: str = "final",
    readout_window: int = 1,
    primary_average: str = "weighted",
    profile: bool = False,
    return_artifacts: bool = False,
) -> dict[str, Any]:
    """Stratified k-fold cross-validation.

    Parameters
    ----------
    reservoir : Node
        Template reservoir. Every fold runs a copy of it: one reservoir per run, and the
        folds differ only in their data. (v0.1 meant to rebuild it per fold with a fold
        seed, but the rebuild always failed and fell back to the same copy.)
    readout : Node
        Template readout; every fold fits its own copy.
    X, Y : np.ndarray
        Full dataset (train + test concatenated).  Labels must be one-hot or
        class-index.  Augmentation is applied per-fold to the training split
        only.
    folds : int
        Number of CV folds.
    trial_name : str, optional
        When given, all fold artefacts are saved under
        ``results/runs/{trial_name}/`` **after all folds complete**.  If the
        mean F1 across folds is below ``save_threshold``, nothing is saved.
        This all-or-none behaviour avoids partial trial artefacts.
    save_threshold : float, optional
        Minimum mean F1 required to persist artefacts.  ``None`` means always
        save when ``trial_name`` is set.
    save_states : bool, default False
        Whether to include reservoir states in the saved artefacts.  See
        :class:`~utils.results.FoldArtifact` for when this is needed.
    fold_backend : {"sequential", "loky"}
        ``"sequential"`` runs folds in the current process (safe with
        ``numba parallel=True``).  ``"loky"`` spawns one process per fold via
        joblib (safe with OpenMP; requires ``joblib``).
    profile : bool
        Thread profiling through each fold; results are averaged across folds
        and returned under ``"profiling"`` in the output dict.

    Returns
    -------
    Dict[str, Any]
        Mean classification metrics across all folds, optionally with a
        ``"profiling"`` key when ``profile=True``.
    """
    if noise_rate or noise_ratio:
        logger.info(
            "CV will apply augmentation to each training fold only "
            "(noise_rate=%s noise_ratio=%s)",
            noise_rate,
            noise_ratio,
        )

    folds_list = _iter_folds(X, Y, folds=folds, seed=seed)

    if fold_backend not in ("sequential", "loky"):
        raise ValueError(f"Unknown fold_backend '{fold_backend}'")

    if fold_backend == "sequential":
        artifacts: list[FoldArtifact] = []
        for fold in folds_list:
            X_tr, Y_tr, X_va, Y_va = _prepare_fold_data(
                X, Y, fold,
                noise_rate=noise_rate,
                noise_ratio=noise_ratio,
                seed=seed,
                scaler_type=scaler_type,
            )
            artifacts.append(run_fold(
                reservoir, readout,
                X_tr, Y_tr, X_va, Y_va,
                fold_index=fold.fold_index,
                trial_name=trial_name or "",
                save_states=save_states,
                readout_aggregation=readout_aggregation,
                readout_window=readout_window,
                primary_average=primary_average,
                profile=profile,
            ))
    else:
        # loky: separate process per fold (avoids OpenMP fork issues)
        try:
            from joblib import Parallel, delayed
        except Exception as e:  # pragma: no cover
            raise RuntimeError("joblib is required for fold_backend='loky'.") from e

        n_jobs = int(fold_workers) if fold_workers else folds
        artifacts = list(Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(run_fold)(
                reservoir,
                readout,
                *_prepare_fold_data(
                    X, Y, fold,
                    noise_rate=noise_rate,
                    noise_ratio=noise_ratio,
                    seed=seed,
                    scaler_type=scaler_type,
                ),
                fold_index=fold.fold_index,
                trial_name=trial_name or "",
                save_states=save_states,
                readout_aggregation=readout_aggregation,
                readout_window=readout_window,
                primary_average=primary_average,
                profile=profile,
            )
            for fold in folds_list
        ))

    for artifact, fold in zip(artifacts, folds_list, strict=True):
        artifact.test_index = fold.val_idx

    # ---- Trial-level save decision (all-or-none) --------------------------
    # Evaluated after ALL folds complete so a partial trial is never written.
    if trial_name:
        mean_f1 = sum(fa.metrics.f1 for fa in artifacts) / len(artifacts)
        if save_threshold is None or mean_f1 >= save_threshold:
            for fa in artifacts:
                fa.save(save_states=save_states)
            logger.info(
                "Saved %d fold artefact(s) for trial '%s' (mean F1=%.4f).",
                len(artifacts), trial_name, mean_f1,
            )
        else:
            logger.info(
                "Trial '%s' not saved: mean F1=%.4f < threshold=%.4f.",
                trial_name, mean_f1, save_threshold,
            )

    return _return_cv_result(artifacts, profile=profile, return_artifacts=return_artifacts)


def _return_cv_result(
    artifacts, *, profile: bool, return_artifacts: bool
):
    """Helper: return averaged metrics, optionally with raw fold artifacts."""
    result = _average_fold_artifacts(artifacts, profile=profile)
    if return_artifacts:
        return result, artifacts
    return result


def fit_on_features(features: np.ndarray, Y: np.ndarray, readout: Node, fold: Fold, *,
                    save_states: bool = False, primary_average: str = "weighted",
                    labels: np.ndarray | None = None,
                    model_hypers: dict[str, Any] | None = None) -> FoldArtifact:
    """Fit a copy of ``readout`` on the fold's training rows and score its test rows."""
    readout = readout.copy()
    train_rows = [features[i][np.newaxis] for i in fold.train_idx]
    test_rows = [features[i][np.newaxis] for i in fold.val_idx]
    Y_train, Y_test = Y[fold.train_idx], Y[fold.val_idx]
    started = time.perf_counter()
    fit_readout(readout, train_rows, Y_train)
    Y_pred = _stack_runs(predict(readout, test_rows))
    runtime = round(time.perf_counter() - started, 4)
    selected = selected_readout_params(readout)
    return FoldArtifact(
        trial_name="",
        fold_index=fold.fold_index,
        Y_train=Y_train,
        Y_test=Y_test,
        Y_pred=Y_pred,
        metrics=evaluate_performance(Y_test, Y_pred, runtime,
                                     primary_average=primary_average, labels=labels),
        train_states=np.stack(train_rows) if save_states else None,
        test_states=np.stack(test_rows) if save_states else None,
        model_hypers={**(model_hypers or {}), **readout.hypers,
                      **({"selected_readout_params": selected} if selected else {})},
        test_index=fold.val_idx,
    )


def cross_validate_features(features: np.ndarray, Y: np.ndarray, readout: Node, *,
                            folds: int = 5, seed: int = config.SEED,
                            primary_average: str = "weighted", save_states: bool = False,
                            model_hypers: dict[str, Any] | None = None):
    """Cross-validate ``readout`` on features computed once per beat.

    ``features[i]`` belongs to beat ``i`` (:func:`training.simulate.simulate_beats`). The
    folds are those of :func:`cross_validate` for the same labels and seed, and every
    fold fits its own copy of the readout. Returns ``(result, artifacts)``, as
    ``cross_validate(..., return_artifacts=True)`` does.
    """
    artifacts = [
        fit_on_features(features, Y, readout, fold, save_states=save_states,
                        primary_average=primary_average, model_hypers=model_hypers)
        for fold in _iter_folds(features, Y, folds=folds, seed=seed)
    ]
    return _return_cv_result(artifacts, profile=False, return_artifacts=True)
# ---------------------------------------------------------------------------

def classify(
    reservoir: Node,
    readout: Node,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    folds: int | None = None,
    trial_name: str | None = None,
    save_threshold: float | None = None,
    save_states: bool = False,
    noise_rate: float = 0,
    noise_ratio: float = 0,
    readout_aggregation: str = "final",
    readout_window: int = 1,
    primary_average: str = "weighted",
) -> dict[str, Any]:
    """Train and evaluate a reservoir classifier.

    When ``folds > 1`` the full dataset (train + test) is pooled and passed
    to :func:`cross_validate`.  Otherwise a single train/test split is used.

    Parameters
    ----------
    reservoir, readout : Node
        As for :func:`cross_validate`.
    X_train, Y_train, X_test, Y_test : np.ndarray
        Dataset splits.
    folds : int, optional
        Number of CV folds.  If ``None`` or 1, no cross-validation is performed.
    trial_name : str, optional
        Saves artefacts under ``results/runs/{trial_name}/`` when provided.
        Threshold is evaluated per-trial (all-or-none in CV mode; single
        threshold check in single-split mode).
    save_threshold : float, optional
        Minimum F1 to trigger saving.  ``None`` = always save.
    save_states : bool, default False
        Include reservoir states in saved artefacts.
    noise_rate, noise_ratio : float
        Augmentation parameters (applied per-fold in CV mode).
    readout_aggregation : str, default "final"
        State aggregation strategy used when extracting one readout vector per
        instance in the single-split path.
    readout_window : int, default 1
        Final-window aggregation size used by :func:`training.pipeline.train`
        in the single-split path.

    Returns
    -------
    Dict[str, Any]
        Classification metrics.
    """
    logger.debug("----- Model Hyperparameters -----")
    for k, v in {**reservoir.hypers, **readout.hypers}.items():
        logger.debug("%s: %s", k, v)
    logger.debug("-------------------------------")

    if folds and folds > 1:
        X = np.concatenate((X_train, X_test), axis=0)
        Y = np.concatenate((Y_train, Y_test), axis=0)
        metrics = cross_validate(
            reservoir, readout, X, Y,
            folds=folds,
            trial_name=trial_name,
            save_threshold=save_threshold,
            save_states=save_states,
            noise_rate=noise_rate,
            noise_ratio=noise_ratio,
            readout_aggregation=readout_aggregation,
            readout_window=readout_window,
            fold_workers=0,
            fold_backend="sequential",
            primary_average=primary_average,
        )
    else:
        t0 = time.perf_counter()
        train_states = train(
            reservoir,
            X_train,
            readout_aggregation=readout_aggregation,
            readout_window=readout_window,
        )
        fit_readout(readout, train_states, Y_train)
        test_states = train(
            reservoir,
            X_test,
            readout_aggregation=readout_aggregation,
            readout_window=readout_window,
        )
        Y_pred_list = predict(readout, test_states)
        Y_pred = _stack_runs(Y_pred_list)
        runtime = round(time.perf_counter() - t0, 4)
        fold_metrics = evaluate_performance(
            Y_test, Y_pred, runtime, primary_average=primary_average
        )

        if trial_name and (save_threshold is None or fold_metrics.f1 >= save_threshold):
            FoldArtifact(
                trial_name   = trial_name,
                fold_index   = 0,
                Y_train      = Y_train,
                Y_test       = Y_test,
                Y_pred       = Y_pred,
                metrics      = fold_metrics,
                train_states = train_states if save_states else None,
                test_states  = test_states  if save_states else None,
                model_hypers = {**reservoir.hypers, **readout.hypers},
            ).save(save_states=save_states)

        metrics = fold_metrics.to_dict()

    logger.info(
        "Metrics: acc=%.4f f1=%.4f recall=%.4f precision=%.4f runtime=%.3fs",
        float(metrics.get("accuracy",  0.0)),
        float(metrics.get("f1",        0.0)),
        float(metrics.get("recall",    0.0)),
        float(metrics.get("precision", 0.0)),
        float(metrics.get("runtime",   0.0)),
    )
    logger.debug("Full metrics: %s", metrics)
    return metrics
