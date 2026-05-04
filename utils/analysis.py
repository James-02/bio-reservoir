"""utils.analysis — performance metrics and aggregation helpers.

Provides :class:`Metrics`, the canonical typed container for classification
run results, and helper functions used throughout the project for evaluating
and averaging fold metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, fields as _dc_fields
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

import logging

logger = logging.getLogger(__name__)


_SUPPORTED_AVERAGES = ("weighted", "macro")


def normalize_average_name(average: str) -> str:
    """Return a validated average name used for aggregate classification metrics."""
    average = str(average).strip().lower()
    if average not in _SUPPORTED_AVERAGES:
        raise ValueError(
            f"Unsupported average {average!r}. Supported: {_SUPPORTED_AVERAGES}."
        )
    return average


def _nanmean_attr(metrics_list: List["Metrics"], attr_name: str) -> float:
    values = np.array([getattr(m, attr_name, float("nan")) for m in metrics_list], dtype=float)
    if values.size == 0 or np.isnan(values).all():
        return float("nan")
    return float(np.nanmean(values))


# ---------------------------------------------------------------------------
# Metrics dataclass
# ---------------------------------------------------------------------------

@dataclass
class Metrics:
    """Typed container for a single classification run's performance metrics.

    All scalar fields are plain Python ``float`` values.  The two structured
    fields carry the richer diagnostics needed for plotting and per-class
    analysis.

    Attributes
    ----------
    f1 : float
        Weighted F1 score across all classes.
    accuracy : float
        Overall classification accuracy (fraction of correct predictions).
    precision : float
        Weighted precision.
    recall : float
        Weighted recall.
    mcc : float
        Matthews correlation coefficient (−1 to +1).
    kappa : float
        Cohen's kappa (inter-rater agreement).
    runtime : float
        Wall-clock time for the run in seconds.
    confusion_matrix : np.ndarray, shape (n_classes, n_classes)
        ``confusion_matrix[i, j]`` is the number of samples with true class
        ``i`` predicted as class ``j``.
    class_metrics : dict
        Per-class precision / recall / f1 / support from
        :func:`sklearn.metrics.classification_report` (``output_dict=True``).
        Keyed by string class index, e.g. ``"0"``, ``"1"``.

    Notes
    -----
    Supports dict-style access for backward compatibility with code that
    previously received a plain ``dict`` from :func:`evaluate_performance`::

        m = evaluate_performance(Y_true, Y_pred)
        m.f1               # typed attribute access
        m["f1"]            # dict-style key access
        m.get("f1", 0.0)   # with default
        "accuracy" in m    # membership test
        list(m.items())    # iterate key/value pairs
        m.to_dict()        # convert to plain dict for serialisation
    """

    f1: float
    accuracy: float
    precision: float
    recall: float
    mcc: float
    kappa: float
    runtime: float
    confusion_matrix: np.ndarray
    class_metrics: Dict[str, Any]
    primary_average: str = "weighted"
    f1_macro: float = float("nan")
    f1_weighted: float = float("nan")
    precision_macro: float = float("nan")
    precision_weighted: float = float("nan")
    recall_macro: float = float("nan")
    recall_weighted: float = float("nan")

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain ``dict`` (e.g. for NPZ storage or Optuna attrs)."""
        return {f.name: getattr(self, f.name) for f in _dc_fields(self)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Metrics":
        """Reconstruct from a dict, silently ignoring unknown keys.

        Provides backward compatibility with older NPZ artefacts that stored
        ``mse``/``rmse``/``r2_score`` instead of ``mcc``/``kappa``.
        """
        known = {f.name for f in _dc_fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known}
        # Default mcc/kappa to NaN when loading legacy artefacts.
        filtered.setdefault("mcc", float("nan"))
        filtered.setdefault("kappa", float("nan"))
        filtered.setdefault("primary_average", "weighted")
        filtered.setdefault("f1_macro", float("nan"))
        filtered.setdefault("f1_weighted", filtered.get("f1", float("nan")))
        filtered.setdefault("precision_macro", float("nan"))
        filtered.setdefault("precision_weighted", filtered.get("precision", float("nan")))
        filtered.setdefault("recall_macro", float("nan"))
        filtered.setdefault("recall_weighted", filtered.get("recall", float("nan")))
        return cls(**filtered)

    # ------------------------------------------------------------------
    # Dict-style access (backward compatibility)
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and hasattr(self, key)

    def keys(self) -> Iterator[str]:
        return (f.name for f in _dc_fields(self))

    def values(self) -> Iterator[Any]:
        return (getattr(self, f.name) for f in _dc_fields(self))

    def items(self) -> Iterator[Tuple[str, Any]]:
        return ((f.name, getattr(self, f.name)) for f in _dc_fields(self))

    def __repr__(self) -> str:
        return (
            f"Metrics(f1={self.f1:.4f}, average={self.primary_average}, accuracy={self.accuracy:.4f}, "
            f"precision={self.precision:.4f}, recall={self.recall:.4f}, "
            f"runtime={self.runtime:.2f}s)"
        )

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def log_metrics(metrics: Metrics) -> None:
    """Log the scalar fields of a :class:`Metrics` object at INFO level."""
    logger.info("----- Classification Report -----")
    logger.info("primary_average: %s", metrics.primary_average)
    for key in ("f1", "accuracy", "precision", "recall", "mcc", "kappa", "runtime"):
        logger.info("%s: %.3f", key, metrics[key])
    logger.info("f1_macro: %.3f", metrics.f1_macro)
    logger.info("f1_weighted: %.3f", metrics.f1_weighted)
    logger.info("---------------------------------")


def log_params(params: Dict[str, Any], title: str = "Hyperparameters") -> None:
    """Log hyperparameter key/value pairs at DEBUG level."""
    logger.debug("----- %s -----", title)
    for key, value in params.items():
        logger.debug("%s: %s", key, value)
    logger.debug("--------------------------------")


def evaluate_performance(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    time: float = 0,
    primary_average: str = "weighted",
) -> Metrics:
    """Compute classification metrics from true and predicted label arrays.

    Parameters
    ----------
    Y_true : np.ndarray
        True labels in any shape accepted by argmax reduction, e.g. one-hot
        ``(N, 1, C)`` or class-index ``(N,)``.
    Y_pred : np.ndarray
        Predicted scores / one-hot outputs.  Argmax is taken to obtain the
        predicted class index.
    time : float
        Wall-clock runtime in seconds stored directly in the returned
        :class:`Metrics` object; not computed here.

    Returns
    -------
    Metrics
    """
    primary_average = normalize_average_name(primary_average)
    logger.info("Calculating model performance metrics (primary_average=%s)", primary_average)
    yt = np.array([np.argmax(y_t) for y_t in Y_true])
    yp = np.array([np.argmax(y_p) for y_p in Y_pred])

    f1_macro = float(f1_score(yt, yp, average="macro"))
    f1_weighted = float(f1_score(yt, yp, average="weighted"))
    precision_macro = float(precision_score(yt, yp, average="macro"))
    precision_weighted = float(precision_score(yt, yp, average="weighted"))
    recall_macro = float(recall_score(yt, yp, average="macro"))
    recall_weighted = float(recall_score(yt, yp, average="weighted"))

    if primary_average == "macro":
        f1 = f1_macro
        precision = precision_macro
        recall = recall_macro
    else:
        f1 = f1_weighted
        precision = precision_weighted
        recall = recall_weighted

    return Metrics(
        runtime          = float(time),
        f1               = f1,
        accuracy         = float(accuracy_score(yt, yp)),
        precision        = precision,
        recall           = recall,
        mcc              = float(matthews_corrcoef(yt, yp)),
        kappa            = float(cohen_kappa_score(yt, yp)),
        confusion_matrix = confusion_matrix(yt, yp),
        class_metrics    = classification_report(yt, yp, output_dict=True),
        primary_average  = primary_average,
        f1_macro         = f1_macro,
        f1_weighted      = f1_weighted,
        precision_macro  = precision_macro,
        precision_weighted = precision_weighted,
        recall_macro     = recall_macro,
        recall_weighted  = recall_weighted,
    )

def _is_class_key(key: str) -> bool:
    """Return True if *key* looks like a per-class index (e.g. ``"0"``, ``"3"``)."""
    return isinstance(key, str) and key.isdigit()


def _extract_class_keys_and_metrics(
    dicts_list: List[Dict],
) -> Tuple[List[str], List[str]]:
    """Discover per-class keys and metric names from a list of class_metrics dicts.

    Returns (sorted class keys, metric names) or ([], []) when the input is
    empty or contains no per-class sub-dicts.
    """
    keys = sorted({str(k) for d in dicts_list for k in d if _is_class_key(str(k))})
    if not keys:
        return [], []

    metric_keys: List[str] = []
    for d in dicts_list:
        for k in keys:
            v = d.get(k)
            if isinstance(v, dict) and v:
                metric_keys = list(v.keys())
                break
        if metric_keys:
            break
    return keys, metric_keys


def compute_mean_dicts(dicts_list: List[Dict]) -> Dict[str, Dict[str, float]]:
    """
    Compute the mean of dictionaries for integer based keys, representing each class in the dataset.

    Args:
        dicts_list (list): List of dictionaries to compute mean from.

    Returns:
        dict: Mean values for each key across dictionaries.
    """
    keys, metric_keys = _extract_class_keys_and_metrics(dicts_list)
    if not keys:
        return {}

    count = len(dicts_list)
    sum_values: Dict[str, Dict[str, float]] = {
        key: {mk: 0.0 for mk in metric_keys} for key in keys
    }

    # compute the average of each metric across each class
    for d in dicts_list:
        for key, value in d.items():
            if str(key) in sum_values and isinstance(value, dict):
                for inner_key, inner_value in value.items():
                    if inner_key in sum_values[str(key)]:
                        sum_values[str(key)][inner_key] += inner_value
    return {key: {mk: sv / count for mk, sv in mvs.items()} for key, mvs in sum_values.items()}


def compute_class_dicts(
    dicts_list: List[Dict],
) -> Dict[str, Dict[str, float]]:
    """Compute per-class metric means **and** standard deviations across folds.

    Returns a dict keyed by class index (``"0"``, ``"1"``, …) where each
    value is a dict with keys like ``precision_mean``, ``precision_std``,
    ``recall_mean``, ``recall_std``, ``f1_mean``, ``f1_std``,
    ``support_mean``.
    """
    keys, metric_keys = _extract_class_keys_and_metrics(dicts_list)
    if not keys:
        return {}

    result: Dict[str, Dict[str, float]] = {}
    for cls in keys:
        cls_result: Dict[str, float] = {}
        for mk in metric_keys:
            vals = [
                d[cls][mk]
                for d in dicts_list
                if cls in d and isinstance(d[cls], dict) and mk in d[cls]
            ]
            if vals:
                arr = np.array(vals, dtype=float)
                cls_result[f"{mk}_mean"] = float(arr.mean())
                cls_result[f"{mk}_std"] = float(arr.std(ddof=0))
        result[cls] = cls_result
    return result

def compute_mean_metrics(metrics_list: List[Metrics]) -> Metrics:
    """Average a list of :class:`Metrics` instances across folds.

    Scalar fields are arithmetically averaged.  ``confusion_matrix`` is
    element-wise **summed** (not divided) so the aggregate represents all
    test-set predictions combined.  ``class_metrics`` is averaged via
    :func:`compute_mean_dicts`.

    Parameters
    ----------
    metrics_list : list of Metrics
        One :class:`Metrics` per fold.

    Returns
    -------
    Metrics
    """
    n = len(metrics_list)
    primary_average = normalize_average_name(metrics_list[0].primary_average) if metrics_list else "weighted"
    if metrics_list and any(normalize_average_name(m.primary_average) != primary_average for m in metrics_list):
        logger.warning("Mixed primary_average values detected across folds; using %s from the first fold.", primary_average)

    return Metrics(
        f1               = sum(m.f1        for m in metrics_list) / n,
        accuracy         = sum(m.accuracy  for m in metrics_list) / n,
        precision        = sum(m.precision for m in metrics_list) / n,
        recall           = sum(m.recall    for m in metrics_list) / n,
        mcc              = sum(m.mcc       for m in metrics_list) / n,
        kappa            = sum(m.kappa     for m in metrics_list) / n,
        runtime          = sum(m.runtime   for m in metrics_list) / n,
        confusion_matrix = sum(m.confusion_matrix for m in metrics_list),
        class_metrics    = compute_mean_dicts([m.class_metrics for m in metrics_list]),
        primary_average  = primary_average,
        f1_macro         = _nanmean_attr(metrics_list, "f1_macro"),
        f1_weighted      = _nanmean_attr(metrics_list, "f1_weighted"),
        precision_macro  = _nanmean_attr(metrics_list, "precision_macro"),
        precision_weighted = _nanmean_attr(metrics_list, "precision_weighted"),
        recall_macro     = _nanmean_attr(metrics_list, "recall_macro"),
        recall_weighted  = _nanmean_attr(metrics_list, "recall_weighted"),
    )

def count_labels(Y: np.ndarray | List[np.ndarray]) -> Dict[int, int]:
    """
    Count the occurrences of each class label, converts one-hot encoding if given.

    Args:
        Y: Array or list of target arrays (one-hot or class index).

    Returns:
        dict: Mapping from class label to occurrence count.
    """
    Y = np.array([targets[0] for targets in Y])
    if len(Y.shape) != 1:
        Y = np.argmax(Y, axis=1)
    return {int(label): int(count) for label, count in zip(*np.unique(Y, return_counts=True))}

def measure_dataset_deviation(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Measure the mean and standard deviation of features in a dataset.

    Args:
        X: Input data of shape ``(N, ...)``.

    Returns:
        Tuple of (mean, std) arrays along axis 0.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std

def measure_class_deviation(
    X: List[np.ndarray],
    Y: List[np.ndarray],
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Measure the mean and standard deviation of features for each class in a dataset.

    Args:
        X: List of feature arrays.
        Y: List of target arrays.

    Returns:
        Tuple of (class_means, class_stds) dicts keyed by class label.
    """
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    class_means = {}
    class_stds = {}

    for label in np.unique(Y):
        indices = np.where(Y == label)[0]
        class_means[label] = round(np.mean(X[indices], axis=0).tolist()[0], 4)
        class_stds[label] = round(np.std(X[indices], axis=0).tolist()[0], 4)

    return class_means, class_stds
