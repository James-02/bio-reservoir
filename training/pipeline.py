"""training.pipeline — per-instance reservoir training helpers.

This module contains the functions that operate on *individual ECG instances*:
driving the reservoir, aggregating the resulting state trajectory, fitting the
readout weight matrix, generating predictions, and stacking outputs.

These are pure functional helpers with no global state.  They are composed by
:mod:`training.cross_validate` to build the full CV pipeline.
"""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
from reservoirpy.node import Node
from sklearn.base import is_classifier

from training.profiling import Profiler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State aggregation
# ---------------------------------------------------------------------------

def _aggregate_states(states: np.ndarray, mode: str, x: np.ndarray) -> np.ndarray:
    """Aggregate a ``(T, state_dim)`` trajectory into a ``(1, state_dim)`` readout vector.

    Parameters
    ----------
    states : np.ndarray, shape (T, state_dim)
        Full reservoir state trajectory for one instance (one external timestep
        per row — NOT RK4 substeps).
    mode : str
        Aggregation strategy:

        ``"final"``
            Last timestep only.  Default; preserves pre-existing behaviour.
        ``"mean"``
            Mean over all T timesteps.  Not recommended when zero-padding is
            present; the settling tail dilutes class-discriminative states.
        ``"signal_mean"``
            Mean over non-zero-padded timesteps only.  The zero-padded tail is
            detected from the raw input ``x`` (absolute value > 1e-8).
        ``"first_N:K"``
            Mean over the first K timesteps, e.g. ``"first_N:120"``.
        ``"last_N:K"``
            Mean over the final K timesteps, e.g. ``"last_N:20"``.
            Equivalent to passing ``readout_window=K`` to :func:`train`.
    x : np.ndarray
        Raw input sequence for this instance; used only for ``"signal_mean"``
        mode.  Shape ``(T, 1)`` or ``(T,)``.
    """
    if mode == "final":
        return states[-1, np.newaxis]

    if mode == "mean":
        return states.mean(axis=0, keepdims=True)

    if mode.startswith("first_N:"):
        try:
            k = int(mode.split(":")[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(
                f"Invalid first_N mode '{mode}'; expected 'first_N:K' e.g. 'first_N:120'"
            ) from exc
        k = min(k, len(states))
        return states[:k].mean(axis=0, keepdims=True)

    if mode == "last_N":
        # Bare "last_N" without a window value — produced when readout_window=1
        # is sampled alongside readout_aggregation="last_N".  A window of 1 is
        # equivalent to "final"; fall back silently rather than crashing.
        import logging as _logging
        _logging.getLogger(__name__).debug(
            "readout_aggregation='last_N' with no window; falling back to 'final'."
        )
        return states[-1:]

    if mode.startswith("last_N:"):
        try:
            k = int(mode.split(":")[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(
                f"Invalid last_N mode '{mode}'; expected 'last_N:K' e.g. 'last_N:20'"
            ) from exc
        k = min(k, len(states))
        return states[-k:].mean(axis=0, keepdims=True)

    if mode == "signal_mean":
        # Detect the zero-padded tail: last timestep where abs(input) > threshold.
        # Zero-padding in the Kachuee ECG preprocessing is literal zeros; after
        # per-sequence z-scoring the padded region may shift, so we use the
        # *original* input x rather than the scaled version seen by the reservoir.
        signal = np.asarray(x).ravel()
        nonzero_mask = np.abs(signal) > 1e-8
        if nonzero_mask.any():
            last_signal_t = int(np.flatnonzero(nonzero_mask)[-1]) + 1
        else:
            last_signal_t = len(states)  # fallback: no padding detected
        last_signal_t = max(1, min(last_signal_t, len(states)))
        return states[:last_signal_t].mean(axis=0, keepdims=True)

    raise ValueError(
        f"Unknown readout_aggregation mode '{mode}'. "
        "Choose from: 'final', 'mean', 'signal_mean', 'first_N:K', 'last_N:K'."
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    reservoir: Node,
    X_train: np.ndarray,
    readout_aggregation: str = "final",
    readout_window: int = 1,
    profiler: Optional[Profiler] = None,
) -> List[np.ndarray]:
    """Drive the reservoir with each instance in ``X_train`` and return readout vectors.

    Parameters
    ----------
    reservoir : Node
        Initialised BioReservoir (or any ReservoirPy Node).
    X_train : np.ndarray, shape (N, T, input_dim) or list of (T, input_dim) arrays
        Training instances.  Each instance is run independently with
        ``reservoir.run(x, reset=True)``.
    readout_aggregation : str
        How to reduce the ``(T, state_dim)`` trajectory to a single readout
        vector.  Ignored when ``readout_window > 1``.
        See :func:`_aggregate_states` for available modes.
    readout_window : int
        If > 1, overrides ``readout_aggregation`` and uses the mean of the
        final ``readout_window`` external timesteps.  Operates over the full
        187-step ECG trajectory (not over RK4 substeps within one step).
        Default 1 = final timestep only.
    profiler : Profiler, optional
        When provided and ``profiler.enabled`` is ``True``, per-instance
        wall-time is recorded under ``"instance/reservoir_run"`` and
        ``"instance/state_aggregation"`` (calls accumulate, so
        ``mean_s ≈ per-instance cost``).

    Returns
    -------
    List[np.ndarray]
        One ``(1, state_dim)`` readout vector per instance.
    """
    mode = f"last_N:{readout_window}" if readout_window > 1 else readout_aggregation
    logger.info(
        "Training reservoir: units=%s instances=%s aggregation=%s",
        reservoir.units, len(X_train), mode,
    )

    if profiler is None or not profiler.enabled:
        return [
            _aggregate_states(reservoir.run(x, reset=True), mode, x)
            for x in X_train
        ]

    results: List[np.ndarray] = []
    for x in X_train:
        with profiler.section("instance/reservoir_run"):
            states = reservoir.run(x, reset=True)
        with profiler.section("instance/state_aggregation"):
            agg = _aggregate_states(states, mode, x)
        results.append(agg)
    return results


# ---------------------------------------------------------------------------
# Readout fit and predict
# ---------------------------------------------------------------------------

def _is_sklearn_classifier(readout: Node) -> bool:
    """Return *True* if *readout* wraps a scikit-learn classifier.

    ReservoirPy's :class:`ScikitLearnNode` stores the underlying sklearn
    estimator class as a hyper-parameter called ``model``.
    """
    model_cls = readout.hypers.get("model")
    if model_cls is None:
        return False
    try:
        return is_classifier(model_cls)
    except Exception:
        return False


def fit_readout(
    readout: Node,
    trained_states: List[np.ndarray],
    Y_train: np.ndarray,
) -> None:
    """Fit the readout using pre-computed reservoir states.

    For scikit-learn classifiers wrapped in a :class:`ScikitLearnNode`,
    one-hot labels are converted to integer class indices before fitting
    to avoid the reservoirpy multi-output bug (classifiers that advertise
    ``multioutput=True``, such as :class:`RandomForestClassifier`, would
    otherwise be trained as five independent binary classifiers instead of
    a single five-class classifier).

    Parameters
    ----------
    readout : Node
        Uninitialised Ridge (or compatible) ReservoirPy Node.
    trained_states : list of np.ndarray
        Output of :func:`train`.
    Y_train : np.ndarray
        One-hot or class-index labels matching ``trained_states``.
    """
    logger.info("Fitting readout: n_states=%s", len(trained_states))

    if _is_sklearn_classifier(readout):
        # --- bypass ScikitLearnNode entirely ------------------------------
        # Create a fresh sklearn instance from the class + hypers stored
        # in ScikitLearnNode's hypers dict.  This avoids triggering the
        # buggy init/backward path which crashes SVM / LR (single-sample
        # one-hot → "only one class" error) and produces wrong multi-output
        # behaviour for RF / DT / GB.
        from copy import deepcopy

        X = np.vstack([s.reshape(1, -1) for s in trained_states])
        Y = np.vstack([y.reshape(1, -1) for y in Y_train])
        n_classes = Y.shape[-1]
        y_int = np.argmax(Y, axis=-1) if n_classes > 1 else Y.ravel()

        model_cls = readout.hypers["model"]
        model_hypers = deepcopy(readout.hypers.get("model_hypers", {}))
        instance = model_cls(**model_hypers)
        instance.fit(X, y_int)

        # Store on the node for predict() to retrieve.
        readout._fitted_sklearn = instance
        readout._n_classes = int(n_classes)
    else:
        readout.fit(trained_states, Y_train)


def predict(readout: Node, states: List[np.ndarray]) -> List[np.ndarray]:
    """Run the fitted readout on a list of state vectors.

    For scikit-learn classifiers, predictions are converted from integer
    labels back to one-hot to match the downstream ``evaluate_performance``
    expectation.

    Parameters
    ----------
    readout : Node
        Fitted readout Node.
    states : list of np.ndarray
        Output of :func:`train` on the validation/test set.

    Returns
    -------
    List[np.ndarray]
        One ``(1, C)`` prediction array per instance.
    """
    logger.info("Predicting: n_instances=%s", len(states))

    if _is_sklearn_classifier(readout):
        n_classes = getattr(readout, "_n_classes", None)
        instance = getattr(readout, "_fitted_sklearn", None)
        if instance is None:
            # Fallback: node was fitted before this fix (e.g. from NPZ).
            instance = readout.params["instances"]
            if isinstance(instance, list):
                instance = instance[0]
        X = np.vstack([s.reshape(1, -1) for s in states])
        y_int = instance.predict(X)
        if n_classes is not None and y_int.ndim == 1:
            Y_pred = np.eye(n_classes, dtype=np.float64)[y_int.astype(int)]
        else:
            Y_pred = y_int
        return [Y_pred[i : i + 1] for i in range(len(Y_pred))]

    return [readout.run(state, reset=True) for state in states]


def _stack_runs(outputs: List[np.ndarray]) -> np.ndarray:
    """Stack a list of ReservoirPy readout outputs into ``(N, 1, C)``.

    Normalises the common shapes returned by ``readout.run(...)``:
    ``(1, C)``, ``(1, 1, C)``, ``(N, 1, C)``.
    """
    if len(outputs) == 0:
        return np.empty((0, 1, 0))
    arr = np.asarray(outputs)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0, :, :]
    if arr.ndim == 3:
        return arr
    if arr.ndim == 2:
        return arr.reshape(arr.shape[0], 1, arr.shape[1])
    raise ValueError(f"Unexpected output stack shape: {arr.shape}")
