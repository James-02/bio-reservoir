"""Linear surrogate of a reservoir: the best linear imitation of its features, fitted in each
fold.

The control asks whether the readout needs the reservoir's nonlinear dynamics, or whether a
linear filter of the beat would do as well. For each outer fold of a saved run
(``reservoir.states_from``), using its training beats only:

1. a ridge regression (:class:`sklearn.linear_model.RidgeCV`, input and target standardised,
   strength chosen by leave-one-out among :data:`ALPHAS`) maps the scaled beat to the run's
   observed features;
2. its predictions replace the features of the training and test beats, and the readout is
   fitted and scored on them as on the run's own features.

The test beats' features enter only the reported fit quality (R^2 on the test beats), never
the fit. With ``features.measurement_noise`` the surrogate's features are measured like the
reservoir's (:func:`training.observation.measure`): with the same run seed, a beat gets the
same noise draws as in a run of the reservoir. Select it in a run config with

.. code-block:: yaml

    reservoir: {states_from: <run with outputs.save_states>, surrogate: linear}
"""

from __future__ import annotations

import dataclasses
import json
import logging
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)

SURROGATES = ("linear",)

# Ridge strengths of the surrogate regression: 10^-2 to 10^4, two per decade.
ALPHAS = np.logspace(-2, 4, 13)


def check_surrogate(value: Any) -> str:
    """The canonical name of a ``reservoir.surrogate`` setting."""
    kind = str(value).strip().lower()
    if kind not in SURROGATES:
        raise ValueError(f"Unknown reservoir.surrogate {value!r}; choose from "
                         f"{', '.join(SURROGATES)}.")
    return kind


def fit_surrogate(beats: np.ndarray, states: np.ndarray):
    """Fit the linear map from ``beats`` (n, T) to ``states`` (n, F); return the fitted parts.

    Inputs and targets are standardised on these rows, and one ridge strength, chosen by
    leave-one-out among :data:`ALPHAS`, serves all targets.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    beat_scaler = StandardScaler().fit(beats)
    state_scaler = StandardScaler().fit(states)
    ridge = RidgeCV(alphas=ALPHAS).fit(beat_scaler.transform(beats),
                                       state_scaler.transform(states))
    return beat_scaler, state_scaler, ridge


def predict_surrogate(fitted, beats: np.ndarray) -> np.ndarray:
    """The surrogate's features of ``beats`` (n, T), in the units of the fitted states."""
    beat_scaler, state_scaler, ridge = fitted
    return state_scaler.inverse_transform(ridge.predict(beat_scaler.transform(beats)))


def r2_scores(states: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Coefficient of determination of each feature (column)."""
    total = np.maximum(((states - states.mean(axis=0)) ** 2).sum(axis=0), 1e-30)
    return 1.0 - ((states - predicted) ** 2).sum(axis=0) / total


def linear_surrogate(train_beats: np.ndarray, train_states: np.ndarray,
                     test_beats: np.ndarray, test_states: np.ndarray | None = None
                     ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """The surrogate features of a fold's training and test beats, fitted on the training rows.

    ``test_states`` is used only to report R^2 on the test beats. Returns
    ``(train_features, test_features, info)``.
    """
    fitted = fit_surrogate(train_beats, train_states)
    train = predict_surrogate(fitted, train_beats)
    test = predict_surrogate(fitted, test_beats)
    info: dict[str, Any] = {"alpha": float(fitted[2].alpha_)}
    if test_states is not None:
        scores = r2_scores(test_states, test)
        info.update(r2_median=float(np.median(scores)), r2_mean=float(np.mean(scores)))
    return train, test, info


def source_beats(states_from: str) -> tuple[np.ndarray, np.ndarray, list]:
    """The scaled beats, labels and folds of the run whose features a surrogate imitates.

    The pool and folds are rebuilt from the run's config (as :func:`training.run.load_task`
    and :func:`training.cross_validate.cross_validate_features` built them) and checked
    against the run's manifest.
    """
    from training import run as run_module
    from training.compare import run_path
    from training.cross_validate import _iter_folds
    from training.simulate import PER_BEAT_SCALERS
    from utils.preprocessing import standardize_data

    path = run_path(states_from)
    source = yaml.safe_load((path / "config.yaml").read_text())
    if "states_from" in source["reservoir"] or "test" in source:
        raise ValueError(f"A surrogate imitates the features of a cross-validated run that "
                         f"simulated its reservoir; {states_from} is not one.")
    task, cv = source["task"], source["cv"]
    if cv.get("noise_rate") or cv.get("noise_ratio"):
        raise ValueError("A surrogate needs a run without augmentation (cv.noise_rate, "
                         "cv.noise_ratio): augmented beats are not in the pool.")
    if task["scaler"] not in PER_BEAT_SCALERS:
        raise ValueError(f"A surrogate needs beats scaled on their own "
                         f"({', '.join(PER_BEAT_SCALERS)}); {states_from} used task.scaler "
                         f"{task['scaler']!r}.")
    X, Y, _ = run_module.load_task(task)
    manifest = path / "manifest.json"
    expected = json.loads(manifest.read_text()).get("data_pool_sha256") \
        if manifest.is_file() else None
    if expected and run_module._pool_sha256(X, Y) != expected:
        raise ValueError(f"The pool of {states_from} cannot be rebuilt: its checksum differs.")
    beats, _ = standardize_data(X, X[:1], scaler_type=str(task["scaler"]))
    beats = np.asarray(beats, dtype=float).reshape(len(beats), -1)
    folds = _iter_folds(X, Y, folds=int(cv["folds"]), seed=int(cv["seed"]))
    return beats, Y, folds


def _classes(Y: np.ndarray) -> np.ndarray:
    return np.argmax(np.asarray(Y).reshape(len(Y), -1), axis=1)


def _measured(rows: np.ndarray, noise: float, seed: int, beats: np.ndarray) -> np.ndarray:
    """``rows`` of the pool's ``beats`` measured as :func:`training.observation.measure` does
    for the whole pool."""
    from training.observation import measure

    if not noise:
        return rows
    return np.vstack([measure(rows[i:i + 1], noise, seed, first_index=int(beat))
                      for i, beat in enumerate(beats)])


def surrogate_folds(cfg: dict[str, Any], source_folds: list) -> list:
    """Copies of a saved run's folds whose states are the linear surrogate's features."""
    beats, Y, folds = source_beats(cfg["reservoir"]["states_from"])
    noise, seed = float(cfg["features"]["measurement_noise"]), int(cfg["seed"])
    out = []
    for source in source_folds:
        fold = folds[source.fold_index]
        train, test = fold.train_idx, fold.val_idx
        if (len(source.train_states) != len(train) or len(source.test_states) != len(test)
                or (source.test_index is not None
                    and not np.array_equal(source.test_index, test))
                or not np.array_equal(_classes(source.Y_train), _classes(Y[train]))
                or not np.array_equal(_classes(source.Y_test), _classes(Y[test]))):
            raise ValueError(f"Fold {source.fold_index} of {cfg['reservoir']['states_from']} "
                             "does not match the fold rebuilt from its config.")
        train_states = source.train_states.reshape(len(train), -1)
        test_states = source.test_states.reshape(len(test), -1)
        train_features, test_features, info = linear_surrogate(
            beats[train], train_states, beats[test], test_states)
        info["measurement_noise"] = noise
        logger.info("Linear surrogate, fold %d: alpha %.3g, R2 on the test beats median %.4f, "
                    "mean %.4f", source.fold_index, info["alpha"], info["r2_median"],
                    info["r2_mean"])
        out.append(dataclasses.replace(
            source,
            train_states=_measured(train_features, noise, seed, train)[:, np.newaxis],
            test_states=_measured(test_features, noise, seed, test)[:, np.newaxis],
            model_hypers={**(source.model_hypers or {}), "surrogate": {"kind": "linear",
                                                                       **info}}))
    return out
