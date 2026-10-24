"""Run a classification task from a config file.

    python -m training.run run CONFIG.yaml [--processes N] [--out DIR] [--allow-dirty]
                                           [--final-test]
    python -m training.run resolve CONFIG.yaml
    python -m training.run compare RUN_A RUN_B [--out DIR] [--margin M] [--ceiling RUN_C]
    python -m training.run study CONFIG.yaml [--processes N] [--trials N]
    python -m training.run study-best CONFIG.yaml [--out BEST.yaml]

A config names the task (labels, pool size, balancing, scaling), the reservoir (explicit
parameters or an Optuna study trial), the state features, the readout (explicit or a
readout-study trial) and the cross-validation; see ``experiments/benchmarks/`` for
examples. ``resolve`` turns study references into explicit values, and a run records
the resolved config, so it can be repeated without the study databases.

A run writes one folder, ``results/runs/<name>-<UTC time>/``: ``config.yaml``,
``metrics.json`` (per fold, mean and sample SD, pooled), ``fold-<k>.npz`` (labels,
predictions and, with ``outputs.save_states``, the state features), ``figures/``,
``run.log`` and ``manifest.json``, and appends a line to the run registry
(``utils.provenance``). With ``reservoir.states_from`` the readout is refitted on the
saved states of another run instead of simulating the reservoir again.

By default every beat is simulated once and the folds share its features
(``cv.simulate: once``, :mod:`training.simulate`), with ``--processes`` worker processes
simulating beats; ``cv.simulate: per_fold`` simulates the beats of every fold again, with
``--processes`` folds in parallel (the ``v0.1`` code path).

The ``features`` section also sets what a measurement sees (:mod:`training.observation`):
the observed variables (``variables: [I]`` for the LuxI reporter alone), K images over the
beat (``aggregation: images:K``), multiplicative measurement noise and the calibration
error of the test fold's colonies.

A config with a ``study`` section defines an Optuna search over its own values:
``study`` runs the trials through this runner's pipeline and ``study-best`` writes the best
trial as a plain config (:mod:`training.study`).

With ``test: {split: <name>}`` the run fits the readout on the whole pool and evaluates it
on a held-out split instead of cross-validating (``utils.heldout``: ``validation-1``,
``validation-2`` or ``test``); ``fold-0.npz`` then holds the split's predictions and
``metrics.json`` a bootstrap 95% CI of its macro F1. The ``test`` split is the final test
set and needs ``--final-test``.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import logging
import resource
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from utils import config as settings
from utils import provenance

logger = logging.getLogger(__name__)

DEFAULTS: dict[str, dict[str, Any]] = {
    "task": {"labels": "five-class", "rows": 4015, "balance_classes": True,
             "max_per_class": None, "cap_includes_test": True, "scaler": "sequence_zscore",
             "data_file": None},
    "features": {"aggregation": "final", "window": 1, "variables": None,
                 "measurement_noise": 0.0, "calibration_error": 0.0},
    "cv": {"folds": 5, "primary_average": "macro", "noise_rate": 0.0, "noise_ratio": 0.0,
           "profile": False, "simulate": "once"},
    "outputs": {"save_states": False},
}

# Parameters read by optimization.reservoirs.build_reservoir, per reservoir type.
RESERVOIR_KEYS = {
    "bioreservoir": (
        "units", "warmup", "rk4_substeps", "input_scaling",
        "input_connectivity", "input_bias", "bias_scaling", "rc_scaling", "dde_scaling",
        "cell_coupling", "noise_in", "noise_rc", "output_variables",
        "param_heterogeneity_cv", "param_heterogeneity_seed", "heterogeneity_mean_preserving",
        "topology_type",
        "interaction_diameters", "fade_alpha", "sparsity", "signed", "p_excite",
        "rc_connectivity", "geometry", "couple_vars", "delay_scheme", "normalize_after_masks",
        "coupling", "diffusion_rate", "input_encoding", "input_range", "input_sign",
        "heterogeneity_density_cv", "heterogeneity_dilution_cv", "input_levels", "reset_noise",
    ),
    "esn": (
        "units", "esn_leak_rate", "esn_spectral_radius", "esn_activation", "input_scaling",
        "input_connectivity", "input_bias", "bias_scaling", "noise_in", "noise_rc",
        "rc_connectivity",
    ),
    # Digital controls (optimization.controls)
    "random_features": ("units", "length", "activation", "input_scaling", "bias_scaling"),
    "leaky_bank": ("units", "tau_min", "tau_max", "band_pass", "expansion", "input_range"),
}

SCALAR_METRICS = ("f1", "f1_macro", "f1_weighted", "accuracy", "precision", "recall",
                  "mcc", "kappa")

# Ridge strengths a RidgeCV readout chooses from: 10^-4 to 10^6, two per decade.
RIDGE_ALPHAS = tuple(float(a) for a in np.logspace(-4, 6, 21))

# Named settings; explicit config values win. ``v0.1`` reproduces the v0.1 code
# path. ``qs_diffusion`` is the structure of the chemically realisable model: colonies
# exchange AHL through symmetric distance-based conductances (so AHL is conserved), and the
# input is a non-negative inducer level delivered with non-negative weights.
PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "v0.1": {
        "task": {"cap_includes_test": False},
        "reservoir": {"geometry": "v0.1", "delay_scheme": "frozen",
                      "normalize_after_masks": False,
                      "heterogeneity_mean_preserving": False},
        "readout": {"scale": False},
        "cv": {"simulate": "per_fold"},
    },
    "qs_diffusion": {
        "reservoir": {"coupling": "diffusion", "topology_type": "distance", "signed": False,
                      "p_excite": 1.0, "sparsity": 1.0, "input_encoding": "nonnegative",
                      "input_sign": "positive"},
    },
}


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict[str, Any]:
    """A config file. With ``base: <file>`` (relative to the config) it is the base config
    with this file's settings merged over it: sections merge key by key, other values replace."""
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if "base" in cfg:
        cfg = _merged(load_config(path.parent / cfg.pop("base")), cfg)
    return cfg


def _merged(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        both = isinstance(value, dict) and isinstance(out.get(key), dict)
        out[key] = _merged(out[key], value) if both else value
    return out


def _plain(value: Any) -> Any:
    """JSON-compatible copy (tuples become lists, numpy scalars become numbers)."""
    return json.loads(json.dumps(value, default=lambda v: v.item() if hasattr(v, "item")
                                 else str(v)))


def dump_config(cfg: dict[str, Any]) -> str:
    return yaml.safe_dump(_plain(cfg), sort_keys=False, default_flow_style=False)


def classifier_class(name: str):
    """The scikit-learn class behind a readout name (the Phase 4 readout-study names,
    RidgeCV: ridge with its strength chosen by leave-one-out on the training data, and
    HistGB: histogram-based gradient boosting)."""
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import (
        LogisticRegression,
        Perceptron,
        RidgeClassifier,
        RidgeClassifierCV,
    )
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    table = {"Ridge": RidgeClassifier, "RidgeCV": RidgeClassifierCV, "LR": LogisticRegression,
             "SVM": SVC, "KNN": KNeighborsClassifier, "RF": RandomForestClassifier,
             "DT": DecisionTreeClassifier, "GB": GradientBoostingClassifier,
             "HistGB": HistGradientBoostingClassifier,
             "MLP": MLPClassifier, "Bayes": GaussianNB, "Perceptron": Perceptron}
    if name not in table:
        raise ValueError(f"Unknown readout classifier {name!r}; choose from {sorted(table)}.")
    return table[name]


def _load_trial(study: str, trial: Any, archive: str | None, storage: str | None = None):
    import optuna

    if storage is None:
        path = settings.optuna_db(study, archive)
        if not path.is_file():
            raise FileNotFoundError(f"No Optuna database for study {study!r}: {path}")
        storage = f"sqlite:///{path}"
    loaded = optuna.load_study(study_name=study, storage=storage)
    if trial in (None, "best"):
        return loaded.best_trial
    matches = [t for t in loaded.trials if t.number == int(trial)]
    if not matches:
        raise ValueError(f"Study {study!r} has no trial {trial}.")
    return matches[0]


def study_params(study: str, trial: Any = "best", archive: str | None = None,
                 storage: str | None = None) -> dict[str, Any]:
    """The parameters a study trial ran with (as rerun-best), with its pool size and
    labels."""
    from optimization.studies import get_study
    from optimization.training import _resolve_params_from_trial

    study_config = get_study(study)
    frozen = _load_trial(study, trial, archive, storage)
    params = _resolve_params_from_trial(frozen, study_config)
    attrs = frozen.user_attrs
    params["_trial"] = frozen.number
    params["_instances"] = attrs.get("dataset_instances",
                                     study_config.get("instances", DEFAULTS["task"]["rows"]))
    params["_binary"] = bool(attrs.get("dataset_binary", study_config.get("binary", False)))
    return params


def _from_study(params: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The task, feature and CV settings that a study trial implies."""
    aggregation = str(params.get("readout_aggregation", "final"))
    return {
        "task": {"labels": "binary" if params["_binary"] else "five-class",
                 "rows": int(params["_instances"]),
                 "balance_classes": bool(params.get("balance_classes", True)),
                 "max_per_class": params.get("max_per_class"),
                 "scaler": str(params.get("scaler_type", "sequence_zscore"))},
        "features": {"aggregation": aggregation,
                     "window": int(params.get("readout_window", 1))
                     if aggregation == "last_N" else 1},
        "cv": {"primary_average": str(params.get("objective_average", "macro")),
               "noise_rate": float(params.get("noise_rate", 0.0)),
               "noise_ratio": float(params.get("noise_ratio", 0.0)),
               # study trials simulated the beats of every fold
               "simulate": "per_fold"},
    }


def resolve(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the explicit config: defaults, then study values, then the config's values."""
    cfg = copy.deepcopy(cfg)
    if not cfg.get("name"):
        raise ValueError("A run config needs a name.")
    seed = int(cfg.get("seed", settings.SEED))
    preset_name = cfg.get("preset")
    if preset_name is not None and preset_name not in PRESETS:
        raise ValueError(f"Unknown preset {preset_name!r}; choose from {sorted(PRESETS)}.")
    preset = PRESETS.get(preset_name, {})
    sections = {key: {**value, **preset.get(key, {})} for key, value in DEFAULTS.items()}

    reservoir = dict(cfg.get("reservoir") or {})
    if "states_from" in reservoir:
        resolved_reservoir = {"states_from": str(reservoir["states_from"])}
    elif reservoir.get("type") == "none":
        other = sorted(set(reservoir) - {"type"})
        if other:
            raise ValueError("reservoir: {type: none} classifies the beat itself; reservoir "
                             f"settings do not apply ({', '.join(other)}).")
        resolved_reservoir = {"type": "none"}
    else:
        params: dict[str, Any] = dict(preset.get("reservoir", {}))
        source = None
        if reservoir.get("study"):
            found = study_params(reservoir["study"], reservoir.get("trial", "best"),
                                 reservoir.get("archive"), reservoir.get("storage"))
            for key, values in _from_study(found).items():
                sections[key].update(values)
            source = {"study": reservoir["study"], "trial": found["_trial"],
                      "archive": reservoir.get("archive")}
            params.update({k: v for k, v in found.items() if not k.startswith("_")})
        params.update(reservoir.get("params") or {})
        kind = str(params.get("reservoir_type", reservoir.get("type", "bioreservoir")))
        from optimization.reservoirs import normalize_reservoir_type
        kind = normalize_reservoir_type(kind)
        from optimization.controls import CONTROL_DEFAULTS
        params = {**CONTROL_DEFAULTS.get(kind, {}), **params}
        ignored = sorted(set(reservoir.get("params") or {}) - set(RESERVOIR_KEYS[kind])
                         - {"reservoir_type"})
        if ignored:
            logger.warning("Ignoring reservoir parameters the %s builder does not read: %s",
                           kind, ", ".join(ignored))
        missing = [k for k in ("units",) if k not in params]
        if missing:
            raise ValueError(f"Reservoir parameters missing: {missing}")
        resolved_reservoir = {
            "type": kind,
            "seed": int(reservoir.get("seed", seed)),
            "params": {k: params[k] for k in RESERVOIR_KEYS[kind] if k in params},
        }
        if source:
            resolved_reservoir["source"] = source
        elif reservoir.get("source"):
            resolved_reservoir["source"] = reservoir["source"]

    readout = dict(cfg.get("readout") or {})
    if readout.get("study"):
        from optimization.readout import choose_classifier

        frozen = _load_trial(readout["study"], readout.get("trial", "best"),
                             readout.get("archive"), readout.get("storage"))
        name = frozen.params["classifier"]
        _, hypers = choose_classifier(frozen, name, seed=int(readout.get("seed", seed)))
        hypers.update(readout.get("params") or {})
        resolved_readout = {"classifier": name, "params": hypers,
                            "source": {"study": readout["study"], "trial": frozen.number,
                                       "archive": readout.get("archive")}}
    elif readout.get("classifier"):
        classifier_class(readout["classifier"])
        resolved_readout = {"classifier": readout["classifier"],
                            "params": dict(readout.get("params") or {})}
        if readout["classifier"] == "RidgeCV":
            resolved_readout["params"].setdefault("alphas", list(RIDGE_ALPHAS))
        if readout.get("source"):
            resolved_readout["source"] = readout["source"]
    else:
        raise ValueError("A run config needs a readout (classifier or study).")
    resolved_readout["scale"] = bool(readout.get("scale", preset.get("readout", {})
                                                 .get("scale", True)))
    if readout.get("search"):
        resolved_readout["search"] = {k: list(v) for k, v in readout["search"].items()}
        resolved_readout["inner_folds"] = int(readout.get("inner_folds", 3))
        resolved_readout["seed"] = int(readout.get("seed", seed))

    for key in ("task", "features", "cv", "outputs"):
        sections[key].update(cfg.get(key) or {})
    sections["task"].setdefault("seed", seed)
    sections["cv"].setdefault("seed", seed)
    if sections["task"]["labels"] not in ("five-class", "binary"):
        raise ValueError("task.labels must be 'five-class' or 'binary'.")
    _check_raw(sections, resolved_reservoir)
    _check_simulation(sections, resolved_reservoir)
    _check_observation(sections, resolved_reservoir)
    split = (cfg.get("test") or {}).get("split")
    if split is not None:
        from utils.heldout import SPLITS

        if split not in SPLITS.values():
            raise ValueError(f"Unknown held-out split {split!r}; choose from "
                             f"{sorted(SPLITS.values())}.")
        if sections["task"]["labels"] != "five-class":
            raise ValueError("The held-out splits hold five-class beats.")
        if "states_from" in resolved_reservoir:
            raise ValueError("A held-out evaluation simulates the reservoir; it cannot reuse "
                             "the states of another run (reservoir.states_from).")

    out = {"name": str(cfg["name"]), "description": str(cfg.get("description", "")).strip(),
           "seed": seed, "task": sections["task"], "reservoir": resolved_reservoir,
           "features": sections["features"], "readout": resolved_readout,
           "cv": sections["cv"], "outputs": sections["outputs"]}
    if split is not None:
        out["test"] = {"split": str(split)}
    if preset_name:
        out["preset"] = preset_name
    if cfg.get("reference"):
        out["reference"] = str(cfg["reference"])
    return _plain(out)


def _check_raw(sections: dict[str, Any], reservoir: dict[str, Any]) -> None:
    """A raw-beat run (``reservoir: {type: none}``) classifies each scaled beat once, as is."""
    if reservoir.get("type") != "none":
        return
    changed = sorted(key for key, value in sections["features"].items()
                     if value != DEFAULTS["features"].get(key))
    if changed:
        raise ValueError("reservoir: {type: none} classifies the beat itself; observation "
                         f"settings do not apply (features.{', features.'.join(changed)}).")
    if sections["cv"]["simulate"] != "once":
        raise ValueError("reservoir: {type: none} computes the features of each beat once; "
                         "it needs cv.simulate: once.")


def _check_simulation(sections: dict[str, Any], reservoir: dict[str, Any]) -> None:
    from training.simulate import PER_BEAT_SCALERS

    simulate = sections["cv"]["simulate"]
    if simulate not in ("once", "per_fold"):
        raise ValueError("cv.simulate must be 'once' or 'per_fold'.")
    if simulate == "per_fold" and float((reservoir.get("params") or {}).get("reset_noise", 0)):
        raise ValueError("reset_noise needs cv.simulate: once, which draws each beat's "
                         "perturbation from the beat's own seed.")
    if simulate == "per_fold" or "states_from" in reservoir:
        return
    if sections["cv"]["noise_rate"] or sections["cv"]["noise_ratio"]:
        raise ValueError("Augmented beats belong to one training fold: augmentation "
                         "(cv.noise_rate, cv.noise_ratio) needs cv.simulate: per_fold.")
    if sections["task"]["scaler"] not in PER_BEAT_SCALERS:
        raise ValueError(f"task.scaler {sections['task']['scaler']!r} is fitted on a training "
                         "fold; cv.simulate: once needs a scaling of each beat on its own "
                         f"({', '.join(PER_BEAT_SCALERS)}).")


def _check_observation(sections: dict[str, Any], reservoir: dict[str, Any]) -> None:
    features = sections["features"]
    aggregation = str(features["aggregation"])
    if aggregation.startswith("images:"):
        count = aggregation.split(":")[1]
        if not count.isdigit() or int(count) < 1:
            raise ValueError(f"features.aggregation {aggregation!r}: expected images:K, K >= 1.")
        if int(features["window"]) > 1:
            raise ValueError("features.window averages the last steps instead of taking "
                             f"images; it cannot be combined with {aggregation!r}.")
    for key in ("measurement_noise", "calibration_error"):
        if float(features[key]) < 0:
            raise ValueError(f"features.{key} must be 0 or more.")
    variables = features["variables"]
    if variables is not None:
        if (isinstance(variables, str) or not variables
                or not set(variables) <= {"A", "I", "Hi", "He"}
                or len(set(variables)) != len(variables)):
            raise ValueError("features.variables must list variables of A, I, Hi and He, "
                             "each once.")
        kind = reservoir.get("type", "bioreservoir")
        if kind != "bioreservoir":
            raise ValueError(f"features.variables needs a bioreservoir: it names variables of "
                             f"the colony model (A, I, Hi, He), which the {kind} reservoir does "
                             "not have. Remove features.variables to observe all of its "
                             "features.")
    if variables is None and not features["measurement_noise"] \
            and not features["calibration_error"]:
        return
    if "states_from" in reservoir:
        raise ValueError("A refit reads the features of another run as they were measured; "
                         "features.variables, measurement_noise and calibration_error "
                         "cannot change them.")
    if sections["cv"]["simulate"] != "once":
        raise ValueError("features.variables, measurement_noise and calibration_error need "
                         "cv.simulate: once.")


# ---------------------------------------------------------------------------
# Data, models and folds
# ---------------------------------------------------------------------------

def load_task(task: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    """The beat pool of a task: (X, Y, source files), as in ``optimize rerun-best``.

    The beats come from the prepared dataset (``ecg/ecg_data.npz`` or
    ``ecg/binary_ecg_data.npz`` in the data folder), or from ``task.data_file`` (relative
    to the data folder) when it is set. ``task.data_sha256``, when set, must be the file's
    SHA-256.
    """
    from utils.preprocessing import load_ecg_data

    binary = task["labels"] == "binary"
    rows, seed = int(task["rows"]), int(task["seed"])
    cache = settings.data_dir() / "ecg" / ("binary_ecg_data.npz" if binary else "ecg_data.npz")
    source: dict[str, str] = {}
    if task.get("data_file"):
        cache = settings.data_dir() / task["data_file"]
        if not cache.is_file():
            raise FileNotFoundError(f"task.data_file not found: {cache}")
        source = {"data_dir": str(cache.parent),
                  "binary_save_file" if binary else "save_file": cache.name}
    X_train, Y_train, X_test, Y_test = load_ecg_data(
        rows=rows, balance_classes=bool(task["balance_classes"]),
        max_per_class=task["max_per_class"], binary=binary, noise_rate=0.0, noise_ratio=0.0,
        scaler_type=str(task["scaler"]), seed=seed, preserve_split=False,
        cap_includes_test=bool(task.get("cap_includes_test", True)), **source,
    )
    if task.get("data_sha256") and provenance.sha256_file(cache) != task["data_sha256"]:
        raise ValueError(f"{cache} does not match task.data_sha256.")
    X = np.concatenate((X_train, X_test), axis=0)
    Y = np.concatenate((Y_train, Y_test), axis=0)
    if len(X) > rows:
        idx = np.sort(np.random.default_rng(seed).permutation(len(X))[:rows])
        X, Y = X[idx], Y[idx]
    return X, Y, [cache] if cache.is_file() else []


def make_readout(readout: dict[str, Any]):
    """The readout node: the classifier, after a StandardScaler when ``scale`` is set, inside
    a grid search over the training fold (inner stratified folds) when ``search`` is set."""
    from reservoirpy.nodes import ScikitLearnNode
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    cls = classifier_class(readout["classifier"])
    params = dict(readout.get("params") or {})
    steps = [("classifier", cls(**params))]
    if readout.get("scale"):
        steps.insert(0, ("scale", StandardScaler()))
    if readout.get("search"):
        from sklearn.model_selection import GridSearchCV, StratifiedKFold

        inner = StratifiedKFold(int(readout["inner_folds"]), shuffle=True,
                                random_state=int(readout["seed"]))
        grid = {f"classifier__{k}": list(v) for k, v in readout["search"].items()}
        return ScikitLearnNode(GridSearchCV, model_hypers={
            "estimator": Pipeline(steps), "param_grid": grid, "scoring": "f1_macro",
            "cv": inner})
    if readout.get("scale"):
        return ScikitLearnNode(Pipeline, model_hypers={"steps": steps})
    return ScikitLearnNode(cls, model_hypers=params)


def _build(cfg: dict[str, Any]):
    """The run's reservoir, or None for a raw-beat run."""
    from optimization.reservoirs import build_reservoir

    if cfg["reservoir"]["type"] == "none":
        return None
    reservoir_cfg = dict(cfg["reservoir"]["params"], reservoir_type=cfg["reservoir"]["type"])
    return build_reservoir(reservoir_cfg, seed=int(cfg["reservoir"]["seed"]))[0]


def _simulate(cfg: dict[str, Any], X: np.ndarray, Y: np.ndarray,
              processes: int) -> tuple[dict[str, Any], list]:
    from training.cross_validate import cross_validate, cross_validate_features

    reservoir = _build(cfg)
    if cfg["cv"]["simulate"] == "once":
        folds = int(cfg["cv"]["folds"])
        features, timing = _features(cfg, X, reservoir, processes)
        result, artifacts = cross_validate_features(
            features, Y, make_readout(cfg["readout"]), folds=folds,
            seed=int(cfg["cv"]["seed"]), primary_average=str(cfg["cv"]["primary_average"]),
            save_states=bool(cfg["outputs"]["save_states"]),
            model_hypers=getattr(reservoir, "hypers", None),
            test_gains=_test_gains(cfg, features, folds))
        result["simulation"] = timing
        return result, artifacts
    return cross_validate(
        reservoir, make_readout(cfg["readout"]), X, Y,
        folds=int(cfg["cv"]["folds"]), seed=int(cfg["cv"]["seed"]),
        save_states=bool(cfg["outputs"]["save_states"]),
        noise_rate=float(cfg["cv"]["noise_rate"]), noise_ratio=float(cfg["cv"]["noise_ratio"]),
        scaler_type=str(cfg["task"]["scaler"]),
        readout_aggregation=str(cfg["features"]["aggregation"]),
        readout_window=int(cfg["features"]["window"]),
        primary_average=str(cfg["cv"]["primary_average"]),
        fold_backend="loky" if processes > 1 else "sequential",
        fold_workers=processes if processes > 1 else None,
        profile=bool(cfg["cv"]["profile"]),
        return_artifacts=True,
    )


def _classes(Y: np.ndarray) -> np.ndarray:
    return np.argmax(np.asarray(Y).reshape(len(Y), -1), axis=1)


def _features(cfg: dict[str, Any], X: np.ndarray, reservoir,
              processes: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Scale the beats, simulate each once on the run's ``reservoir`` and measure it; return
    the features and the timing. Without a reservoir the features are the scaled beats."""
    import time

    from training.observation import measure, observed_columns
    from training.simulate import simulate_beats
    from utils.preprocessing import standardize_data

    X_scaled, _ = standardize_data(X, X[:1], scaler_type=str(cfg["task"]["scaler"]))
    if reservoir is None:
        rows = np.asarray(X_scaled, dtype=float).reshape(len(X), -1)
        return rows, {"beats": int(len(X)), "features": int(rows.shape[1]), "simulate_s": 0.0}
    columns = None
    if cfg["features"]["variables"] is not None:
        columns = observed_columns(reservoir.hypers["output_variables"],
                                   cfg["features"]["variables"], int(reservoir.hypers["units"]))
    started = time.perf_counter()
    features = simulate_beats(
        reservoir, X_scaled, noise_seed=int(cfg["reservoir"]["seed"]),
        aggregation=str(cfg["features"]["aggregation"]),
        window=int(cfg["features"]["window"]), columns=columns, processes=processes)
    elapsed = time.perf_counter() - started
    logger.info("Simulated %d beats once in %.1f s (%d processes)", len(X), elapsed, processes)
    features = measure(features, float(cfg["features"]["measurement_noise"]), int(cfg["seed"]))
    return features, {"beats": int(len(X)), "processes": int(processes),
                      "simulate_s": round(elapsed, 3),
                      "beats_per_s": round(len(X) / elapsed, 2) if elapsed else None}


def _test_gains(cfg: dict[str, Any], features: np.ndarray, folds: int) -> list | None:
    """Calibration gains of each test fold, or None without calibration error."""
    from training.observation import calibration_gains, image_count

    error = float(cfg["features"]["calibration_error"])
    if not error:
        return None
    images = image_count(str(cfg["features"]["aggregation"]))
    return [calibration_gains(features.shape[1] // images, images, error, int(cfg["seed"]),
                              fold) for fold in range(folds)]


def _evaluate_split(cfg: dict[str, Any], X: np.ndarray, Y: np.ndarray, X_test: np.ndarray,
                    Y_test: np.ndarray, beat_ids: np.ndarray, processes: int = 1):
    """Fit the readout on the whole pool and evaluate it on a held-out split, as fold 0.

    The metrics average over the classes of the split. With ``cv.simulate: per_fold`` the
    pool is augmented (``cv.noise_rate``, ``cv.noise_ratio``) and scaled as a training fold.
    Augmentation needs ``per_fold``: its noisy copies are extra training beats of one fold,
    which a feature row per beat, shared by every fold, cannot hold.
    """
    from training.cross_validate import Fold, _prepare_fold_data, run_fold
    from utils.heldout import overlap

    shared = overlap(X, X_test)
    if shared:
        raise ValueError(f"{shared} beats of the held-out split {cfg['test']['split']!r} are "
                         "in the training pool.")
    X_all, Y_all = np.concatenate((X, X_test)), np.concatenate((Y, Y_test))
    fold = Fold(train_idx=np.arange(len(X)), val_idx=np.arange(len(X), len(X_all)),
                fold_index=0)
    reservoir = _build(cfg)
    if cfg["cv"]["simulate"] == "once":
        from training.cross_validate import fit_on_features

        features, _ = _features(cfg, X_all, reservoir, processes)
        gains = _test_gains(cfg, features, folds=1)
        artifact = fit_on_features(features, Y_all, make_readout(cfg["readout"]), fold,
                                   save_states=bool(cfg["outputs"]["save_states"]),
                                   primary_average=str(cfg["cv"]["primary_average"]),
                                   labels=np.unique(_classes(Y_test)),
                                   model_hypers=getattr(reservoir, "hypers", None),
                                   test_gains=None if gains is None else gains[0])
        artifact.test_index = np.asarray(beat_ids, dtype=np.int64)
        return artifact
    data = _prepare_fold_data(X_all, Y_all, fold, noise_rate=float(cfg["cv"]["noise_rate"]),
                              noise_ratio=float(cfg["cv"]["noise_ratio"]),
                              seed=int(cfg["cv"]["seed"]),
                              scaler_type=str(cfg["task"]["scaler"]))
    artifact = run_fold(reservoir, make_readout(cfg["readout"]), *data, fold_index=0,
                        save_states=bool(cfg["outputs"]["save_states"]),
                        readout_aggregation=str(cfg["features"]["aggregation"]),
                        readout_window=int(cfg["features"]["window"]),
                        primary_average=str(cfg["cv"]["primary_average"]),
                        profile=bool(cfg["cv"]["profile"]),
                        labels=np.unique(_classes(Y_test)))
    artifact.test_index = np.asarray(beat_ids, dtype=np.int64)
    return artifact


def refit_readout(cfg: dict[str, Any], source_folds: list, processes: int = 1) -> list:
    """Fit the readout on each fold's saved training states and predict its test states."""
    for source in source_folds:
        if source.train_states is None or source.test_states is None:
            raise SystemExit(f"Fold {source.fold_index} of {cfg['reservoir']['states_from']} "
                             "has no saved states.")
    if processes > 1:
        from joblib import Parallel, delayed

        return list(Parallel(n_jobs=processes, backend="loky")(
            delayed(_refit_fold)(cfg, source) for source in source_folds))
    return [_refit_fold(cfg, source) for source in source_folds]


def _refit_fold(cfg: dict[str, Any], source):
    import time

    from training.pipeline import fit_readout, predict, selected_readout_params
    from utils.analysis import evaluate_performance
    from utils.results import FoldArtifact

    train_states = source.train_states.reshape(len(source.train_states), 1, -1)
    test_states = source.test_states.reshape(len(source.test_states), 1, -1)
    Y_train = source.Y_train.reshape(len(source.Y_train), 1, -1)
    Y_test = source.Y_test.reshape(len(source.Y_test), 1, -1)
    readout = make_readout(cfg["readout"])
    started = time.perf_counter()
    fit_readout(readout, list(train_states), list(Y_train))
    Y_pred = np.stack(predict(readout, list(test_states)))
    runtime = round(time.perf_counter() - started, 4)
    selected = selected_readout_params(readout)
    return FoldArtifact(
        trial_name=source.trial_name, fold_index=source.fold_index,
        Y_train=source.Y_train, Y_test=source.Y_test, Y_pred=Y_pred,
        metrics=evaluate_performance(Y_test, Y_pred, runtime,
                                     primary_average=cfg["cv"]["primary_average"]),
        train_states=source.train_states, test_states=source.test_states,
        test_index=source.test_index,
        model_hypers={**source.model_hypers,
                      "model": classifier_class(cfg["readout"]["classifier"]),
                      "model_hypers": cfg["readout"]["params"],
                      **({"selected_readout_params": selected} if selected else {})},
    )


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def summarise(artifacts: list, primary_average: str = "macro",
              labels: np.ndarray | None = None) -> dict[str, Any]:
    """Per-fold metrics, their mean and sample SD (ddof=1), and pooled metrics.

    ``labels`` restricts the averaged pooled metrics to those classes (a held-out split).
    """
    from utils.analysis import evaluate_performance

    folds = []
    for a in sorted(artifacts, key=lambda a: a.fold_index):
        m = a.metrics
        folds.append({"fold": int(a.fold_index), "n_test": int(len(a.Y_test)),
                      **{k: float(getattr(m, k)) for k in SCALAR_METRICS},
                      "runtime_s": float(m.runtime),
                      "confusion_matrix": np.asarray(m.confusion_matrix).tolist()})
        selected = (a.model_hypers or {}).get("selected_readout_params")
        if selected:
            folds[-1]["selected"] = selected
    values = {k: np.array([f[k] for f in folds]) for k in SCALAR_METRICS}
    mean = {k: float(v.mean()) for k, v in values.items()}
    sd = {k: float(v.std(ddof=1)) if len(v) > 1 else 0.0 for k, v in values.items()}
    pooled_true = np.concatenate([a.Y_test.reshape(len(a.Y_test), -1) for a in artifacts])
    pooled_pred = np.concatenate([a.Y_pred.reshape(len(a.Y_pred), -1) for a in artifacts])
    pooled = evaluate_performance(pooled_true, pooled_pred, 0.0, primary_average=primary_average,
                                  labels=labels)
    return {
        "primary_average": primary_average, "n_folds": len(folds), "folds": folds,
        "mean": mean, "sd": sd,
        "pooled": {**{k: float(getattr(pooled, k)) for k in SCALAR_METRICS},
                   "confusion_matrix": np.asarray(pooled.confusion_matrix).tolist()},
        "summary": {"f1": mean["f1"], "f1_sd": sd["f1"], "accuracy": mean["accuracy"],
                    "mcc": mean["mcc"]},
    }


def plot_run(fig_dir: Path, metrics: dict[str, Any]) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    cm = np.asarray(metrics["pooled"]["confusion_matrix"], dtype=float)
    rows = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    image = ax.imshow(rows, vmin=0, vmax=1, cmap="Blues")
    for (i, j), value in np.ndenumerate(rows):
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                color="white" if value > 0.5 else "black")
    ax.set(xlabel="Predicted class", ylabel="True class",
           title="Confusion matrix (all folds, row-normalised)")
    fig.colorbar(image, ax=ax, fraction=0.046)
    paths = [fig_dir / "confusion_matrix.png"]
    fig.savefig(paths[0], dpi=150, bbox_inches="tight")
    plt.close(fig)

    folds = metrics["folds"]
    fig, ax = plt.subplots(figsize=(5, 3))
    for key, marker in (("f1", "o"), ("accuracy", "s")):
        ax.plot([f["fold"] for f in folds], [f[key] for f in folds], marker=marker, label=key)
    ax.set(xlabel="Fold", ylabel="Score", title="Per-fold metrics")
    ax.legend()
    paths.append(fig_dir / "fold_metrics.png")
    fig.savefig(paths[1], dpi=150, bbox_inches="tight")
    plt.close(fig)
    return paths


@contextlib.contextmanager
def _log_to(path: Path):
    handler = logging.FileHandler(path)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [%(name)s] %(message)s"))
    root = logging.getLogger()
    previous = root.level
    root.addHandler(handler)
    root.setLevel(min(previous, logging.INFO) if previous else logging.INFO)
    try:
        yield
    finally:
        root.removeHandler(handler)
        root.setLevel(previous)
        handler.close()


def _pool_sha256(X: np.ndarray, Y: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (X, Y):
        array = np.ascontiguousarray(array)
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _peak_ram_gb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 3)


def execute(cfg: dict[str, Any], *, out: str | Path | None = None, processes: int = 1,
            allow_dirty: bool = False, registry: Path | None = None,
            final_test: bool = False) -> Path:
    """Run a config and write its run folder; return the folder.

    A config that evaluates on the ``test`` split (the final test set) needs
    ``final_test=True``.
    """
    from training.compare import bootstrap_ci, load_folds

    cfg = resolve(cfg)
    split = cfg.get("test", {}).get("split")
    if split == "test" and not final_test:
        raise SystemExit("The 'test' split is the final test set, evaluated once at the end; "
                         "pass --final-test to evaluate it.")
    git = provenance.git_info()
    citable = provenance.require_clean(git, allow_dirty)
    started = datetime.now(UTC)
    run_dir = Path(out) if out else (
        settings.runs_dir() / f"{cfg['name']}-{started:%Y%m%dT%H%M%SZ}")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"{run_dir} already exists and is not empty.")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(dump_config(cfg))

    data_files: list[Path] = []
    pool_sha256 = None
    beat_ids = None
    labels = None
    evaluation: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    with _log_to(run_dir / "run.log"):
        logger.info("Run %s (%s): %s", run_dir.name, cfg["name"], cfg["description"])
        if "states_from" in cfg["reservoir"]:
            artifacts = refit_readout(cfg, load_folds(cfg["reservoir"]["states_from"],
                                                      states=True), processes)
        else:
            X, Y, data_files = load_task(cfg["task"])
            pool_sha256 = _pool_sha256(X, Y)
            logger.info("Pool: %d beats, class counts %s", len(X),
                        np.bincount(_classes(Y)).tolist())
            if split:
                from utils import heldout

                X_test, Y_test, beat_ids = heldout.load_split(split, cfg["task"]["scaler"])
                data_files = [*data_files, heldout.SPLITS_FILE]
                labels = np.unique(_classes(Y_test))
                logger.info("Held-out split %s: %d beats, class counts %s", split,
                            len(X_test), np.bincount(_classes(Y_test)).tolist())
                artifacts = [_evaluate_split(cfg, X, Y, X_test, Y_test, beat_ids, processes)]
                low, high = bootstrap_ci(_classes(Y_test), _classes(artifacts[0].Y_pred),
                                         seed=int(cfg["seed"]))
                evaluation = {"split": split, "n_beats": int(len(X_test)),
                              "classes": labels.tolist(), "f1_macro_ci95": [low, high]}
            else:
                result, artifacts = _simulate(cfg, X, Y, processes)
                timings = {k: result[k] for k in ("profiling", "fold_runtimes", "simulation")
                           if k in result}
        for artifact in artifacts:
            artifact.trial_name = run_dir.name
            artifact.save(results_dir=str(run_dir.parent),
                          save_states=bool(cfg["outputs"]["save_states"]))
        metrics = {**summarise(artifacts, cfg["cv"]["primary_average"], labels), **timings}
        if evaluation:
            metrics["test"] = evaluation
            metrics["summary"].update(f1_macro_ci95_low=evaluation["f1_macro_ci95"][0],
                                      f1_macro_ci95_high=evaluation["f1_macro_ci95"][1])
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1) + "\n")
        plot_run(run_dir / "figures", metrics)
        if evaluation:
            logger.info("Held-out split %s: F1 %.4f (95%% CI %.4f to %.4f), accuracy %.4f",
                        split, metrics["mean"]["f1"], *evaluation["f1_macro_ci95"],
                        metrics["mean"]["accuracy"])
        else:
            logger.info("F1 %.4f +/- %.4f (sample SD over %d folds), accuracy %.4f",
                        metrics["mean"]["f1"], metrics["sd"]["f1"], metrics["n_folds"],
                        metrics["mean"]["accuracy"])

    ended = datetime.now(UTC)
    outputs = sorted(p for p in run_dir.rglob("*") if p.is_file())
    seeds = sorted({int(cfg["seed"]), int(cfg["task"]["seed"]), int(cfg["cv"]["seed"]),
                    int(cfg["reservoir"].get("seed", cfg["seed"]))})
    manifest = provenance.build_manifest(
        cfg["name"], cfg, seeds=seeds, started=started, ended=ended, data_files=data_files,
        beat_ids=beat_ids, outputs=outputs, peak_ram_gb=_peak_ram_gb(),
        allow_dirty=allow_dirty)
    manifest["data_pool_sha256"] = pool_sha256
    provenance.write_manifest(run_dir, manifest)
    provenance.register(manifest, run_dir, status="ok" if citable else "exploratory",
                        metrics=metrics["summary"], evaluation=split,
                        registry=registry or provenance.REGISTRY_FILE)
    return run_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    from utils.logger import setup_logging

    parser = argparse.ArgumentParser(prog="python -m training.run", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="Run a config.")
    p_run.add_argument("config")
    p_run.add_argument("--out", help="Run folder (default results/runs/<name>-<UTC time>).")
    p_run.add_argument("--processes", type=int, default=1,
                       help="Worker processes: beats simulated in parallel (cv.simulate: "
                            "once, the default) or folds (per_fold).")
    p_run.add_argument("--allow-dirty", action="store_true",
                       help="Run from uncommitted code; the run is marked non-citable.")
    p_run.add_argument("--save-states", action="store_true",
                       help="Keep the state features in the fold files.")
    p_run.add_argument("--final-test", action="store_true",
                       help="Allow a config that evaluates on the test split (the final test "
                            "set, evaluated once at the end).")
    p_res = sub.add_parser("resolve", help="Print the explicit config.")
    p_res.add_argument("config")
    p_cmp = sub.add_parser("compare", help="Compare two runs.")
    p_cmp.add_argument("run_a", help="Run folder, registry run id, or archive:<version>/<run>.")
    p_cmp.add_argument("run_b")
    p_cmp.add_argument("--out", help="Folder for compare.json and figures.")
    p_cmp.add_argument("--bootstrap", type=int, default=2000)
    p_cmp.add_argument("--margin", type=float,
                       help="Non-inferiority margin for B - A, e.g. -0.02.")
    p_cmp.add_argument("--ceiling", help="Run C on the same beats: the share of the gap "
                                          "from A to C that B closes.")
    p_std = sub.add_parser("study", help="Run the Optuna study of a config (training.study).")
    p_std.add_argument("config")
    p_std.add_argument("--processes", type=int, default=1,
                       help="Worker processes of each trial, as for run.")
    p_std.add_argument("--trials", type=int,
                       help="Finished trials to reach (default: study.trials).")
    p_std.add_argument("--storage",
                       help="Optuna storage URL (default results/optuna/optuna-<study>.db).")
    p_std.add_argument("--allow-dirty", action="store_true",
                       help="Run from uncommitted code; trials are marked non-citable.")
    p_bst = sub.add_parser("study-best",
                           help="Print (or write) the best trial of a study as a run config.")
    p_bst.add_argument("config")
    p_bst.add_argument("--out", help="File for the run config.")
    p_bst.add_argument("--storage")
    p_tbl = sub.add_parser("study-table",
                           help="Print the trials of one or more studies as a table "
                                "(training.study).")
    p_tbl.add_argument("configs", nargs="+", metavar="config")
    p_tbl.add_argument("--group", action="store_true",
                       help="One row per point of the searched values, averaged over "
                            "reservoir.seed and over the given studies.")
    p_tbl.add_argument("--csv", help="Also write the table to this CSV file.")
    p_tbl.add_argument("--storage",
                       help="Optuna storage URL (default results/optuna/optuna-<study>.db).")
    args = parser.parse_args(argv)

    setup_logging(level=logging.INFO)
    if args.command == "resolve":
        print(dump_config(resolve(load_config(args.config))), end="")
    elif args.command == "run":
        cfg = load_config(args.config)
        if args.save_states:
            cfg.setdefault("outputs", {})["save_states"] = True
        run_dir = execute(cfg, out=args.out, processes=args.processes,
                          allow_dirty=args.allow_dirty, final_test=args.final_test)
        print(run_dir)
    elif args.command == "study":
        from training.study import run_study

        study = run_study(load_config(args.config), processes=args.processes,
                          trials=args.trials, storage=args.storage,
                          allow_dirty=args.allow_dirty)
        best = study.best_trial
        print(f"{study.study_name}: {len(study.trials)} trials, best {best.value:.4f} "
              f"(trial {best.number})")
    elif args.command == "study-best":
        from training.study import write_best

        print(write_best(load_config(args.config), args.out, args.storage), end="")
    elif args.command == "study-table":
        from training.study import format_table, study_table

        frame = study_table([load_config(path) for path in args.configs], args.storage,
                            group=args.group)
        if args.csv:
            frame.to_csv(args.csv, index=False)
        print(format_table(frame))
    else:
        from training.compare import compare, format_comparison

        result = compare(args.run_a, args.run_b, out=args.out, n_boot=args.bootstrap,
                         margin=args.margin, ceiling=args.ceiling)
        print(format_comparison(result))


if __name__ == "__main__":
    main()
