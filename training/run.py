"""Run a classification task from a config file (revision plan, section 16.4).

    python -m training.run run CONFIG.yaml [--processes N] [--out DIR] [--allow-dirty]
    python -m training.run resolve CONFIG.yaml
    python -m training.run compare RUN_A RUN_B [--out DIR]

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
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import logging
import resource
from datetime import datetime, timezone
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
    "features": {"aggregation": "final", "window": 1},
    "cv": {"folds": 5, "primary_average": "macro", "noise_rate": 0.0, "noise_ratio": 0.0,
           "profile": False},
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
    ),
    "esn": (
        "units", "esn_leak_rate", "esn_spectral_radius", "esn_activation", "input_scaling",
        "input_connectivity", "input_bias", "bias_scaling", "noise_in", "noise_rc",
        "rc_connectivity",
    ),
}

SCALAR_METRICS = ("f1", "f1_macro", "f1_weighted", "accuracy", "precision", "recall",
                  "mcc", "kappa")

# Settings that reproduce the v0.1 code path; explicit config values win.
PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "v0.1": {
        "task": {"cap_includes_test": False},
        "reservoir": {"geometry": "v0.1", "delay_scheme": "frozen",
                      "normalize_after_masks": False,
                      "heterogeneity_mean_preserving": False},
        "readout": {"scale": False},
    },
}


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _plain(value: Any) -> Any:
    """JSON-compatible copy (tuples become lists, numpy scalars become numbers)."""
    return json.loads(json.dumps(value, default=lambda v: v.item() if hasattr(v, "item")
                                 else str(v)))


def dump_config(cfg: dict[str, Any]) -> str:
    return yaml.safe_dump(_plain(cfg), sort_keys=False, default_flow_style=False)


def classifier_class(name: str):
    """The scikit-learn class behind a readout name (the Phase 4 readout-study names)."""
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    table = {"Ridge": RidgeClassifier, "LR": LogisticRegression, "SVM": SVC,
             "KNN": KNeighborsClassifier, "RF": RandomForestClassifier,
             "DT": DecisionTreeClassifier, "GB": GradientBoostingClassifier,
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
    """BASELINE, then the study's fixed values, then the trial's values (as rerun-best)."""
    from optimization.studies import get_study
    from optimization.training import _resolve_params_from_trial

    study_config = get_study(study)
    frozen = _load_trial(study, trial, archive, storage)
    params = _resolve_params_from_trial(frozen, study_config)
    params["_trial"] = frozen.number
    params["_instances"] = study_config.get("instances", DEFAULTS["task"]["rows"])
    params["_binary"] = bool(study_config.get("binary", False))
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
               "noise_ratio": float(params.get("noise_ratio", 0.0))},
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

    out = {"name": str(cfg["name"]), "description": str(cfg.get("description", "")).strip(),
           "seed": seed, "task": sections["task"], "reservoir": resolved_reservoir,
           "features": sections["features"], "readout": resolved_readout,
           "cv": sections["cv"], "outputs": sections["outputs"]}
    if preset_name:
        out["preset"] = preset_name
    if cfg.get("reference"):
        out["reference"] = str(cfg["reference"])
    return _plain(out)


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


def _simulate(cfg: dict[str, Any], X: np.ndarray, Y: np.ndarray,
              processes: int) -> tuple[dict[str, Any], list]:
    from optimization.reservoirs import build_reservoir
    from training.cross_validate import cross_validate

    reservoir_cfg = dict(cfg["reservoir"]["params"], reservoir_type=cfg["reservoir"]["type"])
    reservoir, _ = build_reservoir(reservoir_cfg, seed=int(cfg["reservoir"]["seed"]))
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

    from training.pipeline import fit_readout, predict
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
    selected = getattr(getattr(readout, "_fitted_sklearn", None), "best_params_", None)
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

def summarise(artifacts: list, primary_average: str = "macro") -> dict[str, Any]:
    """Per-fold metrics, their mean and sample SD (ddof=1), and pooled metrics."""
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
    pooled = evaluate_performance(pooled_true, pooled_pred, 0.0, primary_average=primary_average)
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
            allow_dirty: bool = False, registry: Path | None = None) -> Path:
    """Run a config and write its run folder; return the folder."""
    from training.compare import load_folds

    cfg = resolve(cfg)
    git = provenance.git_info()
    citable = provenance.require_clean(git, allow_dirty)
    started = datetime.now(timezone.utc)
    run_dir = Path(out) if out else (
        settings.runs_dir() / f"{cfg['name']}-{started:%Y%m%dT%H%M%SZ}")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"{run_dir} already exists and is not empty.")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(dump_config(cfg))

    data_files: list[Path] = []
    pool_sha256 = None
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
                        np.bincount(np.argmax(Y.reshape(len(Y), -1), axis=1)).tolist())
            result, artifacts = _simulate(cfg, X, Y, processes)
            timings = {k: result[k] for k in ("profiling", "fold_runtimes") if k in result}
        for artifact in artifacts:
            artifact.trial_name = run_dir.name
            artifact.save(results_dir=str(run_dir.parent),
                          save_states=bool(cfg["outputs"]["save_states"]))
        metrics = {**summarise(artifacts, cfg["cv"]["primary_average"]), **timings}
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1) + "\n")
        plot_run(run_dir / "figures", metrics)
        logger.info("F1 %.4f +/- %.4f (sample SD over %d folds), accuracy %.4f",
                    metrics["mean"]["f1"], metrics["sd"]["f1"], metrics["n_folds"],
                    metrics["mean"]["accuracy"])

    ended = datetime.now(timezone.utc)
    outputs = sorted(p for p in run_dir.rglob("*") if p.is_file())
    seeds = sorted({int(cfg["seed"]), int(cfg["task"]["seed"]), int(cfg["cv"]["seed"]),
                    int(cfg["reservoir"].get("seed", cfg["seed"]))})
    manifest = provenance.build_manifest(
        cfg["name"], cfg, seeds=seeds, started=started, ended=ended, data_files=data_files,
        outputs=outputs, peak_ram_gb=_peak_ram_gb(), allow_dirty=allow_dirty)
    manifest["data_pool_sha256"] = pool_sha256
    provenance.write_manifest(run_dir, manifest)
    provenance.register(manifest, run_dir, status="ok" if citable else "exploratory",
                        metrics=metrics["summary"],
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
                       help="Folds run in parallel worker processes.")
    p_run.add_argument("--allow-dirty", action="store_true",
                       help="Run from uncommitted code; the run is marked non-citable.")
    p_run.add_argument("--save-states", action="store_true",
                       help="Keep the state features in the fold files.")
    p_res = sub.add_parser("resolve", help="Print the explicit config.")
    p_res.add_argument("config")
    p_cmp = sub.add_parser("compare", help="Compare two runs.")
    p_cmp.add_argument("run_a", help="Run folder, registry run id, or archive:<version>/<run>.")
    p_cmp.add_argument("run_b")
    p_cmp.add_argument("--out", help="Folder for compare.json and figures.")
    p_cmp.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args(argv)

    setup_logging(level=logging.INFO)
    if args.command == "resolve":
        print(dump_config(resolve(load_config(args.config))), end="")
    elif args.command == "run":
        cfg = load_config(args.config)
        if args.save_states:
            cfg.setdefault("outputs", {})["save_states"] = True
        run_dir = execute(cfg, out=args.out, processes=args.processes,
                          allow_dirty=args.allow_dirty)
        print(run_dir)
    else:
        from training.compare import compare, format_comparison

        result = compare(args.run_a, args.run_b, out=args.out, n_boot=args.bootstrap)
        print(format_comparison(result))


if __name__ == "__main__":
    main()
