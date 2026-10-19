"""Optuna studies defined in a run config.

A run config with a ``study`` section defines a search. Every trial is the run config with
the trial's values put in at dotted paths (``reservoir.params.input_scaling``), run through
the same pipeline as a single run (task, reservoir, features, readout, cross-validation) and
scored by a cross-validation metric. Trials record their resolved config, so any trial can
be run again as a plain config; ``study-best`` writes the best one, and its run gives the
reported numbers.

.. code-block:: yaml

    study:
      trials: 300
      metric: f1                # a key of the run's mean metrics (cv.primary_average)
      direction: maximize
      sampler: {type: tpe, seed: 6337, multivariate: true, n_startup_trials: 20}
      enqueue:                  # points to try first
        - {reservoir.params.input_scaling: 0.083}
      search:
        reservoir.params.input_scaling: {type: float, low: 0.005, high: 0.5, log: true}
        reservoir.params.units: {type: categorical, choices: [256, 1024]}

``search`` uses the suggest specs of :mod:`optimization.studies` (``float`` with
``low``, ``high`` and ``log``; ``int`` with ``low``, ``high`` and ``step``;
``categorical`` with ``choices``). Samplers: ``tpe`` (``seed``, ``multivariate``,
``n_startup_trials``, ``constant_liar``), ``random`` (``seed``) and ``grid`` (every
combination of ``choices``, which every parameter then needs). Trials run one at a time,
each with ``--processes`` workers, so a study with a seeded sampler runs the same trials
again; several processes can share one study through its storage.

    python -m training.run study CONFIG.yaml [--processes N] [--trials N] [--storage URL]
    python -m training.run study-best CONFIG.yaml [--out BEST.yaml] [--storage URL]
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import yaml

from utils import config as settings
from utils import provenance

logger = logging.getLogger(__name__)

SECTIONS = ("task", "reservoir", "features", "readout", "cv")


def split(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The run config without its ``study`` section, and the section."""
    run_cfg = copy.deepcopy(cfg)
    study = run_cfg.pop("study", None)
    if not study or not study.get("search"):
        raise ValueError("A study config needs a study section with a search.")
    return run_cfg, study


def apply(cfg: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    """A copy of ``cfg`` with each value put in at its dotted path."""
    cfg = copy.deepcopy(cfg)
    for path, value in values.items():
        keys = path.split(".")
        if keys[0] not in SECTIONS or len(keys) < 2:
            raise ValueError(f"Search path {path!r} must start with one of {SECTIONS}.")
        node = cfg
        for key in keys[:-1]:
            node = node.setdefault(key, {})
            if not isinstance(node, dict):
                raise ValueError(f"Search path {path!r} goes through a value.")
        node[keys[-1]] = value
    return cfg


def check(run_cfg: dict[str, Any], study: dict[str, Any]) -> None:
    """Refuse search paths the run would ignore, and incomplete grids."""
    from training.run import RESERVOIR_KEYS, resolve

    kind = resolve(run_cfg)["reservoir"].get("type")
    for path, spec in study["search"].items():
        apply(run_cfg, {path: None})
        keys = path.split(".")
        if keys[:2] == ["reservoir", "params"] and keys[2] not in RESERVOIR_KEYS.get(kind, ()):
            raise ValueError(f"Search path {path!r}: the {kind} builder has no "
                             f"parameter {keys[2]!r}.")
        if spec.get("type") not in ("float", "int", "categorical"):
            raise ValueError(f"Search path {path!r}: type must be float, int or categorical.")
        if _sampler_type(study) == "grid" and "choices" not in spec:
            raise ValueError(f"Search path {path!r}: a grid needs choices.")


def _sampler_type(study: dict[str, Any]) -> str:
    return str((study.get("sampler") or {}).get("type", "tpe"))


def make_sampler(study: dict[str, Any]):
    import optuna

    spec = dict(study.get("sampler") or {})
    kind = spec.pop("type", "tpe")
    if kind == "tpe":
        return optuna.samplers.TPESampler(**spec)
    if kind == "random":
        return optuna.samplers.RandomSampler(**spec)
    if kind == "grid":
        grid = {path: list(p["choices"]) for path, p in study["search"].items()}
        return optuna.samplers.GridSampler(grid, **spec)
    raise ValueError(f"Unknown sampler {kind!r}; choose tpe, random or grid.")


def suggest(trial, search: dict[str, Any]) -> dict[str, Any]:
    values = {}
    for path, spec in search.items():
        if spec["type"] == "categorical" or "choices" in spec:
            values[path] = trial.suggest_categorical(path, list(spec["choices"]))
        elif spec["type"] == "float":
            values[path] = trial.suggest_float(path, float(spec["low"]), float(spec["high"]),
                                               log=bool(spec.get("log", False)))
        else:
            values[path] = trial.suggest_int(path, int(spec["low"]), int(spec["high"]),
                                             step=int(spec.get("step", 1)),
                                             log=bool(spec.get("log", False)))
    return values


def name(cfg: dict[str, Any]) -> str:
    return str((cfg.get("study") or {}).get("name") or cfg["name"])


def storage_url(cfg: dict[str, Any], storage: str | None = None) -> str:
    if storage:
        return storage
    path = settings.optuna_db(name(cfg))
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path}"


def run_study(cfg: dict[str, Any], *, processes: int = 1, trials: int | None = None,
              storage: str | None = None, allow_dirty: bool = False):
    """Run the study of ``cfg`` until it has its number of finished trials; return it."""
    import optuna

    from training.run import _pool_sha256, _simulate, dump_config, load_task, resolve, summarise

    run_cfg, study_cfg = split(cfg)
    check(run_cfg, study_cfg)
    git = provenance.git_info()
    citable = provenance.require_clean(git, allow_dirty)
    template = resolve(run_cfg)
    X, Y, _ = load_task(template["task"])
    metric = str(study_cfg.get("metric", "f1"))
    study = optuna.create_study(
        study_name=name(cfg), storage=storage_url(cfg, storage), load_if_exists=True,
        direction=str(study_cfg.get("direction", "maximize")), sampler=make_sampler(study_cfg))
    study.set_user_attr("config", dump_config(cfg))
    study.set_user_attr("data_pool_sha256", _pool_sha256(X, Y))
    queued = [t.params or getattr(t, "system_attrs", {}).get("fixed_params", {})
              for t in study.trials]
    for point in study_cfg.get("enqueue") or []:
        if point not in queued:
            study.enqueue_trial(point)

    def objective(trial) -> float:
        resolved = resolve(apply(run_cfg, suggest(trial, study_cfg["search"])))
        result, artifacts = _simulate(resolved, X, Y, processes)
        metrics = summarise(artifacts, resolved["cv"]["primary_average"])
        trial.set_user_attr("config", dump_config(resolved))
        trial.set_user_attr("summary", metrics["summary"])
        trial.set_user_attr("fold_f1", [f["f1"] for f in metrics["folds"]])
        trial.set_user_attr("code", {"commit": git.get("commit"), "citable": citable})
        if "simulation" in result:
            trial.set_user_attr("simulation", result["simulation"])
        return float(metrics["mean"][metric])

    target = int(trials if trials is not None else study_cfg.get("trials", 100))
    finished = sum(t.state.is_finished() for t in study.trials)
    if _sampler_type(study_cfg) == "grid":
        target = min(target, _grid_size(study_cfg))
    if target > finished:
        logger.info("Study %s: %d trials done, running %d", name(cfg), finished,
                    target - finished)
        study.optimize(objective, n_trials=target - finished)
    return study


def _grid_size(study: dict[str, Any]) -> int:
    size = 1
    for spec in study["search"].values():
        size *= len(spec["choices"])
    return size


def best_config(cfg: dict[str, Any], storage: str | None = None) -> dict[str, Any]:
    """The resolved config of the study's best trial, as a plain run config."""
    import optuna

    study = optuna.load_study(study_name=name(cfg), storage=storage_url(cfg, storage))
    best = study.best_trial
    out = yaml.safe_load(best.user_attrs["config"])
    out["name"] = f"{name(cfg)}-best"
    out["description"] = (f"Trial {best.number} of study {name(cfg)} "
                          f"({study.direction.name.lower()} {best.value:.6g}).")
    return out


def write_best(cfg: dict[str, Any], out: str | Path | None = None,
               storage: str | None = None) -> str:
    from training.run import dump_config

    text = dump_config(best_config(cfg, storage))
    if out:
        Path(out).write_text(text)
    return text
