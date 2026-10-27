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
again. Run one process per study: several processes sharing a study would run more trials
than asked and could repeat points. The trial count includes failed trials. A trial whose
reservoir the solver cannot integrate (``UnstableIntegrationError``: the RK4 step check
refuses it, or its states blow up) is recorded as failed and the study goes on; any other
error stops the study, and running the command again resumes it. Each trial records the
machine it ran on (``environment``: CPU, cores, memory, OS and package versions). Only
``study.trials`` may change between runs of a study. With ``reservoir.states_from`` the
trials refit only the readout on the saved states of a run (no simulation), so only
``readout`` settings can be searched.

    python -m training.run study CONFIG.yaml [--processes N] [--trials N] [--storage URL]
    python -m training.run study-best CONFIG.yaml [--out BEST.yaml] [--storage URL]
    python -m training.run study-table CONFIG.yaml [CONFIG.yaml ...] [--group] [--csv FILE]

``study-table`` lists the trials of one or more studies (:func:`trials_table`): the searched
values, the metric's mean and SD over the folds, the reservoir seed and the trial's wall
time. ``--group`` averages the trials that differ only in ``reservoir.seed``, or only in the
study they belong to (:func:`seed_table`), so a grid run at three seeds reads as one row per
point with the spread over seeds.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import yaml

from utils import config as settings
from utils import provenance

logger = logging.getLogger(__name__)

SECTIONS = ("task", "reservoir", "features", "readout", "cv")

# Settings a search may set in the sections without defaults (task, features and cv take
# the keys of training.run.DEFAULTS).
_SETTINGS = {"reservoir": ("params", "seed", "trial"),
             "readout": ("classifier", "params", "scale", "search", "inner_folds", "seed",
                         "trial")}
_EXTRA_SETTINGS = {"task": ("seed", "data_sha256", "scramble"), "cv": ("seed",)}
# Environment keys a trial leaves out: the code is in its ``code`` attribute, and these vary.
_UNRECORDED = ("timestamp_utc", "ram_available_gb", "git_commit", "git_dirty", "git_branch")


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
        if path == "seed":
            cfg["seed"] = value
            continue
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
    """Refuse search paths the run would ignore, incomplete specs and unknown metrics."""
    from training.run import DEFAULTS, RESERVOIR_KEYS, SCALAR_METRICS, resolve

    metric = str(study.get("metric", "f1"))
    if metric not in SCALAR_METRICS:
        raise ValueError(f"study.metric {metric!r} is not a run metric; choose from "
                         f"{', '.join(SCALAR_METRICS)}.")
    resolved = resolve(run_cfg)
    kind = resolved["reservoir"].get("type")
    for path, spec in study["search"].items():
        apply(run_cfg, {path: None})
        keys = path.split(".")
        if "states_from" in resolved["reservoir"] and keys[0] != "readout":
            raise ValueError(f"Search path {path!r}: a study on saved states "
                             "(reservoir.states_from) can only search readout settings.")
        if kind == "none" and keys[0] in ("reservoir", "features"):
            raise ValueError(f"Search path {path!r}: a raw-beat run (reservoir: {{type: none}}) "
                             "has no reservoir or observation settings.")
        if path == "seed":
            if "seed" not in (run_cfg.get("task") or {}) or "seed" not in (run_cfg.get("cv") or {}):
                raise ValueError("Search path 'seed': set task.seed and cv.seed, so that every "
                                 "trial has the same pool and folds.")
        else:
            known = (set(DEFAULTS[keys[0]]) | set(_EXTRA_SETTINGS.get(keys[0], ()))
                     if keys[0] in DEFAULTS else set(_SETTINGS[keys[0]]))
            if keys[1] not in known:
                raise ValueError(f"Search path {path!r}: {keys[0]} has no setting "
                                 f"{keys[1]!r}.")
        if keys[:2] == ["reservoir", "params"] and (
                len(keys) < 3 or keys[2] not in RESERVOIR_KEYS.get(kind, ())):
            raise ValueError(f"Search path {path!r}: the {kind} builder has no such "
                             "parameter.")
        if spec.get("type") not in ("float", "int", "categorical"):
            raise ValueError(f"Search path {path!r}: type must be float, int or categorical.")
        if "choices" in spec:
            continue
        if _sampler_type(study) == "grid" or spec["type"] == "categorical":
            raise ValueError(f"Search path {path!r}: needs choices.")
        if not ("low" in spec and "high" in spec
                and float(spec["low"]) < float(spec["high"])):
            raise ValueError(f"Search path {path!r}: needs low < high.")


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

    from bioreservoir.dde import UnstableIntegrationError
    from optimization.environment import get_environment
    from training.compare import load_folds
    from training.run import (
        _pool_sha256,
        _simulate,
        dump_config,
        load_task,
        refit_readout,
        resolve,
        summarise,
    )

    run_cfg, study_cfg = split(cfg)
    check(run_cfg, study_cfg)
    git = provenance.git_info()
    citable = provenance.require_clean(git, allow_dirty)
    template = resolve(run_cfg)
    saved = (load_folds(template["reservoir"]["states_from"], states=True)
             if "states_from" in template["reservoir"] else None)
    pools: dict[str, tuple] = {}

    def pool(task: dict[str, Any]) -> tuple:
        key = json.dumps(task, sort_keys=True)
        if key not in pools:
            X, Y, _ = load_task(task)
            pools[key] = (X, Y, _pool_sha256(X, Y))
        return pools[key]

    metric = str(study_cfg.get("metric", "f1"))
    url = storage_url(cfg, storage)
    study = optuna.create_study(
        study_name=name(cfg), storage=url, load_if_exists=True,
        direction=str(study_cfg.get("direction", "maximize")), sampler=make_sampler(study_cfg))
    stored = study.user_attrs.get("config")
    if stored is not None and _definition(yaml.safe_load(stored)) != _definition(cfg):
        raise ValueError(f"Study {name(cfg)!r} in {url} was run with another config; only "
                         "study.trials may change. Give the study a new name.")
    study.set_user_attr("config", dump_config(cfg))
    if saved is None:
        study.set_user_attr("data_pool_sha256", pool(template["task"])[2])
    queued = [t.params or getattr(t, "system_attrs", {}).get("fixed_params", {})
              for t in study.trials]
    for point in study_cfg.get("enqueue") or []:
        if not any(all(q.get(k) == v for k, v in point.items()) for q in queued):
            study.enqueue_trial(point)

    def objective(trial) -> float:
        trial.set_user_attr("environment", {k: v for k, v in get_environment().items()
                                            if k not in _UNRECORDED})
        resolved = resolve(apply(run_cfg, suggest(trial, study_cfg["search"])))
        if saved is None:
            X, Y, pool_sha256 = pool(resolved["task"])
            result, artifacts = _simulate(resolved, X, Y, processes)
            trial.set_user_attr("data_pool_sha256", pool_sha256)
        else:
            result, artifacts = {}, refit_readout(resolved, saved, processes)
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
        study.optimize(objective, n_trials=target - finished,
                       catch=(UnstableIntegrationError,))
    return study


def _grid_size(study: dict[str, Any]) -> int:
    size = 1
    for spec in study["search"].values():
        size *= len(spec["choices"])
    return size


def _definition(cfg: dict[str, Any]) -> dict[str, Any]:
    """The parts of a study config that its trials depend on (all but trials and text)."""
    from training.run import dump_config

    cfg = yaml.safe_load(dump_config(cfg))
    cfg.pop("description", None)
    cfg.get("study", {}).pop("trials", None)
    return cfg


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


# ---------------------------------------------------------------------------
# Tables of trials
# ---------------------------------------------------------------------------

SEED_PATH = "reservoir.seed"


def trials_table(cfg: dict[str, Any], storage: str | None = None):
    """The trials of the study of ``cfg``, one row each, as a pandas DataFrame.

    Columns: ``trial``, ``state``, the searched values (one column per dotted path),
    ``reservoir.seed`` (from the trial's resolved config when the search does not set it),
    the study metric's mean over the folds (the trial's value, named after the metric), its
    sample SD over the folds (``<metric>_sd``; trials record it for ``f1``), ``seconds``
    (the trial's wall time), ``simulate_s`` (of which simulating the beats) and ``commit``.
    """
    import optuna
    import pandas as pd

    run_cfg, study_cfg = split(cfg)
    metric = str(study_cfg.get("metric", "f1"))
    if storage is None:
        path = settings.optuna_db(name(cfg))
        if not path.is_file():
            raise FileNotFoundError(f"No Optuna database for study {name(cfg)!r}: {path}")
        storage = f"sqlite:///{path}"
    study = optuna.load_study(study_name=name(cfg), storage=storage)
    paths = [p for p in study_cfg["search"] if p != SEED_PATH]
    fixed_seed = (run_cfg.get("reservoir") or {}).get("seed", run_cfg.get("seed", settings.SEED))
    rows = []
    for trial in study.trials:
        attrs = trial.user_attrs
        seed = trial.params.get(SEED_PATH, fixed_seed)
        if "config" in attrs:
            seed = yaml.safe_load(attrs["config"])["reservoir"].get("seed")
        started, ended = trial.datetime_start, trial.datetime_complete
        rows.append({
            "trial": trial.number, "state": trial.state.name,
            **{p: trial.params.get(p) for p in paths}, SEED_PATH: seed,
            metric: trial.value,
            f"{metric}_sd": (attrs.get("summary") or {}).get(f"{metric}_sd"),
            "seconds": (ended - started).total_seconds() if started and ended else None,
            "simulate_s": (attrs.get("simulation") or {}).get("simulate_s"),
            "commit": (attrs.get("code") or {}).get("commit"),
        })
    columns = ["trial", "state", *paths, SEED_PATH, metric, f"{metric}_sd", "seconds",
               "simulate_s", "commit"]
    frame = pd.DataFrame(rows, columns=columns)
    for column in (metric, f"{metric}_sd", "seconds", "simulate_s"):
        frame[column] = frame[column].astype(float)
    return frame


def seed_table(trials, metric: str = "f1"):
    """The finished trials of :func:`trials_table` averaged over reservoir seeds.

    One row per point: a combination of the searched values other than ``reservoir.seed``.
    Trials of several studies with the same searched paths group together, so studies that
    differ only in their seeds read as one. Columns: the searched values, ``n`` (trials
    averaged), the metric's mean over them, its sample SD over them (``<metric>_seed_sd``),
    the mean of their SDs over folds (``<metric>_fold_sd``), their reservoir ``seeds`` and
    their mean ``seconds``.
    """
    import pandas as pd

    done = trials[trials["state"] == "COMPLETE"]
    keys = [c for c in trials.columns if "." in c and c != SEED_PATH]
    groups = done.groupby(keys, sort=True, dropna=False) if keys else [((), done)]
    rows = []
    for key, part in groups:
        if not len(part):
            continue
        values = part[metric].astype(float)
        seeds = sorted({int(s) for s in part[SEED_PATH].dropna()})
        rows.append({
            **dict(zip(keys, key if isinstance(key, tuple) else (key,), strict=True)),
            "n": len(part),
            metric: values.mean(),
            f"{metric}_seed_sd": values.std(ddof=1) if len(part) > 1 else float("nan"),
            f"{metric}_fold_sd": part[f"{metric}_sd"].astype(float).mean(),
            "seeds": " ".join(str(s) for s in seeds), "seconds": part["seconds"].mean(),
        })
    return pd.DataFrame(rows, columns=[*keys, "n", metric, f"{metric}_seed_sd",
                                       f"{metric}_fold_sd", "seeds", "seconds"])


def study_table(cfgs: list[dict[str, Any]], storage: str | None = None, group: bool = False):
    """:func:`trials_table` of each config, one after another (with a ``study`` column when
    there are several), or with ``group`` their :func:`seed_table`."""
    import pandas as pd

    metrics = {str(split(cfg)[1].get("metric", "f1")) for cfg in cfgs}
    if len(metrics) != 1:
        raise ValueError(f"The studies score different metrics: {sorted(metrics)}.")
    frames = [trials_table(cfg, storage) for cfg in cfgs]
    if len(frames) > 1:
        for cfg, frame in zip(cfgs, frames, strict=True):
            frame.insert(0, "study", name(cfg))
    frame = pd.concat(frames, ignore_index=True)
    return seed_table(frame, metrics.pop()) if group else frame


def format_table(frame) -> str:
    """``frame`` as text: dotted paths shown by their last key when that is unique, metrics
    to four decimals, times to 0.1 s, commits to seven characters."""
    import math

    last = [c.rsplit(".", 1)[-1] for c in frame.columns]
    shown = frame.rename(columns={c: s for c, s in zip(frame.columns, last, strict=True)
                                  if last.count(s) == 1})

    def number(digits: str):
        return lambda v: "" if v is None or (isinstance(v, float) and math.isnan(v)) \
            else format(v, digits)

    formatters = {}
    for column, label in zip(frame.columns, shown.columns, strict=True):
        if column in ("seconds", "simulate_s"):
            formatters[label] = number(".1f")
        elif column == "commit":
            formatters[label] = lambda v: str(v)[:7] if v else ""
        elif frame[column].dtype.kind == "f":
            # undotted float columns are the metric's mean and SDs; dotted ones searched values
            formatters[label] = number(".6g" if "." in column else ".4f")
    return shown.to_string(index=False, formatters=formatters)
