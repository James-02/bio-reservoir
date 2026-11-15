"""Sweeps: one config file that expands into many runs or studies.

A config with a ``sweep`` section stands for every combination of its axes. Its other
settings (and its ``base``) are the starting config; each axis then merges one of its
variants over it, in the order the axes are listed, so a later axis wins::

    sweep:
      name: final-{task}-{model}-s{seed}
      model:                                   # named variants: a config file or settings
        lattice: designs/lattice.yaml
        lattice-forest: {base: designs/lattice.yaml, readout: {classifier: RF}}
      task:
        five-class: {}
        binary: {task.labels: binary}          # a dotted path sets one setting
      seed: {reservoir.seed: [1, 2, 3]}        # one setting, a list of values

An axis is either named variants (``label: file or settings``), one setting with a list of
values (``path: [values]``, labelled by the value), or a list of labels that only the
templates use. ``name`` (required) and
``description`` are templates of the axes' labels, and so is any other text setting: a
``{axis}`` in it becomes that axis's label (``states_from: final-{task}-{model}-s{seed}``),
and a setting that is only ``{axis}`` takes the axis's value (``choices: ['{seed}']``).
``sweep`` can also be a list of such blocks, whose runs are concatenated; YAML anchors share
axes between blocks. Every run of a sweep is an ordinary config: ``expand`` returns them,
and ``training.run sweep`` runs them (a config with a ``study`` section runs as a study).
"""

from __future__ import annotations

import copy
import itertools
import re
from pathlib import Path
from typing import Any

from training.run import _merged, load_config


def _fill(value: Any, labels: dict[str, Any]) -> Any:
    """``value`` with each ``{axis}`` of a known axis in its text replaced by the label; a
    text that is only ``{axis}`` becomes the axis's value itself (a number stays a number)."""
    if isinstance(value, str):
        whole = re.fullmatch(r"\{([\w-]+)\}", value)
        if whole and whole[1] in labels:
            return labels[whole[1]]
        return re.sub(r"\{([\w-]+)\}", lambda m: str(labels.get(m[1], m[0])), value)
    if isinstance(value, dict):
        return {k: _fill(v, labels) for k, v in value.items()}
    if isinstance(value, list):
        return [_fill(v, labels) for v in value]
    return value


def _nested(settings: dict[str, Any]) -> dict[str, Any]:
    """Settings with dotted paths (``reservoir.seed: 1``) as nested sections."""
    out: dict[str, Any] = {}
    for path, value in settings.items():
        *parents, leaf = str(path).split(".")
        node = out
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    return out


def _axis(name: str, spec: Any, folder: Path) -> list[tuple[Any, dict[str, Any]]]:
    """The (label, settings) variants of one axis; a value or listed label keeps its type."""
    if isinstance(spec, list) and spec:
        return [(label, {}) for label in spec]
    if not isinstance(spec, dict) or not spec:
        raise ValueError(f"Sweep axis {name!r} must map labels to variants, map one setting "
                         f"to a list of values, or list labels.")
    if len(spec) == 1 and isinstance(next(iter(spec.values())), list):
        (path, values), = spec.items()
        return [(value, _nested({path: value})) for value in values]
    variants = []
    for label, variant in spec.items():
        if isinstance(variant, str):
            settings = load_config(folder / variant)
        elif isinstance(variant, dict):
            settings = _nested({k: v for k, v in variant.items() if k != "base"})
            if "base" in variant:
                settings = _merged(load_config(folder / variant["base"]), settings)
        else:
            raise ValueError(f"Variant {label!r} of sweep axis {name!r} must be a config "
                             f"file or a mapping of settings.")
        variants.append((str(label), settings))
    return variants


def expand(path: str | Path) -> list[dict[str, Any]]:
    """The configs of a sweep file, in order; a config without ``sweep`` is returned alone.

    Each config records the labels it was made from in ``sweep_labels``, which ``select``
    filters on and removes.
    """
    path = Path(path)
    cfg = load_config(path)
    blocks = cfg.pop("sweep", None)
    if blocks is None:
        return [cfg]
    configs: list[dict[str, Any]] = []
    for block in blocks if isinstance(blocks, list) else [blocks]:
        block = dict(block)
        if "name" not in block:
            raise ValueError(f"{path}: a sweep block needs a name template.")
        name, description = block.pop("name"), block.pop("description", None)
        axes = [(axis, _axis(axis, spec, path.parent)) for axis, spec in block.items()]
        for combination in itertools.product(*(variants for _, variants in axes)):
            values = {axis: label for (axis, _), (label, _) in zip(axes, combination,
                                                                    strict=True)}
            run = copy.deepcopy(cfg)
            for _, settings in combination:
                run = _merged(run, copy.deepcopy(settings))
            run = _fill(run, values)
            run["name"] = _fill(str(name), {k: str(v) for k, v in values.items()})
            if description is not None:
                run["description"] = _fill(str(description),
                                           {k: str(v) for k, v in values.items()})
            run["sweep_labels"] = {k: str(v) for k, v in values.items()}
            configs.append(run)
    names = [c["name"] for c in configs]
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise ValueError(f"{path}: the sweep makes several runs named {', '.join(repeated)}.")
    return configs


def select(configs: list[dict[str, Any]], only: list[str] | None) -> list[dict[str, Any]]:
    """The configs whose labels match every ``axis=label`` filter (several labels of one axis
    match any of them), without their ``sweep_labels``."""
    wanted: dict[str, set[str]] = {}
    for item in only or []:
        axis, sep, label = item.partition("=")
        if not sep:
            raise ValueError(f"--only takes axis=label, got {item!r}.")
        wanted.setdefault(axis, set()).add(label)
    known = set().union(*(c.get("sweep_labels", {}) for c in configs)) if configs else set()
    unknown = sorted(set(wanted) - known)
    if unknown:
        raise ValueError(f"Unknown sweep axes {unknown}; the axes are {sorted(known)}.")
    return [{k: v for k, v in c.items() if k != "sweep_labels"} for c in configs
            if all(c.get("sweep_labels", {}).get(axis) in labels
                   for axis, labels in wanted.items())]


def run_sweep(configs: list[dict[str, Any]], *, out_dir: str | Path | None = None,
              processes: int = 1, allow_dirty: bool = False,
              final_test: bool = False) -> list[str]:
    """Run each config of ``select`` (a study if it has a ``study`` section) and return what
    was done.

    A run goes to ``<out_dir>/<name>`` (default the runs folder), and a run whose folder
    already holds ``metrics.json`` is skipped, so a stopped sweep resumes; a study resumes
    by itself.
    """
    from training.run import execute
    from training.study import run_study
    from utils import config as settings

    out_dir = Path(out_dir) if out_dir else settings.runs_dir()
    done = []
    for cfg in configs:
        if "study" in cfg:
            run_study(cfg, processes=processes, allow_dirty=allow_dirty)
            done.append(f"study {cfg['name']}")
            continue
        folder = out_dir / cfg["name"]
        if (folder / "metrics.json").exists():
            done.append(f"skipped {folder} (done)")
            continue
        execute(cfg, out=folder, processes=processes, allow_dirty=allow_dirty,
                final_test=final_test)
        done.append(f"run {folder}")
    return done
