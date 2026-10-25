"""Checks of the campaign configs in ``experiments/v0.2/``.

Every config resolves; every search path names a setting that its run reads; names are
unique and start with their folder's prefix (``P<k>-`` in ``phase-<k>/``, ``tracking-`` in
``tracking/``); every study is seeded, grids run each point once, and no trial of a coupled
array is refused by the RK4 stability check, which would stop its study.
"""

import itertools
import math
import re
from pathlib import Path

import pytest

from optimization.reservoirs import build_reservoir
from training import study as study_module
from training.run import load_config, resolve

ROOT = Path(__file__).resolve().parents[2] / "experiments" / "v0.2"
CONFIGS = sorted(ROOT.rglob("*.yaml"))
STUDIES = [p for p in CONFIGS if "study" in load_config(p)]


def _ids(paths):
    return [str(p.relative_to(ROOT)) for p in paths]


def _prefix(path: Path) -> str:
    folder = path.relative_to(ROOT).parts[0]
    phase = re.fullmatch(r"phase-(\d)", folder)
    return f"P{phase[1]}-" if phase else re.escape(folder) + "-"


def _get(cfg, path):
    for key in path.split("."):
        cfg = cfg[key]
    return cfg


def _read_by_builder(reservoir, key):
    """Whether ``build_reservoir`` reads reservoir parameter ``key`` with these settings: some
    parameters act only with one coupling or layout (optimization/reservoirs.py)."""
    params = reservoir["params"]
    if reservoir["type"] != "bioreservoir":
        return True
    coupling = params.get("coupling", "abstract")
    topology = params.get("topology_type", "distance")
    if key == "diffusion_rate":
        return coupling == "diffusion"
    if key in ("rc_scaling", "noise_rc"):
        return coupling == "abstract"
    if key in ("interaction_diameters", "fade_alpha"):
        return coupling != "none" and topology == "distance"
    if key == "rc_connectivity":
        return coupling != "none" and topology == "random"
    if key == "param_heterogeneity_seed":
        return float(params.get("param_heterogeneity_cv", 0.0)) > 0
    return True


def _points(study):
    """The points a study runs: every grid point, or the corners of a search and its
    enqueued points."""
    search = study["search"]
    if study["sampler"]["type"] == "grid":
        return [dict(zip(search, values, strict=True))
                for values in itertools.product(*(spec["choices"] for spec in search.values()))]
    corners = [{path: spec["choices"][i] if "choices" in spec else spec[end]
                for path, spec in search.items()} for i, end in ((0, "low"), (-1, "high"))]
    return corners + list(study.get("enqueue") or [])


def test_the_campaign_configs_are_found():
    assert {p.relative_to(ROOT).parts[0] for p in CONFIGS} >= {"tracking", "phase-1", "phase-2"}


@pytest.mark.parametrize("path", CONFIGS, ids=_ids(CONFIGS))
def test_each_config_resolves_and_never_scores_the_test_split(path):
    cfg = load_config(path)
    if "study" in cfg:
        run_cfg, study = study_module.split(cfg)
        study_module.check(run_cfg, study)
    else:
        run_cfg = cfg
    resolved = resolve(run_cfg)
    assert resolved.get("test", {}).get("split") != "test"


@pytest.mark.parametrize("path", STUDIES, ids=_ids(STUDIES))
def test_every_point_puts_its_values_where_the_run_reads_them(path):
    run_cfg, study = study_module.split(load_config(path))
    for point in _points(study):
        resolved = resolve(study_module.apply(run_cfg, point))
        for key, value in point.items():
            assert _get(resolved, key) == value, key
            if key.startswith("reservoir.params."):
                assert _read_by_builder(resolved["reservoir"], key.rsplit(".", 1)[-1]), key


def test_names_are_unique_and_carry_their_experiment_prefix():
    names = {}
    for path in CONFIGS:
        cfg = load_config(path)
        for name in {cfg["name"], study_module.name(cfg) if "study" in cfg else cfg["name"]}:
            assert re.match(_prefix(path), name), (path, name)
            assert name not in names, (path, names.get(name))
            names[name] = path


@pytest.mark.parametrize("path", STUDIES, ids=_ids(STUDIES))
def test_each_study_is_seeded_and_runs_its_intended_trials(path):
    study = load_config(path)["study"]
    sampler = study["sampler"]
    assert "seed" in sampler
    if sampler["type"] == "grid":
        for spec in study["search"].values():
            assert len(set(map(repr, spec["choices"]))) == len(spec["choices"])
        assert study["trials"] == math.prod(len(s["choices"]) for s in study["search"].values())
        return
    assert study["trials"] > sampler.get("n_startup_trials", 10)
    for point in study.get("enqueue") or []:
        assert set(point) == set(study["search"])
        for key, value in point.items():
            spec = study["search"][key]
            assert spec["low"] <= value <= spec["high"], key


def _stiffest_points(study):
    """Points of a study with the fastest AHL exchange: the largest coupling rate and
    heterogeneity, the fewest substeps and the shortest Gaussian range, at each array size
    and reservoir seed of the study."""
    extreme = {"diffusion_rate": max, "param_heterogeneity_cv": max, "rk4_substeps": min,
               "interaction_diameters": min}
    fixed, each = {}, {}
    for path, spec in study["search"].items():
        values = spec.get("choices") or [spec["low"], spec["high"]]
        key = path.rsplit(".", 1)[-1]
        if key in extreme:
            fixed[path] = extreme[key](values)
        elif path == "reservoir.seed" or key == "units":
            each[path] = values
    return [dict(fixed, **dict(zip(each, values, strict=True)))
            for values in itertools.product(*each.values())]


COUPLED = [p for p in STUDIES
           if resolve(study_module.split(load_config(p))[0])["reservoir"].get("params", {})
           .get("coupling") == "diffusion"]


@pytest.mark.parametrize("path", COUPLED, ids=_ids(COUPLED))
def test_no_trial_is_refused_by_the_rk4_stability_check(path):
    run_cfg, study = study_module.split(load_config(path))
    for point in _stiffest_points(study):
        reservoir = resolve(study_module.apply(run_cfg, point))["reservoir"]
        # Building without warm-up runs the check without simulating.
        build_reservoir(dict(reservoir["params"], warmup=0, reservoir_type=reservoir["type"]),
                        seed=reservoir["seed"])
