"""Unit tests for training.study (Optuna studies defined in run configs)."""

import importlib
from pathlib import Path

import numpy as np
import optuna
import pytest
import yaml

from training import run as run_module
from training import study as study_module
from utils import provenance

cv_module = importlib.import_module("training.cross_validate")

RESERVOIR = {
    "units": 3, "warmup": 0, "rk4_substeps": 16,
    "input_scaling": 1.0, "input_connectivity": 1.0, "input_bias": False, "bias_scaling": 0.0,
    "rc_scaling": 0.1, "dde_scaling": 0.01, "cell_coupling": 1.0, "noise_in": 0.0,
    "noise_rc": 0.0, "output_variables": "He", "topology_type": "distance",
    "interaction_diameters": 3.0, "fade_alpha": 0.05, "sparsity": 1.0, "signed": False,
    "p_excite": 1.0,
}


def _pool(n_per_class=12, steps=8, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, steps)
    X = np.concatenate([t + 0.3 * rng.normal(size=(n_per_class, steps)),
                        1.0 - t + 0.3 * rng.normal(size=(n_per_class, steps))])[..., None]
    return X, np.eye(2)[np.repeat([0, 1], n_per_class)][:, None, :]


def _config(search=None, sampler=None, **study):
    return {
        "name": "unit-study", "seed": 3, "reservoir": {"params": dict(RESERVOIR)},
        "readout": {"classifier": "Ridge", "params": {"alpha": 1e-3}},
        "task": {"rows": 24}, "cv": {"folds": 3},
        "study": {"trials": 4,
                  "sampler": sampler or {"type": "tpe", "seed": 1, "n_startup_trials": 2},
                  "search": search or {
                      "reservoir.params.input_scaling": {"type": "float", "low": 0.1,
                                                         "high": 10.0, "log": True},
                      "readout.params.alpha": {"type": "categorical",
                                               "choices": [1e-3, 1.0]}},
                  **study},
    }


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("BIORESERVOIR_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(provenance, "REGISTRY_FILE", tmp_path / "registry.jsonl")
    monkeypatch.setattr(provenance, "git_info",
                        lambda: {"commit": "abc", "describe": "abc", "dirty": False})
    monkeypatch.setattr(run_module, "load_task", lambda task: (*_pool(), []))
    monkeypatch.setattr(cv_module, "setup_logging", lambda **kwargs: None)
    return tmp_path


def test_values_go_in_at_their_dotted_paths():
    cfg = study_module.apply({"reservoir": {"params": {"units": 3}}},
                             {"reservoir.params.units": 5, "cv.folds": 2})
    assert cfg == {"reservoir": {"params": {"units": 5}}, "cv": {"folds": 2}}


@pytest.mark.parametrize("path, message", [
    ("model.units", "must start with"),
    ("reservoir.params.unknown", "no such parameter"),
    ("reservoir.params.units.size", "goes through a value"),
    ("features.aggregations", "no setting"),
    ("cv.fold", "no setting"),
    ("reservoir.units", "no setting"),
])
def test_search_paths_the_run_would_ignore_are_refused(path, message):
    cfg = _config(search={path: {"type": "int", "low": 1, "high": 4}})
    with pytest.raises(ValueError, match=message):
        study_module.run_study(cfg)


@pytest.mark.parametrize("spec, study, message", [
    ({"type": "float", "low": 1.0}, {}, "low < high"),
    ({"type": "float", "low": 2.0, "high": 1.0}, {}, "low < high"),
    ({"type": "categorical"}, {}, "needs choices"),
    ({"type": "float", "low": 1.0, "high": 2.0}, {"sampler": {"type": "grid"}},
     "needs choices"),
    ({"type": "float", "low": 1.0, "high": 2.0}, {"metric": "auc"}, "not a run metric"),
])
def test_incomplete_specs_and_unknown_metrics_are_refused_before_any_trial(spec, study,
                                                                          message):
    cfg = _config(search={"reservoir.params.input_scaling": spec}, **study)
    with pytest.raises(ValueError, match=message):
        study_module.run_study(cfg)
    assert not list(Path(study_module.storage_url(cfg).removeprefix("sqlite:///")).parent
                    .glob("*.db"))


def test_a_grid_runs_every_point_once_and_records_resolved_configs():
    cfg = _config(search={"reservoir.params.input_scaling": {"type": "float",
                                                             "choices": [0.5, 2.0]},
                          "readout.params.alpha": {"type": "float", "choices": [1e-3, 1.0]}},
                  sampler={"type": "grid"}, trials=10)
    study = study_module.run_study(cfg)
    points = sorted(tuple(t.params.values()) for t in study.trials)
    assert points == [(0.5, 1e-3), (0.5, 1.0), (2.0, 1e-3), (2.0, 1.0)]
    for trial in study.trials:
        resolved = yaml.safe_load(trial.user_attrs["config"])
        assert resolved["reservoir"]["params"]["input_scaling"] == \
            trial.params["reservoir.params.input_scaling"]
        assert resolved["readout"]["params"]["alpha"] == trial.params["readout.params.alpha"]
        assert len(trial.user_attrs["fold_f1"]) == 3
        assert trial.value == pytest.approx(np.mean(trial.user_attrs["fold_f1"]))


def test_a_seeded_study_runs_the_same_trials_again(tmp_path):
    first = study_module.run_study(_config(), storage=f"sqlite:///{tmp_path / 'a.db'}")
    second = study_module.run_study(_config(), storage=f"sqlite:///{tmp_path / 'b.db'}")
    assert [t.params for t in first.trials] == [t.params for t in second.trials]
    assert [t.value for t in first.trials] == [t.value for t in second.trials]


def test_enqueued_points_run_first_and_a_study_resumes():
    point = {"reservoir.params.input_scaling": 1.5, "readout.params.alpha": 1.0}
    study = study_module.run_study(_config(enqueue=[point]), trials=2)
    assert study.trials[0].params == point and len(study.trials) == 2
    study = study_module.run_study(_config(enqueue=[point]), trials=3)
    assert len(study.trials) == 3
    assert sum(t.params == point for t in study.trials) == 1


def test_a_partly_given_point_is_enqueued_once():
    point = {"reservoir.params.input_scaling": 1.5}
    for trials in (1, 2, 3):
        study = study_module.run_study(_config(enqueue=[point]), trials=trials)
    assert sum(t.params["reservoir.params.input_scaling"] == 1.5 for t in study.trials) == 1


def test_a_study_keeps_its_definition_but_may_run_more_trials():
    study_module.run_study(_config(), trials=1)
    assert len(study_module.run_study(_config(trials=7), trials=2).trials) == 2
    changed = _config()
    changed["cv"] = {"folds": 4}
    with pytest.raises(ValueError, match="another config"):
        study_module.run_study(changed, trials=3)


def test_each_trial_runs_on_the_pool_of_its_own_task(monkeypatch):
    loaded = []

    def load_task(task):
        loaded.append(task["rows"])
        return (*_pool(n_per_class=task["rows"] // 2), [])

    monkeypatch.setattr(run_module, "load_task", load_task)
    cfg = _config(search={"task.rows": {"type": "int", "choices": [18, 24]}},
                  sampler={"type": "grid"})
    study = study_module.run_study(cfg)
    assert sorted(loaded) == [18, 24]
    hashes = {t.params["task.rows"]: t.user_attrs["data_pool_sha256"] for t in study.trials}
    assert len(set(hashes.values())) == 2


def test_the_best_trial_becomes_a_plain_run_config(tmp_path):
    study = study_module.run_study(_config())
    best = study_module.best_config(_config())
    assert "study" not in best and best["name"] == "unit-study-best"
    assert best["reservoir"]["params"]["input_scaling"] == \
        study.best_trial.params["reservoir.params.input_scaling"]
    run_dir = run_module.execute(best, out=tmp_path / "best", allow_dirty=True)
    assert (run_dir / "metrics.json").is_file()


def _readout_study(states_from, search):
    return {"name": "readout-study", "reservoir": {"states_from": str(states_from)},
            "readout": {"classifier": "Ridge", "params": {"alpha": 1e-3}},
            "study": {"trials": 2, "sampler": {"type": "grid"}, "search": search}}


def test_a_study_on_saved_states_refits_only_the_readout(tmp_path, monkeypatch):
    from training.compare import load_folds

    source_cfg = {k: v for k, v in _config().items() if k != "study"}
    source = run_module.execute(dict(source_cfg, outputs={"save_states": True}),
                                out=tmp_path / "source", allow_dirty=True)
    monkeypatch.setattr(run_module, "_simulate", lambda *args: pytest.fail("simulated"))
    search = {"readout.params.alpha": {"type": "float", "choices": [1e-3, 1.0]}}
    study = study_module.run_study(_readout_study(source, search))
    assert len(study.trials) == 2
    for trial in study.trials:
        cfg = run_module.resolve(study_module.apply(
            study_module.split(_readout_study(source, search))[0], trial.params))
        refit = run_module.refit_readout(cfg, load_folds(source, states=True))
        assert trial.value == pytest.approx(run_module.summarise(refit)["mean"]["f1"])


def test_a_study_on_saved_states_only_searches_the_readout(tmp_path):
    search = {"reservoir.params.input_scaling": {"type": "float", "low": 0.1, "high": 1.0}}
    with pytest.raises(ValueError, match="only search readout settings"):
        study_module.run_study(_readout_study(tmp_path, search))


def test_the_command_line_runs_a_study_and_writes_its_best_config(tmp_path, capsys):
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(_config()))
    run_module.main(["study", str(path), "--trials", "2"])
    assert "2 trials" in capsys.readouterr().out
    run_module.main(["study-best", str(path), "--out", str(tmp_path / "best.yaml")])
    assert yaml.safe_load((tmp_path / "best.yaml").read_text())["name"] == "unit-study-best"
    assert isinstance(optuna.load_study(study_name="unit-study",
                                        storage=study_module.storage_url(_config())),
                      optuna.Study)
