"""Sweeps (training.sweep): one config file that expands into many runs or studies."""

import json

import pytest
import yaml

from training import run as run_module
from training import study as study_module
from training.sweep import expand, run_sweep, select

RAW = {"task": {"labels": "five-class", "rows": 100}, "reservoir": {"type": "none"},
       "readout": {"classifier": "RidgeCV"}, "cv": {"folds": 2}}


def _write(path, cfg):
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def test_a_config_without_a_sweep_is_returned_alone(tmp_path):
    path = _write(tmp_path / "one.yaml", {"name": "one", **RAW})
    assert expand(path) == [{"name": "one", **RAW}]


def test_every_combination_of_the_axes_in_order(tmp_path):
    _write(tmp_path / "model-b.yaml", {"readout": {"classifier": "RF"}, "seed": 9})
    path = _write(tmp_path / "sweep.yaml", {**RAW, "sweep": {
        "name": "S-{task}-{model}-s{seed}",
        "description": "{model} on {task}",
        "model": {"a": {}, "b": "model-b.yaml"},
        "task": {"five": {}, "two": {"task.labels": "binary"}},
        "seed": {"reservoir.seed": [1, 2]},
    }})
    configs = expand(path)
    assert [c["name"] for c in configs] == [
        "S-five-a-s1", "S-five-a-s2", "S-two-a-s1", "S-two-a-s2",
        "S-five-b-s1", "S-five-b-s2", "S-two-b-s1", "S-two-b-s2"]
    last = configs[-1]
    assert last["description"] == "b on two"
    assert last["sweep_labels"] == {"model": "b", "task": "two", "seed": "2"}
    assert last["task"] == {"labels": "binary", "rows": 100}
    assert last["reservoir"] == {"type": "none", "seed": 2}
    assert last["readout"] == {"classifier": "RF"} and last["seed"] == 9
    assert configs[0]["readout"] == {"classifier": "RidgeCV"}


def test_a_variant_can_start_from_a_base_and_later_axes_win(tmp_path):
    _write(tmp_path / "design.yaml", {"name": "design", **RAW, "cv": {"folds": 5}})
    path = _write(tmp_path / "sweep.yaml", {"sweep": {
        "name": "S-{model}-{folds}",
        "model": {"d": {"base": "design.yaml", "outputs": {"save_states": True}}},
        "folds": {"cv.folds": [3]},
    }})
    (cfg,) = expand(path)
    assert cfg["cv"] == {"folds": 3}
    assert cfg["outputs"] == {"save_states": True}
    assert cfg["task"] == RAW["task"]


def test_text_settings_use_the_labels_and_other_braces_stay(tmp_path):
    path = _write(tmp_path / "sweep.yaml", {**RAW, "description": "{model} {other}", "sweep": {
        "name": "R-{model}-s{seed}",
        "model": {"m": {"reservoir": {"states_from": "S-{model}-s{seed}"},
                         "cv": {"folds": "{seed}"}}},
        "seed": [4, 5],
    }})
    first, second = expand(path)
    assert first["reservoir"] == {"type": "none", "states_from": "S-m-s4"}
    assert first["cv"] == {"folds": 4} and second["cv"] == {"folds": 5}
    assert second["name"] == "R-m-s5" and "seed" not in second
    assert first["description"] == "m {other}"


def test_blocks_are_concatenated_and_share_axes_through_anchors(tmp_path):
    (tmp_path / "sweep.yaml").write_text(
        "task: {labels: five-class}\nreservoir: {type: none}\n"
        "readout: {classifier: RidgeCV}\n"
        "sweep:\n"
        "  - name: A-{model}-s{seed}\n"
        "    model: &models {x: {}, y: {cv.folds: 3}}\n"
        "    seed: {reservoir.seed: [1, 2]}\n"
        "  - name: B-{model}\n"
        "    model: *models\n")
    assert [c["name"] for c in expand(tmp_path / "sweep.yaml")] == [
        "A-x-s1", "A-x-s2", "A-y-s1", "A-y-s2", "B-x", "B-y"]


@pytest.mark.parametrize("sweep, message", [
    ({"model": {"a": {}}}, "needs a name template"),
    ({"name": "S", "model": {"a": {}, "b": {}}}, "several runs named S"),
    ({"name": "S-{model}", "model": []}, "must map labels to variants"),
    ({"name": "S-{model}", "model": {"a": 3}}, "must be a config file or a mapping"),
])
def test_a_malformed_sweep_is_refused(tmp_path, sweep, message):
    path = _write(tmp_path / "sweep.yaml", {**RAW, "sweep": sweep})
    with pytest.raises(ValueError, match=message):
        expand(path)


def test_select_filters_on_labels_and_drops_them(tmp_path):
    path = _write(tmp_path / "sweep.yaml", {**RAW, "sweep": {
        "name": "S-{task}-s{seed}", "task": {"a": {}, "b": {}, "c": {}},
        "seed": {"reservoir.seed": [1, 2]}}})
    configs = expand(path)
    chosen = select(configs, ["task=a", "task=c", "seed=2"])
    assert [c["name"] for c in chosen] == ["S-a-s2", "S-c-s2"]
    assert all("sweep_labels" not in c for c in chosen)
    assert len(select(configs, None)) == 6
    with pytest.raises(ValueError, match="Unknown sweep axes"):
        select(configs, ["model=a"])
    with pytest.raises(ValueError, match="axis=label"):
        select(configs, ["task"])


def test_run_sweep_runs_studies_and_skips_finished_runs(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(run_module, "execute",
                        lambda cfg, out, **kw: calls.append(("run", cfg["name"], out, kw)))
    monkeypatch.setattr(study_module, "run_study",
                        lambda cfg, **kw: calls.append(("study", cfg["name"])))
    done = tmp_path / "runs" / "R-done"
    done.mkdir(parents=True)
    (done / "metrics.json").write_text(json.dumps({}))
    configs = [{"name": "R-done"}, {"name": "R-new"}, {"name": "S-study", "study": {}}]
    lines = run_sweep(configs, out_dir=tmp_path / "runs", processes=3, final_test=True)
    assert calls == [
        ("run", "R-new", tmp_path / "runs" / "R-new",
         {"processes": 3, "allow_dirty": False, "final_test": True}),
        ("study", "S-study")]
    assert lines[0].startswith("skipped")


def test_the_cli_lists_and_resolves_a_sweep_and_refuses_to_run_it_as_one(tmp_path, capsys):
    path = _write(tmp_path / "sweep.yaml", {**RAW, "sweep": {
        "name": "S-s{seed}", "seed": {"seed": [1, 2]}}})
    run_module.main(["sweep", str(path), "--list"])
    assert capsys.readouterr().out.split("\n")[:2] == ["run  S-s1", "run  S-s2"]
    run_module.main(["resolve", str(path), "--only", "seed=2"])
    resolved = yaml.safe_load(capsys.readouterr().out)
    assert resolved["name"] == "S-s2" and resolved["seed"] == 2
    with pytest.raises(SystemExit, match="is a sweep"):
        run_module.main(["run", str(path)])
