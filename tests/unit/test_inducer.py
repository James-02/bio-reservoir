"""Unit tests for training.inducer (designed inducer-pattern tasks for slow sensing)."""

import numpy as np
import pytest

from training import inducer
from training.run import class_names, load_task, resolve


def _task(**settings):
    return inducer.resolve_task({"type": "inducer", "pattern": "freq", "n_per_class": 20,
                                 **settings}, seed=6337)


@pytest.mark.parametrize("pattern, classes", [
    ("freq", 5), ("ramp", 2), ("order", 2), ("shape", 4), ("xor", 2), ("parity3", 2),
    ("xor_tr", 2), ("parity3_tr", 2),
])
@pytest.mark.parametrize("condition", inducer.CONDITIONS)
def test_each_task_has_balanced_classes_of_four_hour_traces(pattern, classes, condition):
    X, Y, sources = load_task(_task(pattern=pattern, condition=condition))
    assert X.shape == (20 * classes, 240, 1) and Y.shape == (20 * classes, 1, classes)
    np.testing.assert_array_equal(Y.sum(axis=(0, 1)), np.full(classes, 20))
    assert sources == [] and len(inducer.CLASS_NAMES[pattern]) == classes


@pytest.mark.parametrize("pattern", ["freq", "order", "xor", "xor_tr", "parity3_tr"])
def test_the_mean_input_does_not_give_the_class_away(pattern):
    X, Y, _ = load_task(_task(pattern=pattern, n_per_class=200))
    means = [X[Y[:, 0, k] == 1].mean() for k in range(Y.shape[-1])]
    assert np.ptp(means) < 0.15 * np.mean(means)


def test_the_time_reversed_pair_has_the_same_values_in_another_order():
    X, Y, _ = load_task(_task(pattern="xor_tr", n_per_class=200))
    labels = Y[:, 0].argmax(1)
    for statistic in (np.mean, np.std, np.max):
        values = [statistic(X[labels == k], axis=(1, 2)).mean() for k in (0, 1)]
        assert values[0] == pytest.approx(values[1], rel=0.05)


def test_a_seed_gives_the_same_task_and_another_seed_another():
    first, again = (load_task(_task())[0] for _ in range(2))
    other = load_task(_task(seed=6338))[0]
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)


def test_the_device_delivers_held_levels_mixed_in_the_chamber():
    x = np.zeros(40)
    x[10:30] = 1.0
    delivered = inducer.device(x, np.random.default_rng(0))
    assert delivered.min() >= 0 and delivered.max() <= 1.6
    # First-order mixing (3 min): the pulse arrives late and leaves late.
    assert delivered[10] < 0.5 < delivered[29] and delivered[30] > 0.5 > delivered[39]


def test_the_traces_are_divided_by_the_pool_sd_or_scaled_by_a_number():
    assert load_task(_task())[0].std() == pytest.approx(1.0)
    unscaled = load_task(_task(scale=1.0))[0]
    np.testing.assert_allclose(load_task(_task(scale=2.0))[0], 2 * unscaled)


def test_a_run_config_resolves_an_inducer_task_without_the_ecg_settings():
    cfg = {"name": "inducer-run", "task": {"type": "inducer", "pattern": "order"},
           "reservoir": {"type": "none"}, "readout": {"classifier": "RidgeCV"}}
    assert resolve(cfg)["task"] == {"type": "inducer", "pattern": "order",
                                    "condition": "ideal", "n_per_class": 400, "scale": "sd",
                                    "image_delay": 0, "scaler": "none", "seed": 6337}


@pytest.mark.parametrize("condition", inducer.CONDITIONS)
def test_an_image_delay_appends_baseline_without_changing_the_pattern_or_its_scale(condition):
    plain = load_task(_task(pattern="order", condition=condition))[0]
    X, Y, _ = load_task(_task(pattern="order", condition=condition, image_delay=60))
    assert X.shape == (40, 300, 1)
    np.testing.assert_array_equal(Y, load_task(_task(pattern="order"))[1])
    assert X[:, :240].std() == pytest.approx(1.0)
    # After the pattern only the delivery noise is left (the pump quantises it upwards).
    assert abs(X[:, 270:].mean()) < 0.2 * plain.mean()
    assert X[:, 270:].std() < 0.25 * plain.std()


def test_without_an_image_delay_the_task_is_unchanged():
    np.testing.assert_array_equal(load_task(_task(image_delay=0))[0], load_task(_task())[0])


def test_the_classes_are_named_after_the_pattern():
    assert class_names(_task(pattern="order")) == ("A then B", "B then A")
    assert class_names({"labels": "binary"}) == ("Normal", "Arrhythmia")


@pytest.mark.parametrize("task, message", [
    ({"pattern": "chirp"}, "task.pattern"),
    ({"pattern": "freq", "condition": "lab"}, "task.condition"),
    ({"pattern": "freq", "rows": 100}, "no setting rows"),
    ({"pattern": "freq", "scaler": "sequence_zscore"}, "no setting scaler"),
    ({"pattern": "freq", "n_per_class": 3}, "at least 5"),
    ({"pattern": "freq", "scale": 0}, "task.scale"),
    ({"pattern": "freq", "image_delay": -5}, "task.image_delay"),
    ({"pattern": "freq", "image_delay": 2.5}, "task.image_delay"),
])
def test_invalid_inducer_tasks_are_refused(task, message):
    with pytest.raises(ValueError, match=message):
        resolve({"name": "bad", "task": {"type": "inducer", **task},
                 "reservoir": {"type": "none"}, "readout": {"classifier": "RidgeCV"}})


def test_unknown_task_types_and_held_out_splits_are_refused():
    run = {"name": "bad", "reservoir": {"type": "none"}, "readout": {"classifier": "RidgeCV"}}
    with pytest.raises(ValueError, match="task.type"):
        resolve({**run, "task": {"type": "mnist"}})
    with pytest.raises(ValueError, match="five-class"):
        resolve({**run, "task": {"type": "inducer", "pattern": "freq"},
                 "test": {"split": "validation-1"}})


def test_a_study_runs_each_pattern_on_its_own_traces(tmp_path, monkeypatch):
    import importlib

    from training import study
    from utils import provenance

    monkeypatch.setenv("BIORESERVOIR_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(provenance, "REGISTRY_FILE", tmp_path / "registry.jsonl")
    monkeypatch.setattr(provenance, "git_info",
                        lambda: {"commit": "abc", "describe": "abc", "dirty": False})
    monkeypatch.setattr(importlib.import_module("training.cross_validate"), "setup_logging",
                        lambda **kwargs: None)
    cfg = {"name": "inducer-study",
           "task": {"type": "inducer", "pattern": "order", "n_per_class": 10},
           "reservoir": {"type": "none"}, "readout": {"classifier": "RidgeCV"},
           "study": {"trials": 2, "sampler": {"type": "grid"},
                     "search": {"task.pattern": {"type": "categorical",
                                                 "choices": ["order", "shape"]}}}}
    trials = study.run_study(cfg).trials
    assert [t.state.name for t in trials] == ["COMPLETE", "COMPLETE"]
    assert len({t.user_attrs["data_pool_sha256"] for t in trials}) == 2
    assert all(0 <= t.value <= 1 for t in trials)
