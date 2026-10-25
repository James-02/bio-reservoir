"""Unit tests for training.simulate (each beat simulated once) and the cross-validation of
cached features."""

import numpy as np
import pytest

from optimization.reservoirs import build_reservoir
from training import run as run_module
from training.cross_validate import _iter_folds, cross_validate_features
from training.observation import (
    calibration_gains,
    image_count,
    image_times,
    measure,
    observed_columns,
)
from training.pipeline import _aggregate_states, train
from training.simulate import beat_seed, reseed_noise, simulate_beats

PARAMS = {
    "reservoir_type": "bioreservoir", "units": 4, "warmup": 3, "rk4_substeps": 16,
    "input_scaling": 1.0, "input_connectivity": 1.0, "input_bias": False, "bias_scaling": 0.0,
    "rc_scaling": 0.1, "dde_scaling": 0.01, "cell_coupling": 1.0, "noise_in": 0.0,
    "noise_rc": 0.0, "output_variables": "IHe", "topology_type": "distance",
    "interaction_diameters": 3.0, "fade_alpha": 0.05, "sparsity": 1.0, "signed": False,
    "p_excite": 1.0,
}


def _beats(n=6, steps=10, seed=0):
    return np.random.default_rng(seed).normal(size=(n, steps, 1))


def _reservoir(params=PARAMS, seed=2):
    return build_reservoir(params, seed=seed)[0]


def test_beat_seeds_depend_only_on_the_run_seed_and_the_beat():
    assert beat_seed(3, 0) == beat_seed(3, 0)
    assert len({beat_seed(3, 0), beat_seed(3, 1), beat_seed(4, 0)}) == 3


def test_reseeding_keeps_the_noise_settings_and_repeats_the_draws():
    node = build_reservoir(dict(PARAMS, noise_in=0.1), seed=1)[0]
    keywords = {k: v for k, v in node.hypers["noise_generator"].keywords.items() if k != "rng"}
    draws = []
    for _ in range(2):
        reseed_noise(node, 7)
        generator = node.hypers["noise_generator"]
        assert {k: v for k, v in generator.keywords.items() if k != "rng"} == keywords
        draws.append(generator(dist="normal", shape=(5,), gain=1.0))
    np.testing.assert_array_equal(*draws)


@pytest.mark.parametrize("aggregation", ["final", "mean"])
def test_without_noise_the_features_are_those_of_the_per_fold_path(aggregation):
    X = _beats()
    features = simulate_beats(_reservoir(), X, noise_seed=5, aggregation=aggregation)
    per_fold = train(_reservoir(), X, readout_aggregation=aggregation)
    np.testing.assert_array_equal(features, np.vstack(per_fold))


def test_with_noise_a_beat_gets_the_same_features_in_any_batch():
    reservoir = _reservoir(dict(PARAMS, noise_in=0.2, noise_rc=0.2))
    X = _beats()
    whole = simulate_beats(reservoir, X, noise_seed=5)
    part = simulate_beats(reservoir, X[2:4], noise_seed=5, first_index=2)
    np.testing.assert_array_equal(whole[2:4], part)
    assert not np.array_equal(whole, simulate_beats(reservoir, X, noise_seed=6))


def test_beats_run_without_progress_bars(capfd):
    import reservoirpy

    reservoir = _reservoir()
    previous = reservoirpy.verbosity()
    reservoirpy.verbosity(1)
    try:
        capfd.readouterr()
        simulate_beats(reservoir, _beats(), noise_seed=5)
        assert "Running" not in capfd.readouterr().err
    finally:
        reservoirpy.verbosity(previous)


def test_beats_run_on_a_copy_of_the_warmed_up_reservoir(monkeypatch):
    from bioreservoir.node import BioReservoir

    warm_ups = []
    warm_up = BioReservoir._warmup_nodes
    monkeypatch.setattr(BioReservoir, "_warmup_nodes",
                        lambda node: warm_ups.append(node) or warm_up(node))
    reservoir = _reservoir(dict(PARAMS, noise_in=0.2))
    generator = reservoir.hypers["noise_generator"]
    simulate_beats(reservoir, _beats(), noise_seed=5)
    assert len(warm_ups) == 1 and warm_ups[0] is reservoir
    assert reservoir.hypers["noise_generator"] is generator


def test_workers_read_the_reservoir_from_a_file(monkeypatch):
    import multiprocessing

    initargs = []
    get_context = multiprocessing.get_context

    class Context:
        def __init__(self, method):
            self.context = get_context(method)

        def Pool(self, processes, **kwargs):
            initargs.append(kwargs["initargs"])
            return self.context.Pool(processes, **kwargs)

    monkeypatch.setattr(multiprocessing, "get_context", Context)
    reservoir = _reservoir(dict(PARAMS, noise_in=0.2))
    X = _beats()
    np.testing.assert_array_equal(simulate_beats(reservoir, X, noise_seed=5, processes=2),
                                  simulate_beats(reservoir, X, noise_seed=5))
    # sent with each worker's start-up, the reservoir made the workers start one at a time
    assert [type(arg) for args in initargs for arg in args] == [str]


def test_cross_validated_features_test_every_beat_once_with_a_fitted_readout_per_fold():
    from sklearn.linear_model import RidgeClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    labels = np.repeat([0, 1], 15)
    features = rng.normal(size=(30, 6)) + labels[:, None]
    Y = np.eye(2)[labels][:, None, :]
    readout = run_module.make_readout({"classifier": "Ridge", "params": {"alpha": 1.0},
                                       "scale": True})
    result, artifacts = cross_validate_features(features, Y, readout, folds=3, seed=4,
                                                save_states=True)
    tested = np.concatenate([a.test_index for a in artifacts])
    assert sorted(tested) == list(range(30))
    for artifact, fold in zip(artifacts, _iter_folds(features, Y, folds=3, seed=4),
                              strict=True):
        model = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))
        model.fit(features[fold.train_idx], labels[fold.train_idx])
        np.testing.assert_array_equal(np.argmax(artifact.Y_pred.reshape(-1, 2), axis=1),
                                      model.predict(features[fold.val_idx]))
        assert artifact.train_states.shape == (len(fold.train_idx), 1, 6)
    assert result["f1"] == pytest.approx(np.mean([a.metrics.f1 for a in artifacts]))


def test_test_gains_multiply_only_the_test_rows_of_their_fold():
    rng = np.random.default_rng(1)
    labels = np.repeat([0, 1], 9)
    features = rng.normal(size=(18, 4)) + labels[:, None]
    Y = np.eye(2)[labels][:, None, :]
    readout = run_module.make_readout({"classifier": "Ridge", "params": {}, "scale": True})
    gains = [np.full(4, 1.0 + fold) for fold in range(3)]
    _, artifacts = cross_validate_features(features, Y, readout, folds=3, seed=0,
                                           save_states=True, test_gains=gains)
    for artifact, fold in zip(artifacts, _iter_folds(features, Y, folds=3, seed=0),
                              strict=True):
        np.testing.assert_array_equal(artifact.test_states[:, 0],
                                      features[fold.val_idx] * gains[fold.fold_index])
        np.testing.assert_array_equal(artifact.train_states[:, 0], features[fold.train_idx])


def test_a_reference_divides_every_row_and_cancels_the_test_gains():
    rng = np.random.default_rng(2)
    labels = np.repeat([0, 1], 9)
    features = rng.uniform(1, 2, size=(18, 4)) + labels[:, None]
    reference = rng.uniform(0.5, 2, size=4)
    Y = np.eye(2)[labels][:, None, :]
    readout = run_module.make_readout({"classifier": "Ridge", "params": {}, "scale": True})
    gains = [1.0 + 0.3 * rng.standard_normal(4) for _ in range(3)]
    runs = [cross_validate_features(features, Y, readout, folds=3, seed=0, save_states=True,
                                    test_gains=g, reference=reference)[1]
            for g in (gains, None)]
    for with_gains, without, fold in zip(*runs, _iter_folds(features, Y, folds=3, seed=0),
                                         strict=True):
        np.testing.assert_array_equal(with_gains.train_states[:, 0],
                                      features[fold.train_idx] / reference)
        np.testing.assert_allclose(with_gains.test_states[:, 0],
                                   features[fold.val_idx] / reference, rtol=1e-15)
        np.testing.assert_array_equal(with_gains.Y_pred, without.Y_pred)


def test_images_are_evenly_spaced_and_end_at_the_last_timestep():
    states = np.arange(187 * 2).reshape(187, 2)
    np.testing.assert_array_equal(_aggregate_states(states, "images:4", None),
                                  states[[45, 92, 139, 186]].reshape(1, -1))
    np.testing.assert_array_equal(_aggregate_states(states, "images:1", None), states[-1:])
    with pytest.raises(ValueError, match="between 1 and 187"):
        _aggregate_states(states, "images:188", None)
    assert (image_count("images:4"), image_count("final")) == (4, 1)


def test_images_at_chosen_times_take_the_state_after_that_many_steps():
    states = np.arange(187 * 2).reshape(187, 2)
    np.testing.assert_array_equal(_aggregate_states(states, "images_at:46,93,140,187", None),
                                  _aggregate_states(states, "images:4", None))
    np.testing.assert_array_equal(_aggregate_states(states, "images_at:187", None),
                                  _aggregate_states(states, "final", None))
    np.testing.assert_array_equal(_aggregate_states(states, "images_at:1,155", None),
                                  states[[0, 154]].reshape(1, -1))
    with pytest.raises(ValueError, match="187 steps long"):
        _aggregate_states(states, "images_at:155,188", None)
    assert image_times("images_at:155,187") == [155, 187]
    assert (image_count("images_at:155,187"), image_count("images_at:187")) == (2, 1)


@pytest.mark.parametrize("mode, message", [
    ("images_at:187,155", "strictly increasing"),
    ("images_at:155,155", "strictly increasing"),
    ("images_at:0,187", "1 or more"),
    ("images_at:", "Invalid images_at"),
    ("images_at:1.5", "Invalid images_at"),
    ("images_at:-3,4", "Invalid images_at"),
])
def test_image_times_are_whole_increasing_minutes(mode, message):
    with pytest.raises(ValueError, match=message):
        image_times(mode)


def test_observed_columns_take_the_variables_of_every_colony():
    np.testing.assert_array_equal(observed_columns(("I", "He"), ["I"], 3), [0, 2, 4])
    np.testing.assert_array_equal(observed_columns(("I", "He"), ["He", "I"], 2), [1, 0, 3, 2])
    with pytest.raises(ValueError, match="output variables"):
        observed_columns(("I", "He"), ["A"], 3)
    columns = observed_columns(("I", "He"), ["I"], PARAMS["units"])
    X = _beats()
    reservoir = _reservoir()
    np.testing.assert_array_equal(
        simulate_beats(reservoir, X, noise_seed=5, aggregation="images:3", columns=columns),
        simulate_beats(reservoir, X, noise_seed=5, aggregation="images:3").reshape(
            len(X), 3, -1)[:, :, columns].reshape(len(X), -1))


def test_each_beat_is_measured_with_its_own_draws():
    features = np.random.default_rng(0).uniform(1, 2, size=(400, 50))
    assert measure(features, 0.0, seed=1) is features
    measured = measure(features, 0.1, seed=1)
    np.testing.assert_array_equal(measured[5:9], measure(features[5:9], 0.1, seed=1,
                                                         first_index=5))
    assert np.std(measured / features) == pytest.approx(0.1, rel=0.05)
    assert not np.array_equal(measured, measure(features, 0.1, seed=2))


def test_a_channel_has_one_calibration_gain_in_every_image():
    gains = calibration_gains(1000, 3, 0.05, seed=1, fold=0).reshape(3, 1000)
    np.testing.assert_array_equal(gains[0], gains[2])
    assert np.std(gains[0]) == pytest.approx(0.05, rel=0.1)
    assert not np.array_equal(gains[0], calibration_gains(1000, 1, 0.05, seed=1, fold=1))
