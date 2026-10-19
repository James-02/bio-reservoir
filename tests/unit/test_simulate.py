"""Unit tests for training.simulate (each beat simulated once) and the cross-validation of
cached features."""

import numpy as np
import pytest

from optimization.reservoirs import build_reservoir
from training import run as run_module
from training.cross_validate import _iter_folds, cross_validate_features
from training.pipeline import train
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
