"""Unit tests for the linear surrogate control (training.surrogate)."""

import dataclasses
import importlib
import json

import numpy as np
import pytest

from training import run as run_module
from training import study as study_module
from training import surrogate as surrogate_module
from training.compare import load_folds
from training.cross_validate import _iter_folds
from training.observation import measure
from utils import provenance
from utils.preprocessing import standardize_data

cv_module = importlib.import_module("training.cross_validate")


def _pool(n_per_class=12, steps=8, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, steps)
    X = np.concatenate([t + 0.3 * rng.normal(size=(n_per_class, steps)),
                        1.0 - t + 0.3 * rng.normal(size=(n_per_class, steps))])[..., None]
    return X, np.eye(2)[np.repeat([0, 1], n_per_class)][:, None, :]


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Results and registry under tmp_path; a synthetic two-class pool of 8-step beats."""
    monkeypatch.setenv("BIORESERVOIR_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("BIORESERVOIR_ARCHIVE_DIR", str(tmp_path / "archive"))
    monkeypatch.setattr(provenance, "REGISTRY_FILE", tmp_path / "registry.jsonl")
    monkeypatch.setattr(provenance, "git_info",
                        lambda: {"commit": "abc", "describe": "abc", "dirty": False})
    monkeypatch.setattr(run_module, "load_task", lambda task: (*_pool(), []))
    monkeypatch.setattr(cv_module, "setup_logging", lambda **kwargs: None)
    return tmp_path


READOUT = {"classifier": "RidgeCV", "scale": True}


def _source(tmp_path, name="source", **sections):
    """A saved run of a small nonlinear feature map, whose states a surrogate imitates."""
    cfg = {"name": name, "seed": 3,
           "reservoir": {"type": "random_features", "params": {"units": 10, "length": 8}},
           "readout": READOUT, "task": {"rows": 24}, "cv": {"folds": 3},
           "outputs": {"save_states": True}, **sections}
    return run_module.execute(cfg, out=tmp_path / name, allow_dirty=True)


def _surrogate(source, **sections):
    return {"name": "surrogate", "seed": 3,
            "reservoir": {"states_from": str(source), "surrogate": "linear"},
            "readout": READOUT, "outputs": {"save_states": True}, **sections}


def _states(run_dir, kind):
    return [np.load(p, allow_pickle=True)[f"{kind}_states"][:, 0]
            for p in sorted(run_dir.glob("fold-*.npz"))]


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------

def test_the_fit_recovers_a_linear_map_and_reports_a_poor_fit_to_a_nonlinear_one():
    rng = np.random.default_rng(37)
    beats = rng.normal(size=(120, 7))
    states = beats @ rng.normal(size=(7, 12)) + 20.0
    _, test, info = surrogate_module.linear_surrogate(beats[:80], states[:80], beats[80:],
                                                      states[80:])
    assert info["r2_median"] > 0.999
    np.testing.assert_allclose(test, states[80:], rtol=1e-3)
    _, _, info = surrogate_module.linear_surrogate(beats[:80], beats[:80, :3] ** 2,
                                                   beats[80:], beats[80:, :3] ** 2)
    assert info["r2_median"] < 0.5


def test_the_test_beats_features_do_not_enter_the_fit():
    rng = np.random.default_rng(1)
    beats = rng.normal(size=(60, 9))
    states = np.tanh(beats @ rng.normal(size=(9, 5)))
    train, test, _ = surrogate_module.linear_surrogate(beats[:40], states[:40], beats[40:],
                                                       states[40:])
    permuted = states[40:][rng.permutation(20)]
    train2, test2, _ = surrogate_module.linear_surrogate(beats[:40], states[:40], beats[40:],
                                                         permuted + 1000.0)
    np.testing.assert_array_equal(train, train2)
    np.testing.assert_array_equal(test, test2)


def test_permuted_test_features_of_a_saved_run_give_the_same_surrogate(tmp_path):
    source = _source(tmp_path)
    folds = load_folds(source, states=True)
    rng = np.random.default_rng(0)
    shuffled = [dataclasses.replace(f, test_states=f.test_states[rng.permutation(
        len(f.test_states))] * 3.0) for f in folds]
    cfg = run_module.resolve(_surrogate(source))
    for a, b in zip(surrogate_module.surrogate_folds(cfg, folds),
                    surrogate_module.surrogate_folds(cfg, shuffled), strict=True):
        np.testing.assert_array_equal(a.train_states, b.train_states)
        np.testing.assert_array_equal(a.test_states, b.test_states)


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def test_a_surrogate_run_fits_each_fold_on_its_training_beats(tmp_path):
    source = _source(tmp_path)
    run_dir = run_module.execute(_surrogate(source), out=tmp_path / "surrogate",
                                 allow_dirty=True)
    X, Y = _pool()
    beats = standardize_data(X, X[:1], scaler_type="sequence_zscore")[0].reshape(len(X), -1)
    folds = _iter_folds(X, Y, folds=3, seed=3)
    for fold, real, train, test in zip(folds, _states(source, "train"),
                                       _states(run_dir, "train"), _states(run_dir, "test"),
                                       strict=True):
        fitted = surrogate_module.fit_surrogate(beats[fold.train_idx], real)
        np.testing.assert_allclose(train, surrogate_module.predict_surrogate(
            fitted, beats[fold.train_idx]), rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(test, surrogate_module.predict_surrogate(
            fitted, beats[fold.val_idx]), rtol=1e-10, atol=1e-12)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert len(metrics["surrogate"]) == 3
    for fit in metrics["surrogate"]:
        assert fit["kind"] == "linear" and fit["measurement_noise"] == 0.0
        assert fit["alpha"] in surrogate_module.ALPHAS
        assert -1.0 < fit["r2_median"] <= 1.0


def test_measurement_noise_measures_the_surrogate_like_the_reservoir(tmp_path):
    source = _source(tmp_path)
    clean = run_module.execute(_surrogate(source), out=tmp_path / "clean", allow_dirty=True)
    noisy = run_module.execute(_surrogate(source, features={"measurement_noise": 0.05}),
                               out=tmp_path / "noisy", allow_dirty=True)
    # The draws a run of the reservoir with seed 3 multiplies the pool's features by.
    gains = measure(np.ones((24, 10)), 0.05, 3)
    folds = _iter_folds(*_pool(), folds=3, seed=3)
    for fold, c, n in zip(folds, _states(clean, "test"), _states(noisy, "test"), strict=True):
        np.testing.assert_allclose(n / c, gains[fold.val_idx], rtol=1e-9)
    for fold, c, n in zip(folds, _states(clean, "train"), _states(noisy, "train"),
                          strict=True):
        np.testing.assert_allclose(n / c, gains[fold.train_idx], rtol=1e-9)


def test_a_study_on_saved_states_uses_the_surrogate(tmp_path):
    source = _source(tmp_path)
    cfg = dict(_surrogate(source), readout={"classifier": "Ridge", "params": {"alpha": 1.0}},
               study={"trials": 2, "sampler": {"type": "grid"},
                      "search": {"readout.params.alpha": {"type": "float",
                                                          "choices": [1e-3, 1.0]}}})
    study = study_module.run_study(cfg)
    for trial in study.trials:
        resolved = run_module.resolve(study_module.apply(study_module.split(cfg)[0],
                                                         trial.params))
        assert resolved["reservoir"]["surrogate"] == "linear"
        refit = run_module.refit_readout(resolved, load_folds(source, states=True))
        assert all("surrogate" in a.model_hypers for a in refit)
        assert trial.value == pytest.approx(run_module.summarise(refit)["mean"]["f1"])


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reservoir, features, message", [
    ({"surrogate": "linear", "type": "random_features", "params": {"units": 3}}, {},
     "needs reservoir.states_from"),
    ({"states_from": "run", "surrogate": "quadratic"}, {}, "Unknown reservoir.surrogate"),
    ({"states_from": "run", "surrogate": "linear"}, {"calibration_error": 0.01}, "refit"),
    ({"states_from": "run", "surrogate": "linear"}, {"variables": ["I"]}, "refit"),
    ({"states_from": "run"}, {"measurement_noise": 0.01}, "refit"),
])
def test_invalid_surrogate_configs_are_refused(reservoir, features, message):
    cfg = {"name": "x", "reservoir": reservoir, "readout": READOUT, "features": features}
    with pytest.raises(ValueError, match=message):
        run_module.resolve(cfg)


def test_a_surrogate_needs_a_run_that_simulated_its_reservoir(tmp_path):
    source = _source(tmp_path)
    refit = run_module.execute({"name": "refit", "reservoir": {"states_from": str(source)},
                                "readout": READOUT, "outputs": {"save_states": True}},
                               out=tmp_path / "refit", allow_dirty=True)
    with pytest.raises(ValueError, match="cross-validated run that simulated"):
        run_module.execute(_surrogate(refit), out=tmp_path / "surrogate", allow_dirty=True)


def test_a_surrogate_refuses_a_pool_that_differs_from_its_source(tmp_path, monkeypatch):
    source = _source(tmp_path)
    monkeypatch.setattr(run_module, "load_task", lambda task: (*_pool(seed=1), []))
    with pytest.raises(ValueError, match="checksum"):
        run_module.execute(_surrogate(source), out=tmp_path / "surrogate", allow_dirty=True)
