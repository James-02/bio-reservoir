"""Unit tests for raw-beat runs (reservoir: {type: none}): the readout classifies the scaled
beat itself."""

import json

import numpy as np
import pytest
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold

from training import run as run_module
from training import study as study_module
from utils import provenance
from utils.preprocessing import standardize_data

RESERVOIR = {
    "units": 3, "warmup": 0, "rk4_substeps": 16, "input_scaling": 1.0,
    "input_connectivity": 1.0, "input_bias": False, "bias_scaling": 0.0, "rc_scaling": 0.1,
    "dde_scaling": 0.01, "cell_coupling": 1.0, "noise_in": 0.0, "noise_rc": 0.0,
    "output_variables": "He", "topology_type": "distance", "interaction_diameters": 3.0,
    "fade_alpha": 0.05, "sparsity": 1.0, "signed": False, "p_excite": 1.0,
}


def _three_classes(n_per_class=10, steps=8, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, steps)
    shapes = (t, 1.0 - t, np.sin(np.pi * t))
    X = np.concatenate([2.0 * s + 1.0 + 0.3 * rng.normal(size=(n_per_class, steps))
                        for s in shapes])
    return X[..., None], np.eye(3)[np.repeat([0, 1, 2], n_per_class)][:, None, :]


POOL = _three_classes()


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Results and registry under tmp_path; a synthetic three-class pool."""
    monkeypatch.setenv("BIORESERVOIR_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("BIORESERVOIR_ARCHIVE_DIR", str(tmp_path / "archive"))
    monkeypatch.setattr(provenance, "REGISTRY_FILE", tmp_path / "registry.jsonl")
    monkeypatch.setattr(run_module, "load_task", lambda task: (*POOL, []))
    return tmp_path


def _raw_config(**sections):
    cfg = {"name": "raw", "seed": 3, "reservoir": {"type": "none"},
           "readout": {"classifier": "Ridge", "params": {"alpha": 1.0}},
           "task": {"rows": 30}, "cv": {"folds": 3}}
    cfg.update(sections)
    return cfg


def _execute(cfg, out, **kwargs):
    return run_module.execute(cfg, out=out, allow_dirty=True, **kwargs)


def _folds(run_dir):
    return [np.load(p, allow_pickle=True) for p in sorted(run_dir.glob("fold-*.npz"))]


class TestResolve:
    def test_the_raw_beat_has_no_reservoir_settings(self):
        resolved = run_module.resolve(_raw_config())
        assert resolved["reservoir"] == {"type": "none"}
        assert resolved["cv"]["simulate"] == "once"
        assert run_module.resolve(resolved) == resolved

    @pytest.mark.parametrize("reservoir", [
        {"type": "none", "params": {"units": 10}},
        {"type": "none", "seed": 4},
        {"type": "none", "study": "P2-distance"},
    ])
    def test_reservoir_settings_are_refused(self, reservoir):
        with pytest.raises(ValueError, match="reservoir settings do not apply"):
            run_module.resolve(_raw_config(reservoir=reservoir))

    @pytest.mark.parametrize("features, key", [
        ({"aggregation": "mean"}, "aggregation"),
        ({"window": 5}, "window"),
        ({"variables": ["I"]}, "variables"),
        ({"measurement_noise": 0.01}, "measurement_noise"),
        ({"calibration_error": 0.01}, "calibration_error"),
    ])
    def test_observation_settings_are_refused(self, features, key):
        with pytest.raises(ValueError, match=f"observation settings do not apply.*{key}"):
            run_module.resolve(_raw_config(features=features))

    def test_the_default_observation_may_be_written_out(self):
        run_module.resolve(_raw_config(features={"aggregation": "final", "window": 1,
                                                 "measurement_noise": 0}))

    def test_per_fold_simulation_is_refused(self):
        with pytest.raises(ValueError, match="cv.simulate: once"):
            run_module.resolve(_raw_config(cv={"folds": 3, "simulate": "per_fold"}))
        with pytest.raises(ValueError, match="cv.simulate: once"):
            run_module.resolve(_raw_config(preset="v0.1"))

    def test_scalings_fitted_on_a_training_fold_are_refused(self):
        with pytest.raises(ValueError, match="fitted on a training fold"):
            run_module.resolve(_raw_config(task={"rows": 30, "scaler": "standard"}))


class TestRun:
    def test_the_features_are_the_scaled_beats(self, tmp_path):
        run_dir = _execute(_raw_config(outputs={"save_states": True}), tmp_path / "run")
        scaled = standardize_data(POOL[0], POOL[0][:1], "sequence_zscore")[0][..., 0]
        for fold in _folds(run_dir):
            np.testing.assert_array_equal(fold["test_states"][:, 0],
                                          scaled[fold["test_index"]])
        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert metrics["simulation"]["features"] == 8
        assert metrics["mean"]["f1"] > 0.8

    def test_predictions_are_those_of_the_classifier_on_the_same_folds(self, tmp_path):
        run_dir = _execute(_raw_config(readout={"classifier": "Ridge", "params": {"alpha": 1.0},
                                                "scale": False}), tmp_path / "run")
        X = standardize_data(POOL[0], POOL[0][:1], "sequence_zscore")[0][..., 0]
        y = np.argmax(POOL[1][:, 0], axis=1)
        folds = StratifiedKFold(3, shuffle=True, random_state=3).split(X, y)
        for fold, (train, test) in zip(_folds(run_dir), folds, strict=True):
            np.testing.assert_array_equal(fold["test_index"], test)
            expected = RidgeClassifier(alpha=1.0).fit(X[train], y[train]).predict(X[test])
            np.testing.assert_array_equal(np.argmax(fold["Y_pred"][:, 0], axis=1), expected)

    def test_raw_and_reservoir_runs_share_their_folds(self, tmp_path):
        raw_run = _execute(_raw_config(), tmp_path / "raw")
        reservoir_run = _execute({**_raw_config(), "reservoir": {"params": dict(RESERVOIR)}},
                                 tmp_path / "reservoir")
        for a, b in zip(_folds(raw_run), _folds(reservoir_run), strict=True):
            np.testing.assert_array_equal(a["test_index"], b["test_index"])
            np.testing.assert_array_equal(a["Y_test"], b["Y_test"])

    def test_histogram_gradient_boosting_classifies_the_raw_beat(self, tmp_path):
        run_dir = _execute(_raw_config(readout={"classifier": "HistGB", "params": {
            "max_iter": 20, "min_samples_leaf": 2}}), tmp_path / "run")
        assert json.loads((run_dir / "metrics.json").read_text())["mean"]["f1"] > 0.8

    def test_saved_raw_features_can_be_refitted(self, tmp_path):
        first = _execute(_raw_config(outputs={"save_states": True}), tmp_path / "first")
        refit = _execute({"name": "refit", "reservoir": {"states_from": str(first)},
                          "readout": _raw_config()["readout"]}, tmp_path / "refit")
        for a, b in zip(_folds(first), _folds(refit), strict=True):
            np.testing.assert_array_equal(a["Y_pred"], b["Y_pred"])

    def test_a_held_out_split_is_scored(self, tmp_path, monkeypatch):
        from utils import heldout

        X_split, Y_split = _three_classes(n_per_class=4, seed=1)
        ids = np.arange(200, 200 + len(X_split))
        monkeypatch.setattr(heldout, "load_split",
                            lambda name, scaler_type: (X_split, Y_split, ids))
        run_dir = _execute(_raw_config(test={"split": "validation-1"}), tmp_path / "run")
        (fold,) = _folds(run_dir)
        assert len(fold["Y_train"]) == len(POOL[0])
        np.testing.assert_array_equal(fold["test_index"], ids)
        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert metrics["test"]["split"] == "validation-1"
        assert metrics["mean"]["f1"] > 0.8


class TestStudy:
    def test_a_study_searches_the_readout_of_the_raw_beat(self, isolated):
        cfg = _raw_config(study={"name": "raw-grid", "trials": 2, "sampler": {"type": "grid"},
                                 "search": {"readout.params.alpha": {
                                     "type": "categorical", "choices": [0.1, 10.0]}}})
        study = study_module.run_study(cfg, allow_dirty=True)
        assert sorted(t.params["readout.params.alpha"] for t in study.trials) == [0.1, 10.0]
        assert all(t.user_attrs["summary"]["f1"] > 0.5 for t in study.trials)

    @pytest.mark.parametrize("path", ["reservoir.params.units", "reservoir.seed",
                                      "features.measurement_noise"])
    def test_a_study_cannot_search_reservoir_or_observation_settings(self, path):
        cfg = _raw_config(study={"search": {path: {"type": "categorical", "choices": [2, 3]}}})
        with pytest.raises(ValueError, match="has no reservoir or observation settings"):
            study_module.check(*study_module.split(cfg))
