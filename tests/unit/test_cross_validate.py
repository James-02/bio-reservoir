"""Unit tests for training.cross_validate (behaviour of v0.1)."""

import importlib
import os

import numpy as np
import pytest
from reservoirpy.nodes import Ridge

from bioreservoir import BioReservoir
from training.profiling import _merge_profiles
from utils.analysis import evaluate_performance
from utils.preprocessing import standardize_data
from utils.results import FoldArtifact

# training/__init__.py re-exports a function named cross_validate, which hides the module.
cv = importlib.import_module("training.cross_validate")
_limit_native_threads = cv._limit_native_threads


@pytest.fixture(autouse=True)
def _keep_global_state(monkeypatch):
    """run_fold configures logging and thread limits for worker processes."""
    monkeypatch.setattr(cv, "setup_logging", lambda **kwargs: None)
    monkeypatch.setattr(cv, "_limit_native_threads", lambda: None)


@pytest.fixture
def saved(monkeypatch):
    """Record FoldArtifact.save calls instead of writing files."""
    calls = []

    def record(self, results_dir=None, save_states=False):
        calls.append(self.fold_index)

    monkeypatch.setattr(FoldArtifact, "save", record)
    return calls


def _dataset(n_per_class=12, steps=8, seed=0):
    """Two classes of noisy ramps, rising and falling; one-hot labels of shape (N, 1, 2)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, steps)
    X = np.concatenate([t + 0.3 * rng.normal(size=(n_per_class, steps)),
                        1.0 - t + 0.3 * rng.normal(size=(n_per_class, steps))])[..., None]
    labels = np.repeat([0, 1], n_per_class)
    return X, np.eye(2)[labels][:, None, :], labels


def _reservoir():
    # RK4 needs about 16 substeps per input step for the default kinetics; with 8 or fewer
    # the states are wrong or diverge.
    return BioReservoir(units=3, warmup=0, rk4_substeps=16, seed=1, input_dim=1,
                        output_variables="He", parallel=False)


def _readout():
    return Ridge(ridge=1e-6)


def test_thread_limits_default_to_one_but_keep_user_settings(monkeypatch):
    others = ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS")
    for var in others:
        monkeypatch.setenv(var, "")  # records the original value so it is restored
        monkeypatch.delenv(var)
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    _limit_native_threads()
    assert os.environ["OMP_NUM_THREADS"] == "4"
    assert all(os.environ[var] == "1" for var in others)


class TestLabels:
    def test_labels_become_class_indices(self):
        labels = np.array([2, 0, 1, 2])
        one_hot = np.eye(3)[labels]
        for Y in (labels, one_hot, one_hot[:, None, :]):
            np.testing.assert_array_equal(cv._labels_to_1d(Y), labels)

    def test_other_shapes_are_rejected(self):
        with pytest.raises(ValueError, match="Unsupported"):
            cv._labels_to_1d(np.zeros((2, 1, 1, 3)))


class TestFolds:
    def test_validation_sets_partition_the_data(self):
        X, Y, _ = _dataset(n_per_class=20)
        folds = cv._iter_folds(X, Y, folds=4, seed=0)
        assert [f.fold_index for f in folds] == [0, 1, 2, 3]
        np.testing.assert_array_equal(np.sort(np.concatenate([f.val_idx for f in folds])),
                                      np.arange(40))
        for f in folds:
            np.testing.assert_array_equal(np.sort(np.concatenate([f.train_idx, f.val_idx])),
                                          np.arange(40))

    def test_folds_are_stratified(self):
        X, Y, labels = _dataset(n_per_class=20)
        for f in cv._iter_folds(X, Y, folds=4, seed=0):
            assert np.bincount(labels[f.val_idx]).tolist() == [5, 5]

    def test_folds_depend_on_the_seed(self):
        X, Y, _ = _dataset(n_per_class=20)
        a, b, c = (cv._iter_folds(X, Y, folds=4, seed=s) for s in (3, 3, 4))
        assert all(np.array_equal(x.val_idx, y.val_idx) for x, y in zip(a, b, strict=True))
        assert not all(np.array_equal(x.val_idx, y.val_idx) for x, y in zip(a, c, strict=True))

    def test_one_fold_is_a_stratified_hold_out(self):
        X, Y, labels = _dataset(n_per_class=20)
        (fold,) = cv._iter_folds(X, Y, folds=1, seed=0)
        assert np.bincount(labels[fold.val_idx]).tolist() == [4, 4]
        assert len(fold.train_idx) == 32
        assert not set(fold.train_idx) & set(fold.val_idx)


class TestFoldData:
    def test_without_augmentation_the_fold_is_only_scaled(self):
        X, Y, _ = _dataset()
        fold = cv._iter_folds(X, Y, folds=3, seed=0)[1]
        X_tr, Y_tr, X_va, Y_va = cv._prepare_fold_data(X, Y, fold, noise_rate=0,
                                                       noise_ratio=0, seed=0)
        expected_tr, expected_va = standardize_data(X[fold.train_idx], X[fold.val_idx])
        np.testing.assert_array_equal(X_tr, expected_tr)
        np.testing.assert_array_equal(X_va, expected_va)
        np.testing.assert_array_equal(Y_tr, Y[fold.train_idx])
        np.testing.assert_array_equal(Y_va, Y[fold.val_idx])

    def test_augmentation_only_adds_training_samples(self):
        X, Y, _ = _dataset()
        fold = cv._iter_folds(X, Y, folds=3, seed=0)[1]
        X_tr, Y_tr, X_va, Y_va = cv._prepare_fold_data(X, Y, fold, noise_rate=0.2,
                                                       noise_ratio=0.5, seed=0)
        expected_tr, expected_va = standardize_data(X[fold.train_idx], X[fold.val_idx])
        n_train = len(fold.train_idx)
        assert len(X_tr) == len(Y_tr) == n_train + n_train // 2
        np.testing.assert_array_equal(X_tr[:n_train], expected_tr)
        np.testing.assert_array_equal(X_va, expected_va)
        np.testing.assert_array_equal(Y_va, Y[fold.val_idx])
        again = cv._prepare_fold_data(X, Y, fold, noise_rate=0.2, noise_ratio=0.5, seed=0)
        np.testing.assert_array_equal(again[0], X_tr)


class TestNewNodeLike:
    def test_rebuilds_the_node_with_the_fold_seed(self):
        class Seeded:
            def __init__(self, size=1, seed=None):
                self.hypers = {"size": size}
                self.seed = seed

            def copy(self):
                raise AssertionError("the node should be rebuilt, not copied")

        node = cv._new_node_like(Seeded(size=3, seed=1), seed=7)
        assert node.hypers == {"size": 3} and node.seed == 7

    def test_falls_back_to_a_copy_when_rebuilding_fails(self):
        class Unbuildable:
            def __init__(self, seed=None):
                self.hypers = {"unknown": 1}

            def copy(self):
                return "copy"

        assert cv._new_node_like(Unbuildable(), seed=7) == "copy"


def _artifact(fold_index, predicted, runtime=1.0, profiling=None):
    Y = np.eye(2)[[0, 0, 1, 1]][:, None, :]
    P = np.eye(2)[predicted][:, None, :]
    return FoldArtifact(trial_name="t", fold_index=fold_index, Y_train=Y, Y_test=Y, Y_pred=P,
                        metrics=evaluate_performance(Y, P, runtime), profiling=profiling)


class TestAverageFoldArtifacts:
    def test_scalar_metrics_are_fold_means(self):
        folds = [_artifact(0, [0, 0, 1, 1]), _artifact(1, [0, 1, 1, 1])]
        result = cv._average_fold_artifacts(folds)
        assert result["accuracy"] == pytest.approx(0.875)
        assert result["f1"] == pytest.approx((folds[0].metrics.f1 + folds[1].metrics.f1) / 2)
        assert [f["fold_index"] for f in result["folds"]] == [0, 1]
        assert [f["accuracy"] for f in result["folds"]] == [1.0, 0.75]
        assert "profiling" not in result and "fold_runtimes" not in result

    def test_profiles_and_runtimes_are_summarised_on_request(self):
        profiles = [{"fit": {"total_s": 1.0, "calls": 1, "mean_s": 1.0}},
                    {"fit": {"total_s": 3.0, "calls": 1, "mean_s": 3.0}}]
        folds = [_artifact(0, [0, 0, 1, 1], runtime=2.0, profiling=profiles[0]),
                 _artifact(1, [0, 0, 1, 1], runtime=4.0, profiling=profiles[1])]
        result = cv._average_fold_artifacts(folds, profile=True)
        assert result["profiling"] == _merge_profiles(profiles)
        runtimes = result["fold_runtimes"]
        assert (runtimes["mean_s"], runtimes["min_s"], runtimes["max_s"]) == (3.0, 2.0, 4.0)
        assert runtimes["per_fold"] == [2.0, 4.0]


@pytest.fixture(scope="module")
def fold_data():
    X, Y, _ = _dataset()
    fold = cv._iter_folds(X, Y, folds=3, seed=0)[0]
    return cv._prepare_fold_data(X, Y, fold, noise_rate=0, noise_ratio=0, seed=0)


class TestRunFold:
    def test_fold_scores_its_validation_predictions(self, fold_data):
        X_tr, Y_tr, X_va, Y_va = fold_data
        art = cv.run_fold(_reservoir(), _readout(), X_tr, Y_tr, X_va, Y_va, fold_index=2,
                          trial_name="t")
        assert (art.trial_name, art.fold_index) == ("t", 2)
        assert art.Y_pred.shape == (len(X_va), 1, 2)
        np.testing.assert_array_equal(art.Y_train, Y_tr)
        np.testing.assert_array_equal(art.Y_test, Y_va)
        expected = evaluate_performance(Y_va, art.Y_pred, 0.0)
        assert (art.metrics.f1, art.metrics.accuracy) == (expected.f1, expected.accuracy)
        assert art.train_states is None and art.test_states is None and art.profiling is None

    def test_states_and_timings_are_kept_on_request(self, fold_data):
        X_tr, Y_tr, X_va, Y_va = fold_data
        art = cv.run_fold(_reservoir(), _readout(), X_tr, Y_tr, X_va, Y_va, fold_index=0,
                          save_states=True, profile=True)
        assert len(art.train_states) == len(X_tr) and len(art.test_states) == len(X_va)
        assert art.train_states[0].shape == (1, 3)
        assert {"train_reservoir", "fit_readout", "val_reservoir", "predict"} <= set(art.profiling)
        assert art.profiling["instance/reservoir_run"]["calls"] == len(X_tr) + len(X_va)

    def test_templates_are_left_untouched(self, fold_data):
        reservoir, readout = _reservoir(), _readout()
        cv.run_fold(reservoir, readout, *fold_data, fold_index=0)
        assert not reservoir.is_initialized
        assert not readout.fitted

    def test_prediction_count_must_match_the_labels(self, fold_data, monkeypatch):
        real_predict = cv.predict

        def predict_one_too_few(readout, states):
            return real_predict(readout, states)[:-1]

        monkeypatch.setattr(cv, "predict", predict_one_too_few)
        with pytest.raises(ValueError, match="length mismatch"):
            cv.run_fold(_reservoir(), _readout(), *fold_data, fold_index=0)


class TestCrossValidate:
    def test_folds_receive_their_own_data(self):
        X, Y, _ = _dataset()
        result, artifacts = cv.cross_validate(_reservoir(), _readout(), X, Y, folds=3, seed=5,
                                              return_artifacts=True)
        assert [a.fold_index for a in artifacts] == [0, 1, 2]
        for art, fold in zip(artifacts, cv._iter_folds(X, Y, folds=3, seed=5), strict=True):
            np.testing.assert_array_equal(art.Y_train, Y[fold.train_idx])
            np.testing.assert_array_equal(art.Y_test, Y[fold.val_idx])
        assert result["accuracy"] == pytest.approx(np.mean([a.metrics.accuracy for a in artifacts]))

    def test_results_are_reproducible(self):
        X, Y, _ = _dataset()
        runs = [cv.cross_validate(_reservoir(), _readout(), X, Y, folds=3, seed=0,
                                  return_artifacts=True)[1] for _ in range(2)]
        for a, b in zip(*runs, strict=True):
            np.testing.assert_array_equal(a.Y_pred, b.Y_pred)

    # loky starts workers with fork followed immediately by exec, so the warning about
    # forking a multi-threaded process (numba threads from earlier tests) does not apply.
    @pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
    def test_worker_processes_give_the_same_predictions(self):
        from joblib.externals.loky import get_reusable_executor

        X, Y, _ = _dataset()
        options = dict(folds=2, seed=0, return_artifacts=True)
        _, local = cv.cross_validate(_reservoir(), _readout(), X, Y, **options)
        try:
            _, workers = cv.cross_validate(_reservoir(), _readout(), X, Y, fold_backend="loky",
                                           fold_workers=2, **options)
        finally:
            get_reusable_executor().shutdown(wait=True)
        for a, b in zip(local, workers, strict=True):
            np.testing.assert_array_equal(a.Y_pred, b.Y_pred)

    def test_unknown_backend_is_rejected(self):
        X, Y, _ = _dataset()
        with pytest.raises(ValueError, match="fold_backend"):
            cv.cross_validate(_reservoir(), _readout(), X, Y, folds=2, fold_backend="threads")

    @pytest.mark.parametrize("threshold, n_saved", [(None, 3), (0.0, 3), (1.01, 0)])
    def test_trial_is_saved_all_or_nothing(self, saved, threshold, n_saved):
        X, Y, _ = _dataset()
        cv.cross_validate(_reservoir(), _readout(), X, Y, folds=3, trial_name="t",
                          save_threshold=threshold)
        assert len(saved) == n_saved

    def test_nothing_is_saved_without_a_trial_name(self, saved):
        X, Y, _ = _dataset()
        cv.cross_validate(_reservoir(), _readout(), X, Y, folds=2)
        assert saved == []

    def test_profiling_on_request(self):
        X, Y, _ = _dataset()
        result = cv.cross_validate(_reservoir(), _readout(), X, Y, folds=2, profile=True)
        assert {"profiling", "fold_runtimes"} <= set(result)
        assert result["profiling"]["train_reservoir"]["n_folds"] == 2


class TestClassify:
    def test_several_folds_pool_the_splits(self, monkeypatch):
        X, Y, _ = _dataset()
        seen = {}

        def fake_cross_validate(reservoir, readout, X_all, Y_all, **kwargs):
            seen.update(X=X_all, Y=Y_all, **kwargs)
            return {"accuracy": 1.0}

        monkeypatch.setattr(cv, "cross_validate", fake_cross_validate)
        result = cv.classify(_reservoir(), _readout(), X[:16], Y[:16], X[16:], Y[16:], folds=3)
        assert result == {"accuracy": 1.0}
        np.testing.assert_array_equal(seen["X"], X)
        np.testing.assert_array_equal(seen["Y"], Y)
        assert seen["folds"] == 3 and seen["fold_backend"] == "sequential"

    @pytest.fixture
    def shuffled(self):
        X, Y, _ = _dataset()
        order = np.random.default_rng(0).permutation(len(X))
        return X[order], Y[order]

    def test_single_split_scores_the_test_set(self, shuffled, saved):
        X, Y = shuffled
        result = cv.classify(_reservoir(), _readout(), X[:16], Y[:16], X[16:], Y[16:],
                             trial_name="t")
        assert np.sum(result["confusion_matrix"]) == 8
        assert 0.0 <= result["accuracy"] <= 1.0
        assert saved == [0]

    def test_single_split_below_the_threshold_is_not_saved(self, shuffled, saved):
        X, Y = shuffled
        cv.classify(_reservoir(), _readout(), X[:16], Y[:16], X[16:], Y[16:], trial_name="t",
                    save_threshold=1.01)
        assert saved == []
