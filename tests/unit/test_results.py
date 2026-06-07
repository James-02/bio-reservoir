"""Unit tests for utils.results — FoldArtifact save/load round-trip."""

import os
import pytest
import numpy as np

from utils.analysis import Metrics
from utils.results import FoldArtifact


@pytest.fixture
def sample_artifact(tmp_path):
    """A minimal FoldArtifact for testing."""
    return FoldArtifact(
        trial_name="test-trial",
        fold_index=0,
        Y_train=np.eye(3)[[0, 1, 2, 0]],
        Y_test=np.eye(3)[[1, 2]],
        Y_pred=np.eye(3)[[1, 2]],
        metrics=Metrics(
            f1=0.9, accuracy=0.85, precision=0.88, recall=0.87,
            mcc=0.75, kappa=0.74, runtime=1.5,
            confusion_matrix=np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]]),
            class_metrics={"0": {"precision": 1.0}, "1": {"precision": 0.8}},
        ),
    )


class TestFoldArtifactSaveLoad:
    def test_round_trip(self, sample_artifact, tmp_path):
        """Save, then load — check all fields match."""
        path = sample_artifact.save(results_dir=str(tmp_path))
        assert os.path.isfile(path)

        loaded = FoldArtifact.load(path)
        assert loaded.trial_name == "test-trial"
        assert loaded.fold_index == 0
        np.testing.assert_array_equal(loaded.Y_train, sample_artifact.Y_train)
        np.testing.assert_array_equal(loaded.Y_test, sample_artifact.Y_test)
        np.testing.assert_array_equal(loaded.Y_pred, sample_artifact.Y_pred)
        assert loaded.metrics.f1 == pytest.approx(0.9)
        assert loaded.metrics.kappa == pytest.approx(0.74)

    def test_save_creates_directory(self, sample_artifact, tmp_path):
        nested = tmp_path / "deep" / "nested"
        sample_artifact.save(results_dir=str(nested))
        assert (nested / "test-trial" / "fold-0.npz").exists()

    def test_states_not_saved_by_default(self, sample_artifact, tmp_path):
        sample_artifact.train_states = np.ones((4, 10))
        sample_artifact.test_states = np.ones((2, 10))
        path = sample_artifact.save(results_dir=str(tmp_path), save_states=False)
        loaded = FoldArtifact.load(path)
        assert loaded.train_states is None
        assert loaded.test_states is None

    def test_states_saved_when_requested(self, sample_artifact, tmp_path):
        sample_artifact.train_states = np.ones((4, 10))
        sample_artifact.test_states = np.ones((2, 10))
        path = sample_artifact.save(results_dir=str(tmp_path), save_states=True)
        loaded = FoldArtifact.load(path)
        assert loaded.train_states is not None
        np.testing.assert_array_equal(loaded.train_states, np.ones((4, 10)))

    def test_empty_trial_name_raises(self, tmp_path):
        fa = FoldArtifact(
            trial_name="",
            fold_index=0,
            Y_train=np.zeros((1, 2)),
            Y_test=np.zeros((1, 2)),
            Y_pred=np.zeros((1, 2)),
            metrics=Metrics(f1=0, accuracy=0, precision=0, recall=0,
                            mcc=0, kappa=0, runtime=0,
                            confusion_matrix=np.zeros((2, 2)),
                            class_metrics={}),
        )
        with pytest.raises(ValueError, match="trial_name"):
            fa.save(results_dir=str(tmp_path))


class TestFoldArtifactLoadTrial:
    def test_load_multiple_folds(self, tmp_path):
        for fold in range(3):
            fa = FoldArtifact(
                trial_name="multi",
                fold_index=fold,
                Y_train=np.eye(2)[[0, 1]],
                Y_test=np.eye(2)[[0]],
                Y_pred=np.eye(2)[[0]],
                metrics=Metrics(f1=0.8 + fold * 0.01, accuracy=0.8, precision=0.8,
                                recall=0.8, mcc=0.7, kappa=0.7, runtime=1.0,
                                confusion_matrix=np.eye(2),
                                class_metrics={}),
            )
            fa.save(results_dir=str(tmp_path))

        folds = FoldArtifact.load_trial("multi", results_dir=str(tmp_path))
        assert len(folds) == 3
        assert [f.fold_index for f in folds] == [0, 1, 2]

    def test_load_trial_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No fold files"):
            FoldArtifact.load_trial("nonexistent", results_dir=str(tmp_path))


class TestFoldArtifactListTrials:
    def test_list_empty_dir(self, tmp_path):
        assert FoldArtifact.list_trials(results_dir=str(tmp_path)) == []

    def test_list_discovers_trials(self, tmp_path):
        for name in ["alpha", "beta"]:
            fa = FoldArtifact(
                trial_name=name,
                fold_index=0,
                Y_train=np.zeros((1, 2)),
                Y_test=np.zeros((1, 2)),
                Y_pred=np.zeros((1, 2)),
                metrics=Metrics(f1=0, accuracy=0, precision=0, recall=0,
                                mcc=0, kappa=0, runtime=0,
                                confusion_matrix=np.zeros((2, 2)),
                                class_metrics={}),
            )
            fa.save(results_dir=str(tmp_path))
        trials = FoldArtifact.list_trials(results_dir=str(tmp_path))
        assert "alpha" in trials
        assert "beta" in trials


class TestFoldArtifactProperties:
    def test_convenience_properties(self, sample_artifact):
        assert sample_artifact.f1 == pytest.approx(0.9)
        assert sample_artifact.accuracy == pytest.approx(0.85)
        assert sample_artifact.runtime == pytest.approx(1.5)

    def test_repr(self, sample_artifact):
        r = repr(sample_artifact)
        assert "test-trial" in r
        assert "fold=0" in r
