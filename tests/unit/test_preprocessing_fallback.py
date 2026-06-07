"""Tests for the no-NPZ / no-CSV fallback paths in utils.preprocessing.

Scenarios covered:

1. ``test_loads_from_csv_when_no_npz_exists``
   When neither ecg_data.npz nor binary_ecg_data.npz exist in the data dir,
   load_ecg_data must read the raw CSV files and produce valid outputs.

2. ``test_npz_cache_is_created_after_csv_load``
   After a successful CSV load the NPZ cache file must exist on disk so that
   subsequent calls do not re-parse the CSV.

3. ``test_npz_cache_is_used_on_second_call``
   After the NPZ is created the CSVs can be removed; a second call must still
   succeed by reading from the cache.

4. ``test_binary_loads_from_csv_when_no_npz_exists``
   Same as (1) but with binary=True; the correct binary cache file must be
   created.

5. ``test_kaggle_fallback_when_no_local_csvs``
   When neither NPZ nor local CSV files are present, ensure_kaggle_heartbeat_files
   must call kagglehub.dataset_download. The test monkeypatches that call so no
   real network request is made.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from utils.preprocessing import load_ecg_data, ensure_kaggle_heartbeat_files


# ---------------------------------------------------------------------------
# Synthetic-data helpers
# ---------------------------------------------------------------------------

#: Number of ECG feature columns used in the MIT-BIH Kaggle dataset.
N_FEATURES = 187
#: Number of arrhythmia classes (0 = Normal, 1-4 = arrhythmia variants).
N_CLASSES = 5
#: Samples per class per CSV file.  Two files are merged at load time, so the
#: total per class after the merge is 2 × this value.  Keep this small to make
#: the tests fast while remaining large enough for balanced splitting.
N_PER_CLASS_PER_FILE = 25


def _make_synthetic_ecg_csv(dest_dir: Path, filename: str, seed: int) -> Path:
    """Write a synthetic ECG CSV to *dest_dir/filename* and return the path.

    The format matches the MIT-BIH Kaggle files: 187 float feature columns
    followed by a single integer label column (0–4), no header row.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for class_label in range(N_CLASSES):
        for _ in range(N_PER_CLASS_PER_FILE):
            features = rng.uniform(0.0, 1.0, N_FEATURES).tolist()
            rows.append(features + [float(class_label)])

    # Shuffle so the CSV doesn't have all class-0 rows first (mirrors reality).
    rng.shuffle(rows)
    df = pd.DataFrame(rows)
    path = dest_dir / filename
    df.to_csv(path, header=False, index=False)
    return path


def _setup_csv_data_dir(tmp_path: Path, train_seed: int = 1, test_seed: int = 2) -> Path:
    """Create a temporary data dir containing only the two raw CSV files."""
    data_dir = tmp_path / "ecg"
    data_dir.mkdir()
    _make_synthetic_ecg_csv(data_dir, "mitbih_train.csv", seed=train_seed)
    _make_synthetic_ecg_csv(data_dir, "mitbih_test.csv", seed=test_seed)
    return data_dir


def _call_load_ecg(**kwargs):
    """Convenience wrapper with sensible defaults for fallback tests."""
    defaults = dict(
        rows=50,
        test_ratio=0.2,
        encode_labels=True,
        standardize=False,
        shuffle=True,
        binary=False,
        noise_rate=0.0,
        noise_ratio=0.0,
        balance_classes=True,
        seed=42,
    )
    defaults.update(kwargs)
    return load_ecg_data(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNoNpzFallback:
    """load_ecg_data must work from raw CSV files when no NPZ cache exists."""

    def test_loads_from_csv_when_no_npz_exists(self, tmp_path):
        data_dir = _setup_csv_data_dir(tmp_path)

        # Guard: no NPZ files present before the call.
        assert not (data_dir / "ecg_data.npz").exists()
        assert not (data_dir / "binary_ecg_data.npz").exists()

        X_train, Y_train, X_test, Y_test = _call_load_ecg(
            data_dir=str(data_dir),
        )

        X_train = np.asarray(X_train)
        X_test = np.asarray(X_test)
        Y_train = np.asarray(Y_train)
        Y_test = np.asarray(Y_test)

        # Shape checks: (N, T, 1) for X, (N, 1, C) for one-hot Y.
        assert X_train.ndim == 3 and X_train.shape[2] == 1
        assert X_train.shape[1] == N_FEATURES
        assert X_test.ndim == 3 and X_test.shape[2] == 1
        assert Y_train.ndim == 3 and Y_train.shape[1] == 1
        assert Y_test.ndim == 3 and Y_test.shape[1] == 1

        # Both splits must be non-empty.
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0

        # All five classes must appear in the test set (balanced split).
        test_labels = np.argmax(Y_test[:, 0, :], axis=1)
        assert set(np.unique(test_labels)) == set(range(N_CLASSES))

    def test_npz_cache_is_created_after_csv_load(self, tmp_path):
        """After loading from CSV the NPZ cache file must appear on disk."""
        data_dir = _setup_csv_data_dir(tmp_path)

        _call_load_ecg(data_dir=str(data_dir), binary=False)

        assert (data_dir / "ecg_data.npz").exists(), (
            "ecg_data.npz was not created after loading from CSV"
        )

    def test_binary_npz_cache_is_created_after_csv_load(self, tmp_path):
        """binary=True must create binary_ecg_data.npz, not ecg_data.npz."""
        data_dir = _setup_csv_data_dir(tmp_path)

        _call_load_ecg(data_dir=str(data_dir), binary=True)

        assert (data_dir / "binary_ecg_data.npz").exists(), (
            "binary_ecg_data.npz was not created after loading from CSV with binary=True"
        )
        # The multiclass cache should *not* have been created.
        assert not (data_dir / "ecg_data.npz").exists()

    def test_npz_cache_is_used_on_second_call(self, tmp_path):
        """After the NPZ is created, subsequent calls must use it even if CSVs are gone."""
        data_dir = _setup_csv_data_dir(tmp_path)

        # First call: reads CSVs and creates the cache.
        X_train1, Y_train1, X_test1, Y_test1 = _call_load_ecg(
            data_dir=str(data_dir),
            rows=None,        # no row cap so shapes are deterministic
            balance_classes=False,
            shuffle=False,
            seed=0,
        )

        assert (data_dir / "ecg_data.npz").exists()

        # Remove CSV files so the second call MUST use the cache.
        (data_dir / "mitbih_train.csv").unlink()
        (data_dir / "mitbih_test.csv").unlink()

        # Second call: no CSVs present but NPZ cache exists.
        X_train2, Y_train2, X_test2, Y_test2 = _call_load_ecg(
            data_dir=str(data_dir),
            rows=None,
            balance_classes=False,
            shuffle=False,
            seed=0,
        )

        # The loaded raw data is identical; shapes must match.
        assert np.asarray(X_train1).shape == np.asarray(X_train2).shape
        assert np.asarray(X_test1).shape == np.asarray(X_test2).shape

    def test_full_dataset_size_reflects_both_csvs(self, tmp_path):
        """The merged dataset should contain samples from both train and test CSVs."""
        data_dir = _setup_csv_data_dir(tmp_path)

        # Disable balancing/limiting so we get the full merged pool.
        X_train, _, X_test, _ = _call_load_ecg(
            data_dir=str(data_dir),
            rows=None,
            balance_classes=False,
            shuffle=False,
            seed=0,
        )

        total = np.asarray(X_train).shape[0] + np.asarray(X_test).shape[0]
        expected_total = N_PER_CLASS_PER_FILE * 2 * N_CLASSES  # both files merged
        assert total == expected_total, (
            f"Expected {expected_total} total instances after merging both CSVs, "
            f"got {total}"
        )


class TestKaggleFallback:
    """When no local CSVs exist, load_ecg_data must download them from Kaggle."""

    def test_kaggle_fallback_when_no_local_csvs(self, tmp_path, monkeypatch):
        """Monkeypatch kagglehub so no real network request is made."""
        import kagglehub

        # Workspace data dir is empty (no CSVs, no NPZ).
        data_dir = tmp_path / "workspace_ecg"
        data_dir.mkdir()

        # Kaggle "cache" dir contains the CSV files.
        kaggle_cache = tmp_path / "kaggle_cache"
        kaggle_cache.mkdir()
        _make_synthetic_ecg_csv(kaggle_cache, "mitbih_train.csv", seed=10)
        _make_synthetic_ecg_csv(kaggle_cache, "mitbih_test.csv", seed=11)

        monkeypatch.delenv("KAGGLE_HEARTBEAT_DIR", raising=False)
        monkeypatch.setattr(
            kagglehub,
            "dataset_download",
            lambda handle: str(kaggle_cache),
        )

        X_train, Y_train, X_test, Y_test = _call_load_ecg(
            data_dir=str(data_dir),
        )

        assert np.asarray(X_train).shape[0] > 0
        assert np.asarray(X_test).shape[0] > 0

    def test_ensure_kaggle_heartbeat_files_raises_without_kagglehub(
        self, tmp_path, monkeypatch
    ):
        """FileNotFoundError is raised when files are missing and kagglehub is absent."""
        import builtins

        real_import = builtins.__import__

        def _block_kagglehub(name, *args, **kwargs):
            if name == "kagglehub":
                raise ImportError("kagglehub not available")
            return real_import(name, *args, **kwargs)

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        monkeypatch.delenv("KAGGLE_HEARTBEAT_DIR", raising=False)
        monkeypatch.setattr(builtins, "__import__", _block_kagglehub)

        with pytest.raises(FileNotFoundError, match="kagglehub is not installed"):
            ensure_kaggle_heartbeat_files(
                ["mitbih_train.csv", "mitbih_test.csv"],
                data_dir=str(empty_dir),
            )

    def test_ensure_kaggle_heartbeat_files_raises_when_download_missing_files(
        self, tmp_path, monkeypatch
    ):
        """FileNotFoundError is raised when the Kaggle download doesn't have expected files."""
        import kagglehub

        empty_cache = tmp_path / "empty_cache"
        empty_cache.mkdir()

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        monkeypatch.delenv("KAGGLE_HEARTBEAT_DIR", raising=False)
        monkeypatch.setattr(
            kagglehub,
            "dataset_download",
            lambda handle: str(empty_cache),
        )

        with pytest.raises(FileNotFoundError, match="required files were missing"):
            ensure_kaggle_heartbeat_files(
                ["mitbih_train.csv", "mitbih_test.csv"],
                data_dir=str(empty_dir),
            )
