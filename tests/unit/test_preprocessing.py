import numpy as np
import pytest

from utils.preprocessing import ensure_kaggle_heartbeat_files, load_ecg_data


def _flatten_onehot_labels(Y):
    """Flatten one-hot labels to class indices.

    Supports either:
      - list of shape [(1, C), ...]
      - array of shape (N, 1, C)
    """
    if isinstance(Y, list):
        return np.array([int(np.argmax(y[0])) for y in Y], dtype=int)
    Y = np.asarray(Y)
    assert Y.ndim == 3 and Y.shape[1] == 1
    return np.argmax(Y[:, 0, :], axis=1).astype(int)


def _flatten_scalar_labels(Y):
    """Flatten scalar labels.

    Supports either:
      - list of shape [(1,), ...]
      - array of shape (N, 1, 1) or (N, 1)
    """
    if isinstance(Y, list):
        out = []
        for y in Y:
            y0 = y[0]
            out.append(int(y0 if np.isscalar(y0) else y0[0]))
        return np.array(out, dtype=int)
    Y = np.asarray(Y)
    if Y.ndim == 3:
        assert Y.shape[1] == 1 and Y.shape[2] == 1
        return Y[:, 0, 0].astype(int)
    assert Y.ndim == 2 and Y.shape[1] == 1
    return Y[:, 0].astype(int)


def _stack_X(X):
    """Stack X into (N, T) regardless of list/array representation."""
    if isinstance(X, list):
        return np.stack([x[:, 0] for x in X], axis=0)
    X = np.asarray(X)
    assert X.ndim == 3 and X.shape[2] == 1
    return X[:, :, 0]


@pytest.mark.parametrize("seed", [0, 1337])
def test_load_ecg_data_shapes_and_types(seed):
    X_train, Y_train, X_test, Y_test = load_ecg_data(
        rows=200,
        test_ratio=0.2,
        encode_labels=True,
        standardize=True,
        shuffle=True,
        binary=False,
        noise_rate=0.0,
        noise_ratio=0.0,
        scaler_type="sequence_zscore",
        balance_classes=True,
        seed=seed,
    )

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    Y_train = np.asarray(Y_train)
    Y_test = np.asarray(Y_test)

    assert X_train.ndim == 3 and X_train.shape[2] == 1
    assert X_test.ndim == 3 and X_test.shape[2] == 1
    assert Y_train.ndim == 3 and Y_train.shape[1] == 1
    assert Y_test.ndim == 3 and Y_test.shape[1] == 1

    num_classes = Y_train.shape[2]
    assert num_classes in (2, 5)


def test_binary_flag_produces_two_classes():
    X_train, Y_train, X_test, Y_test = load_ecg_data(
        rows=200,
        test_ratio=0.2,
        encode_labels=True,
        shuffle=True,
        binary=True,
        standardize=True,
        noise_rate=0.0,
        noise_ratio=0.0,
        scaler_type="sequence_zscore",
        balance_classes=True,
        seed=123,
    )

    assert Y_train[0].shape[1] == 2
    labels_train = _flatten_onehot_labels(Y_train)
    labels_test = _flatten_onehot_labels(Y_test)
    assert set(np.unique(labels_train)).issubset({0, 1})
    assert set(np.unique(labels_test)).issubset({0, 1})


def test_shuffle_is_reproducible_with_seed_and_changes_with_different_seed():
    # Same seed => identical ordering
    X_train1, Y_train1, X_test1, Y_test1 = load_ecg_data(
        rows=120,
        test_ratio=0.2,
        encode_labels=True,
        shuffle=True,
        binary=False,
        standardize=False,
        noise_rate=0.0,
        noise_ratio=0.0,
        balance_classes=True,
        seed=999,
    )
    X_train2, Y_train2, X_test2, Y_test2 = load_ecg_data(
        rows=120,
        test_ratio=0.2,
        encode_labels=True,
        shuffle=True,
        binary=False,
        standardize=False,
        noise_rate=0.0,
        noise_ratio=0.0,
        balance_classes=True,
        seed=999,
    )

    # Compare full arrays for stability
    assert np.allclose(_stack_X(X_train1), _stack_X(X_train2))
    assert np.array_equal(_flatten_onehot_labels(Y_train1), _flatten_onehot_labels(Y_train2))
    assert np.allclose(_stack_X(X_test1), _stack_X(X_test2))
    assert np.array_equal(_flatten_onehot_labels(Y_test1), _flatten_onehot_labels(Y_test2))

    # Different seed => very likely different ordering (not guaranteed, but extremely likely)
    X_train3, Y_train3, *_ = load_ecg_data(
        rows=120,
        test_ratio=0.2,
        encode_labels=True,
        shuffle=True,
        binary=False,
        standardize=False,
        noise_rate=0.0,
        noise_ratio=0.0,
        balance_classes=True,
        seed=1000,
    )

    assert not np.allclose(_stack_X(X_train1), _stack_X(X_train3))


def test_train_test_split_balanced_test_set_counts():
    # Use no training cap so we can check the test set is balanced across classes.
    X_train, Y_train, X_test, Y_test = load_ecg_data(
        rows=None,
        test_ratio=0.2,
        encode_labels=True,
        shuffle=True,
        binary=False,
        standardize=False,
        noise_rate=0.0,
        noise_ratio=0.0,
        balance_classes=True,
        seed=42,
    )

    y_test = _flatten_onehot_labels(Y_test)
    classes, counts = np.unique(y_test, return_counts=True)

    # Must be balanced: all counts equal
    assert len(set(counts.tolist())) == 1


def test_one_hot_encoding_toggle():
    # encoded
    _, Y_train_oh, _, _ = load_ecg_data(
        rows=100,
        test_ratio=0.2,
        encode_labels=True,
        shuffle=True,
        binary=False,
        standardize=False,
        noise_rate=0.0,
        noise_ratio=0.0,
        balance_classes=True,
        seed=7,
    )
    assert Y_train_oh[0].shape[0] == 1
    assert Y_train_oh[0].ndim == 2

    # not encoded
    _, Y_train_raw, _, _ = load_ecg_data(
        rows=100,
        test_ratio=0.2,
        encode_labels=False,
        shuffle=True,
        binary=False,
        standardize=False,
        noise_rate=0.0,
        noise_ratio=0.0,
        balance_classes=True,
        seed=7,
    )
    # Scalar label in shape (N, 1) once reshaped
    Y_train_raw = np.asarray(Y_train_raw)
    assert Y_train_raw.ndim == 2
    assert Y_train_raw.shape[1] == 1


@pytest.mark.parametrize("scaler_type", ["sequence_zscore", "none", "standard", "minmax"])
def test_standardization_modes_run_and_have_reasonable_effect(scaler_type):
    # Keep noise off so scaling properties are easier to reason about.
    X_train, Y_train, X_test, Y_test = load_ecg_data(
        rows=200,
        test_ratio=0.2,
        encode_labels=True,
        shuffle=True,
        binary=False,
        standardize=(scaler_type != "none"),
        noise_rate=0.0,
        noise_ratio=0.0,
        scaler_type=scaler_type,
        balance_classes=True,
        seed=1234,
    )

    Xtr = _stack_X(X_train)

    if scaler_type == "sequence_zscore":
        # Per-sequence mean ~0 and std ~1
        means = Xtr.mean(axis=1)
        stds = Xtr.std(axis=1)
        assert np.allclose(means.mean(), 0.0, atol=0.25)
        assert np.all(stds > 0.5)
        assert np.all(stds < 2.0)
    elif scaler_type == "minmax":
        assert np.isfinite(Xtr).all()
        assert Xtr.min() >= -1e-6
        assert Xtr.max() <= 1.0 + 1e-6
    elif scaler_type == "standard":
        assert np.isfinite(Xtr).all()
        # Global standardization isn't per-sequence; just check not exploding.
        assert abs(Xtr.mean()) < 1.0
        assert 0.1 < Xtr.std() < 10.0
    elif scaler_type == "none":
        # No scaling; still must be finite.
        assert np.isfinite(Xtr).all()


def test_noise_augmentation_increases_training_size_and_keeps_label_shape():
    # Compare identical run with and without augmentation. Since seed is set,
    # the base training subset should match; the augmented run should add samples.
    X_train0, Y_train0, X_test0, Y_test0 = load_ecg_data(
        rows=300,
        test_ratio=0.2,
        encode_labels=True,
        shuffle=True,
        binary=False,
        standardize=False,
        noise_rate=0.0,
        noise_ratio=0.0,
        balance_classes=True,
        seed=2024,
    )

    X_train1, Y_train1, X_test1, Y_test1 = load_ecg_data(
        rows=300,
        test_ratio=0.2,
        encode_labels=True,
        shuffle=True,
        binary=False,
        standardize=False,
        noise_rate=0.1,
        noise_ratio=0.1,
        balance_classes=True,
        seed=2024,
    )

    assert len(X_train1) > len(X_train0)
    assert Y_train1[0].shape == Y_train0[0].shape

    # Augmentation should not affect test set size
    assert len(X_test1) == len(X_test0)
    assert np.array_equal(_flatten_onehot_labels(Y_test1), _flatten_onehot_labels(Y_test0))


def test_ensure_kaggle_heartbeat_files_prefers_workspace_directory(tmp_path):
    data_dir = tmp_path / "ecg"
    data_dir.mkdir()
    train = data_dir / "mitbih_train.csv"
    test = data_dir / "mitbih_test.csv"
    train.write_text("1,2,0\n")
    test.write_text("3,4,1\n")

    resolved = ensure_kaggle_heartbeat_files(
        ["mitbih_train.csv", "mitbih_test.csv"],
        data_dir=str(data_dir),
    )

    assert resolved == [str(train.resolve()), str(test.resolve())]


def test_ensure_kaggle_heartbeat_files_uses_env_root(tmp_path, monkeypatch):
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    env_dir = tmp_path / "kaggle-cache"
    env_dir.mkdir()
    train = env_dir / "mitbih_train.csv"
    test = env_dir / "mitbih_test.csv"
    train.write_text("1,2,0\n")
    test.write_text("3,4,1\n")
    monkeypatch.setenv("KAGGLE_HEARTBEAT_DIR", str(env_dir))

    resolved = ensure_kaggle_heartbeat_files(
        ["mitbih_train.csv", "mitbih_test.csv"],
        data_dir=str(workspace_dir),
    )

    assert resolved == [str(train.resolve()), str(test.resolve())]
