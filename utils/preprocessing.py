from typing import Tuple, List, Optional, Sequence
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import logging

from utils.analysis import count_labels

logger = logging.getLogger(__name__)


# Public API of this module (keeps a clean surface while still enabling
# unit tests to import and test individual steps).
__all__ = [
    "load_ecg_data",
    "ensure_kaggle_heartbeat_files",
    "standardize_data",
    "augment_data",
    "save_npz",
    "load_npz",
]


def ensure_kaggle_heartbeat_files(
    filenames: Sequence[str],
    data_dir: str = "data/ecg",
    dataset_handle: str = "shayanfazeli/heartbeat",
) -> List[str]:
    """Resolve heartbeat CSV files from the workspace or Kaggle cache."""
    wanted = [str(name) for name in filenames]
    candidate_roots = [Path(data_dir)]

    env_root = os.getenv("KAGGLE_HEARTBEAT_DIR")
    if env_root:
        candidate_roots.append(Path(env_root))

    for root in candidate_roots:
        resolved = [root / name for name in wanted]
        if all(path.exists() for path in resolved):
            return [str(path.resolve()) for path in resolved]

    try:
        import kagglehub
    except ImportError as exc:
        missing = ", ".join(wanted)
        raise FileNotFoundError(
            f"Could not find {missing} under '{data_dir}' and kagglehub is not installed."
        ) from exc

    cache_root = Path(kagglehub.dataset_download(dataset_handle))
    resolved = [cache_root / name for name in wanted]
    if not all(path.exists() for path in resolved):
        missing = ", ".join(str(path) for path in resolved if not path.exists())
        raise FileNotFoundError(
            f"Downloaded Kaggle dataset '{dataset_handle}' but required files were missing: {missing}"
        )

    logger.info("Resolved Kaggle heartbeat dataset via cache at %s", cache_root)
    return [str(path.resolve()) for path in resolved]


def _rng_from_seed(seed: Optional[int]) -> np.random.Generator:
    """Return a dedicated RNG for this call.

    We avoid using the process-global RNG where possible so that callers can
    reason about determinism (and unit tests can be precise).
    """
    if seed is None:
        # Non-deterministic, but still avoids mutating global np.random state.
        return np.random.default_rng()
    return np.random.default_rng(int(seed))

def save_npz(filename: str, **kwargs) -> None:
    """
    Save numpy arrays to a compressed file atomically.

    We write to a temporary file and then atomically replace the target, so
    readers never observe a partially written NPZ.
    """
    import tempfile
    dest_dir = os.path.dirname(filename) or '.'
    os.makedirs(dest_dir, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(suffix=".npz", dir=dest_dir)
    os.close(fd)
    try:
        np.savez(tmp_name, **kwargs)
        os.replace(tmp_name, filename)  # atomic rename
    except BaseException:
        os.unlink(tmp_name)
        raise

def load_npz(filename: str, allow_pickle: bool = True) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Load numpy arrays from a compressed file.

    Args:
        filename (str): Name of the file.
        allow_pickle (bool, optional): Allow loading pickled objects (default is True).

    Returns:
        Optional[Tuple[np.ndarray, np.ndarray]]: Tuple containing loaded arrays if successful, None otherwise.
    """
    try:
        return np.load(filename, allow_pickle=allow_pickle)
    except Exception as e:
        logger.error(f"Error loading data from: {filename}\n{e}")
        return None

def load_ecg_data(
    rows: Optional[int] = None,
    test_ratio: float = 0.2,
    encode_labels: bool = True,
    standardize: bool = True,
    repeat_targets: bool = False,
    shuffle: bool = True,
    binary: bool = False,
    noise_rate: float = 0,
    noise_ratio: float = 0,
    data_dir: str = "data/ecg",
    train_file: str = "mitbih_train.csv",
    test_file: str = "mitbih_test.csv",
    save_file: str = "ecg_data.npz",
    binary_save_file: str = "binary_ecg_data.npz",
    scaler_type: str = "sequence_zscore",
    balance_classes: bool = True,
    max_per_class: Optional[int] = None,
    seed: Optional[int] = 6337,
    preserve_split: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load ECG dataset from file, encode labels, shuffle, augment, scale, balance classes, and split into training and testing subsets.

    Args:
        rows (int, optional): Number of rows to limit the dataset to, defaults to the maximum.
        test_ratio (float, optional): Ratio of testing data to total data. Defaults to 0.2.
        encode_labels (bool, optional): Whether to one-hot encode labels. Defaults to True.
        standardize (bool, optional): Whether to standardize the input values. Defaults to True.
        repeat_targets (bool, optional): Whether to repeat targets. Defaults to False.
        shuffle (bool, optional): Whether to shuffle instances. Defaults to True.
        binary (bool, optional): Whether to load dataset as binary targets. Defaults to False.
        noise_rate (float, optional): Rate of noise augmentation. Defaults to 0.
        noise_ratio (float, optional): Ratio of noise augmentation. Defaults to 0.
        data_dir (str, optional): Directory containing ECG data files. Defaults to "data/ecg".
        train_file (str, optional): Filename of the training data file. Defaults to "ecg_train.csv".
        test_file (str, optional): Filename of the testing data file. Defaults to "ecg_test.csv".
        save_file (str, optional): Filename to save the preprocessed data. Defaults to "ecg_data.npz".
        binary_save_file (str, optional): Filename to save the preprocessed data with binary targets. Defaults to "binary_ecg_data.npz".

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing X_train, Y_train, X_test, Y_test.
    """
    train_file_path, test_file_path = ensure_kaggle_heartbeat_files(
        [train_file, test_file],
        data_dir=data_dir,
    )
    save_file_path = os.path.join(data_dir, save_file)
    save_file_path_binary = os.path.join(data_dir, binary_save_file)

    logger.info("Loading ECG dataset")

    if preserve_split:
        X_train_raw, Y_train_raw, X_test, Y_test = _load_raw_ecg_splits(
            binary, train_file_path, test_file_path
        )
        X_loaded = np.concatenate((X_train_raw, X_test), axis=0)
        Y_loaded = np.concatenate((Y_train_raw, Y_test), axis=0)
    else:
        X_loaded, Y_loaded = _load_raw_ecg_dataset(
            binary, train_file_path, test_file_path, save_file_path, save_file_path_binary
        )

    # Debug: raw data is often minmax-scaled to [0, 1] in the source CSV/NPZ.
    # This is *before* any standardization (e.g., sequence_zscore).
    try:
        x_min = float(np.min(X_loaded))
        x_max = float(np.max(X_loaded))
        x_mean = float(np.mean(X_loaded))
        x_std = float(np.std(X_loaded))
        neg_frac = float(np.mean(np.asarray(X_loaded) < 0))
        logger.debug(
            "Raw ECG loaded (pre-standardize) shape=%s min=%.6g max=%.6g mean=%.6g std=%.6g neg_frac=%.4f",
            np.asarray(X_loaded).shape,
            x_min,
            x_max,
            x_mean,
            x_std,
            neg_frac,
        )
    except Exception:
        # Don't fail data loading if debug stats can't be computed.
        pass

    rng = _rng_from_seed(seed)

    # If rows is not given, set to all rows
    if rows is None:
        rows = len(X_loaded)
        logger.warning("Number of rows not specified, implicitly using all rows.")

    if preserve_split:
        logger.info("Preserving original train/test split from source CSV files")
        if shuffle:
            X_train_raw, Y_train_raw = _shuffle_dataset(X_train_raw, Y_train_raw, rng)
            X_test, Y_test = _shuffle_dataset(X_test, Y_test, rng)
    else:
        # First, create a balanced test set that is limited by the smallest class.
        # Then, apply optional class balancing and row limiting to the remaining
        # data for training only. This keeps the test set balanced while preserving
        # the existing behaviour of the training set (balanced up to max_per_class).
        logger.info("Preparing train/test split (balanced test set)")
        X_train_raw, Y_train_raw, X_test, Y_test = _train_test_split_balanced_test(
            X_loaded, Y_loaded, test_size=test_ratio, shuffle=shuffle, rng=rng
        )

    logger.info("Balancing classes for training")
    if balance_classes:
        X_balanced, Y_balanced = _balance_classes(X_train_raw, Y_train_raw, max_per_class=max_per_class, rng=rng)
    else:
        X_balanced, Y_balanced = X_train_raw, Y_train_raw

    logger.info("Limiting training instances to %s", rows)
    X_limited, Y_limited = _limit_instances(X_balanced, Y_balanced, rows, rng=rng)
    num_rows = X_limited.shape[0]
    if num_rows < rows:
        logger.warning(f"The {rows} rows requested were capped at {num_rows} rows to keep classes balanced.")

    X_train, Y_train = X_limited, Y_limited

    if encode_labels:
        num_classes = len(np.unique(Y_loaded))
        logger.debug("One-hot encoding labels.")
        Y_train = _one_hot_encode(Y_train, num_classes)
        Y_test = _one_hot_encode(Y_test, num_classes)

    logger.debug("Reshaping data into time-series")
    X_train, Y_train = _reshape_data(X_train, Y_train)
    X_test, Y_test = _reshape_data(X_test, Y_test)

    # Repeat targets if specified
    if repeat_targets:
        logger.debug(f"Repeating train and test targets to size: {X_train[0].shape[0]}")
        Y_train = [np.repeat(Y_instance, X_instance.shape[0], axis=0) for X_instance, Y_instance in zip(X_train, Y_train)]
        Y_test = [np.repeat(Y_instance, X_instance.shape[0], axis=0) for X_instance, Y_instance in zip(X_test, Y_test)]

    # add augmentation (noise) to training set
    if noise_rate and noise_ratio:
        X_train, Y_train = augment_data(np.array(X_train), np.array(Y_train), noise_rate, noise_ratio, rng=rng)

    if standardize:
        X_train, X_test = standardize_data(X_train, X_test, scaler_type=scaler_type)

        # Debug: confirm scaling did what we expect.
        if str(scaler_type) == "sequence_zscore":
            try:
                # Compute per-sequence mean/std along time axis to confirm means are ~0.
                X_dbg = np.asarray(X_train, dtype=np.float64)
                seq_means = X_dbg.mean(axis=1)
                seq_stds = X_dbg.std(axis=1)
                logger.debug(
                    "After sequence_zscore (train): shape=%s global_min=%.6g global_max=%.6g global_mean=%.6g global_std=%.6g neg_frac=%.4f",
                    X_dbg.shape,
                    float(np.min(X_dbg)),
                    float(np.max(X_dbg)),
                    float(np.mean(X_dbg)),
                    float(np.std(X_dbg)),
                    float(np.mean(X_dbg < 0)),
                )
                logger.debug(
                    "After sequence_zscore (train): per-seq mean range [%.6g, %.6g] | per-seq std range [%.6g, %.6g]",
                    float(np.min(seq_means)),
                    float(np.max(seq_means)),
                    float(np.min(seq_stds)),
                    float(np.max(seq_stds)),
                )
            except Exception:
                pass

    # Ensure consistent return types regardless of which pipeline branches ran.
    # X: (N, T, 1)
    # Y: (N, 1, C) if one-hot else (N, 1)
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    Y_train = np.asarray(Y_train)
    Y_test = np.asarray(Y_test)

    train_shapes = (X_train[0].shape, Y_train[0].shape)
    test_shapes = (X_test[0].shape, Y_test[0].shape)

    train_instances = len(X_train)
    test_instances = len(X_test)
    total_instances = train_instances + test_instances

    params = {
        # Total number of instances returned by this loader.
        # (Historically this field was training-only; keep separate train/test fields below.)
        "instances": total_instances,
        "total_instances": total_instances,
        "encode_labels": encode_labels,
        "repeat_targets": repeat_targets,
        "standardize": standardize,
        "test_ratio": test_ratio,
        "shuffle": shuffle,
        "binary": binary,
        "preserve_split": preserve_split,
        "noise_rate": noise_rate,
        "noise_ratio": noise_ratio,
        "train_instances": train_instances,
        "test_instances": test_instances,
        # How many training instances were requested/kept after balancing/limiting.
        "requested_train_instances": rows,
        "effective_train_instances": num_rows,
        "train_shapes": train_shapes,
        "test_shapes": test_shapes,
        "train_labels": count_labels(Y_train),
        "test_labels": count_labels(Y_test),
    }

    # Log dataset parameters (debug-level; can be noisy).
    logger.debug("----- Dataset Parameters -----")
    for key, value in params.items():
        logger.debug("%s: %s", key, value)
    logger.debug("------------------------------")

    return X_train, Y_train, X_test, Y_test


def _shuffle_dataset(
    X: np.ndarray,
    Y: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Shuffle X/Y in unison without changing split membership."""
    rng = rng or np.random.default_rng()
    indices = rng.permutation(len(X))
    return X[indices], Y[indices]

def _sequence_zscore(X: np.ndarray) -> np.ndarray:
    """Per-sequence z-score normalization: (x - mean) / std along time axis.

    Expects X with shape (N, T, 1) and returns the same shape.
    """
    X = np.array(X)
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std[std < 1e-8] = 1.0
    return (X - mean) / std


def _sequence_minmax(X: np.ndarray, a: float = 0.0, b: float = 1.0) -> np.ndarray:
    """Per-sequence min-max normalization into the range [a, b] along the time axis.

    Expects X with shape (N, T, 1) and returns the same shape.

    Scaling into a fully positive range such as [0.1, 0.9] avoids the zero-crossing
    artefact that arises with z-score: when u_t = 0 the input term alpha_u * Win @ u
    contributes exactly zero to the reservoir drive, making the reservoir blind to the
    baseline region between beats.  With a positive-only range every timestep retains
    a non-zero contribution proportional to its local amplitude.
    """
    if b <= a:
        raise ValueError(f"_sequence_minmax requires a < b; got a={a}, b={b}")
    X = np.array(X)
    x_min = X.min(axis=1, keepdims=True)
    x_max = X.max(axis=1, keepdims=True)
    denom = x_max - x_min
    denom[denom < 1e-8] = 1.0  # flat sequences stay flat
    return a + (b - a) * (X - x_min) / denom


def standardize_data(X_train: np.ndarray, X_test: np.ndarray, scaler_type: str = "sequence_zscore") -> Tuple[np.ndarray, np.ndarray]:
    """Standardize input data using configurable scaling.

    scaler_type:
        - "minmax": global Min-Max scaling per time index across the dataset.
        - "standard": global z-score scaling per time index across the dataset.
        - "sequence_zscore": per-sequence z-score normalization.
        - "sequence_minmax": per-sequence min-max normalization into [0, 1].
        - "sequence_minmax_pos": per-sequence min-max into [0.1, 0.9] (fully positive;
          recommended for reservoir computing to avoid zero-drive at baseline).
        - "none": no scaling (input returned unchanged).
    """
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    if scaler_type == "sequence_zscore":
        return _sequence_zscore(X_train), _sequence_zscore(X_test)

    if scaler_type == "sequence_minmax":
        return _sequence_minmax(X_train, 0.0, 1.0), _sequence_minmax(X_test, 0.0, 1.0)

    if scaler_type == "sequence_minmax_pos":
        return _sequence_minmax(X_train, 0.1, 0.9), _sequence_minmax(X_test, 0.1, 0.9)

    if scaler_type == "none":
        return X_train, X_test

    # Flatten (N, T, 1) -> (N, T)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    if scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown scaler_type '{scaler_type}'")

    scaler.fit(X_train_flat)
    X_train_scaled = scaler.transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)

    # Reshape back to (N, T, 1)
    X_train_scaled = X_train_scaled.reshape(X_train.shape[0], -1, 1)
    X_test_scaled = X_test_scaled.reshape(X_test.shape[0], -1, 1)

    return X_train_scaled, X_test_scaled

def _balance_classes(
    X: np.ndarray,
    Y: np.ndarray,
    max_per_class: Optional[int] = None,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balance classes by undersampling.

    Args:
        X (np.ndarray): Input data.
        Y (np.ndarray): Target labels.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Balanced input data and target labels.
    """
    Y = Y.astype(int)
    classes, counts = np.unique(Y, return_counts=True)
    if max_per_class is None:
        # Default behavior: fully balance to the smallest class
        min_instances_per_class = min(counts)
        target_per_class = {c: min_instances_per_class for c in classes}
    else:
        # Cap each class at max_per_class while keeping minority classes intact
        target_per_class = {c: min(count, max_per_class) for c, count in zip(classes, counts)}

    rng = rng or np.random.RandomState()
    balanced_indices = [rng.choice(np.where(Y == c)[0], target_per_class[c], replace=False) for c in classes]
    return X[np.concatenate(balanced_indices)], Y[np.concatenate(balanced_indices)]

def augment_data(
    X: np.ndarray,
    Y: np.ndarray,
    noise_rate: float,
    noise_ratio: float,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment training data with Gaussian distributed noise, at a specific rate, for a specific ratio of the training data.

    Args:
        X (np.ndarray): Training data.
        Y (np.ndarray): Training target labels.
        noise_rate (float): Rate of noise augmentation, where an increasing value from 0 - 1 indicates more noise.
        noise_ratio (float): Ratio of data augmentation defining the amount of augmentated instances to add to each class.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Augmented input data and target labels.
    """
    if not noise_rate or not noise_ratio:
        logger.debug("Noise rate or ratio of 0, no data was augmented.")
        return X, Y

    rng = rng or np.random.RandomState()

    # Flatten Y to handle both one-hot encoded and categorical labels
    Y_flat = np.argmax([y[0] for y in Y], axis=1) if len(Y.shape) > 1 else Y

    # Make augmentation proportional to each class size instead of assuming
    # perfectly balanced classes. For each class c with count N_c, we
    # generate approximately ``noise_ratio * N_c`` augmented samples.
    classes, counts = np.unique(Y_flat, return_counts=True)
    num_classes = len(classes)

    X_augmented = np.empty((0, X.shape[1], X.shape[2]))
    Y_augmented = np.empty(0)

    for class_label, count_c in zip(classes, counts):
        class_instances = X[Y_flat == class_label]

        # Number of augmented samples for this class, proportional to its size
        n_aug_c = int(noise_ratio * count_c)
        if n_aug_c <= 0:
            continue

        indices = rng.choice(len(class_instances), n_aug_c, replace=True)
        instances = class_instances[indices]

        # Generate Gaussian noise scaled to a fraction (noise_rate) of each
        # sequence's standard deviation. This keeps perturbations
        # proportional to signal amplitude without over-normalizing.
        seq_std = instances.std(axis=1, keepdims=True)
        seq_std[seq_std < 1e-8] = 1.0
        noise = rng.normal(0.0, 1.0, instances.shape) * (noise_rate * seq_std)
        noisy_instances = instances + noise

        X_augmented = np.concatenate((X_augmented, noisy_instances), axis=0)

        augmented_labels = np.full(n_aug_c, class_label)
        Y_augmented = np.concatenate((Y_augmented, augmented_labels))

    shuffle_indices = rng.permutation(len(X_augmented))
    X_augmented = X_augmented[shuffle_indices]
    Y_augmented = Y_augmented[shuffle_indices]

    if len(Y.shape) > 1:
        Y_augmented = _one_hot_encode(Y_augmented.reshape(-1, 1), num_classes)

    X_augmented = np.concatenate((X, X_augmented), axis=0)
    Y_augmented = np.concatenate((Y, Y_augmented))

    return X_augmented, Y_augmented

def _merge_dataset(train_df: pd.DataFrame, test_df: pd.DataFrame, binary: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess the dataset by concatenating train and test datasets.

    Args:
        train_df (pd.DataFrame): Training dataset.
        test_df (pd.DataFrame): Testing dataset.
        binary (bool): Whether to merge arrhythmic classes into one.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Processed input data and target labels.
    """
    # Concatenate train and test datasets
    X_combined = pd.concat([train_df.iloc[:, :-1], test_df.iloc[:, :-1]], axis=0)
    Y_combined = pd.concat([train_df.iloc[:, -1], test_df.iloc[:, -1]], axis=0)

    # If binary flag is enabled, merge arrhythmic classes into one
    if binary:
        logger.debug("Merging all arrhythmia variations into one target for binary classification")
        Y_combined[Y_combined != 0] = 1

    return X_combined.values, Y_combined.values


def _load_raw_ecg_splits(
    binary: bool,
    train_file_path: str,
    test_file_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the original Kachuee train/test CSV split without recombining it."""
    logger.debug("Loading raw ECG train/test split from: %s and %s", train_file_path, test_file_path)
    train_df = pd.read_csv(train_file_path, header=None)
    test_df = pd.read_csv(test_file_path, header=None)

    X_train = train_df.iloc[:, :-1].values
    Y_train = train_df.iloc[:, -1].values
    X_test = test_df.iloc[:, :-1].values
    Y_test = test_df.iloc[:, -1].values

    if binary:
        logger.debug("Merging all arrhythmia variations into one target for binary classification")
        Y_train = Y_train.copy()
        Y_test = Y_test.copy()
        Y_train[Y_train != 0] = 1
        Y_test[Y_test != 0] = 1

    return X_train, Y_train, X_test, Y_test

def _load_raw_ecg_dataset(binary: bool, train_file_path: str, test_file_path: str, save_file_path: str, save_file_path_binary: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load or preprocess ECG dataset.

    Args:
        binary (bool): Whether to load dataset with binary targets.
        train_file_path (str): File path for training data.
        test_file_path (str): File path for testing data.
        save_file_path (str): File path to save preprocessed data.
        save_file_path_binary (str): File path to save preprocessed data with binary targets.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Processed input data and target labels.
    """
    file_path = save_file_path_binary if binary else save_file_path

    if os.path.exists(file_path):
        logger.debug(f"Loading preprocessed dataset from: {file_path}")
        data = load_npz(file_path)
        if data is None:
            logger.warning(f"Could not load dataset at: {file_path}")
        else:
            return data["X"], data["Y"]

    logger.debug(f"Preprocessing datasets from: {train_file_path} and {test_file_path}")
    train_df = pd.read_csv(train_file_path, header=None)
    test_df = pd.read_csv(test_file_path, header=None)

    X_loaded, Y_loaded = _merge_dataset(train_df, test_df, binary)
    save_npz(file_path, **{"X": X_loaded, "Y": Y_loaded})
    logger.debug(f"Saving preprocessed dataset to: {file_path}")

    return X_loaded, Y_loaded

def _train_test_split(
    X: np.ndarray,
    Y: np.ndarray,
    test_size: float,
    shuffle: bool,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset into training and testing sets.

    Args:
        X (np.ndarray): Input data.
        Y (np.ndarray): Target labels.
        test_size (float): Ratio of testing data to total data.
        shuffle (bool): Whether to shuffle instances.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Split training and testing data.
    """
    logger.debug(f"Train:Test dataset ratio: [{int((1 - test_size) * 100)}:{int(test_size * 100)}]")
    Y = Y.astype(int)
    classes, counts = np.unique(Y, return_counts=True)
    rng = rng or np.random.RandomState()
    test_class_counts = np.floor(counts * test_size).astype(int)
    train_indices, test_indices = [], []

    for c, count in zip(classes, test_class_counts):
        class_indices = np.where(Y == c)[0]
        selected_indices = rng.choice(class_indices, count, replace=False)
        test_indices.extend(selected_indices)
        train_indices.extend(np.setdiff1d(class_indices, selected_indices))

    if shuffle:
        rng.shuffle(train_indices)
        rng.shuffle(test_indices)

    return X[train_indices], Y[train_indices], X[test_indices], Y[test_indices]

def _train_test_split_balanced_test(
    X: np.ndarray,
    Y: np.ndarray,
    test_size: float,
    shuffle: bool,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split dataset into train and *balanced* test sets.

    The test set will contain the same number of samples per class, limited by
    the smallest class count and the desired test_size. Concretely, for each
    class ``c`` with ``count_c`` samples, the number of test samples is::

        test_per_class = floor(min(counts) * test_size)

    where ``min(counts)`` is the size of the smallest class. This ensures that
    all classes contribute equally to the test set while respecting the
    available data in the limiting class. The remaining samples are used for
    training.

    Args:
        X (np.ndarray): Input data.
        Y (np.ndarray): Target labels.
        test_size (float): Ratio of testing data to total data.
        shuffle (bool): Whether to shuffle instances within train and test.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            X_train, Y_train, X_test, Y_test
    """
    logger.debug(
        f"Creating balanced test set with per-class size based on smallest class and test_size={test_size}"
    )
    Y = Y.astype(int)
    classes, counts = np.unique(Y, return_counts=True)
    rng = rng or np.random.RandomState()

    min_count = int(counts.min())
    test_per_class = int(np.floor(min_count * test_size))
    if test_per_class <= 0:
        raise ValueError(
            "test_size too small to allocate at least one sample per class in the balanced test set."
        )

    test_indices = []
    train_indices = []

    for c in classes:
        class_indices = np.where(Y == c)[0]
        if len(class_indices) < test_per_class:
            raise ValueError(
                f"Not enough instances in class {c} to create balanced test set: "
                f"required {test_per_class}, found {len(class_indices)}."
            )
        selected_test = rng.choice(class_indices, test_per_class, replace=False)
        test_indices.extend(selected_test)
        # Remaining go to train
        remaining = np.setdiff1d(class_indices, selected_test)
        train_indices.extend(remaining)

    if shuffle:
        rng.shuffle(train_indices)
        rng.shuffle(test_indices)

    return X[train_indices], Y[train_indices], X[test_indices], Y[test_indices]

def _one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Perform one-hot encoding of target labels.

    Args:
        labels (np.ndarray): Target labels.
        num_classes (int): Number of classes.

    Returns:
        np.ndarray: One-hot encoded target labels.
    """
    return np.eye(num_classes)[labels.astype(int)]

def _reshape_data(X: np.ndarray, Y: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Reshape input and target data for reservoir processing.

    Args:
        X (np.ndarray): Input data.
        Y (np.ndarray): Target labels.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: Reshaped input and target data.
    """
    # Reshape X to have shape (num_instances, timesteps, features)
    X_reshaped = [instance.T.reshape(-1, 1) for instance in X]

    # Create a list of arrays for Y
    Y_reshaped = [np.array([label]) for label in Y]

    return X_reshaped, Y_reshaped

def _limit_instances(
    X: np.ndarray,
    Y: np.ndarray,
    rows: int,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Limit the total number of instances without re-balancing classes.

    This function randomly selects up to ``rows`` samples from the dataset,
    preserving the original class distribution (up to sampling noise).

    Args:
        X (np.ndarray): Input data.
        Y (np.ndarray): Target labels.
        rows (int): Maximum number of instances to keep.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Limited input data and target labels.
    """
    if rows is None or rows >= len(X):
        return X, Y

    rng = rng or np.random.RandomState()
    indices = np.arange(len(X))
    rng.shuffle(indices)
    selected = indices[:rows]
    return X[selected], Y[selected]
