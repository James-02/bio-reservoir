"""Convenience script to generate visualisations for the ECG classification dataset.

This script uses ``load_ecg_data`` from ``utils.preprocessing`` and the
visualisation helpers in ``utils.visualisation`` to give an overview of:

- class/sample distribution
- class-wise mean and variance
- average waveform per class
- effect of noise augmentation on a single sample
- effect of different scalers (min-max vs standard / z-score) on a single sample
"""

import argparse
import os
from typing import Optional, Dict, List

import numpy as np

from utils.preprocessing import load_ecg_data, standardize_data
from utils.visualisation import (
    plot_data_distribution,
    plot_noise,
    plot_average_instances_with_band_and_exemplars,
    _save_figure,
)


_ECG_SAMPLING_FREQ_HZ = 125


def _time_axis_ms(num_samples: int, sampling_freq_hz: float = _ECG_SAMPLING_FREQ_HZ) -> np.ndarray:
    """Return a milliseconds time axis for a uniformly sampled signal."""
    sampling_freq_hz = float(sampling_freq_hz)
    if sampling_freq_hz <= 0:
        raise ValueError(f"sampling_freq_hz must be > 0; got {sampling_freq_hz}")
    return (np.arange(int(num_samples), dtype=np.float64) / sampling_freq_hz) * 1000.0


def _select_single_sample(
    X,
    Y,
    *,
    index: int = 0,
    target_class: Optional[int] = None,
):
    """Pick a single (X, y) pair, optionally constrained to a given class index.

    X is expected to be a list/array of sequences shaped (T, 1) and Y a list/array
    of labels (either class indices or one-hot vectors).
    """
    X_arr = np.array(X, dtype=object)
    Y_arr = np.array(Y)

    # If one-hot, convert to class indices
    if Y_arr.ndim > 1:
        labels = np.array([np.argmax(y) for y in Y_arr])
    else:
        labels = Y_arr.astype(int)

    n = int(len(X_arr))
    if n == 0:
        raise ValueError("Empty dataset: X has no samples")

    # If a target class is requested, interpret `index` as an index *within that class*.
    if target_class is not None:
        idx_candidates = np.where(labels == int(target_class))[0]
        if len(idx_candidates) == 0:
            raise ValueError(f"No samples found for class {target_class}.")
        k = int(index)
        if k < 0 or k >= len(idx_candidates):
            raise ValueError(
                f"--single-index out of range for class {target_class}: {k} (class count={len(idx_candidates)})"
            )
        idx = int(idx_candidates[k])
    else:
        idx = int(index)
        if idx < 0 or idx >= n:
            raise ValueError(f"--single-index out of range: {idx} (dataset size={n})")

    x = np.asarray(X_arr[idx]).reshape(-1)
    y = int(labels[idx])
    return x, y


def _make_noise_example(x: np.ndarray, noise_std: float = 0.1, seed: Optional[int] = 0):
    """Generate a simple additive Gaussian noise example for a 1D signal x."""
    rng = np.random.RandomState(seed)
    noise = rng.normal(scale=noise_std, size=x.shape)
    noisy_x = x + noise
    return noise, noisy_x


def _class_names(*, binary: bool) -> List[str]:
    # Keep this local to avoid importing globals from utils.visualisation.
    return ["Normal", "Arrhythmia"] if bool(binary) else ["Normal", "Supraventricular", "Ventricular", "Fusion", "Unknown"]


def _labels_to_class_index(Y: np.ndarray) -> np.ndarray:
    """Convert Y in common repo formats to a 1D integer class index per sample."""
    Y_np = np.asarray(Y)
    if Y_np.ndim > 2:
        # one-hot per timestep (N, T, C) -> class id from first timestep
        return np.array([int(np.argmax(y_seq[0])) for y_seq in Y_np], dtype=int)
    if Y_np.ndim == 2 and Y_np.shape[1] > 1:
        # one-hot per sample (N, C)
        return np.array([int(np.argmax(y_row)) for y_row in Y_np], dtype=int)
    # label per timestep (N, T) or (N, 1)
    return Y_np.reshape(len(Y_np), -1)[:, 0].astype(int)


def _save_ecg_sample_plot(
    *,
    x: np.ndarray,
    class_name: str,
    index_within_class: int,
    filename: str,
    show: bool = False,
    sampling_freq_hz: float = _ECG_SAMPLING_FREQ_HZ,
) -> None:
    """Save a single ECG trace as a clean, paper-friendly plot."""
    import matplotlib.pyplot as plt

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    t_ms = _time_axis_ms(len(x), sampling_freq_hz=float(sampling_freq_hz))

    fig, ax = plt.subplots(figsize=(7.5, 2.6), constrained_layout=True)
    ax.plot(t_ms, x, color="#CC0000", linewidth=1.4)
    ax.grid(True, alpha=0.18)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("ECG")
    ax.text(
        0.01,
        0.88,
        f"{class_name}  (idx={int(index_within_class)})",
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2},
    )

    _save_figure(filename=filename)
    if bool(show):
        plt.show()
    plt.close(fig)


def visualise_dataset(args: argparse.Namespace) -> None:
    # Load a reasonably sized dataset for visualisation (no noise/augmentation here).
    # We load the *raw* waveforms (no standardization) so we can control scaling
    # per-figure: average/noise use sequence_zscore, and scaler_comparison shows
    # minmax vs standard vs sequence_zscore on the same raw sample.
    X_train, Y_train, X_test, Y_test = load_ecg_data(
        rows=args.rows,
        test_ratio=args.test_ratio,
        encode_labels=args.encode_labels,
        standardize=False,           # we apply scaling explicitly per-plot
        repeat_targets=False,
        shuffle=True,
        binary=args.binary,
        noise_rate=0.0,
        noise_ratio=0.0,
        scaler_type="minmax",       # not used when standardize=False
        balance_classes=True,
        max_per_class=None,
    )

    # 1) Class / sample distribution (train + test combined labels)
    # ``plot_data_distribution`` expects a sequence where each element is
    # indexable (targets[0]) and, for one-hot, possibly a sequence itself.
    # We therefore build a simple (N, 1) integer-label array that is
    # compatible with ``count_labels``.
    if args.encode_labels:
        # Y_train/Y_test are one-hot sequences per timestep; reduce to class per sample
        train_labels = [int(np.argmax(y_seq[0])) for y_seq in Y_train]
        test_labels = [int(np.argmax(y_seq[0])) for y_seq in Y_test]
    else:
        train_labels = [int(y_seq[0]) for y_seq in Y_train]
        test_labels = [int(y_seq[0]) for y_seq in Y_test]

    all_labels = np.array(train_labels + test_labels).reshape(-1, 1)
    # plot_data_distribution(all_labels, filename="ecg_data_distribution.png", show=False)

    # 2) Average instances figure (mean + quantile band + multiple exemplars)
    #    We compute it on sequence_zscore-scaled waveforms.
    X_train_np = np.asarray(X_train, dtype=np.float64)
    X_train_scaled, _ = standardize_data(X_train_np, X_train_np, scaler_type="sequence_zscore")
    plot_average_instances_with_band_and_exemplars(
        X_train_scaled,
        np.asarray(Y_train),
        filename="ecg_average_instances.png",
        show=False,
        sampling_freq_hz=_ECG_SAMPLING_FREQ_HZ,
        example_traces=int(args.example_traces),
        q_low=0.10,
        q_high=0.90,
        seed=int(args.seed),
        binary=bool(args.binary),
    )

    # 2b) Save per-instance ECG sample plots
    if bool(args.save_samples) or bool(args.save_all_samples):
        labels = _labels_to_class_index(np.asarray(Y_train))
        class_names = _class_names(binary=bool(args.binary))
        unique_classes = sorted(set(int(c) for c in np.unique(labels)))

        # Build index lists for each class.
        idx_by_class: Dict[int, List[int]] = {}
        for c in unique_classes:
            idx_by_class[int(c)] = [int(i) for i in np.where(labels == int(c))[0].tolist()]

        rng = np.random.RandomState(int(args.seed))
        for c in unique_classes:
            idxs = idx_by_class.get(int(c), [])
            if len(idxs) == 0:
                continue

            # Choose which indices to save.
            if bool(args.save_all_samples):
                picked = idxs
            else:
                k = int(args.samples_per_class)
                k = max(1, k)  # default: 1 per class
                k = min(k, len(idxs))
                picked = [int(i) for i in rng.choice(idxs, size=k, replace=False).tolist()]

            name = class_names[int(c)] if int(c) < len(class_names) else f"class_{int(c)}"
            # Folder must be safe for filesystem.
            folder = name.replace(" ", "_")
            for i in picked:
                x = np.asarray(X_train[i]).reshape(-1)
                # Save to results/dataset/samples/<class>/<index>.png
                out = os.path.join("dataset", "samples", folder, f"{int(i)}.png")
                _save_ecg_sample_plot(
                    x=x,
                    class_name=name,
                    index_within_class=int(i),
                    filename=out,
                    show=False,
                    sampling_freq_hz=_ECG_SAMPLING_FREQ_HZ,
                )

    # 3) Noise example on a single sample
    x_single, y_single = _select_single_sample(
        X_train,
        Y_train,
        index=int(args.single_index),
        target_class=(int(args.single_class) if args.single_class is not None else None),
    )
    # Noise plot uses sequence_zscore-scaled input (same scaling convention as average-instance figs)
    # Per-sequence z-score *for this single instance* (avoid object-dtype pitfalls).
    x_single = np.asarray(x_single, dtype=np.float64).reshape(-1)
    mu = float(np.mean(x_single))
    sigma = float(np.std(x_single))
    if sigma < 1e-8:
        sigma = 1.0
    x_single = (x_single - mu) / sigma
    noise, noisy_x = _make_noise_example(x_single, noise_std=args.noise_std, seed=args.seed)
    plot_noise(
        x_single,
        noise,
        noisy_x,
        noise_std=float(args.noise_std),
        filename="ecg_noise_example.png",
        show=False,
    )

    # 4) Comparison of scaling methods for a single sample
    #    We'll take the same sample, put it in a tiny dataset, and run both scalers.
    # For the comparison plot we want to compare scalers starting from the same *raw* signal.
    # Build a small raw subset for fitting scaler parameters.
    X_single_dataset = np.array(X_train[: min(len(X_train), 128)])  # small subset for fitting scalers
    X_single_dataset = X_single_dataset.reshape(X_single_dataset.shape[0], -1, 1)

    # Ensure the selected instance is the one we plot after scaling by placing it at index 0.
    # (standardize_data returns scaled arrays aligned with the input order.)
    # IMPORTANT: `x_single` above is sequence_zscore-scaled for the noise figure.
    # For the scaler comparison, we need the raw selected sample again.
    x_single_raw, _ = _select_single_sample(
        X_train,
        Y_train,
        index=int(args.single_index),
        target_class=(int(args.single_class) if args.single_class is not None else None),
    )
    X_single_dataset[0] = np.asarray(x_single_raw).reshape(-1, 1)

    # Original unscaled version of the chosen sample
    x_orig = X_single_dataset[0].reshape(-1)

    # Min-max scaling
    X_mm, _ = standardize_data(X_single_dataset, X_single_dataset, scaler_type="sequence_minmax_pos")
    x_mm = X_mm[0].reshape(-1)

    # Standard (z-score) scaling
    X_std, _ = standardize_data(X_single_dataset, X_single_dataset, scaler_type="standard")
    x_std = X_std[0].reshape(-1)

    # Per-sequence (z-score) scaling
    X_zscore, _ = standardize_data(X_single_dataset, X_single_dataset, scaler_type="sequence_zscore")
    x_zscore = X_zscore[0].reshape(-1)

    # Plot comparison
    import matplotlib.pyplot as plt
    from utils.visualisation import _save_figure

    # Accessibility + readability: small multiples (4 subplots) instead of 4 overlapping lines.
    t_ms = _time_axis_ms(len(x_orig))

    # Okabe–Ito palette (https://jfly.uni-koeln.de/color/)
    palette = {
        "black": "#000000",
        "blue": "#0072B2",
        "vermillion": "#D55E00",
        "bluish_green": "#009E73",
    }

    # Single overlay plot is clearer here: we distinguish methods using linestyle/alpha,
    # and we emphasize the chosen scaler (per-sequence z-score).
    # Match the noise figure height so the waveform is easy to read.
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    l_mm = ax.plot(
        t_ms,
        x_mm,
        color=palette["blue"],
        alpha=0.28,
        linewidth=1.6,
        label="Min-max (reference)",
    )[0]
    l_std = ax.plot(
        t_ms,
        x_std,
        color=palette["vermillion"],
        linewidth=1.7,
        linestyle=(0, (4, 2)),
        label="Z-score (standard)",
    )[0]
    l_z = ax.plot(
        t_ms,
        x_zscore,
        color=palette["bluish_green"],
        linewidth=2.1,
        linestyle="-",
        label="Z-score (per-seq) [used]",
    )[0]

    ax.grid(True, alpha=0.2)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Value")
    ax.legend(handles=[l_mm, l_std, l_z], loc="upper right", fontsize=9, frameon=True)

    # No figure title (paper-friendly and avoids overlap)
    _save_figure(filename="preprocessing/ecg_scalers_comparison.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise ECG classification dataset")
    parser.add_argument("--rows", type=int, default=5000, help="Number of training rows to load")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--encode-labels", action="store_true", help="One-hot encode labels (matches training setup)")
    parser.add_argument("--binary", action="store_true", help="Use binary classification version of the ECG dataset")
    parser.add_argument("--noise-std", type=float, default=0.2, help="Std dev of additive Gaussian noise for single-sample example")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for noise example")
    parser.add_argument(
        "--example-traces",
        type=int,
        default=5,
        help="Number of example ECG instances to overlay per class in the average-instances figure",
    )
    parser.add_argument(
        "--single-index",
        type=int,
        default=0,
        help=(
            "Index of the single ECG instance used for the noise + scaler comparison figures. "
            "If --single-class is also set, this is the index *within that class* (after shuffling)."
        ),
    )
    parser.add_argument(
        "--single-class",
        type=int,
        default=None,
        help=(
            "Optional class index constraint for selecting the single ECG instance. "
            "If set, the instance is chosen from this class only (and --single-index is within-class)."
        ),
    )

    # Save per-instance ECG trace plots under results/dataset/samples/<class>/<index>.png
    parser.add_argument(
        "--save-samples",
        action="store_true",
        help="Save ECG instance plots under results/dataset/samples/<class>/<index>.png (defaults to 1 per class)",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=1,
        help="How many ECG instances to save per class (ignored if --save-all-samples)",
    )
    parser.add_argument(
        "--save-all-samples",
        action="store_true",
        help="Save *all* ECG instances per class (can generate lots of files)",
    )

    args = parser.parse_args()
    visualise_dataset(args)


if __name__ == "__main__":
    main()
