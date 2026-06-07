from typing import List, Dict, Any, Optional
import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from utils.analysis import count_labels, measure_class_deviation

# Set global visualization config
sns.set_style("ticks")

plt.rcParams.update({'font.size': 15})

DPI = 800
RESULTS_DIR = "results/"

# Define class labels and palettes
binary_classes = ['Normal', 'Arrhythmia']
binary_palette = sns.color_palette("husl", n_colors=len(binary_classes))[::-1]

classes = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown']
categorical_palette = sns.color_palette("husl", n_colors=len(classes))[::-1]


# ----------------------------------------------------------------------------
# Display names for oscillator variables / genes
# ----------------------------------------------------------------------------

# We prefer using full gene names in filenames, legends, and axis labels.
# Note: this repo historically uses both single-letter variables and (Hi/He).
GENE_DISPLAY_NAME = {
    "I": "luxI",
    "A": "aiiA",
    "Hi": "Hi",
    "He": "He",
    # Allow passing full names through unchanged.
    "luxI": "luxI",
    "aiiA": "aiiA",
}


def gene_full_name(name: str) -> str:
    """Return the preferred display name for a gene/state variable."""
    return GENE_DISPLAY_NAME.get(str(name), str(name))


def format_sci(x: float, *, max_decimals: int = 2) -> str:
    """Format numbers with scientific notation when they would be too long.

    Rule: if rounding to `max_decimals` would change the value materially (or
    the value is very small), we switch to compact e-notation.

    Examples
    --------
    - 0.001  -> 1e-3
    - 0.0001 -> 1e-4
    - 0.0123 -> 0.01 (if max_decimals=2)
    - 1.234  -> 1.23
    """
    x = float(x)
    if x == 0.0:
        return "0"
    # Prefer e-notation for very small values.
    if abs(x) < 10 ** (-(max_decimals + 1)):
        return f"{x:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    rounded = round(x, int(max_decimals))
    if rounded != x:
        return f"{x:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    return f"{rounded:.{int(max_decimals)}f}".rstrip("0").rstrip(".")

def _save_figure(filename: str) -> None:
    """
    Save the current figure with a given filename.

    Args:
        filename (str): The name of the file to save.
    """
    file_path = os.path.join(RESULTS_DIR, filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.savefig(file_path, bbox_inches="tight", dpi=DPI)

def plot_states(states: np.ndarray, labels: List[str] = None, xlabel: str = "Time", 
         ylabel: str = "State", filename: str = "states.png", legend: bool = True, show: bool = True) -> None:
    """
    Plot a generic line graph of the evolution of system states over time.

    Args:
        states (np.ndarray): Time-series states to plot.
        labels (List[str], optional): Labels for the data.
        xlabel (str, optional): Label for the x-axis.
        ylabel (str, optional): Label for the y-axis.
        filename (str, optional): Name of the file to save.
        legend (bool, optional): Whether to show the legend.
        show (bool, optional): Whether to display the plot.
    """
    timesteps = len(states)
    time = np.linspace(0, timesteps, timesteps)

    plt.figure(figsize=(10, 6))
    for i in range(states.shape[1]):
        plt.plot(time, states[:, i], label=labels[i] if labels else None)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    if legend:
        plt.legend()
    _save_figure(filename=os.path.join("states/", filename))
    
    if show:
        plt.show()


def plot_dde_overlay(
    *,
    t_axis: np.ndarray,
    baseline_trace: np.ndarray,
    driven_traces: List[np.ndarray],
    driven_applied: List[np.ndarray],
    amps: List[float],
    warmup: int,
    overlay_variable: str,
    filename: str,
    show: bool = False,
    show_peaks: bool = False,
) -> None:
    """Plot the main DDE overlay figure used in Methods.

    Creates a single figure with two stacked axes:
    - top: chosen state variable overlayed across amplitudes
    - bottom: applied DDE drive (after preprocessing) for each amplitude

    Parameters
    ----------
    t_axis:
        1D array of timesteps covering warmup + driven segment.
    baseline_trace:
        Arrays for the 0-input baseline.
    driven_traces / driven_applied:
        Lists aligned with `amps`.
    amps:
        Amplitudes (excluding 0) corresponding to driven_* series.
    warmup:
        Warmup length (rendered dashed for baseline).
    filename:
        Path relative to results/ (caller typically uses `states/...png`).
    """
    warmup = int(warmup)

    # Small multiples (amplitude rows) to avoid relying on color for discrimination.
    # We intentionally *omit* the right-column input subplots: in practice they add
    # little information for these figures, and removing them simplifies the layout.
    n_rows = 1 + int(len(amps))
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=1,
        sharex=True,
        figsize=(12.5, 2.2 * n_rows),
        gridspec_kw={"hspace": 0.12},
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = np.array([axes])

    # Okabe–Ito palette (colorblind-friendly) for subtle accents; not required for discrimination.
    okabe_ito = [
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
        "#F0E442",
        "#000000",
    ]

    # Baseline row
    ax_var0 = axes[0]
    if warmup > 0:
        ax_var0.plot(t_axis[:warmup], baseline_trace[:warmup], color="black", linestyle="--", linewidth=2.0)
        ax_var0.plot(t_axis[warmup - 1 :], baseline_trace[warmup - 1 :], color="black", linewidth=1.8)
        ax_var0.axvline(warmup - 1, color="black", alpha=0.30, linewidth=1.0)
    else:
        ax_var0.plot(t_axis, baseline_trace, color="black", linewidth=1.8)

    ax_var0.text(
        0.01,
        0.88,
        "amp=0 (baseline)",
        transform=ax_var0.transAxes,
        fontsize=9,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2},
    )

    # Driven rows
    for i, (a, trace, applied) in enumerate(zip(amps, driven_traces, driven_applied), start=1):
        col = okabe_ito[(i - 1) % len(okabe_ito)]
        ax_var = axes[i]

        if warmup > 0:
            # Show baseline warmup segment as a reference (dashed), then driven trace.
            ax_var.plot(t_axis[:warmup], baseline_trace[:warmup], color="black", linestyle="--", linewidth=1.4, alpha=0.6)
            ax_var.plot(t_axis[warmup - 1 :], trace[warmup - 1 :], color=col, linewidth=1.8)
            ax_var.axvline(warmup - 1, color="black", alpha=0.30, linewidth=1.0)
        else:
            ax_var.plot(t_axis, trace, color=col, linewidth=1.8)

        if bool(show_peaks):
            t_peaks, y_peaks = _peak_envelope(
                trace,
                t_axis,
                start=max(0, warmup),
                min_separation=5,
            )
            if t_peaks.size >= 2:
                ax_var.plot(
                    t_peaks,
                    y_peaks,
                    color=col,
                    linewidth=1.2,
                    linestyle=":",
                    alpha=0.95,
                )

    # (input trace intentionally not plotted)

        ax_var.text(
            0.01,
            0.88,
            f"amp={format_sci(a)}",
            transform=ax_var.transAxes,
            fontsize=9,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2},
        )

    # Shared labels/formatting
    for r in range(n_rows):
        axes[r].grid(True, alpha=0.22)

    # Label shared axes once (centered) instead of repeating on every subplot.
    fig.supxlabel("Timestep")
    fig.supylabel(f"State of {gene_full_name(overlay_variable)}")

    # Legend to explain warmup linestyle.
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.8, label="Warmup (0 input)"),
        Line2D([0], [0], color="black", linestyle="-", linewidth=1.8, label="Trace (sine input)"),
    ]
    axes[0].legend(handles=legend_items, loc="upper right", fontsize=8, frameon=True)

    _save_figure(filename=filename)
    if show:
        plt.show()


def plot_dde_ecg_multiscale(
    *,
    t_axis: np.ndarray,
    warmup: int,
    baseline_states_by_var: Dict[str, np.ndarray],
    driven_states_by_scale: Dict[float, Dict[str, np.ndarray]],
    driven_input_by_scale: Dict[float, np.ndarray],
    input_trace_unscaled: Optional[np.ndarray] = None,
    ecg_class_label: Optional[str] = None,
    filename: str,
    show: bool = False,
) -> None:
    """Plot a compact ECG-driven overlay figure across multiple input scales.

    Layout
    ------
    - Row 1: baseline (0 input) for luxI + aiiA.
    - Rows 2..K: ECG-driven for each requested input scale.
    - Final row: the ECG input trace (for the *largest* scale), plotted once.

    Notes
    -----
    - Warmup (first ``warmup`` steps) is drawn dashed (baseline shown as a reference).
    - Colors are fixed across all rows to keep the figure consistent and greyscale-friendly.
    """
    import numpy as _np
    from matplotlib.lines import Line2D

    warmup = int(warmup)
    t_axis = _np.asarray(t_axis)

    # Fixed, colourblind-friendly palette; still readable in greyscale thanks to linestyles.
    okabe_ito = {
        "blue": "#0072B2",
        "vermillion": "#D55E00",
        "black": "#000000",
    }
    styles = {
        "I": dict(color=okabe_ito["blue"], linewidth=1.8, linestyle="-"),
        "A": dict(color=okabe_ito["vermillion"], linewidth=1.8, linestyle=(0, (4, 2))),
    }

    scales = sorted(float(s) for s in driven_states_by_scale.keys())
    if len(scales) == 0:
        raise ValueError("driven_states_by_scale must contain at least one scale")

    # Choose a single input trace to show once.
    # We often want to show the ECG in a fixed, "default" scale, since each state row already
    # corresponds to a different input scaling.
    if input_trace_unscaled is not None:
        u_show = _np.asarray(input_trace_unscaled, dtype=_np.float64).reshape(-1)
    else:
        scale_for_input = max(scales)
        u_show = _np.asarray(driven_input_by_scale[scale_for_input], dtype=_np.float64).reshape(-1)

    n_rows = 1 + len(scales) + 1
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=1,
        sharex=True,
        figsize=(12.0, 2.0 * n_rows),
        gridspec_kw={"hspace": 0.10, "height_ratios": [2.2] * (n_rows - 1) + [1.2]},
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = _np.array([axes])

    def _plot_states_row(ax, states_by_var: Dict[str, _np.ndarray], *, show_baseline_warmup: bool) -> None:
        # Only luxI (I) + aiiA (A)
        for v in ("I", "A"):
            if v not in states_by_var:
                continue
            y = _np.asarray(states_by_var[v], dtype=_np.float64).reshape(-1)
            if warmup > 0 and show_baseline_warmup:
                y_base = _np.asarray(baseline_states_by_var[v], dtype=_np.float64).reshape(-1)
                ax.plot(t_axis[:warmup], y_base[:warmup], color="black", linestyle="--", linewidth=1.4, alpha=0.75)
                ax.plot(t_axis[warmup - 1 :], y[warmup - 1 :], **styles[v])
                ax.axvline(warmup - 1, color="black", alpha=0.25, linewidth=1.0)
            elif warmup > 0:
                ax.plot(t_axis[:warmup], y[:warmup], color="black", linestyle="--", linewidth=1.4, alpha=0.75)
                ax.plot(t_axis[warmup - 1 :], y[warmup - 1 :], **styles[v])
                ax.axvline(warmup - 1, color="black", alpha=0.25, linewidth=1.0)
            else:
                ax.plot(t_axis, y, **styles[v])

        ax.grid(True, alpha=0.22)

    # Row 0: baseline
    _plot_states_row(axes[0], baseline_states_by_var, show_baseline_warmup=False)
    axes[0].text(
        0.01,
        0.86,
        "Baseline (0 input)",
        transform=axes[0].transAxes,
        fontsize=9,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2},
    )

    # Rows 1..len(scales): driven
    for i, s in enumerate(scales, start=1):
        _plot_states_row(axes[i], driven_states_by_scale[s], show_baseline_warmup=True)
        axes[i].text(
            0.01,
            0.86,
            f"Scale={format_sci(s)}",
            transform=axes[i].transAxes,
            fontsize=9,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2},
        )

    # Bottom row: input trace (shown once) – visually separated from state panels.
    ax_inp = axes[-1]
    # Plot warmup explicitly as dashed black with a divider, then the ECG drive.
    if warmup > 0:
        ax_inp.plot(
            t_axis[:warmup],
            u_show[:warmup],
            color="black",
            linestyle="--",
            linewidth=1.4,
            alpha=0.85,
        )
        # Divider marking the start of the driven (ECG) segment.
        ax_inp.axvline(warmup - 1, color="black", alpha=0.35, linewidth=1.0)
        # Use a distinct red (separate from aiiA's orange/vermillion) to avoid confusion.
        ax_inp.plot(t_axis[warmup - 1 :], u_show[warmup - 1 :], color="#CC0000", linewidth=1.6)
    else:
        ax_inp.plot(t_axis, u_show, color="#CC0000", linewidth=1.6)
    ax_inp.grid(True, alpha=0.18)
    ax_inp.set_ylabel("ECG Input")
    ax_inp.set_xlabel("")

    # Legend: show ECG class/label and distinguish warmup vs driven segment.
    if ecg_class_label is not None and str(ecg_class_label).strip() != "":
        ecg_label_txt = f"ECG Class: {str(ecg_class_label)}"
    else:
        ecg_label_txt = "ECG"

    warmup_handle_inp = Line2D([0], [0], color="black", linestyle="--", linewidth=1.4, label="Warmup (0 input)")
    ecg_handle_inp = Line2D([0], [0], color="#CC0000", linestyle="-", linewidth=1.6, label=ecg_label_txt)
    ax_inp.legend(
        handles=[ecg_handle_inp, warmup_handle_inp],
        loc="upper right",
        fontsize=8,
        frameon=True,
    )

    # Shared labels once (centered) and no per-axes titles.
    fig.supxlabel("Timestep")
    fig.supylabel("Genetic Biopixel States")

    # Legend: variable styles + warmup explanation.
    warmup_handle = Line2D([0], [0], color="black", linestyle="--", linewidth=1.4, label="Warmup (0 input)")
    luxi_handle = Line2D([0], [0], **styles["I"], label=gene_full_name("I"))
    aiiA_handle = Line2D([0], [0], **styles["A"], label=gene_full_name("A"))
    axes[0].legend(
        handles=[luxi_handle, aiiA_handle, warmup_handle],
        loc="upper right",
        fontsize=8,
        frameon=True,
    )

    _save_figure(filename=filename)
    if show:
        plt.show()



def _peak_envelope(
    y: np.ndarray,
    t: np.ndarray,
    *,
    start: int = 0,
    min_separation: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a simple 'peak-to-peak trend' line.

    We detect local maxima in ``y`` and return their (t, y) coordinates.
    The caller can then plot these points connected by a line to visualize the
    envelope/trend across peaks.

    Notes
    -----
    - This is intentionally lightweight (no scipy dependency).
    - ``min_separation`` prevents dense peak picks on noisy signals.
    """
    y = np.asarray(y)
    t = np.asarray(t)
    n = int(y.shape[0])
    start = max(0, int(start))
    if n < 3 or start >= n - 2:
        return np.empty((0,), dtype=t.dtype), np.empty((0,), dtype=y.dtype)

    # Local maxima: y[i-1] < y[i] >= y[i+1]
    idx = np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1
    idx = idx[idx >= start]
    if idx.size == 0:
        return np.empty((0,), dtype=t.dtype), np.empty((0,), dtype=y.dtype)

    idx_kept = [int(idx[0])]
    last = int(idx[0])
    sep = max(1, int(min_separation))
    for i in idx[1:]:
        i = int(i)
        if i - last >= sep:
            idx_kept.append(i)
            last = i

    idx_kept = np.asarray(idx_kept, dtype=np.int64)
    return t[idx_kept], y[idx_kept]


def plot_data_distribution(Y: list, filename: str = "data-distribution.png", show: bool = True):
    """
    Generate a pie chart to visualize the distribution of data labels.

    Args:
        Y (list): List of data labels.
        filename (str, optional): Filename to save the generated plot. Default is "data-distribution.png".
        show (bool, optional): Whether to display the plot. Default is True.
    """
    label_counts = count_labels(Y)
    if len(label_counts) > 2:
        class_labels = [classes[label] for label in label_counts.keys()]
        colors = categorical_palette
    else:
        class_labels = [binary_classes[label] for label in label_counts.keys()]
        colors = binary_palette

    plt.rcParams.update({'font.size': 14})
    _, ax = plt.subplots(figsize=(14, 8))
    _, _, autotexts = ax.pie(label_counts.values(), autopct='%1.1f%%', colors=colors)

    for autotext in autotexts:
        autotext.set_color('black')

    plt.legend(class_labels, loc="best", fontsize='large')
    plt.tight_layout()

    _save_figure(filename=os.path.join("preprocessing/", filename))

    if show:
        plt.show()

def plot_dataset_info(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray, 
                      show=True, filename="dataset-table.png") -> None:
    """
    Create a table to display information about the dataset.

    Args:
        X_train (np.ndarray): Training data.
        Y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Testing data.
        Y_test (np.ndarray): Testing labels.
        show (bool, optional): Whether to display the table. Default is True.
        filename (str, optional): Filename to save the generated table. Default is "dataset-table.png".
    """
    train_label_counts = count_labels(Y_train)
    test_label_counts = count_labels(Y_test)

    # Create figure and axis
    _, ax = plt.subplots(figsize=(10, 3))

    # Create table
    table_data = [
        ["", "Training", "Testing"],
        ["Instances", len(X_train), len(X_test)],
        ["Targets Shape", str(X_train[0].shape), str(Y_train[0].shape)],
        ["Labels Shape", str(X_test[0].shape), str(Y_test[0].shape)],
        ["Class Size", train_label_counts[0], test_label_counts[0]]
    ]

    table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.2, 0.2, 0.2])

    # Hide axes
    ax.axis('off')

    # Styling the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 1.5)

    _save_figure(filename=os.path.join("preprocessing/", filename))
    if show:
        plt.show()

def plot_class_std(X: np.ndarray, Y: np.ndarray, show: bool = True, filename: str = "class_std.png"):
    """
    Plot the mean standard deviation of each class.

    Args:
        X (np.ndarray): Input data.
        Y (np.ndarray): Labels.
        show (bool, optional): Whether to display the plot. Default is True.
        filename (str, optional): Filename to save the plot. Default is "class_std.png".
    """
    # if one-hot encoded, reverse encoding
    if len(Y.shape) > 2:
        Y = np.array([np.argmax(y, axis=1) for y in Y])

    _, class_std = measure_class_deviation(X, Y)
    mean_class_std = {label: np.mean(std) for label, std in class_std.items()}

    if len(class_std) > 2:
        class_labels = [classes[label] for label in class_std.keys()]
        colors = categorical_palette
    else:
        class_labels = [binary_classes[label] for label in class_std.keys()]
        colors = binary_palette

    _, ax = plt.subplots(figsize=(8, 6))
    bar_width = 0.5
    ax.bar(np.arange(len(class_labels)), mean_class_std.values(), color=colors, width=bar_width)
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=20)
    ax.set_xlabel('Class')
    ax.set_ylabel('Mean Standard Deviation')
    plt.tight_layout()

    _save_figure(filename=os.path.join("preprocessing/", filename))
    if show:
        plt.show()

def plot_class_mean(X: np.ndarray, Y: np.ndarray, show: bool = True, filename: str = "class_means.png") -> None:
    """
    Plot the mean value of each class.

    Args:
        X (np.ndarray): Input data.
        Y (np.ndarray): Labels.
        show (bool, optional): Whether to display the plot. Default is True.
        filename (str, optional): Filename to save the plot. Default is "class_means.png".
    """
    if len(Y.shape) > 2:
        Y = np.array([np.argmax(y, axis=1) for y in Y])

    class_means, _ = measure_class_deviation(X, Y)
    mean_class_means = {label: np.mean(means) for label, means in class_means.items()}

    if len(class_means) > 2:
        class_labels = [classes[label] for label in class_means.keys()]
        colors = categorical_palette
    else:
        class_labels = [binary_classes[label] for label in class_means.keys()]
        colors = binary_palette

    _, ax = plt.subplots(figsize=(8, 6))
    bar_width = 0.5
    ax.bar(np.arange(len(class_labels)), mean_class_means.values(), color=colors, width=bar_width)
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=20)
    ax.set_xlabel('Class')
    ax.set_ylabel('Mean Value')
    plt.tight_layout()

    _save_figure(filename=os.path.join("preprocessing/", filename))
    if show:
        plt.show()


def plot_average_instances_with_band_and_exemplars(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    filename: str = "ecg_average_instances.png",
    show: bool = False,
    sampling_freq_hz: float = 125.0,
    example_traces: int = 5,
    q_low: float = 0.10,
    q_high: float = 0.90,
    seed: int = 0,
    binary: Optional[bool] = None,
) -> None:
    """Paper-ready average ECG instances figure.

    Layout: one row per class, one column:
    - class mean with quantile band (q_low..q_high)

    Notes
    -----
    - Expects X to be numeric and shaped (N, T, 1) or (N, T).
    - If Y is one-hot per timestep (N, T, C) we reduce to class indices.
    """
    X_np = np.asarray(X, dtype=np.float64)
    if X_np.ndim == 3:
        X_np = X_np[..., 0]
    if X_np.ndim != 2:
        raise ValueError(f"X must have shape (N, T) or (N, T, 1); got {X_np.shape}")

    Y_np = np.asarray(Y)
    if Y_np.ndim > 2:
        # one-hot per timestep -> class id per sample using first timestep
        y_classes = np.array([int(np.argmax(y_seq[0])) for y_seq in Y_np], dtype=int)
    elif Y_np.ndim == 2 and Y_np.shape[1] > 1:
        # one-hot per sample
        y_classes = np.array([int(np.argmax(y_row)) for y_row in Y_np], dtype=int)
    else:
        y_classes = Y_np.reshape(-1).astype(int)

    unique_classes = np.unique(y_classes)
    if unique_classes.size == 0:
        raise ValueError("No classes found in Y")

    # Decide label set
    if binary is None:
        binary = unique_classes.size <= 2
    class_names = binary_classes if binary else classes

    # Time axis in ms
    sampling_freq_hz = float(sampling_freq_hz)
    if sampling_freq_hz <= 0:
        raise ValueError(f"sampling_freq_hz must be > 0; got {sampling_freq_hz}")
    t_ms = (np.arange(X_np.shape[1], dtype=np.float64) / sampling_freq_hz) * 1000.0

    # Collect per-class stats
    class_rows = []  # list[(class_idx, mean, q_lo, q_hi)]
    for c in unique_classes:
        mask = y_classes == int(c)
        seqs = X_np[mask].astype(np.float64, copy=False)
        if seqs.size == 0:
            continue
        mean_wave = np.asarray(seqs.mean(axis=0), dtype=np.float64)
        q_lo = np.asarray(np.quantile(seqs, float(q_low), axis=0), dtype=np.float64)
        q_hi = np.asarray(np.quantile(seqs, float(q_high), axis=0), dtype=np.float64)

        class_rows.append((int(c), mean_wave, q_lo, q_hi))

    if len(class_rows) == 0:
        raise ValueError("No class rows could be constructed (empty masks?)")

    # Consistent y-limits across panels
    all_y = np.concatenate(
        [np.asarray(y).reshape(-1) for _, y, *_ in class_rows]
        + [np.asarray(qlo).reshape(-1) for _, _, qlo, *_ in class_rows]
        + [np.asarray(qhi).reshape(-1) for _, _, _, qhi, *_ in class_rows]
    )
    ymin = float(np.nanmin(all_y))
    ymax = float(np.nanmax(all_y))
    pad = 0.05 * (ymax - ymin) if ymax > ymin else 1.0
    ylim = (ymin - pad, ymax + pad)

    # Okabe–Ito palette (colorblind-friendly)
    okabe_ito = [
        "#0072B2",  # blue
        "#D55E00",  # vermillion
        "#009E73",  # bluish green
        "#CC79A7",  # reddish purple
        "#E69F00",  # orange
        "#56B4E9",  # sky blue
        "#F0E442",  # yellow
        "#000000",  # black
    ]

    fig, axes = plt.subplots(
        nrows=len(class_rows),
        ncols=1,
        sharex=True,
        sharey=True,
        figsize=(10.0, 2.0 * len(class_rows) + 0.9),
        gridspec_kw={"hspace": 0.12},
        constrained_layout=True,
    )
    if len(class_rows) == 1:
        axes = np.array([axes])

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    band_label = f"{int(q_low*100)}–{int(q_high*100)}% band"

    for i, (c, mean_wave, q_lo, q_hi) in enumerate(class_rows):
        col = okabe_ito[i % len(okabe_ito)]
        ax_mean = axes[i]

        ax_mean.fill_between(t_ms, q_lo, q_hi, color=col, alpha=0.18, linewidth=0)
        ax_mean.plot(t_ms, mean_wave, color=col, linewidth=1.9)

        ax_mean.set_ylim(*ylim)
        ax_mean.grid(True, alpha=0.2)

        name = class_names[c] if int(c) < len(class_names) else f"class {c}"
        ax_mean.text(
            0.01,
            0.88,
            name,
            transform=ax_mean.transAxes,
            fontsize=10,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2},
        )

        # Per-row legend that explains the mark types (mean vs band).
        # Use the same colour as the plotted traces for consistency.
        mean_label = f"Mean ({name})"
        handles = [
            Line2D([0], [0], color=col, linewidth=1.9, linestyle="-", label=mean_label),
            Patch(facecolor=col, edgecolor="none", alpha=0.18, label=band_label),
        ]
        ax_mean.legend(handles=handles, loc="upper right", fontsize=8, frameon=True)

    axes[-1].set_xlabel("Time (ms)")
    fig.supylabel("Value")
    _save_figure(filename=os.path.join("preprocessing/", filename))

    if show:
        plt.show()

def plot_confusion_matrix(confusion_matrix: np.ndarray, cmap=plt.cm.Blues, 
                          show: bool = True, filename: str = "confusion_matrix.png") -> None:
    """
    Plot the confusion matrix.

    Args:
        confusion_matrix (np.ndarray): Confusion matrix.
        cmap: Colormap.
        show (bool, optional): Whether to display the plot. Default is True.
        filename (str, optional): Filename to save the plot. Default is "confusion_matrix.png".
    """
    confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    labels = confusion_matrix.shape[0]
    if labels > 2:
        class_labels = [classes[i] for i in range(labels)]
    else:
        class_labels = [binary_classes[i] for i in range(labels)]

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, cmap=cmap, xticklabels=class_labels, yticklabels=class_labels)

    # Adjust font size and rotation for better readability
    plt.xticks(rotation=30, ha='right', fontsize=14)
    plt.yticks(rotation=0, fontsize=14)

    plt.xlabel('Predicted Label', fontsize=16)
    plt.ylabel('True Label', fontsize=16)
    plt.tight_layout()

    _save_figure(filename=os.path.join("metrics/", filename))
    if show:
        plt.show()

def plot_metrics_across_folds(metrics: List[Dict[str, any]], metric_names: List[str] = ["accuracy", "f1", "mcc", "kappa"], 
                              show: bool = True, filename: str = "metrics_folds.png") -> None:
    """
    Plot the metrics across different folds.

    Args:
        metrics (List[Dict[str, any]]): List of dictionaries containing metric values for each fold.
        metric_names (List[str], optional): List of metric names to plot. Default is ["accuracy", "mse", "rmse", "f1"].
        show (bool, optional): Whether to display the plot. Default is True.
        filename (str, optional): Filename to save the plot. Default is "metrics_folds.png".
    """
    num_folds = len(metrics)
    num_metrics = len(metric_names)
    bar_width = 0.2
    index = np.arange(num_folds)

    plt.figure(figsize=(12, 6))
    for i in range(num_metrics):
        metric_values = [metric[metric_names[i]] for metric in metrics]
        plt.bar(index + i * bar_width, metric_values, bar_width, label=metric_names[i])

    plt.xlabel('Fold')
    plt.ylabel('Value')
    plt.xticks(index + bar_width * (num_metrics - 1) / 2, range(1, num_folds + 1))
    plt.legend()
    plt.tight_layout()

    _save_figure(filename=os.path.join("metrics/", filename))
    if show:
        plt.show()

def plot_class_metrics(metrics: Dict[str, Dict[str, float]], show: bool = True, filename: str = "class_metrics.png") -> None:
    """
    Plot the metrics for each class.

    Args:
        metrics (Dict[str, Dict[str, float]]): Dictionary containing metrics for each class.
        filename (str, optional): Filename to save figure to. Default is "class_metrics.png".
        show (bool, optional): Whether to display the plot. Default is True.
    """
    # Extract metrics for each class
    class_metrics = {key: value for key, value in metrics.items() if key.isdigit()}
    precisions = [metric['precision'] for metric in class_metrics.values()]
    recalls = [metric['recall'] for metric in class_metrics.values()]
    f1_scores = [metric['f1-score'] for metric in class_metrics.values()]

    class_labels = classes if len(class_metrics) > 2 else binary_classes

    # Create x-axis ticks and labels
    x_ticks = range(len(class_labels))
    x_labels = [class_labels[int(label)] for label in class_metrics.keys()]

    # Plot metrics
    plt.figure(figsize=(10, 6))
    plt.bar(x_ticks, precisions, width=0.2, label='Precision', align='center')
    plt.bar([x + 0.2 for x in x_ticks], recalls, width=0.2, label='Recall', align='center')
    plt.bar([x + 0.4 for x in x_ticks], f1_scores, width=0.2, label='F1-score', align='center')

    # Add labels and legend
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.xticks([x + 0.2 for x in x_ticks], x_labels, rotation=45, ha='right')
    plt.legend()

    # Show plot
    plt.tight_layout()

    _save_figure(filename=os.path.join("metrics/", filename))

    if show:
        plt.show()

def plot_noise(
    X: np.ndarray,
    noise: np.ndarray,
    noisy_X: np.ndarray,
    *,
    noise_std: Optional[float] = None,
    show: bool = True,
    filename: str = "noise.png",
) -> None:
    """
    Plot the noise in training data.

    Args:
        X (np.ndarray): Training data.
        noise (np.ndarray): Noise data.
        noisy_X (np.ndarray): Augmented data.
        show (bool, optional): Whether to display the plot. Default is True.
        filename (str, optional): Filename to save figure to. Default is "noise.png".
    """
    sampling_freq = 125
    num_samples = len(X)
    time_ms = np.arange(num_samples) / sampling_freq * 1000

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(time_ms, X, label="Training Data", linestyle="solid", color="orange")
    noise_label = "Noise"
    if noise_std is not None:
        try:
            noise_label = f"Noise ($\\sigma$={float(noise_std):g})"
        except Exception:
            pass
    ax.plot(time_ms, noise, linestyle="dotted", label=noise_label, color="green")
    ax.plot(time_ms, noisy_X, linestyle="dashed", label="Augmented Data", color="blue")
    plt.ylabel('Value')
    plt.xlabel('Time (ms)')
    plt.xticks()
    plt.yticks()

    ax.legend()

    _save_figure(filename=os.path.join("preprocessing/", filename))

    if show:
        plt.show()
    plt.close(fig)

def plot_tsne_clustering(Y_pred: np.ndarray, Y_true: np.ndarray, show: bool = True, filename: str = "clustering.png") -> None:
    # --- debug: inspect incoming shapes -----------------------------------
    print(f"[TSNE] raw Y_pred type={type(Y_pred)}, shape={getattr(Y_pred, 'shape', None)}")
    print(f"[TSNE] raw Y_true type={type(Y_true)}, shape={getattr(Y_true, 'shape', None)}")

    Y_pred_flat = np.array(Y_pred).reshape(len(Y_pred), -1)
    print(f"[TSNE] Y_pred_flat shape={Y_pred_flat.shape}")

    # Perform t-SNE embedding
    tsne = TSNE(
        n_components=2,
        perplexity=30,          # or smaller, e.g. 10
        learning_rate=200,      # or adjust upward
        n_iter=2000,
        metric="euclidean",     # default, but you could try 'cosine'
        init="random",          # maybe random instead of PCA
    )    
    tsne_embeddings = tsne.fit_transform(Y_pred_flat)
    print(f"[TSNE] tsne_embeddings shape={tsne_embeddings.shape}")

    Y_pred = np.array([np.argmax(y_p) for y_p in Y_pred]).reshape(-1, 1)
    Y_true = np.array([np.argmax(y_t) for y_t in Y_true]).reshape(-1, 1)
    print(f"[TSNE] Y_pred (classes) shape={Y_pred.shape}, unique={np.unique(Y_pred)}")
    print(f"[TSNE] Y_true (classes) shape={Y_true.shape}, unique={np.unique(Y_true)}")

    ticks = np.unique(Y_pred)

    class_names = classes
    colors = ListedColormap(categorical_palette)
    if len(ticks) == 2:
        class_names = binary_classes
        colors = ListedColormap(binary_palette)

    # Identify correct and incorrect predictions
    correct_preds = (Y_pred == Y_true)
    incorrect_preds = ~correct_preds

    n_total = Y_pred.shape[0]
    n_correct = int(correct_preds.sum())
    n_incorrect = int(incorrect_preds.sum())
    print(f"[TSNE] n_total={n_total}, n_correct={n_correct}, n_incorrect={n_incorrect}")

    # Optional: how many unique embedded coords?
    # This will confirm whether thousands of points have collapsed to only a few distinct locations.
    uniq_coords = np.unique(np.round(tsne_embeddings, decimals=3), axis=0)
    print(f"[TSNE] unique embedded coords (rounded 1e-3): {uniq_coords.shape[0]}")

    plt.figure(figsize=(10, 8))

    # Create empty scatter plots with black markers for legend
    scatter_correct = plt.scatter([], [], marker='o', label='Correct Predictions', color='k')
    scatter_incorrect = plt.scatter([], [], marker='x', label='Incorrect Predictions', color='k')

    # Use smaller markers and transparency to reveal density
    alpha_correct = 0.4
    alpha_incorrect = 0.8
    s_correct = 10
    s_incorrect = 20

    # Plot correct predictions
    plt.scatter(
        x=tsne_embeddings[correct_preds[:, 0], 0],
        y=tsne_embeddings[correct_preds[:, 0], 1],
        c=Y_true[correct_preds[:, 0]],
        cmap=colors,
        marker='o',
        s=s_correct,
        alpha=alpha_correct,
        linewidths=0,
    )

    # Plot incorrect predictions
    plt.scatter(
        x=tsne_embeddings[incorrect_preds[:, 0], 0],
        y=tsne_embeddings[incorrect_preds[:, 0], 1],
        c=Y_true[incorrect_preds[:, 0]],
        cmap=colors,
        marker='x',
        s=s_incorrect,
        alpha=alpha_incorrect,
        linewidths=0.7,
    )

    plt.legend(handles=[scatter_correct, scatter_incorrect])

    # Add color bar representing class labels
    cbar = plt.colorbar(ticks=ticks)
    cbar.set_ticklabels([class_names[int(t)] for t in ticks])
    cbar.set_label('Class')

    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')

    _save_figure(filename=os.path.join("metrics/", filename))
    if show:
        plt.show()


def _one_hot_to_class_indices(Y: np.ndarray) -> np.ndarray:
    """Convert one-hot labels of shape (N, 1, C) or (N, C) to class indices (N,)."""

    if Y.ndim == 3 and Y.shape[1] == 1:
        Y = Y[:, 0, :]
    if Y.ndim != 2:
        raise ValueError(f"Expected labels of shape (N, C) or (N, 1, C), got {Y.shape}")
    return np.argmax(Y, axis=1)


def plot_tsne_states(
    X_states: np.ndarray,
    Y_true: np.ndarray,
    show: bool = True,
    filename: str = "tsne_states.png",
) -> None:
    """t-SNE projection of reservoir states coloured by true class.

    Parameters
    ----------
    X_states : np.ndarray
        Array of reservoir states, e.g. shape (N, D) or (N, T, D).
    Y_true : np.ndarray
        One-hot labels matching X_states; shape (N, 1, C) or (N, C).
    show : bool
        Whether to display the figure interactively.
    filename : str
        File name (relative to ``results/metrics/``) used for saving.
    """

    # Flatten any trailing dimensions of states so we have (N, D)
    if X_states.ndim != 2:
        X = X_states.reshape(X_states.shape[0], -1)
    else:
        X = X_states

    y_classes = _one_hot_to_class_indices(Y_true)

    tsne = TSNE(
        n_components=2,
        learning_rate="auto",
        init="pca",
        perplexity=30,
        random_state=42,
    )
    emb = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    num_classes = int(np.max(y_classes)) + 1

    for c in range(num_classes):
        mask = y_classes == c
        if not np.any(mask):
            continue
        plt.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=10,
            alpha=0.6,
            label=f"class {c}",
        )

    plt.title("t-SNE of reservoir states (coloured by true class)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(markerscale=1.5, fontsize="small", loc="best")

    _save_figure(filename=os.path.join("metrics/", filename))
    if show:
        plt.show()


def plot_distance_histograms(units: int = 250,
                             seed: int = 0,
                             spatial_dim: int = 2,
                             bins: int = 50,
                             filename: str = "distance_histograms.png",
                             show: bool = True) -> Dict[str, float]:
    """Visualise the distribution of pairwise and nearest-neighbour distances.

    This helps understand the geometry of the randomly generated node cloud
    before applying any kernel.

    Args:
        units: Number of nodes.
        seed: Random seed.
        spatial_dim: Spatial dimension for the embedding.
        bins: Number of histogram bins.
        filename: File name (relative to ``results/``) used for saving.
        show: Whether to display the plot interactively.

    Returns:
        Basic statistics of the nearest-neighbour distribution as a dict with
        keys ``mean``, ``std``, ``min`` and ``max``.
    """

    rng = np.random.RandomState(seed)
    coords = rng.uniform(0.0, 1.0, size=(units, spatial_dim))

    diff = coords[:, None, :] - coords[None, :, :]
    D = np.linalg.norm(diff, axis=-1)
    D_sorted = np.sort(D, axis=1)
    d1 = D_sorted[:, 1]

    # Stats for nearest-neighbour distances
    stats = {
        "mean": float(d1.mean()),
        "std": float(d1.std()),
        "min": float(d1.min()),
        "max": float(d1.max()),
    }

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.hist(D.flatten(), bins=bins, alpha=0.7)
    plt.xlabel("Pairwise distance")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    plt.hist(d1, bins=bins, alpha=0.7, color="tab:orange")
    plt.xlabel("Nearest-neighbour distance")
    plt.ylabel("Count")
    plt.axvline(stats["mean"], color="k", linestyle="--", label="mean")
    plt.legend()

    plt.tight_layout()
    _save_figure(filename=os.path.join("topology/", filename))
    if show:
        plt.show()

    return stats


def _compute_distance_matrices_from_coords(coords: np.ndarray) -> Dict[str, np.ndarray]:
    """Utility: compute full and nearest-neighbour distance matrices from coords."""

    diff = coords[:, None, :] - coords[None, :, :]
    D = np.linalg.norm(diff, axis=-1)
    D_sorted = np.sort(D, axis=1)
    d1 = D_sorted[:, 1]
    return {"D": D, "d1": d1}


def plot_weight_vs_distance_from_matrix(
    W: np.ndarray,
    coords: np.ndarray,
    diameters: float = 8.2,
    filename: str = "weight_vs_distance.png",
    show: bool = True,
 ) -> Dict[str, float]:
    """Scatter plot of distance vs. raw kernel weight from an existing matrix.

    This variant does *not* generate weights; it assumes you already have a
    distance-based weight matrix ``W`` and corresponding node coordinates
    ``coords``.

    Args:
        W: Weight matrix of shape ``(N, N)``.
        coords: Coordinates of shape ``(N, D)`` for the same nodes as in ``W``.
        diameters: Number of mean nearest-neighbour distances used to define
            the reference distance ``R = diameters * mean_nn``.
        filename: File name (relative to ``results/``) used for saving.
        show: Whether to display the plot interactively.

    Returns:
        Dictionary with basic distance statistics: ``mean_nn`` (mean nearest
        neighbour distance), ``R`` (reference distance) and ``diameters``.
    """

    mats = _compute_distance_matrices_from_coords(coords)
    D, d1 = mats["D"], mats["d1"]

    mean_nn = float(d1.mean())
    R = float(diameters * mean_nn)

    # Flatten and drop self-distances (diagonal). Support both dense and
    # scipy.sparse csr_matrix inputs for W.
    N = D.shape[0]
    off_diag = ~np.eye(N, dtype=bool)
    d_flat = D[off_diag]
    if hasattr(W, "toarray"):
        # Sparse matrix: convert to dense array for this diagnostic plot.
        W_dense = W.toarray()
        w_flat = W_dense[off_diag]
    else:
        w_flat = np.asarray(W)[off_diag]

    plt.figure(figsize=(8, 6))
    plt.scatter(d_flat, w_flat, s=3, alpha=0.2)
    plt.axvline(mean_nn, color="g", linestyle="dotted", label="~1 diameter (mean NN)")
    plt.axvline(R, color="r", linestyle="dashed", label=f"~{diameters:.1f} diameters")
    plt.xlabel("Distance")
    plt.ylabel("Weight (raw)")
    plt.legend()
    plt.tight_layout()

    _save_figure(filename=os.path.join("topology/", filename))
    if show:
        plt.show()

    return {"mean_nn": mean_nn, "R": R, "diameters": float(diameters)}




def plot_spatial_topology(
    W: np.ndarray,
    coords: np.ndarray,
    max_edges_per_node: int = 5,
    thresh_quantile: float = 0.9,
    filename: str = "spatial_topology.png",
    show: bool = True,
) -> None:
    """Plot nodes in space and overlay a subset of strongest edges.

    Edges are filtered by taking, for each node, up to ``max_edges_per_node``
    outgoing edges above the ``thresh_quantile`` of absolute weight.

    Args:
        W: Weight matrix of shape ``(N, N)``.
        coords: Coordinates of shape ``(N, 2)`` (2D plot).
        max_edges_per_node: Maximum number of strongest edges drawn per node.
        thresh_quantile: Quantile of absolute weights used as global threshold.
        filename: File name (relative to ``results/``) used for saving.
        show: Whether to display the plot interactively.
    """

    assert coords.shape[1] == 2, "plot_spatial_topology currently supports 2D coords only"

    N = W.shape[0]
    if hasattr(W, "toarray"):
        W_dense = W.toarray()
    else:
        W_dense = W

    abs_W = np.abs(W_dense)

    # Exclude diagonal when computing threshold
    abs_offdiag = abs_W.copy()
    np.fill_diagonal(abs_offdiag, 0.0)
    thresh = np.quantile(abs_offdiag[abs_offdiag > 0], thresh_quantile) if np.any(abs_offdiag > 0) else 0.0

    plt.figure(figsize=(6, 6))

    # Draw edges
    for i in range(N):
        row = np.array(abs_W[i, :]).ravel().copy()

        # Indices of edges above threshold
        candidates = np.where(row >= thresh)[0]
        # Keep up to max_edges_per_node strongest
        if len(candidates) > max_edges_per_node:
            top_idx = np.argsort(row[candidates])[-max_edges_per_node:]
            candidates = candidates[top_idx]

        for j in candidates:
            x = [coords[i, 0], coords[j, 0]]
            y = [coords[i, 1], coords[j, 1]]
            plt.plot(x, y, color="lightgray", linewidth=0.5)

    # Draw nodes on top
    plt.scatter(coords[:, 0], coords[:, 1], s=10, c="tab:blue")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Spatial topology (strong edges)")
    plt.tight_layout()

    _save_figure(filename=os.path.join("topology/", filename))
    if show:
        plt.show()


# ---------------------------------------------------------------------------
# Heterogeneity robustness figure  (h1-heterogeneity study)
# ---------------------------------------------------------------------------

def plot_heterogeneity_robustness(
    db_path: str = "results/optimization/h1-heterogeneity.db",
    study_name: str = "h1-heterogeneity",
    output_dir: str = "results/diagnostics/",
    filename: str = "heterogeneity_robustness",
    show: bool = False,
) -> None:
    """Plot F1 vs kinetic-parameter heterogeneity (CV).

    Reads completed trials from the Optuna study and produces a line-plot
    with individual seed scatter, mean ± 1σ band, a baseline reference
    line, and a shaded annotation for the biologically realistic CV range.

    Parameters
    ----------
    db_path : str
        Path to the Optuna SQLite database.
    study_name : str
        Optuna study name inside the database.
    output_dir : str
        Directory to write ``{filename}.pdf`` and ``{filename}.png``.
    filename : str
        Base filename (without extension).
    show : bool
        Whether to call ``plt.show()`` after saving.
    """
    import collections
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.load_study(
        study_name=study_name,
        storage=f"sqlite:///{db_path}",
    )
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]

    # ---- collect data ----
    by_cv: Dict[float, list] = collections.defaultdict(list)
    all_cvs_raw, all_f1s = [], []
    for t in completed:
        cv = t.params["param_heterogeneity_cv"]
        by_cv[cv].append(t.value)
        all_cvs_raw.append(cv)
        all_f1s.append(t.value)

    cv_levels = sorted(by_cv)
    means = [np.mean(by_cv[cv]) for cv in cv_levels]
    stds  = [np.std(by_cv[cv])  for cv in cv_levels]

    # ---- evenly-spaced x positions (avoids label overlap) ----
    x_pos = np.arange(len(cv_levels))
    cv_to_x = {cv: i for i, cv in enumerate(cv_levels)}

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.8))

    # Biological range shading (CV ≈ 0.1–0.4 for gene expression noise)
    bio_lo = cv_to_x.get(0.1, 3) - 0.35
    bio_hi = cv_to_x.get(0.3, 6) + 0.35
    ax.axvspan(bio_lo, bio_hi, alpha=0.08, color="#2ca02c", zorder=0)
    ax.text(
        (bio_lo + bio_hi) / 2, ax.get_ylim()[0] if ax.get_ylim()[0] else 0.91,
        "Biological\nrange", ha="center", va="bottom",
        fontsize=7.5, color="#2ca02c", alpha=0.7, style="italic",
    )

    # Baseline reference line (shown in legend, no floating text)
    baseline = means[0]
    ax.axhline(
        baseline, color="#888888", ls="--", lw=0.8, alpha=0.6, zorder=1,
        label=f"Baseline F\u2081 = {baseline:.3f}",
    )

    # Individual seed scatter (slight jitter)
    rng = np.random.default_rng(42)
    seed_x = np.array([cv_to_x[cv] for cv in all_cvs_raw])
    jitter = rng.uniform(-0.15, 0.15, len(seed_x))
    ax.scatter(
        seed_x + jitter, all_f1s,
        s=18, alpha=0.45, color="#1f77b4", edgecolors="none",
        zorder=3, label="Individual seeds",
    )

    # Mean ± 1σ
    ax.plot(
        x_pos, means, "o-", color="#d62728", ms=5, lw=1.5,
        zorder=4, label="Mean F\u2081 (\u00b11\u03c3)",
    )
    ax.fill_between(
        x_pos,
        [m - s for m, s in zip(means, stds)],
        [m + s for m, s in zip(means, stds)],
        alpha=0.15, color="#d62728", zorder=2,
    )

    # Axis formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{cv:g}" for cv in cv_levels], fontsize=8.5)
    ax.set_xlabel("Coefficient of Variation (CV)", fontsize=10)
    ax.set_ylabel("Weighted F\u2081 Score", fontsize=10)
    ax.set_xlim(x_pos[0] - 0.5, x_pos[-1] + 0.5)
    ax.set_ylim(0.80, 0.92)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Re-place bio label after ylim is set
    ax.texts[-1].set_position(((bio_lo + bio_hi) / 2, 0.912))

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(
        os.path.join(output_dir, f"{filename}.png"),
        bbox_inches="tight", dpi=300,
    )
    if show:
        plt.show()
    plt.close(fig)
