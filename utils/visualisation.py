"""Figure helpers shared by the library and the figure scripts.

The ``v0.1`` figures use seaborn's "ticks" style with 15 pt text (:data:`V0_1_STYLE`). The
helpers below that draw them apply it themselves, and the scripts that draw the figures of
the ``v0.1`` paper call :func:`use_v0_1_style`; importing this module changes no matplotlib
setting. Figures are saved under ``results/figures/`` (``BIORESERVOIR_RESULTS_DIR``).
"""

import functools
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE

from utils import config

#: The look of the ``v0.1`` figures: seaborn's "ticks" style with 15 pt text.
V0_1_STYLE = {**sns.axes_style("ticks"), "font.size": 15}

DPI = 800

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


def use_v0_1_style() -> None:
    """Draw every later figure of this process in the ``v0.1`` style (:data:`V0_1_STYLE`).

    The scripts that draw the figures of the ``v0.1`` paper call it first.
    """
    plt.rcParams.update(V0_1_STYLE)


def _in_v0_1_style(plot):
    """Decorator: ``plot`` draws in the ``v0.1`` style and leaves the settings as it found them."""

    @functools.wraps(plot)
    def wrapper(*args, **kwargs):
        with plt.rc_context(V0_1_STYLE):
            return plot(*args, **kwargs)

    return wrapper


def _save_figure(filename: str) -> None:
    """
    Save the current figure under ``results/figures/``.

    Args:
        filename (str): Path relative to the figures directory, e.g. ``states/x.png``.
    """
    file_path = os.path.join(config.figures_dir(), filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.savefig(file_path, bbox_inches="tight", dpi=DPI)

@_in_v0_1_style
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

@_in_v0_1_style
def plot_metrics_across_folds(metrics: list[dict[str, any]], metric_names: list[str] | None = None,
                              show: bool = True, filename: str = "metrics_folds.png") -> None:
    """
    Plot the metrics across different folds.

    Args:
        metrics (List[Dict[str, any]]): List of dictionaries containing metric values for each fold.
        metric_names (List[str], optional): Metric names to plot. Default: accuracy, f1, mcc, kappa.
        show (bool, optional): Whether to display the plot. Default is True.
        filename (str, optional): Filename to save the plot. Default is "metrics_folds.png".
    """
    if metric_names is None:
        metric_names = ["accuracy", "f1", "mcc", "kappa"]
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

@_in_v0_1_style
def plot_class_metrics(metrics: dict[str, dict[str, float]], show: bool = True, filename: str = "class_metrics.png") -> None:
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

@_in_v0_1_style
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


# ---------------------------------------------------------------------------
# Heterogeneity robustness figure  (P3-heterogeneity study)
# ---------------------------------------------------------------------------

@_in_v0_1_style
def plot_heterogeneity_robustness(
    db_path: str | None = None,
    study_name: str = "P3-heterogeneity",
    output_dir: str | None = None,
    filename: str = "heterogeneity_robustness",
    show: bool = False,
) -> None:
    """Plot F1 vs kinetic-parameter heterogeneity (CV).

    Reads completed trials from the Optuna study and produces a line-plot
    with individual seed scatter, mean ± 1σ band, a baseline reference
    line, and a shaded annotation for the biologically realistic CV range.

    Parameters
    ----------
    db_path : str, optional
        Path to the Optuna SQLite database.  Defaults to ``results/optuna/optuna-<study>.db``.
    study_name : str
        Optuna study name inside the database.
    output_dir : str, optional
        Directory for ``{filename}.png``.  Defaults to ``results/figures/diagnostics``.
    filename : str
        Base filename (without extension).
    show : bool
        Whether to call ``plt.show()`` after saving.
    """
    import collections

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    db_path = db_path or str(config.optuna_db(study_name))
    output_dir = output_dir or str(config.figures_dir("diagnostics"))
    study = optuna.load_study(
        study_name=study_name,
        storage=f"sqlite:///{db_path}",
    )
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]

    # ---- collect data ----
    by_cv: dict[float, list] = collections.defaultdict(list)
    all_cvs_raw, all_f1s = [], []
    for t in completed:
        cv = t.params["param_heterogeneity_cv"]
        by_cv[cv].append(t.value)
        all_cvs_raw.append(cv)
        all_f1s.append(t.value)

    cv_levels = sorted(by_cv)
    means = [np.mean(by_cv[cv]) for cv in cv_levels]
    stds  = [np.std(by_cv[cv], ddof=1)  for cv in cv_levels]

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
        [m - s for m, s in zip(means, stds, strict=False)],
        [m + s for m, s in zip(means, stds, strict=False)],
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
