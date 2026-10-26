"""Figure helpers shared by the library and the figure scripts.

The ``v0.1`` figures that these helpers draw use seaborn's "ticks" style with 15 pt text
(:data:`V0_1_STYLE`): the helpers apply it themselves, and the scripts that draw those
figures call :func:`use_v0_1_style` first. Importing this module changes no matplotlib
setting. Figures are saved under ``results/figures/`` (``BIORESERVOIR_RESULTS_DIR``).
"""

import functools
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils import config

#: The look of the ``v0.1`` figures: seaborn's "ticks" style with 15 pt text.
V0_1_STYLE = {**sns.axes_style("ticks"), "font.size": 15}

DPI = 800

# Class labels
binary_classes = ['Normal', 'Arrhythmia']

classes = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown']


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

# ---------------------------------------------------------------------------
# Building blocks for figure scripts: each draws on the Axes it is given
# ---------------------------------------------------------------------------

def plot_sweep(ax, scores, *, reference=None, reference_label=None, seed_label="Seeds",
               mean_label="Mean (\u00b11 SD)", ddof=1, color="#d62728", seed_color="#1f77b4",
               jitter=0.15, jitter_seed=42):
    """Draw a score against the levels of a swept setting, one dot per seed.

    Level ``i`` in sorted order is drawn at ``x = i`` and labelled with its value, so an
    uneven grid such as 0, 0.025, 0.05, 0.1, 0.5 stays readable. Each level gets one dot per
    seed, shifted sideways by up to ``jitter``, and a line through the seed means with a band
    of plus and minus one SD (no band at a level with one seed).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Where to draw.
    scores : mapping of float to sequence of float
        The scores at each level, one per seed: for example the rows of
        :func:`training.study.trials_table` grouped by the swept value, or a grid of run
        folders read with :class:`utils.results.FoldArtifact`.
    reference : float, optional
        Value of a dashed horizontal line, for example the score without the swept effect.
    reference_label, seed_label, mean_label : str
        Legend labels of the reference line, the dots and the mean line.
    ddof : int
        Delta degrees of freedom of the SD: 1 for the sample SD over the seeds, 0 for the
        population SD.
    color, seed_color : str
        Colours of the mean (line and band) and of the dots.
    jitter : float
        Largest sideways shift of a dot, in level spacings.
    jitter_seed : int
        Seed of the shifts, so that a figure is the same every time it is drawn.

    Returns
    -------
    matplotlib.axes.Axes
        ``ax``.

    Examples
    --------
    >>> trials = trials_table(cfg)  # one row per trial: the swept value, the seed, f1
    >>> scores = trials.groupby("reservoir.params.param_heterogeneity_cv")["f1"].apply(list)
    >>> plot_sweep(ax, scores.to_dict(), reference=np.mean(scores[0.0]))
    """
    levels = sorted(scores)
    x = np.arange(len(levels))
    values = [np.asarray(scores[level], dtype=float) for level in levels]
    means = np.array([v.mean() for v in values])
    sds = np.array([v.std(ddof=ddof) if v.size > 1 else np.nan for v in values])
    if reference is not None:
        ax.axhline(reference, color="#888888", ls="--", lw=0.8, alpha=0.6, zorder=1,
                   label=reference_label)
    dots_x = np.concatenate([np.full(v.size, i) for i, v in enumerate(values)])
    shift = np.random.default_rng(jitter_seed).uniform(-jitter, jitter, dots_x.size)
    ax.scatter(dots_x + shift, np.concatenate(values), s=18, alpha=0.45, color=seed_color,
               edgecolors="none", zorder=3, label=seed_label)
    ax.plot(x, means, "o-", color=color, ms=5, lw=1.5, zorder=4, label=mean_label)
    ax.fill_between(x, means - sds, means + sds, alpha=0.15, color=color, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{level:g}" for level in levels])
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    return ax


def save_figure(fig, name, *, topic="", formats=("png", "pdf"), dpi=300):
    """Save ``fig`` as ``results/figures/<topic>/<name>.<format>`` in each format; close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    name : str
        File name without its extension.
    topic : str
        Sub-folder of ``results/figures/`` (see :mod:`utils.config`).
    formats : sequence of str
        File formats: PNG to look at, PDF (vector) for a paper.
    dpi : float
        Resolution of the raster formats, in dots per inch.

    Returns
    -------
    list of pathlib.Path
        The files written, in the order of ``formats``.
    """
    folder = config.figures_dir(topic)
    folder.mkdir(parents=True, exist_ok=True)
    paths = [folder / f"{name}.{fmt}" for fmt in formats]
    for path in paths:
        fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return paths
