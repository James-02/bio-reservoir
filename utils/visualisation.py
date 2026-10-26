"""Figure helpers: the house style, the names and colours of the classes, and shared plots.

The house style :data:`STYLE` follows the figure rules of ACS Synthetic Biology. A figure is
one column (:data:`SINGLE_COLUMN`, 3.25 in) or two columns (:data:`DOUBLE_COLUMN`, 7 in) wide
and at most :data:`MAX_HEIGHT` (9.5 in) high; its lettering is sans-serif (Arial or Helvetica
where installed, else matplotlib's DejaVu Sans) and at least 5 pt at that size; its lines are
at least 1 pt wide. For readers with a colour-vision deficiency, categories take the
Okabe-Ito colours (:data:`OKABE_ITO`), each ECG class always the same one
(:data:`CLASS_COLOURS`), and continuous values a single-hue or perceptually uniform colour
map ("Blues", "viridis"). Figure scripts call ``plt.style.use(STYLE)`` and draw at the printed
size. :func:`save_figure` writes a PNG and a PDF whose text stays text (TrueType), and warns
about anything that :func:`figure_problems` finds. Importing this module changes no
matplotlib setting.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.text import Text

from utils import config

logger = logging.getLogger(__name__)

#: Widths of one and two columns of an ACS page and the page height, in inches.
SINGLE_COLUMN, DOUBLE_COLUMN, MAX_HEIGHT = 3.25, 7.0, 9.5
#: The smallest text size and line width at the printed size, in points.
MIN_FONT_SIZE, MIN_LINE_WIDTH = 5.0, 1.0
#: Resolution of the PNG files, in dots per inch.
DPI = 600

#: The Okabe-Ito colours: blue, vermillion, bluish green, reddish purple, orange, sky blue,
#: black and yellow (last: it is faint on white).
OKABE_ITO = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000",
             "#F0E442")

CLASS_NAMES = ("Normal", "Supraventricular", "Ventricular", "Fusion", "Unknown")
BINARY_CLASS_NAMES = ("Normal", "Arrhythmia")
#: The colour of each class in every figure.
CLASS_COLOURS = {name: OKABE_ITO[i] for i, name in enumerate(CLASS_NAMES)} | {
    "Arrhythmia": OKABE_ITO[1]}

#: Names of the oscillator's variables: the proteins LuxI and AiiA, and the signal AHL inside
#: the cells and in the medium around them.
STATE_NAMES = {"I": "LuxI", "A": "AiiA", "Hi": "AHL (cells)", "He": "AHL (medium)"}

STYLE = {
    "figure.figsize": (SINGLE_COLUMN, 2.4),
    "figure.dpi": 150,
    "figure.constrained_layout.use": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "mathtext.default": "regular",
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "figure.titlesize": 8,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "legend.title_fontsize": 7,
    "legend.frameon": False,
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.prop_cycle": cycler(color=OKABE_ITO),
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
    "patch.linewidth": 1.0,
    "grid.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.minor.width": 1.0,
    "ytick.minor.width": 1.0,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "image.cmap": "viridis",
    "savefig.dpi": DPI,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


def class_names(n_classes: int) -> tuple[str, ...]:
    """The names of the classes of a task with ``n_classes`` classes (two: the binary task)."""
    return BINARY_CLASS_NAMES if n_classes == 2 else CLASS_NAMES[:n_classes]


def figure_problems(fig) -> list[str]:
    """What in ``fig`` breaks the journal's rules at its printed size, one sentence each.

    The figure is measured as :func:`save_figure` crops it: its width and height, the size
    of every visible text, and the width of every line drawn on an Axes or in a legend
    (tick marks and grid lines follow the style).
    """
    fig.draw_without_rendering()
    box = fig.get_tightbbox()
    problems = []
    if box.width > DOUBLE_COLUMN + 0.01:
        problems.append(f"it is {box.width:.2f} in wide; a page is {DOUBLE_COLUMN:g} in wide "
                        f"and a column {SINGLE_COLUMN:g} in")
    if box.height > MAX_HEIGHT + 0.01:
        problems.append(f"it is {box.height:.2f} in high; a page is {MAX_HEIGHT:g} in high")
    small = sorted({(t.get_fontsize(), t.get_text()) for t in fig.findobj(Text)
                    if t.get_visible() and t.get_text().strip()
                    and t.get_fontsize() < MIN_FONT_SIZE})
    if small:
        problems.append(f"{len(small)} texts are smaller than {MIN_FONT_SIZE:g} pt, such as "
                        f"{small[0][1]!r} at {small[0][0]:g} pt")
    legends = [*fig.legends, *(ax.get_legend() for ax in fig.axes if ax.get_legend())]
    lines = [*(line for ax in fig.axes for line in ax.get_lines()),
             *(line for legend in legends for line in legend.get_lines())]
    thin = [line.get_linewidth() for line in lines
            if line.get_visible() and line.get_linestyle() not in ("None", "", " ")
            and 0 < line.get_linewidth() < MIN_LINE_WIDTH]
    if thin:
        problems.append(f"{len(thin)} lines are thinner than {MIN_LINE_WIDTH:g} pt, the "
                        f"thinnest {min(thin):g} pt")
    return problems


def save_figure(fig, name: str, *, topic: str = "", folder: str | Path | None = None,
                formats: Sequence[str] = ("png", "pdf"), dpi: float | None = None) -> list[Path]:
    """Save ``fig`` as ``<folder>/<name>.<format>`` in each format and close it.

    ``folder`` defaults to ``results/figures/<topic>`` (see :mod:`utils.config`). The PDF is
    vector, with its text kept as TrueType text; the PNG has ``dpi`` dots per inch (default
    :data:`DPI`). A warning is logged for each of :func:`figure_problems`. Returns the files
    written, in the order of ``formats``.
    """
    for problem in figure_problems(fig):
        logger.warning("Figure %s: %s.", name, problem)
    folder = Path(folder) if folder is not None else config.figures_dir(topic)
    paths = [folder / f"{name}.{fmt}" for fmt in formats]
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}):
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=dpi or DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return paths


def plot_confusion_matrix(confusion_matrix, ax=None, *, labels: Sequence[str] | None = None,
                          colorbar: bool = True):
    """Draw a confusion matrix with each row divided by its sum, so that row ``i`` shows how
    the beats of true class ``i`` were labelled and the diagonal is the recall of each class.

    ``labels`` defaults to the names of the classes (:func:`class_names`). Without ``ax`` the
    matrix is drawn in a new one-column figure in the house style. Returns the Axes.
    """
    if ax is None:
        with plt.style.context(STYLE):
            _, ax = plt.subplots(figsize=(SINGLE_COLUMN, 2.8))
            return plot_confusion_matrix(confusion_matrix, ax, labels=labels, colorbar=colorbar)
    counts = np.asarray(confusion_matrix, dtype=float)
    shares = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    labels = class_names(len(shares)) if labels is None else labels
    image = ax.imshow(shares, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    for (row, column), share in np.ndenumerate(shares):
        ax.text(column, row, f"{share:.2f}", ha="center", va="center",
                color="white" if share > 0.5 else "black")
    ticks = np.arange(len(labels))
    ax.set_xticks(ticks, labels, rotation=30, ha="right")
    ax.set_yticks(ticks, labels)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    ax.set(xlabel="Predicted label", ylabel="True label")
    if colorbar:
        bar = ax.figure.colorbar(image, ax=ax, label="Fraction of the true class")
        bar.outline.set_visible(False)
    return ax


def plot_sweep(scores: Mapping[float, Sequence[float]], ax=None, *,
               levels: Sequence[float] | None = None, color: str = OKABE_ITO[0],
               marker: str = "o", label: str | None = "Mean \u00b1 SD",
               seed_label: str | None = "Seeds", reference: float | None = None,
               reference_label: str | None = None, jitter: float = 0.15,
               jitter_seed: int = 42):
    """Draw a score against the levels of a swept setting, one dot per seed, and a line
    through the means at each level with a band of plus and minus one sample SD (no band
    where a level has one seed).

    ``scores`` maps each level to its scores, one per seed. Level ``i`` of ``levels``
    (default: the sorted levels of ``scores``) is drawn at ``x = i`` and labelled with its
    value, so that an uneven grid such as 0, 0.025, 0.05, 0.1, 0.5 stays readable; pass the
    same ``levels`` to draw several sweeps on one Axes. Dots are shifted sideways by up to
    ``jitter`` level spacings, the same way every time (``jitter_seed``). ``reference`` adds
    a dashed line, for example the score without the swept effect. Without ``ax`` the sweep
    is drawn in a new one-column figure in the house style. Returns the Axes.

    Examples
    --------
    >>> trials = trials_table(cfg)  # one row per trial: the swept value, the seed, f1
    >>> scores = trials.groupby("reservoir.params.param_heterogeneity_cv")["f1"].apply(list)
    >>> plot_sweep(scores.to_dict(), ax, reference=np.mean(scores[0.0]))
    """
    if ax is None:
        with plt.style.context(STYLE):
            _, ax = plt.subplots()
            return plot_sweep(scores, ax, levels=levels, color=color, marker=marker,
                              label=label, seed_label=seed_label, reference=reference,
                              reference_label=reference_label, jitter=jitter,
                              jitter_seed=jitter_seed)
    levels = sorted(scores) if levels is None else list(levels)
    drawn = [level for level in levels if level in scores]
    x = np.array([levels.index(level) for level in drawn], dtype=float)
    values = [np.asarray(scores[level], dtype=float) for level in drawn]
    means = np.array([v.mean() for v in values])
    sds = np.array([v.std(ddof=1) if v.size > 1 else np.nan for v in values])
    if reference is not None:
        ax.axhline(reference, color="0.4", ls="--", lw=1.0, zorder=1, label=reference_label)
    dots_x = np.concatenate([np.full(v.size, xi) for xi, v in zip(x, values, strict=True)])
    shift = np.random.default_rng(jitter_seed).uniform(-jitter, jitter, dots_x.size)
    ax.scatter(dots_x + shift, np.concatenate(values), s=10, alpha=0.5, color=color,
               marker=marker, edgecolors="none", zorder=3, label=seed_label)
    ax.plot(x, means, color=color, marker=marker, zorder=4, label=label)
    ax.fill_between(x, means - sds, means + sds, alpha=0.2, color=color, lw=0, zorder=2)
    ax.set_xticks(np.arange(len(levels)), [f"{level:g}" for level in levels])
    ax.set_xlim(-0.5, len(levels) - 0.5)
    return ax
