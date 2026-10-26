"""Figure helpers: the house style, the names and colours of the classes, and shared plots.

The house style :data:`STYLE` is the look of the ``v0.1`` figures: seaborn's "ticks" style
with 15 pt text, on figures drawn large (10 to 14 in wide) and reduced to fit the page.
Figure scripts call ``plt.style.use(STYLE)``; importing this module changes no matplotlib
setting. Categories take the Okabe-Ito colours (:data:`OKABE_ITO`), each ECG class always
the same one (:data:`CLASS_COLOURS`). :func:`save_figure` writes a PNG, and on request a PDF
or an SVG whose text stays text (the scripts' ``--formats`` option, :func:`add_formats_option`).
It warns about anything that :func:`figure_problems` finds once the figure is printed: ACS
Synthetic Biology prints a figure at most two columns (7 in) wide and asks for text of at
least 5 pt and lines of at least 1 pt at that size.
"""

from __future__ import annotations

import argparse
import logging
import numbers
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.text import Text

from utils import config

logger = logging.getLogger(__name__)

#: Widths of one and two columns of an ACS page and the page height, in inches.
SINGLE_COLUMN, DOUBLE_COLUMN, MAX_HEIGHT = 3.25, 7.0, 9.5
#: The smallest text size and line width at the printed size, in points.
MIN_FONT_SIZE, MIN_LINE_WIDTH = 5.0, 1.0
#: Resolution of the PNG files at the printed size, in dots per inch.
DPI = 600
#: The file formats that :func:`save_figure` writes: PNG to look at, and PDF and SVG (vector,
#: their text kept as text) for a paper or for editing.
FORMATS = ("png", "pdf", "svg")

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

#: The house style: seaborn's "ticks" style with 15 pt text.
STYLE = {**sns.axes_style("ticks"), "font.size": 15}


def class_names(n_classes: int) -> tuple[str, ...]:
    """The names of the classes of a task with ``n_classes`` classes (two: the binary task)."""
    return BINARY_CLASS_NAMES if n_classes == 2 else CLASS_NAMES[:n_classes]


def printed_width(fig) -> float:
    """The width, in inches, at which ``fig`` is printed: its own as :func:`save_figure`
    crops it, reduced to two columns (7 in) if it is wider."""
    fig.draw_without_rendering()
    return float(min(fig.get_tightbbox().width, DOUBLE_COLUMN))


def figure_problems(fig, width: float | None = None) -> list[str]:
    """What in ``fig`` breaks the journal's rules once printed ``width`` inches wide.

    ``width`` defaults to :func:`printed_width`. The figure is measured as
    :func:`save_figure` crops it and scaled to ``width``, and so are its text and lines.
    Checked: the printed height, the size of every visible text, and the width of every line
    drawn on an Axes or in a legend (tick marks and grid lines follow the style). Returns one
    sentence per problem.
    """
    fig.draw_without_rendering()
    box = fig.get_tightbbox()
    width = printed_width(fig) if width is None else width
    scale = width / box.width
    problems = []
    if width > DOUBLE_COLUMN + 0.01:
        problems.append(f"{width:.2f} in is wider than a page ({DOUBLE_COLUMN:g} in)")
    if box.height * scale > MAX_HEIGHT + 0.01:
        problems.append(f"it is {box.height * scale:.2f} in high; a page is {MAX_HEIGHT:g} in")
    small = sorted({(t.get_fontsize() * scale, t.get_text()) for t in fig.findobj(Text)
                    if t.get_visible() and t.get_text().strip()
                    and t.get_fontsize() * scale < MIN_FONT_SIZE})
    if small:
        problems.append(f"{len(small)} texts are under {MIN_FONT_SIZE:g} pt, such as "
                        f"{small[0][1]!r} at {small[0][0]:.1f} pt")
    legends = [*fig.legends, *(ax.get_legend() for ax in fig.axes if ax.get_legend())]
    lines = [*(line for ax in fig.axes for line in ax.get_lines()),
             *(line for legend in legends for line in legend.get_lines())]
    thin = [line.get_linewidth() * scale for line in lines
            if line.get_visible() and line.get_linestyle() not in ("None", "", " ")
            and 0 < line.get_linewidth() * scale < MIN_LINE_WIDTH]
    if thin:
        problems.append(f"{len(thin)} lines are under {MIN_LINE_WIDTH:g} pt, the thinnest "
                        f"{min(thin):.2f} pt")
    return problems


def figure_formats(text: str) -> tuple[str, ...]:
    """The file formats in ``text``, comma-separated, such as ``"png,pdf"``: the type of a
    script's ``--formats`` option."""
    formats = tuple(dict.fromkeys(f.strip().lower() for f in text.split(",") if f.strip()))
    if not formats or set(formats) - set(FORMATS):
        raise argparse.ArgumentTypeError(
            f"{text!r}: choose one or more of {', '.join(FORMATS)}, comma-separated")
    return formats


def add_formats_option(parser: argparse.ArgumentParser) -> None:
    """Add ``--formats`` to a figure script: the file formats of its figures (default PNG)."""
    parser.add_argument("--formats", type=figure_formats, default=("png",),
                        help="File formats of the figures, comma-separated, from "
                             f"{', '.join(FORMATS)} (default: png).")


def save_figure(fig, name: str, *, topic: str = "", folder: str | Path | None = None,
                formats: Sequence[str] = ("png",), width: float | None = None,
                dpi: float | None = None) -> list[Path]:
    """Save ``fig`` as ``<folder>/<name>.<format>`` in each format and close it.

    ``folder`` defaults to ``results/figures/<topic>`` (see :mod:`utils.config`). ``width``
    is the printed width in inches (default :func:`printed_width`): a warning is logged for
    each of :func:`figure_problems` at that width, and the PNG has :data:`DPI` dots per inch
    at that width unless ``dpi`` is given. PDF and SVG files are vector, with their text kept
    as text (TrueType in the PDF). Returns the files written, in the order of ``formats``.
    """
    width = printed_width(fig) if width is None else width
    for problem in figure_problems(fig, width):
        logger.warning("Figure %s, printed %.2g in wide: %s.", name, width, problem)
    # A plain float: matplotlib compares the dpi with the string "figure".
    dpi = float(dpi or DPI * width / fig.get_tightbbox().width)
    folder = Path(folder) if folder is not None else config.figures_dir(topic)
    paths = [folder / f"{name}.{fmt}" for fmt in formats]
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}):
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return paths


def plot_confusion_matrix(confusion_matrix, ax=None, *, labels: Sequence[str] | None = None,
                          colorbar: bool = True):
    """Draw a confusion matrix with each row divided by its sum, so that row ``i`` shows how
    the beats of true class ``i`` were labelled and the diagonal is the recall of each class.

    ``labels`` defaults to the names of the classes (:func:`class_names`). Without ``ax`` the
    matrix is drawn in a new 10 x 8 in figure in the house style, as in ``v0.1``. Returns the
    Axes.
    """
    if ax is None:
        with plt.style.context(STYLE):
            _, ax = plt.subplots(figsize=(10, 8))
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
               levels: Sequence[float] | None = None, connect: bool = True, ddof: int = 1,
               color: str = "#d62728", seed_color: str = "#1f77b4",
               label: str | None = "Mean (\u00b11 SD)", seed_label: str | None = "Seeds",
               reference: float | None = None, reference_label: str | None = None,
               jitter: float = 0.15, jitter_seed: int = 42):
    """Draw a score against the levels of a swept setting as in Supplementary Fig. S10 of
    ``v0.1``: one dot per seed, and a line through the means at each level with a band of
    plus and minus one SD (no band where a level has one seed, no dots where every level has
    one).

    ``scores`` maps each level, a number or a name, to its scores, one per seed. Level ``i``
    of ``levels`` (default: the sorted levels of ``scores``) is drawn at ``x = i`` and
    labelled with its value (three significant digits), so that an uneven grid such as 0,
    0.025, 0.05, 0.1, 0.5 stays readable. With ``connect=False`` the levels are categories
    without an order, such as metrics or classes: each mean gets its own error bar of one SD
    and nothing joins them. ``ddof`` is 1 for the sample SD over the seeds, 0 for the
    population SD (S10's). Dots are shifted sideways by up to ``jitter`` level spacings, the
    same way every time (``jitter_seed``). ``reference`` adds a dashed line, for example the
    score without the swept effect. Without ``ax`` the sweep is drawn in a new 5.5 x 3.8 in
    figure, S10's size. Returns the Axes.

    Examples
    --------
    >>> trials = trials_table(cfg)  # one row per trial: the swept value, the seed, f1
    >>> scores = trials.groupby("reservoir.params.param_heterogeneity_cv")["f1"].apply(list)
    >>> plot_sweep(scores.to_dict(), ax, reference=np.mean(scores[0.0]))
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 3.8))
    levels = sorted(scores) if levels is None else list(levels)
    drawn = [level for level in levels if level in scores]
    x = np.array([levels.index(level) for level in drawn], dtype=float)
    values = [np.asarray(scores[level], dtype=float) for level in drawn]
    means = np.array([v.mean() for v in values])
    sds = np.array([v.std(ddof=ddof) if v.size > 1 else np.nan for v in values])
    if reference is not None:
        ax.axhline(reference, color="#888888", ls="--", lw=1.0, alpha=0.6, zorder=1,
                   label=reference_label)
    if any(v.size > 1 for v in values):
        dots_x = np.concatenate([np.full(v.size, xi) for xi, v in zip(x, values, strict=True)])
        shift = np.random.default_rng(jitter_seed).uniform(-jitter, jitter, dots_x.size)
        ax.scatter(dots_x + shift, np.concatenate(values), s=18, alpha=0.45, color=seed_color,
                   edgecolors="none", zorder=3, label=seed_label)
    if connect:
        ax.plot(x, means, "o-", color=color, ms=5, lw=1.5, zorder=4, label=label)
        ax.fill_between(x, means - sds, means + sds, alpha=0.15, color=color, zorder=2)
    else:
        ax.errorbar(x, means, yerr=np.nan_to_num(sds), fmt="o", color=color, ms=5, lw=1.5,
                    capsize=4, zorder=4, label=label)
    ax.set_xticks(np.arange(len(levels)), [f"{level:.3g}" if isinstance(level, numbers.Real)
                                           else str(level) for level in levels])
    ax.set_xlim(-0.5, len(levels) - 0.5)
    return ax


#: The scores of a fold that :func:`plot_fold_metrics` draws, with their labels.
FOLD_SCORES = {"f1": "F1", "accuracy": "Accuracy", "precision": "Precision",
               "recall": "Recall", "mcc": "MCC", "kappa": "Cohen's \u03ba"}


def plot_fold_metrics(folds: Sequence[Mapping], average: str = "macro"):
    """Draw the scores of each fold of a run as dots, with their mean and one sample SD, as
    in S10: the scores over all classes (:data:`FOLD_SCORES`; F1, precision and recall are
    averaged over the classes as ``average``) and the F1 of each class, from each fold's
    confusion matrix. The folds have no order, so nothing joins them.

    ``folds`` are mappings with the keys of :data:`FOLD_SCORES` and ``confusion_matrix``,
    such as :class:`utils.analysis.Metrics` or the folds of a run's ``metrics.json``. The F1
    of each class is drawn when every fold has the same classes. Returns the figure, drawn in
    the house style.
    """
    scores = {f"{label} ({average})" if key in ("f1", "precision", "recall") else label:
              [float(fold[key]) for fold in folds] for key, label in FOLD_SCORES.items()}
    matrices = [np.asarray(fold["confusion_matrix"], dtype=float) for fold in folds]
    same_classes = len({m.shape for m in matrices}) == 1
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 1 + same_classes, sharey=True, squeeze=False,
                                 figsize=(13, 5) if same_classes else (7.5, 5),
                                 layout="constrained", width_ratios=[6, 5][:1 + same_classes])
        ax = axes[0, 0]
        plot_sweep(scores, ax, levels=list(scores), connect=False, seed_label="Folds")
        ax.set(title="All classes", ylabel="Score")
        ax.legend(loc="lower left")
        ticks = [(ax, list(scores))]
        if same_classes:
            # F1 = 2 TP / (2 TP + FP + FN): the diagonal over its column and row sums; NaN for
            # a class that a fold neither has nor predicts.
            with np.errstate(invalid="ignore", divide="ignore"):
                f1 = np.array([2 * np.diag(m) / (m.sum(axis=0) + m.sum(axis=1))
                               for m in matrices])
            names = class_names(f1.shape[1])
            per_class = {name: f1[np.isfinite(f1[:, c]), c] for c, name in enumerate(names)}
            plot_sweep(per_class, axes[0, 1], levels=names, connect=False, label=None,
                       seed_label=None)
            axes[0, 1].set_title("F1 of each class")
            ticks.append((axes[0, 1], names))
        for axis, labels in ticks:
            axis.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    return fig
