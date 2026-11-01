"""Draw the figures of the revised paper (design p576) in the style of the v0.1 figures.

Reads the collected results through ``paper_figures.py`` (``BIORESERVOIR_RESULTS_DIR``) and,
for the panels of one beat, ``simulated.npz``, written by ``simulate``:

    python scripts/paper/paper_panels.py simulate        # one beat through the design (~1 min)
    python scripts/paper/paper_panels.py figures --oscillation oscillation_map.csv \\
        --flat-share flat_share.csv --formats png,pdf

Main text: ``fig1_design`` (A is a placeholder for the colony circuit of the v0.1 figure; C
is a drawn sketch of the chip), ``fig2_dynamics``, ``fig3_classification``,
``fig4_requirements``. SI: ``si_class_images``, ``si_confusion_tasks``, ``si_settings``. Run
from the repository root: ``simulate`` reads ``experiments/v0.2/designs/<design>.yaml``, and
Fig. 3 and the SI class images read the task's beats.

Every figure uses one scheme: the design in blue circles and solid lines, a first
alternative in orange squares and dashed lines, a second in pink diamonds and dotted lines,
the design's own setting as a red star, and the raw-beat HistGB as a black dotted line.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullLocator  # noqa: E402
from paper_figures import (  # noqa: E402
    DANINO_PERIODS,
    DESIGNS,
    FOREST,
    HEADLINE,
    ONE,
    RAW,
    SEEDS,
    Results,
    design_point,
    f1_map,
    flat_shares,
    image_times,
    ordered,
    oscillation_map,
    scaled,
    traps,
)

from utils import config  # noqa: E402
from utils.visualisation import (  # noqa: E402
    CLASS_COLOURS,
    CLASS_NAMES,
    OKABE_ITO,
    STATE_NAMES,
    STYLE,
    add_formats_option,
    class_names,
    plot_confusion_matrix,
    save_figure,
)

#: The oscillator's variables in the order of its state (training.figures.VARIABLES).
VARIABLES = ("A", "I", "Hi", "He")
LW = 3.2  # line width on the figures as drawn (up to 18 in wide, printed at 7 in)
THIN = 2.7  # reference lines: 1 pt once printed
SMALL = 13  # legends and labels inside the axes (about 5.5 pt printed)
GRID = 0.22
# The line styles of the v0.1 figures (scripts/figures/visualise_dde.py).
LINES = {"I": {"color": OKABE_ITO[0], "lw": LW, "ls": "-"},
         "A": {"color": OKABE_ITO[1], "lw": LW, "ls": (0, (4, 2))}}
WARMUP = {"color": "black", "lw": THIN, "ls": "--"}
ECG = {"color": "#CC0000", "lw": LW}
BOX = {"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 2}
DESIGN = {"color": OKABE_ITO[0], "marker": "o", "ls": "-"}
FIRST = {"color": OKABE_ITO[4], "marker": "s", "ls": "--"}
SECOND = {"color": OKABE_ITO[3], "marker": "D", "ls": ":"}
STAR = {"marker": "*", "ms": 24, "color": "#d62728", "markeredgecolor": "white", "mew": 1.2,
        "ls": "none", "zorder": 5}


# ---------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------

def panel(ax, letter: str, outside: bool = False):
    if outside:  # left of an axes that carries its own centred title
        ax.text(-0.18, 1.08, letter, transform=ax.transAxes, fontweight="bold", fontsize=20,
                va="bottom")
    else:
        ax.set_title(letter, loc="left", fontweight="bold", fontsize=20)


def tidy(ax):
    ax.grid(True, alpha=GRID)
    ax.spines[["top", "right"]].set_visible(False)


def line(ax, points, style: dict, label=None, band=True):
    """Mean over seeds at each x in ``style`` (``DESIGN``, ``FIRST``, ``SECOND``), with a band
    of one SD."""
    x, m, s = ordered(points)
    if not x:
        return
    ax.plot(x, m, lw=LW, ms=8, label=label, **style)
    if band:
        ax.fill_between(x, np.subtract(m, s), np.add(m, s), color=style["color"], alpha=0.15,
                        lw=0)


def star(ax, points, x):
    """The design's setting ``x`` on a curve, as a red star at its mean."""
    values = points.get(x)
    if values:
        ax.plot(x, np.mean(values), **STAR)


def histgb_line(ax, r: Results):
    value = r.f1(RAW["histgb"])
    if value is not None:
        ax.axhline(value, color="black", ls=":", lw=THIN)


def plain_log_ticks(axis, ticks=None):
    """Numbers instead of powers of ten on a log axis (``ticks``: where)."""
    if ticks is not None:
        axis.set_ticks(ticks)
        axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))


def placeholder(ax, text: str):
    ax.add_patch(Rectangle((0.02, 0.02), 0.96, 0.96, transform=ax.transAxes, fill=False,
                           ls="--", lw=1.5, ec="#999999"))
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
            fontsize=SMALL + 2, color="#666666")
    ax.axis("off")


def load(path: Path | None) -> dict | None:
    if path is None or not path.is_file():
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def confusion(ax, counts: np.ndarray):
    """A confusion matrix (rows: true class) with the classes' initials as labels."""
    names = class_names(len(counts))
    short = [{"Unknown": "Q"}.get(name, name[0]) for name in names]
    plot_confusion_matrix(counts, ax, labels=short, colorbar=False)
    ax.set_xticks(range(len(short)), short, rotation=0, ha="center")
    ax.set_aspect("equal")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    return ax.images[0]


def task_beats(r: Results) -> tuple[np.ndarray, np.ndarray] | None:
    """The balanced pool's beats as the runs scale them, and their classes."""
    from training.run import load_task
    from utils.preprocessing import standardize_data

    cfg = r.settings(RAW["ridge"])
    if cfg is None:
        return None
    X, Y, _ = load_task(cfg["task"])
    X, _ = standardize_data(X, X[:1], scaler_type=str(cfg["task"]["scaler"]))
    return (np.asarray(X, dtype=float).reshape(len(X), -1),
            np.argmax(np.asarray(Y).reshape(len(Y), -1), axis=1))


def saved_features(d: str, seed: int = SEEDS[0], fold: int | None = None):
    """The array's saved features (two images per trap) of the headline run: the training
    and held-out beats of ``fold``, or (``fold=None``) the held-out beats of every fold."""
    run = config.runs_dir() / f"P5-balanced-{d}-headline-s{seed}"
    paths = sorted(run.glob("fold-*.npz")) if fold is None else [run / f"fold-{fold}.npz"]
    parts = []
    for path in paths:
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=True) as z:
            parts.append({k: z[k] for k in ("train_states", "test_states", "Y_train", "Y_test",
                                            "test_index")})
    for part in parts:
        for key in ("train_states", "test_states", "Y_train", "Y_test"):
            part[key] = part[key].reshape(len(part[key]), -1)
        part["y_train"] = np.argmax(part["Y_train"], axis=1)
        part["y_test"] = np.argmax(part["Y_test"], axis=1)
    return parts


# ---------------------------------------------------------------------------------------
# One beat through the design
# ---------------------------------------------------------------------------------------

def simulate(design: Path, seed: int = SEEDS[0], beat_class: int = 0, before: int = 300) -> dict:
    """One typical beat of class ``beat_class`` (the pool beat closest to its class's mean)
    through the design, as a run steps it: one colony alone at 0.1, 1 and 10 times the
    design's input gain (Fig. 1B); every trap's LuxI (Fig. 2B, 2C); the inducer flux into
    the cells and the AHL exchange between traps (Fig. 2A); and the period at rest, from the
    last ``before`` minutes of the warm-up."""
    from scipy.signal import find_peaks

    from bioreservoir.dde import DEFAULT_DDE_PARAMS
    from training.figures import (
        array_states,
        build,
        colony_states,
        coupling,
        encoded_input,
        run_config,
    )
    from training.run import load_task
    from utils.preprocessing import standardize_data

    cfg = run_config(design)
    cfg["reservoir"]["seed"] = seed
    X, Y, _ = load_task(cfg["task"])
    X, _ = standardize_data(X, X[:1], scaler_type=str(cfg["task"]["scaler"]))
    X = np.asarray(X, dtype=float).reshape(len(X), -1)
    labels = np.argmax(np.asarray(Y).reshape(len(Y), -1), axis=1)
    members = np.flatnonzero(labels == beat_class)
    index = int(members[np.argmin(((X[members] - X[members].mean(axis=0)) ** 2).sum(1))])
    beat = X[index].reshape(-1, 1)
    gain = float(cfg["reservoir"]["params"]["input_scaling"])
    scales = [0.1 * gain, gain, 10 * gain]
    warm, rest, driven = colony_states(cfg, beat, index, scales)
    states, terms = array_states(cfg, beat, index)
    node = build(cfg)
    start = np.asarray(node.start_state(input_dim=1)).reshape(-1, 4)
    _, weights = coupling(cfg)
    G = weights.toarray()
    he = states[:, :, VARIABLES.index("He")]
    exchange = he @ G.T - he * G.sum(axis=1)  # sum_j G_ij (He_j - He_i), each trap
    flux = DEFAULT_DDE_PARAMS["D"] * node.cell_coupling * node.dde_scaling * terms[:, 0]
    lux = VARIABLES.index("I")
    free = build(cfg, warmup=0)
    free.initialize(np.zeros((1, 1)))
    history = np.array([free._step(free._warmup_drive)[:, lux]
                        for _ in range(int(cfg["reservoir"]["params"]["warmup"]))])[-before:]
    periods = []
    for x in history.T:
        peaks, _ = find_peaks(x, prominence=0.3 * (x.max() - x.min()))
        if len(peaks) >= 2:
            periods.append(np.median(np.diff(peaks)))
    return {"colony_warm": warm[-100:], "colony_rest": rest,
            "colony_driven": np.stack([driven[s] for s in scales]),
            "colony_scales": np.array(scales), "gain": gain,
            "inducer": encoded_input(cfg, beat), "luxi": states[:, :, lux],
            "start_luxi": start[:, lux], "inducer_flux": flux,
            "exchange": np.mean(np.abs(exchange), axis=1),
            "side": int(round(np.sqrt(states.shape[1]))), "beat_class": beat_class,
            "beat_index": index, "period": float(np.median(periods)) if periods else np.nan}


# ---------------------------------------------------------------------------------------
# Fig. 1: the design
# ---------------------------------------------------------------------------------------

def colony_panel(axes, sim: dict):
    """One colony alone: LuxI and AiiA at the end of the warm-up and during the beat, with no
    inducer and at three input gains; the inducer below (the v0.1 Fig. 3b)."""
    warm, rest = sim["colony_warm"], sim["colony_rest"]
    w, gain = len(warm), float(sim["gain"])
    t = np.arange(-w, len(rest))
    rows = [("No inducer", rest)] + [
        (f"Gain {s:.2g}" + (" (design)" if np.isclose(s, gain) else ""), states)
        for s, states in zip(sim["colony_scales"], sim["colony_driven"], strict=True)]
    for ax, (text, states) in zip(axes[:-1], rows, strict=True):
        for var in ("I", "A"):
            y = np.concatenate([warm[:, VARIABLES.index(var)], states[:, VARIABLES.index(var)]])
            ax.plot(t[:w], y[:w], **WARMUP, alpha=0.75)
            ax.plot(t[w - 1:], y[w - 1:], **LINES[var])
        tidy(ax)
        ax.text(0.01, 0.9, text, transform=ax.transAxes, fontsize=SMALL, va="top", bbox=BOX)
        ax.tick_params(labelbottom=False)
    axes[0].legend(handles=[Line2D([], [], **LINES[v], label=STATE_NAMES[v]) for v in "IA"]
                   + [Line2D([], [], **WARMUP, label="Warm-up")],
                   loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, fontsize=SMALL,
                   frameon=False)
    axes[len(rows) // 2].set_ylabel("Concentration")
    ax = axes[-1]
    u = np.concatenate([np.zeros(w), sim["inducer"]])
    ax.plot(t[:w], u[:w], **WARMUP)
    ax.plot(t[w - 1:], u[w - 1:], **ECG)
    tidy(ax)
    ax.set_ylabel("Inducer")
    ax.set_xlabel("Time (min)")


def draw_chip(ax):
    """A sketch of the chip: inlets and mixer, the channel, 6 x 6 of the 24 x 24 traps with
    AHL exchange between neighbours, and the camera."""
    ax.set_xlim(0, 16)
    ax.set_ylim(-0.3, 7.2)
    ax.set_aspect("equal")
    ax.axis("off")
    water, inducer, edge = "#d8ecf7", "#fbe3c4", "#2b7a3d"
    for y, text, colour in ((6.4, "Medium", water), (5.0, "Medium + AHL", inducer)):
        ax.add_patch(Rectangle((0.2, y - 0.25), 2.2, 0.5, fc=colour, ec="#7f7f7f", lw=1.5))
        ax.text(1.3, y, text, fontsize=SMALL - 1, ha="center", va="center")
    ax.add_patch(FancyBboxPatch((2.5, 4.55), 1.4, 2.2, boxstyle="round,pad=0.05", fc="#eadcf2",
                                ec="#7f7f7f", lw=1.5))
    ax.text(3.2, 5.65, "Mixer", ha="center", va="center", fontsize=SMALL)
    ax.add_patch(Rectangle((3.95, 5.3), 7.9, 0.7, fc=water, ec="#7f7f7f", lw=1.5))
    for x in (4.6, 7.2, 9.8):
        ax.annotate("", xy=(x + 0.9, 5.65), xytext=(x, 5.65),
                    arrowprops={"arrowstyle": "-|>", "color": "#1f77b4", "lw": 2.5})
    ax.text(11.95, 5.65, "Outlet", va="center", fontsize=SMALL)
    ax.text(4.05, 6.1, "Flow (AHL washout, μ)", ha="left", va="bottom", fontsize=SMALL)
    xs, ys = 4.85 + np.arange(6) * 1.05, 0.6 + np.arange(6) * 0.75
    ax.add_patch(Rectangle((4.3, 0.15), 6.45, 4.85, fc=water, ec="none", alpha=0.6))
    for x in xs:
        ax.annotate("", xy=(x, 4.7), xytext=(x, 5.28),
                    arrowprops={"arrowstyle": "-|>", "color": "#1f77b4", "lw": 1.2})
    greens = plt.get_cmap("Greens")
    rng = np.random.default_rng(3)
    for x in xs:
        for y in ys:
            ax.add_patch(FancyBboxPatch((x - 0.3, y - 0.2), 0.6, 0.4,
                                        boxstyle="round,pad=0.04",
                                        fc=greens(rng.uniform(0.25, 0.9)), ec=edge, lw=1.2))
    exchange = {"arrowstyle": "<->", "color": "#d55e00", "lw": 1.4, "shrinkA": 0, "shrinkB": 0}
    for x in xs[:-1]:
        for y in ys:
            ax.annotate("", xy=(x + 1.05 - 0.36, y), xytext=(x + 0.36, y), arrowprops=exchange)
    for x in xs:
        for y in ys[:-1]:
            ax.annotate("", xy=(x, y + 0.75 - 0.25), xytext=(x, y + 0.25), arrowprops=exchange)
    ax.annotate("AHL exchange", xy=(xs[0] + 0.52, ys[-2]), xytext=(0.2, 2.6), fontsize=SMALL,
                arrowprops={"arrowstyle": "-", "color": "#d55e00", "lw": 1.4})
    ax.annotate("Colony (GFP\nreporter of LuxI)", xy=(xs[0] - 0.2, ys[0]), xytext=(0.2, 0.7),
                fontsize=SMALL, arrowprops={"arrowstyle": "-", "color": edge, "lw": 1.4})
    ax.add_patch(Rectangle((13.0, 2.7), 1.6, 1.1, fc="#555555", ec="black", lw=1.5))
    ax.add_patch(Polygon([[13.25, 2.7], [14.35, 2.7], [14.05, 2.15], [13.55, 2.15]],
                         fc="#999999", ec="black", lw=1.5))
    ax.annotate("", xy=(13.0, 3.1), xytext=(10.9, 2.4),
                arrowprops={"arrowstyle": "-|>", "color": edge, "lw": 2.5, "ls": "--"})
    ax.text(13.8, 4.0, "Camera", ha="center", fontsize=SMALL)


def figure_design(sim: dict | None):
    """Fig. 1: the colony circuit (placeholder), one colony during a beat, and the chip."""
    fig = plt.figure(figsize=(16, 13.5))
    outer = fig.add_gridspec(2, 1, height_ratios=[6.4, 5.6], hspace=0.12)
    top = outer[0].subgridspec(1, 2, width_ratios=[1, 1.5], wspace=0.25)
    ax = fig.add_subplot(top[0])
    placeholder(ax, "Colony circuit\n(v0.1 panel)")
    panel(ax, "A")
    rows = top[1].subgridspec(5, 1, height_ratios=[2.2, 2.2, 2.2, 2.2, 1.6], hspace=0.12)
    axes = [fig.add_subplot(rows[i]) for i in range(5)]
    for other in axes[1:]:
        other.sharex(axes[0])
    if sim is not None:
        colony_panel(axes, sim)
    panel(axes[0], "B")
    ax = fig.add_subplot(outer[1])
    draw_chip(ax)
    panel(ax, "C")
    return fig


# ---------------------------------------------------------------------------------------
# Fig. 2: the array during a beat and the operating point
# ---------------------------------------------------------------------------------------

def figure_dynamics(sim: dict | None, r: Results, d: str, oscillation: Path | None):
    fig = plt.figure(figsize=(17, 13.5))
    outer = fig.add_gridspec(2, 2, width_ratios=[1.1, 1], hspace=0.3, wspace=0.28)
    upper = outer[0, 0].subgridspec(2, 1, height_ratios=[2.2, 1.2], hspace=0.08)
    ax, ax_u = fig.add_subplot(upper[0]), fig.add_subplot(upper[1])
    ax_u.sharex(ax)
    bx, cx, dx = (fig.add_subplot(outer[0, 1]), fig.add_subplot(outer[1, 0]),
                  fig.add_subplot(outer[1, 1]))
    if sim is not None:
        t = np.arange(len(sim["inducer"]))
        ax.plot(t, sim["inducer_flux"], color="#E69F00", lw=LW, label="Inducer")
        ax.plot(t, sim["exchange"], color="#56B4E9", lw=LW, ls="--", label="AHL exchange")
        ax.set_yscale("log")
        ax.set_ylabel("Flux per trap (min$^{-1}$)")
        ax.legend(loc="center right", fontsize=SMALL)
        ax.tick_params(labelbottom=False)
        ax_u.plot(t, sim["inducer"], **ECG)
        luxi = sim["luxi"]
        order = np.argsort(np.argmax(luxi[:60], axis=0))  # by the peak in the first hour
        image = bx.imshow(luxi[:, order].T, aspect="auto", origin="lower",
                          interpolation="nearest", cmap="viridis",
                          vmax=np.percentile(luxi, 99))
        fig.colorbar(image, ax=bx, label="LuxI")
        side = int(sim["side"])
        col, row = np.meshgrid(np.arange(side), np.arange(side))
        links = [[(i, j), (i + 1, j)] for i in range(side - 1) for j in range(side)]
        links += [[(i, j), (i, j + 1)] for i in range(side) for j in range(side - 1)]
        cx.add_collection(LineCollection(links, colors="#bbbbbb", linewidths=THIN, zorder=1))
        dots = cx.scatter(col.ravel(), row.ravel(), c=luxi[-1], cmap="viridis", s=95,
                          vmax=np.percentile(luxi[-1], 99), zorder=2)
        fig.colorbar(dots, ax=cx, label="LuxI")
    for a in (ax, ax_u):
        tidy(a)
    ax_u.set_ylabel("Inducer")
    ax_u.set_xlabel("Time (min)")
    bx.set_xlabel("Time (min)")
    bx.set_ylabel("Trap")
    cx.set_aspect("equal")
    cx.axis("off")

    flows, densities, grid = f1_map(r)
    if flows:
        image = dx.imshow(grid, origin="lower", aspect="auto", cmap="viridis",
                          vmin=max(0.75, float(np.nanmin(grid))), vmax=float(np.nanmax(grid)))
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                dark = image.norm(grid[i, j]) < 0.55
                dx.text(j, i, f"{grid[i, j]:.3f}".lstrip("0"), ha="center", va="center",
                        fontsize=SMALL - 1, color="white" if dark else "black")
        fig.colorbar(image, ax=dx, label="Macro F1")
        handles = []
        o_flows, o_densities, share, _ = oscillation_map(oscillation, d)
        if o_flows == flows and o_densities == densities:
            for i, j in zip(*np.nonzero(share < 50), strict=True):
                dx.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///",
                                       ec="white", lw=0))
            handles.append(Rectangle((0, 0), 1, 1, fc="#777777", hatch="///", ec="white",
                                     label="< 50% oscillating"))
        point = design_point(r, d)
        headline = r.seeds(f"P5-balanced-{d}-headline")
        if point is not None:
            dx.plot(np.interp(point[0], flows, range(len(flows))),
                    np.interp(point[1], densities, range(len(densities))), **STAR)
            handles.append(Line2D([], [], **{**STAR, "ms": 16},
                                  label=f"Design ({np.mean(headline):.3f})" if headline
                                  else "Design"))
        dx.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2,
                  fontsize=SMALL, frameon=False)
        dx.set_xticks(range(len(flows)), [f"{f:g}" for f in flows])
        dx.set_yticks(range(len(densities)), [f"{x:g}" for x in densities])
        dx.set_xlabel("AHL washout μ")
        dx.set_ylabel("Cell density")
    for a, letter in ((ax, "A"), (bx, "B"), (cx, "C"), (dx, "D")):
        panel(a, letter)
    return fig


# ---------------------------------------------------------------------------------------
# Fig. 3: classification
# ---------------------------------------------------------------------------------------

def projection(r: Results, d: str, seed: int = SEEDS[0]) -> dict:
    """Held-out beats of fold 0 projected on two linear discriminants fitted on the fold's
    training beats, for the array's features and the raw beat."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler

    def project(train, y_train, test):
        scaler = StandardScaler().fit(train)
        lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto", n_components=2)
        lda.fit(scaler.transform(train), y_train)
        return lda.transform(scaler.transform(test))

    parts, beats = saved_features(d, seed, fold=0), task_beats(r)
    if parts is None or beats is None:
        return {}
    fold, (X, labels) = parts[0], beats
    held_out = fold["test_index"]
    if not np.array_equal(labels[held_out], fold["y_test"]):
        print("Warning: the raw pool's labels differ from the run's held-out labels.",
              file=sys.stderr)
    rest = np.setdiff1d(np.arange(len(X)), held_out)
    return {"array": (project(fold["train_states"], fold["y_train"], fold["test_states"]),
                      fold["y_test"]),
            "raw": (project(X[rest], labels[rest], X[held_out]), labels[held_out])}


def figure_classification(r: Results, d: str):
    fig = plt.figure(figsize=(15.5, 14))
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.05], hspace=0.3, wspace=0.12)
    top = [fig.add_subplot(grid[0, i]) for i in range(2)]
    bottom = [fig.add_subplot(grid[1, i]) for i in range(2)]
    image = None
    for ax, title, runs in ((top[0], "Array", [f"P5-balanced-{d}-headline-s{s}"
                                               for s in SEEDS]),
                            (top[1], "Raw beat", [RAW["ridge"]])):
        matrices = [m for m in (r.confusion(run) for run in runs) if m is not None]
        if matrices:
            image = confusion(ax, np.sum(matrices, axis=0))
        ax.set_title(title, fontsize=SMALL + 4)
    top[1].set_ylabel("")
    top[1].tick_params(labelleft=False)
    if image is not None:
        fig.colorbar(image, cax=fig.add_subplot(grid[0, 2]), label="Fraction of the class")
    found = projection(r, d)
    for ax, key in ((bottom[0], "array"), (bottom[1], "raw")):
        if key in found:
            points, labels = found[key]
            for k, name in enumerate(CLASS_NAMES):
                mine = labels == k
                ax.scatter(points[mine, 0], points[mine, 1], s=14, alpha=0.6, lw=0,
                           color=CLASS_COLOURS[name], label=name)
        ax.set_xlabel("LD 1")
        tidy(ax)
    bottom[0].set_ylabel("LD 2")
    handles, labels = bottom[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=SMALL, frameon=False,
                   markerscale=2.5, bbox_to_anchor=(0.5, 0.01))
    for ax, letter in zip(top + bottom, "ABCD", strict=True):
        panel(ax, letter)
    fig.subplots_adjust(bottom=0.1)
    return fig


# ---------------------------------------------------------------------------------------
# Fig. 4: what a chip needs
# ---------------------------------------------------------------------------------------

def figure_requirements(r: Results, d: str, flat_share: Path | None):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), gridspec_kw={"hspace": 0.35,
                                                                 "wspace": 0.38})
    ax = axes[0, 0]
    curve = traps(r, d)
    line(ax, curve, DESIGN)
    star(ax, curve, 576)
    ax.set_xscale("log")
    plain_log_ticks(ax.xaxis, [100, 1000, 10000])
    ax.set_xlabel("Traps")

    ax = axes[0, 1]
    for agg, label, style in ((HEADLINE, "2 images", DESIGN), (ONE, "1 image", FIRST),
                              (FOREST, "4 images", SECOND)):
        points = scaled(r.sweep(f"P3-{d}-noise", "features.measurement_noise",
                                features__aggregation=agg))
        line(ax, points, style, label)
        if agg == HEADLINE:
            star(ax, points, 1.0)
    ax.set_xlabel("Measurement noise (%)")
    ax.legend(loc="lower left", fontsize=SMALL)

    ax = axes[0, 2]
    table = r.grid("extra-576-connectivity", "reservoir.params.input_connectivity",
                   "reservoir.params.coupling") if d == "p576" else {}
    shares = sorted({k[0] for k in table})
    for coupling, label, style in (("diffusion", "Coupled", DESIGN), ("none", "Isolated", FIRST)):
        points = {100 * s: list(table.get((s, coupling), {}).values()) for s in shares}
        line(ax, points, style, label)
        if coupling == "diffusion":
            star(ax, points, 100)
    ax.set_xscale("log")
    ax.set_xticks([100 * s for s in shares], [f"{100 * s:g}" for s in shares])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Traps receiving inducer (%)")
    ax.legend(loc="lower right", fontsize=SMALL)

    ax = axes[1, 0]
    for study, label, style in ((f"P3-{d}-calibration", "Reference image", DESIGN),
                                (f"P3-{d}-calibration-noref", "No reference", FIRST)):
        points = scaled(r.sweep(study, "features.calibration_error",
                                features__aggregation=HEADLINE))
        line(ax, points, style, label)
        if style is DESIGN:
            star(ax, points, 0.0)
    ax.set_xlabel("Calibration error (%)")
    ax.legend(loc="lower left", fontsize=SMALL)

    ax = axes[1, 1]
    points = r.sweep(f"P3-{d}-heterogeneity", "reservoir.params.param_heterogeneity_cv")
    points.update(r.sweep(f"P3-{d}-heterogeneity-64", "reservoir.params.param_heterogeneity_cv"))
    points = scaled(points)
    flat = flat_shares(flat_share, d)
    if flat:
        x, m, _ = ordered(scaled(flat))
        bars = ax.twinx()
        bars.bar(x, m, width=1.6, color="#cccccc", zorder=0)
        bars.set_ylim(0, 100)
        bars.set_ylabel("Not oscillating (%)", color="#777777")
        bars.tick_params(axis="y", colors="#777777")
        bars.spines[["top"]].set_visible(False)
        ax.set_zorder(bars.get_zorder() + 1)
        ax.patch.set_visible(False)
    line(ax, points, DESIGN)
    star(ax, points, 10.0)
    ax.set_xlabel("Kinetic variability (CV, %)")

    ax = axes[1, 2]
    for study, key, label, style in (
            (f"P3-{d}-density", "density", "Cell density", FIRST),
            (f"P3-{d}-dilution", "dilution", "Washout", SECOND)):
        spread = scaled(r.sweep(study, f"reservoir.params.heterogeneity_{key}_cv"))
        line(ax, spread, style, label)
    star(ax, spread, 0.0)
    ax.set_xlabel("Spread between traps (CV, %)")
    ax.legend(loc="lower left", fontsize=SMALL)

    for ax, letter in zip(axes.flat, "ABCDEF", strict=True):
        histgb_line(ax, r)
        tidy(ax)
        panel(ax, letter)
    for ax in axes[:, 0]:
        ax.set_ylabel("Macro F1")
    fig.legend(handles=[Line2D([], [], **{**STAR, "ms": 16}, label="Design"),
                        Line2D([], [], color="black", ls=":", lw=THIN, label="Raw beat, HistGB")],
               loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=SMALL,
               frameon=False)
    return fig


# ---------------------------------------------------------------------------------------
# SI figures
# ---------------------------------------------------------------------------------------

def figure_class_images(r: Results, d: str):
    """SI: for each class, its beats (mean and middle 80%, as the v0.1 Fig. 5a) and what the
    camera records: the class's average image at 140 and 187 min (log2 of each trap's image
    over its reference), and that average minus the average over all classes."""
    beats, parts = task_beats(r), saved_features(d)
    if beats is None or parts is None:
        return None
    X, labels = beats
    features = np.log2(np.clip(np.concatenate([p["test_states"] for p in parts]), 1e-9, None))
    y = np.concatenate([p["y_test"] for p in parts])
    side = int(round(np.sqrt(features.shape[1] // 2)))
    average = np.array([features[y == k].mean(axis=0) for k in range(len(CLASS_NAMES))])
    deviation = average - features.mean(axis=0)
    fig = plt.figure(figsize=(16, 15))
    grid = fig.add_gridspec(len(CLASS_NAMES) + 1, 6, width_ratios=[3.2, 1, 1, 0.35, 1, 1],
                            height_ratios=[1] * len(CLASS_NAMES) + [0.08], hspace=0.12,
                            wspace=0.06)
    t = np.arange(X.shape[1])
    spans = (np.percentile(np.abs(average), 99), np.percentile(np.abs(deviation), 99))
    shown = [None, None]
    for k, name in enumerate(CLASS_NAMES):
        colour = CLASS_COLOURS[name]
        ax = fig.add_subplot(grid[k, 0])
        mine = X[labels == k]
        low, high = np.quantile(mine, (0.1, 0.9), axis=0)
        ax.fill_between(t, low, high, color=colour, alpha=0.25, lw=0)
        ax.plot(t, mine.mean(axis=0), color=colour, lw=LW)
        for minute in (140, 187):
            ax.axvline(minute - 1, color="#999999", ls=":", lw=THIN)
        ax.text(0.99, 0.92, name, transform=ax.transAxes, ha="right", va="top", fontsize=SMALL)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0, len(t) + 40)
        if k < len(CLASS_NAMES) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Time on the chip (min)")
        if k == len(CLASS_NAMES) // 2:
            ax.set_ylabel("ECG (z-score)")
        for g, (values, column) in enumerate(((average, 1), (deviation, 4))):
            for i in range(2):
                image_ax = fig.add_subplot(grid[k, column + i])
                image = values[k, i * side * side:(i + 1) * side * side].reshape(side, side)
                shown[g] = image_ax.imshow(image, cmap="RdBu_r", vmin=-spans[g], vmax=spans[g],
                                           interpolation="nearest")
                image_ax.set_xticks([])
                image_ax.set_yticks([])
                if k == 0:
                    image_ax.set_title(f"{(140, 187)[i]} min", fontsize=SMALL + 1)
                    if i == 0:
                        panel(image_ax, "BC"[g], outside=True)
        if k == 0:
            panel(ax, "A")
    for g, column in enumerate((1, 4)):
        cax = fig.add_subplot(grid[-1, column:column + 2])
        fig.colorbar(shown[g], cax=cax, orientation="horizontal",
                     label="log$_2$(image / reference)" if g == 0 else "Minus all classes")
    return fig


def figure_confusion_tasks(r: Results, d: str):
    """SI: the array's confusion matrices on the three tasks (ridge on two images)."""
    fig = plt.figure(figsize=(20, 7))
    grid = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.2)
    image = None
    for i, (title, runs) in enumerate((
            ("Balanced", [f"P5-balanced-{d}-headline-s{s}" for s in SEEDS]),
            ("Unbalanced", [f"P5-unbalanced-{d}-headline-s{SEEDS[0]}"]),
            ("Binary", [f"P5-binary-{d}-headline-s{SEEDS[0]}"]))):
        ax = fig.add_subplot(grid[i])
        matrices = [m for m in (r.confusion(run) for run in runs) if m is not None]
        if matrices:
            image = confusion(ax, np.sum(matrices, axis=0))
        ax.set_title(title, fontsize=SMALL + 4)
        ax.set_xlabel("")
        if i:
            ax.set_ylabel("")
        panel(ax, "ABC"[i])
    if image is not None:
        fig.colorbar(image, cax=fig.add_subplot(grid[3]), label="Fraction of the class")
    fig.supxlabel("Predicted", fontsize=15)
    return fig


def figure_settings(r: Results, d: str):
    """SI: the other settings of the array, ridge on two images (mean and SD over the
    reporting seeds); the design's setting as a red star."""
    fig = plt.figure(figsize=(21, 11))
    grid = fig.add_gridspec(2, 8, hspace=0.45, wspace=0.9)
    axes = [fig.add_subplot(grid[0, 2 * i:2 * i + 2]) for i in range(4)]
    axes += [fig.add_subplot(grid[1, 2 * i + 1:2 * i + 3]) for i in range(3)]

    def categories(ax, points, order, labels, design):
        for i, key in enumerate(order):
            values = points.get(key, [])
            if not values:
                continue
            sd = np.std(values, ddof=1) if len(values) > 1 else 0.0
            ax.errorbar(i, np.mean(values), yerr=sd, fmt="none", color=DESIGN["color"],
                        lw=THIN, capsize=6)
            ax.plot(i, np.mean(values), **(STAR if key == design else
                                            {"marker": "o", "ms": 9, "ls": "none",
                                             "color": DESIGN["color"]}))
        ax.set_xticks(range(len(order)), labels)
        ax.set_xlim(-0.6, len(order) - 0.4)

    reset = r.sweep(f"P3-{d}-reset", "reservoir.params.reset_noise")
    order = sorted(k for k in reset if k is not None)
    categories(axes[0], reset, order, [f"{100 * k:g}" for k in order], 0.0)
    axes[0].set_xlabel("Reset error (%)")

    warmup = r.sweep(f"P3-{d}-warmup", "reservoir.params.warmup")
    order = sorted(warmup)
    categories(axes[1], warmup, order, [str(k) for k in order], 900)
    axes[1].set_xlabel("Warm-up (min)")
    axes[1].tick_params(axis="x", labelrotation=45)

    rest = r.sweep(f"P3-{d}-warmup-input", "reservoir.params.warmup_input")
    categories(axes[2], rest, ["zero", "rest"], ["None", "Rest"], "zero")
    axes[2].set_xlabel("Inducer in warm-up")

    levels = r.sweep(f"P3-{d}-input-levels", "reservoir.params.input_levels")
    order = sorted(levels, key=lambda k: (isinstance(k, str), k))
    categories(axes[3], levels, order, [str(k) if k != "continuous" else "Cont." for k in order],
               "continuous")
    axes[3].set_xlabel("Inducer levels")

    layout = r.sweep(f"P6-{d}-gaussian", "reservoir.params.interaction_diameters")
    order = sorted(layout)
    categories(axes[4], layout, order, [f"{k:g}" for k in order], None)
    headline = r.seeds(f"P5-balanced-{d}-headline")
    if headline:
        axes[4].axhline(np.mean(headline), color="#999999", ls="--", lw=THIN)
    axes[4].set_xlabel("Gaussian reach (traps)")

    ax = axes[5]
    sizes = sorted({row["reservoir.params.units"] for row in r.trials(f"P6-{d}-units-images")})
    for agg, label, style, shift in ((ONE, "1", FIRST, -0.15), (HEADLINE, "2", DESIGN, 0.0),
                                     (FOREST, "4", SECOND, 0.15)):
        x, m, s = ordered(r.sweep(f"P6-{d}-units-images", "reservoir.params.units",
                                  features__aggregation=agg))
        ax.errorbar([sizes.index(v) + shift for v in x], m, yerr=s, lw=THIN, ms=8, capsize=4,
                    label=label, **style)
    ax.set_xticks(range(len(sizes)), [f"{v:,}" for v in sizes])
    ax.set_xlabel("Traps")
    ax.legend(title="Images", loc="lower right", fontsize=SMALL, title_fontsize=SMALL, ncol=3)

    times, final = image_times(r, d)
    order = sorted(times)
    categories(axes[6], times, order, [str(t) for t in order], 140)
    if final:
        axes[6].axhline(np.mean(final), color=FIRST["color"], ls="--", lw=THIN)
    axes[6].set_xlabel("Earlier image (min)")
    axes[6].tick_params(axis="x", labelrotation=45)

    for ax, letter in zip(axes, "ABCDEFG", strict=True):
        tidy(ax)
        panel(ax, letter)
    axes[0].set_ylabel("Macro F1")
    axes[4].set_ylabel("Macro F1")
    return fig


# ---------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("simulate", "figures"),
                        help="simulate runs one beat through the design; figures draws.")
    parser.add_argument("--design", choices=sorted(DESIGNS), default="p576",
                        help="The design (default p576).")
    parser.add_argument("--out", type=Path, help="Output folder (default "
                        "results/figures/paper/<design>).")
    parser.add_argument("--flat-share", type=Path,
                        help="CSV of scripts/paper/flat_share.py (Fig. 4E).")
    parser.add_argument("--oscillation", type=Path,
                        help="CSV of scripts/paper/oscillation_map.py (Fig. 2D).")
    add_formats_option(parser)
    args = parser.parse_args(argv)
    d = args.design
    out = args.out or config.figures_dir(f"paper/{d}")
    out.mkdir(parents=True, exist_ok=True)
    simulated = out / "simulated.npz"
    if args.command == "simulate":
        data = simulate(Path("experiments/v0.2/designs") / f"{d}.yaml")
        np.savez(simulated, **data)
        print(f"{simulated}: beat {data['beat_index']} ({CLASS_NAMES[data['beat_class']]}); "
              f"period at rest {data['period']:.1f} min (Danino et al.: "
              f"{DANINO_PERIODS[0]}-{DANINO_PERIODS[1]} min)")
        return 0
    r = Results()
    sim = load(simulated)
    with plt.style.context(STYLE):
        drawn = {"fig1_design": figure_design(sim),
                 "fig2_dynamics": figure_dynamics(sim, r, d, args.oscillation),
                 "fig3_classification": figure_classification(r, d),
                 "fig4_requirements": figure_requirements(r, d, args.flat_share),
                 "si_class_images": figure_class_images(r, d),
                 "si_confusion_tasks": figure_confusion_tasks(r, d),
                 "si_settings": figure_settings(r, d)}
        for name, fig in drawn.items():
            if fig is not None:
                for path in save_figure(fig, name, folder=out, formats=args.formats):
                    print(path)
    if r.missing:
        print(f"\nNot found ({len(r.missing)}): " + ", ".join(sorted(r.missing)[:40])
              + (" ..." if len(r.missing) > 40 else ""), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
