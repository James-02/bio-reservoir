"""Draw the figures of the revised paper (design p576) in the style of the v0.1 figures.

Reads the collected results through ``paper_figures.py`` (``BIORESERVOIR_RESULTS_DIR``) and,
for the panels of one beat, ``simulated.npz``, written by ``simulate``:

    python scripts/paper/paper_panels.py simulate        # one beat through the design (~1 min)
    python scripts/paper/paper_panels.py figures --oscillation oscillation_map.csv \\
        --flat-share flat_share.csv --formats png,pdf

Main text: ``fig1_design`` (A is a placeholder for the colony circuit of the v0.1 figure; C
is a draft of the chip to finish by hand), ``fig2_dynamics``, ``fig3_classification``,
``fig4_requirements``. SI: ``si_confusion_tasks``, ``si_class_images``, ``si_projections``,
``si_settings``. Run from the repository root: ``simulate`` reads
``experiments/v0.2/designs/<design>.yaml``, and Fig. 3 reads the task's beats.
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
    BUILT,
    DANINO_PERIODS,
    DESIGNS,
    FOREST,
    HEADLINE,
    LARGEST,
    ONE,
    RAW,
    SEEDS,
    Results,
    design_point,
    f1_map,
    flat_shares,
    ideal_traps,
    image_times,
    ordered,
    oscillation_map,
    paired,
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
ARRAY, ISOLATED, ALTERNATIVE = OKABE_ITO[0], OKABE_ITO[4], "#8c564b"


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


def line(ax, points, colour, label, *, marker="o", style="-", band=True):
    """Mean over seeds at each x, with a band of one SD."""
    x, m, s = ordered(points)
    if not x:
        return
    ax.plot(x, m, style, marker=marker, color=colour, lw=LW, ms=8, label=label)
    if band:
        ax.fill_between(x, np.subtract(m, s), np.add(m, s), color=colour, alpha=0.15, lw=0)


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


# ---------------------------------------------------------------------------------------
# One beat through the design
# ---------------------------------------------------------------------------------------

def simulate(design: Path, seed: int = SEEDS[0], beat_class: int = 0, before: int = 300) -> dict:
    """One typical beat of class ``beat_class`` (the pool beat closest to its class's mean)
    through the design, as a run steps it: one colony alone at 0.1, 1 and 10 times the
    design's input gain (Fig. 1B); every trap's LuxI and the reference image (Fig. 1D,
    2B, 2C); the inducer flux into the cells and the AHL exchange between traps (Fig. 2A);
    and the period at rest, from the last ``before`` minutes of the warm-up."""
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
        (f"Input gain {s:.2g}" + (" (design)" if np.isclose(s, gain) else ""), states)
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
                   + [Line2D([], [], **WARMUP, label="Warm-up (no inducer)")],
                   loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, fontsize=SMALL,
                   frameon=False)
    axes[len(rows) // 2].set_ylabel("Concentration (model units)")
    ax = axes[-1]
    u = np.concatenate([np.zeros(w), sim["inducer"]])
    ax.plot(t[:w], u[:w], **WARMUP)
    ax.plot(t[w - 1:], u[w - 1:], **ECG,
            label=f"One {CLASS_NAMES[int(sim['beat_class'])].lower()} beat")
    tidy(ax)
    ax.set_ylabel("Inducer")
    ax.set_xlabel("Time (min)")
    ax.legend(loc="upper left", fontsize=SMALL, frameon=False)


def draw_chip(ax):
    """A draft of the chip: inlets and mixer, the channel, 6 x 6 of the 24 x 24 traps with AHL
    exchange between neighbours, and the camera."""
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
    ax.text(3.2, 4.3, "inducer level set\nby the beat, each\nminute", ha="center",
            va="top", fontsize=SMALL - 2)
    ax.add_patch(Rectangle((3.95, 5.3), 7.9, 0.7, fc=water, ec="#7f7f7f", lw=1.5))
    for x in (4.6, 7.2, 9.8):
        ax.annotate("", xy=(x + 0.9, 5.65), xytext=(x, 5.65),
                    arrowprops={"arrowstyle": "-|>", "color": "#1f77b4", "lw": 2.5})
    ax.text(11.95, 5.65, "Outlet", va="center", fontsize=SMALL)
    ax.text(4.05, 6.1, "Flow brings the inducer to every trap and washes AHL out (rate μ)",
            ha="left", va="bottom", fontsize=SMALL - 1)
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
    ax.text(xs.mean(), -0.15, "6 × 6 of the 24 × 24 traps (576 colonies) shown", ha="center",
            va="center", fontsize=SMALL)
    ax.annotate("AHL exchange\nbetween neighbouring\ntraps", xy=(xs[0] + 0.52, ys[-2]),
                xytext=(0.2, 2.3), fontsize=SMALL,
                arrowprops={"arrowstyle": "-", "color": "#d55e00", "lw": 1.4})
    ax.annotate("A colony in each trap;\nGFP reporter of LuxI", xy=(xs[0] - 0.2, ys[0]),
                xytext=(0.2, 0.5), fontsize=SMALL,
                arrowprops={"arrowstyle": "-", "color": edge, "lw": 1.4})
    ax.add_patch(Rectangle((13.0, 2.7), 1.6, 1.1, fc="#555555", ec="black", lw=1.5))
    ax.add_patch(Polygon([[13.25, 2.7], [14.35, 2.7], [14.05, 2.15], [13.55, 2.15]],
                         fc="#999999", ec="black", lw=1.5))
    ax.annotate("", xy=(13.0, 3.1), xytext=(10.9, 2.4),
                arrowprops={"arrowstyle": "-|>", "color": edge, "lw": 2.5, "ls": "--"})
    ax.text(11.95, 2.95, "fluorescence", rotation=18, ha="center", fontsize=SMALL - 1,
            color=edge)
    ax.text(13.8, 4.0, "Camera", ha="center", fontsize=SMALL)
    ax.text(13.8, 1.9, "Reference image (0 min);\nimages at 140 and 187 min", ha="center",
            va="top", fontsize=SMALL - 1)


def readout_strip(fig, axes, sim: dict):
    """What the camera records for one beat and what the readout receives."""
    side = int(sim["side"])
    ref = sim["start_luxi"].reshape(side, side)
    later = {t: sim["luxi"][t - 1].reshape(side, side) for t in (140, 187)}
    images = [("Reference, 0 min", ref), ("140 min", later[140]), ("187 min", later[187])]
    low, high = np.percentile(np.concatenate([im.ravel() for _, im in images]), [1, 99])
    for ax, (title, image) in zip(axes[:3], images, strict=True):
        shown = ax.imshow(image, cmap="viridis", vmin=low, vmax=high, interpolation="nearest")
        ax.set_title(title, fontsize=SMALL)
    fig.colorbar(shown, ax=axes[:3], orientation="horizontal", fraction=0.08, pad=0.06,
                 aspect=40, label="LuxI (model units)")
    ratios = [(f"{t} min / reference", np.log2(later[t] / ref)) for t in (140, 187)]
    span = np.percentile(np.abs(np.concatenate([x.ravel() for _, x in ratios])), 98)
    for ax, (title, image) in zip(axes[3:5], ratios, strict=True):
        shown = ax.imshow(image, cmap="RdBu_r", vmin=-span, vmax=span, interpolation="nearest")
        ax.set_title(title, fontsize=SMALL)
    fig.colorbar(shown, ax=axes[3:5], orientation="horizontal", fraction=0.08, pad=0.06,
                 aspect=27, label="log$_2$ ratio (what the readout gets)")
    for ax in axes[:5]:
        ax.set_xticks([])
        ax.set_yticks([])
    axes[5].axis("off")
    axes[5].text(0.08, 0.5, "1,152 values per beat\n(576 traps × 2 images)\n↓\nridge readout\n↓\n"
                 "N, S, V, F or Q", ha="left", va="center", fontsize=SMALL + 1)


def figure_design(sim: dict | None):
    """Fig. 1: the colony circuit (placeholder), one colony during a beat, the chip, and what
    the camera records."""
    fig = plt.figure(figsize=(16, 19))
    outer = fig.add_gridspec(3, 1, height_ratios=[6.4, 5.6, 3.0], hspace=0.2)
    top = outer[0].subgridspec(1, 2, width_ratios=[1, 1.5], wspace=0.25)
    ax = fig.add_subplot(top[0])
    placeholder(ax, "Colony circuit:\nthe v0.1 panel, unchanged\n(add AHL exchange with\n"
                    "the neighbouring traps)")
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
    strip = outer[2].subgridspec(1, 6, width_ratios=[1, 1, 1, 1, 1, 1.25], wspace=0.3)
    axes = [fig.add_subplot(strip[i]) for i in range(6)]
    if sim is not None:
        readout_strip(fig, axes, sim)
    panel(axes[0], "D", outside=True)
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
        ax.plot(t, sim["inducer_flux"], color="#E69F00", lw=LW,
                label=r"Inducer into the cells: $D\,u_t$")
        ax.plot(t, sim["exchange"], color="#56B4E9", lw=LW, ls="--",
                label=r"AHL exchange: $\sum_j G_{ij}(H_{e,j}-H_{e,i})$")
        ax.set_yscale("log")
        ax.set_ylabel("Mean per trap (per min)")
        ax.legend(loc="center right", fontsize=SMALL)
        ax.tick_params(labelbottom=False)
        ax_u.plot(t, sim["inducer"], **ECG,
                  label=f"One {CLASS_NAMES[int(sim['beat_class'])].lower()} beat")
        ax_u.legend(loc="upper right", fontsize=SMALL, frameon=False)
        luxi = sim["luxi"]
        order = np.argsort(np.argmax(luxi[:60], axis=0))  # by the peak in the first hour
        image = bx.imshow(luxi[:, order].T, aspect="auto", origin="lower",
                          interpolation="nearest", cmap="viridis",
                          vmax=np.percentile(luxi, 99))
        fig.colorbar(image, ax=bx, label="LuxI concentration")
        side = int(sim["side"])
        col, row = np.meshgrid(np.arange(side), np.arange(side))
        links = [[(i, j), (i + 1, j)] for i in range(side - 1) for j in range(side)]
        links += [[(i, j), (i, j + 1)] for i in range(side) for j in range(side - 1)]
        cx.add_collection(LineCollection(links, colors="#bbbbbb", linewidths=THIN, zorder=1))
        dots = cx.scatter(col.ravel(), row.ravel(), c=luxi[-1], cmap="viridis", s=95,
                          vmax=np.percentile(luxi[-1], 99), zorder=2)
        fig.colorbar(dots, ax=cx, label="LuxI at 187 min")
        cx.text(0.5, -0.04, "Each dot a trap; lines: AHL exchange between neighbours",
                transform=cx.transAxes, ha="center", va="top", fontsize=SMALL)
    for a in (ax, ax_u):
        tidy(a)
    ax_u.set_ylabel("Inducer")
    ax_u.set_xlabel("Time (min)")
    bx.set_xlabel("Time (min)")
    bx.set_ylabel("Trap (sorted by its peak time)")
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
        fig.colorbar(image, ax=dx, label="Macro F1 (ridge, 2 images)")
        handles = []
        o_flows, o_densities, share, _ = oscillation_map(oscillation, d)
        if o_flows == flows and o_densities == densities:
            for i, j in zip(*np.nonzero(share < 50), strict=True):
                dx.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///",
                                       ec="white", lw=0))
            handles.append(Rectangle((0, 0), 1, 1, fc="#777777", hatch="///", ec="white",
                                     label="Fewer than half of the colonies oscillate"))
        point = design_point(r, d)
        if point is not None:
            dx.plot(np.interp(point[0], flows, range(len(flows))),
                    np.interp(point[1], densities, range(len(densities))), marker="*", ms=24,
                    color="#d62728", markeredgecolor="white", mew=1.2, ls="none")
            handles.append(Line2D([], [], marker="*", ms=16, color="#d62728", ls="none",
                                  label="Design"))
        dx.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2,
                  fontsize=SMALL, frameon=False)
        dx.set_xticks(range(len(flows)), [f"{f:g}" for f in flows])
        dx.set_yticks(range(len(densities)), [f"{x:g}" for x in densities])
        dx.set_xlabel("AHL washout rate μ (flow)")
        dx.set_ylabel("Cell density")
    for a, letter in ((ax, "A"), (bx, "B"), (cx, "C"), (dx, "D")):
        panel(a, letter)
    return fig


# ---------------------------------------------------------------------------------------
# Fig. 3: classification
# ---------------------------------------------------------------------------------------

def fold_projection(r: Results, d: str, seed: int = SEEDS[0], method: str = "lda") -> dict:
    """Held-out beats of fold 0 projected to two dimensions, for the raw beat and the
    array's features: linear discriminants (``lda``) or principal components (``pca``)
    fitted on the fold's training beats."""
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler

    from training.run import load_task
    from utils.preprocessing import standardize_data

    def project(train, y_train, test):
        scaler = StandardScaler().fit(train)
        fit = (LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto", n_components=2)
               if method == "lda" else PCA(n_components=2))
        fit.fit(scaler.transform(train), y_train)
        return fit.transform(scaler.transform(test))

    path = config.runs_dir() / f"P5-balanced-{d}-headline-s{seed}" / "fold-0.npz"
    cfg = r.settings(RAW["ridge"])
    if not path.is_file() or cfg is None:
        return {}
    with np.load(path, allow_pickle=True) as z:
        train = z["train_states"].reshape(len(z["train_states"]), -1)
        test = z["test_states"].reshape(len(z["test_states"]), -1)
        y_train = np.argmax(z["Y_train"].reshape(len(train), -1), axis=1)
        y_test = np.argmax(z["Y_test"].reshape(len(test), -1), axis=1)
        held_out = z["test_index"]
    X, Y, _ = load_task(cfg["task"])
    X, _ = standardize_data(X, X[:1], scaler_type=str(cfg["task"]["scaler"]))
    X = np.asarray(X, dtype=float).reshape(len(X), -1)
    labels = np.argmax(np.asarray(Y).reshape(len(Y), -1), axis=1)
    if not np.array_equal(labels[held_out], y_test):
        print("Warning: the raw pool's labels differ from the run's held-out labels.",
              file=sys.stderr)
    rest = np.setdiff1d(np.arange(len(X)), held_out)
    return {"raw": (project(X[rest], labels[rest], X[held_out]), labels[held_out]),
            "array": (project(train, y_train, test), y_test)}


def scatter_classes(ax, points, labels, axis_name):
    for k, name in enumerate(CLASS_NAMES):
        mine = labels == k
        ax.scatter(points[mine, 0], points[mine, 1], s=14, alpha=0.6, color=CLASS_COLOURS[name],
                   label=name, lw=0)
    ax.set_xlabel(f"{axis_name} 1")
    ax.set_ylabel(f"{axis_name} 2")
    tidy(ax)


def figure_classification(r: Results, d: str):
    fig, axes = plt.subplots(2, 2, figsize=(15, 14), gridspec_kw={"hspace": 0.42, "wspace": 0.12})
    for ax, (title, runs) in zip(axes[0], (
            ("Array, ridge on two images", [f"P5-balanced-{d}-headline-s{s}" for s in SEEDS]),
            ("Raw beat, ridge", [RAW["ridge"]])), strict=True):
        matrices = [m for m in (r.confusion(run) for run in runs) if m is not None]
        scores = [v for v in (r.f1(run) for run in runs) if v is not None]
        if matrices:
            plot_confusion_matrix(np.sum(matrices, axis=0), ax, colorbar=ax is axes[0, 1])
            ax.set_title(f"{title}\n(macro F1 {np.mean(scores):.3f})", fontsize=SMALL + 2)
    axes[0, 1].set_ylabel("")
    axes[0, 1].tick_params(labelleft=False)
    projection = fold_projection(r, d)
    for ax, key, title in ((axes[1, 0], "raw", "Raw beat (187 samples)"),
                           (axes[1, 1], "array", "Array (576 traps × 2 images)")):
        if key in projection:
            scatter_classes(ax, *projection[key], "Discriminant")
            ax.set_title(title, fontsize=SMALL + 2)
    axes[1, 1].set_ylabel("")
    handles, labels = axes[1, 1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=SMALL, frameon=False,
                   markerscale=2.5, bbox_to_anchor=(0.5, 0.0))
    for ax, letter in zip(axes.flat, "ABCD", strict=True):
        panel(ax, letter)
    fig.subplots_adjust(bottom=0.1)
    return fig


# ---------------------------------------------------------------------------------------
# Fig. 4: what a chip needs
# ---------------------------------------------------------------------------------------

def figure_requirements(r: Results, d: str, flat_share: Path | None):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11.5), gridspec_kw={"hspace": 0.38,
                                                                   "wspace": 0.5})
    ax = axes[0, 0]
    ax.axvspan(*BUILT, color="#dddddd", lw=0, label="Arrays built by Prindle et al.")
    ax.axvline(LARGEST, color="#999999", ls="--", lw=THIN, label="Their largest array")
    line(ax, traps(r, d), ARRAY, "Two images")
    line(ax, ideal_traps(r, d), "black", "Ideal reading (no imaging)", marker="^", style=":",
         band=False)
    headline = r.seeds(f"P5-balanced-{d}-headline")
    if headline:
        ax.plot(576, np.mean(headline), marker="*", ms=24, color="#d62728",
                markeredgecolor="white", mew=1.2, ls="none", label="Design (576)", zorder=4)
    ax.set_xscale("log")
    plain_log_ticks(ax.xaxis, [100, 1000, 10000])
    ax.set_xlabel("Traps in the array")
    ax.legend(loc="lower right", fontsize=SMALL)

    ax = axes[0, 1]
    for agg, label, colour, marker in ((ONE, "1 image (187 min)", OKABE_ITO[5], "v"),
                                       (HEADLINE, "2 images (design)", ARRAY, "o"),
                                       (FOREST, "4 images", OKABE_ITO[3], "D")):
        line(ax, scaled(r.sweep(f"P3-{d}-noise", "features.measurement_noise",
                                features__aggregation=agg)), colour, label, marker=marker)
    ax.axvline(1, color="#bbbbbb", ls="--", lw=THIN)
    ax.set_xlabel("Measurement noise (% of the signal)")
    ax.legend(loc="lower left", fontsize=SMALL)

    ax = axes[0, 2]
    table = r.grid("extra-576-connectivity", "reservoir.params.input_connectivity",
                   "reservoir.params.coupling") if d == "p576" else {}
    shares = sorted({k[0] for k in table})
    for coupling, label, colour, style, marker in (
            ("diffusion", "With AHL exchange (design)", ARRAY, "-", "o"),
            ("none", "Isolated traps", ISOLATED, "--", "s")):
        line(ax, {100 * s: list(table.get((s, coupling), {}).values()) for s in shares}, colour,
             label, style=style, marker=marker)
    for s in shares:
        on, off = table.get((s, "diffusion"), {}), table.get((s, "none"), {})
        gain = paired(on, off)
        if gain:
            middle = (np.mean(list(on.values())) + np.mean(list(off.values()))) / 2
            last = s == shares[-1]
            ax.text(100 * s / 1.12 if last else 100 * s * 1.12, middle,
                    f"+{np.mean(gain):.3f}", ha="right" if last else "left", va="center",
                    fontsize=SMALL, color=ARRAY)
    ax.set_xscale("log")
    ax.set_xlim(8, 125)
    ax.set_xticks([100 * s for s in shares], [f"{100 * s:g}" for s in shares])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Traps the inducer reaches (%)")
    ax.legend(loc="lower right", fontsize=SMALL)

    ax = axes[1, 0]
    for study, label, style in ((f"P3-{d}-calibration", "With the reference image", "-"),
                                (f"P3-{d}-calibration-noref", "Without", ":")):
        line(ax, scaled(r.sweep(study, "features.calibration_error",
                                features__aggregation=HEADLINE)), ARRAY, label, style=style)
    ax.set_xlabel("Calibration error (% SD of a trap's gain)")
    ax.legend(loc="lower left", fontsize=SMALL)

    ax = axes[1, 1]
    points = r.sweep(f"P3-{d}-heterogeneity", "reservoir.params.param_heterogeneity_cv")
    points.update(r.sweep(f"P3-{d}-heterogeneity-64", "reservoir.params.param_heterogeneity_cv"))
    line(ax, scaled(points), ARRAY, "Macro F1")
    ax.axvline(10, color="#bbbbbb", ls="--", lw=THIN)
    ax.set_xlabel("Variability of the colonies' kinetics (CV, %)")
    flat = flat_shares(flat_share, d)
    if flat:
        x, m, _ = ordered(scaled(flat))
        twin = ax.twinx()
        twin.plot(x, m, ":", marker="^", color="#555555", lw=LW, ms=8,
                  label="Colonies not oscillating")
        twin.set_ylabel("Not oscillating (%)", color="#555555")
        twin.set_ylim(0, 100)
        handles = [h for a in (ax, twin) for h in a.get_legend_handles_labels()[0]]
        ax.legend(handles=handles, loc="upper right", fontsize=SMALL)

    ax = axes[1, 2]
    for study, key, label, colour, style, marker in (
            (f"P3-{d}-density", "density", "Density, design", ARRAY, "-", "o"),
            (f"P3-{d}-dilution", "dilution", "Washout, design", ARRAY, ":", "D"),
            ("P3-p576r-density" if d == "p576" else None, "density",
             "Density, alternative b32", ALTERNATIVE, "-", "s")):
        if study is not None:
            line(ax, scaled(r.sweep(study, f"reservoir.params.heterogeneity_{key}_cv")), colour,
                 label, style=style, marker=marker)
    ax.set_xlabel("Spread between traps (CV, %)")
    ax.legend(loc="lower left", fontsize=SMALL)

    for ax, letter in zip(axes.flat, "ABCDEF", strict=True):
        histgb_line(ax, r)
        tidy(ax)
        ax.set_ylabel("Macro F1")
        panel(ax, letter)
    raw, histgb = r.f1(RAW["ridge"]), r.f1(RAW["histgb"])
    if histgb is not None and raw is not None:
        fig.legend(handles=[Line2D([], [], color="black", ls=":", lw=THIN,
                                   label=f"Raw beat, HistGB ({histgb:.3f})")],
                   loc="upper center", bbox_to_anchor=(0.5, 0.995), fontsize=SMALL,
                   frameon=False, title=f"Ridge on the array in every panel; raw-beat ridge "
                                        f"scores {raw:.3f} (below every axis)",
                   title_fontsize=SMALL)
    return fig


# ---------------------------------------------------------------------------------------
# SI figures
# ---------------------------------------------------------------------------------------

def figure_confusion_tasks(r: Results, d: str):
    """SI: the array's confusion matrices on the three tasks (ridge on two images)."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), gridspec_kw={"wspace": 0.45,
                                                                  "width_ratios": [1, 1, 0.6]})
    for ax, (title, runs) in zip(axes, (
            ("Balanced, 5 classes (5 seeds)",
             [f"P5-balanced-{d}-headline-s{s}" for s in SEEDS]),
            ("Unbalanced, 5 classes", [f"P5-unbalanced-{d}-headline-s{SEEDS[0]}"]),
            ("Binary", [f"P5-binary-{d}-headline-s{SEEDS[0]}"])), strict=True):
        matrices = [m for m in (r.confusion(run) for run in runs) if m is not None]
        if matrices:
            total = np.sum(matrices, axis=0)
            plot_confusion_matrix(total, ax, labels=class_names(len(total)))
            ax.set_title(title, fontsize=SMALL + 2)
    for ax, letter in zip(axes, "ABC", strict=True):
        panel(ax, letter)
    return fig


def figure_class_images(r: Results, d: str, seed: int = SEEDS[0]):
    """SI: each class's average image minus the average over all beats, log2 of the image
    over the reference, at 140 and 187 min (the held-out beats of every fold)."""
    folds = sorted((config.runs_dir() / f"P5-balanced-{d}-headline-s{seed}").glob("fold-*.npz"))
    if not folds:
        return None
    features, labels = [], []
    for path in folds:
        with np.load(path, allow_pickle=True) as z:
            test = z["test_states"].reshape(len(z["test_states"]), -1)
            features.append(test)
            labels.append(np.argmax(z["Y_test"].reshape(len(test), -1), axis=1))
    features = np.log2(np.clip(np.concatenate(features), 1e-9, None))
    labels = np.concatenate(labels)
    side = int(round(np.sqrt(features.shape[1] // 2)))
    overall = features.mean(axis=0)
    deviations = np.array([features[labels == k].mean(axis=0) - overall
                           for k in range(len(CLASS_NAMES))])
    span = np.percentile(np.abs(deviations), 99)
    fig, axes = plt.subplots(2, len(CLASS_NAMES), figsize=(17, 7.2),
                             gridspec_kw={"hspace": 0.08, "wspace": 0.08})
    for k, name in enumerate(CLASS_NAMES):
        for row, minute in enumerate((140, 187)):
            image = deviations[k, row * side * side:(row + 1) * side * side].reshape(side, side)
            shown = axes[row, k].imshow(image, cmap="RdBu_r", vmin=-span, vmax=span,
                                        interpolation="nearest")
            axes[row, k].set_xticks([])
            axes[row, k].set_yticks([])
            if row == 0:
                axes[row, k].set_title(name, color=CLASS_COLOURS[name], fontsize=SMALL + 2)
            if k == 0:
                axes[row, k].set_ylabel(f"{minute} min", fontsize=SMALL + 2)
    fig.colorbar(shown, ax=axes, shrink=0.8,
                 label="Class mean minus mean of all beats\n(log$_2$ of image over reference)")
    return fig


def figure_projections(r: Results, d: str):
    """SI: principal components of the held-out beats of fold 0, raw beat and array."""
    projection = fold_projection(r, d, method="pca")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"wspace": 0.3})
    for ax, key, title in ((axes[0], "raw", "Raw beat (187 samples)"),
                           (axes[1], "array", "Array (576 traps × 2 images)")):
        if key in projection:
            scatter_classes(ax, *projection[key], "Principal component")
            ax.set_title(title, fontsize=SMALL + 2)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=SMALL,
                   markerscale=2.5, frameon=False)
    for ax, letter in zip(axes, "AB", strict=True):
        panel(ax, letter)
    return fig


def figure_settings(r: Results, d: str):
    """SI: the other settings of the array, ridge on two images (mean ± SD over seeds)."""
    fig, axes = plt.subplots(2, 4, figsize=(21, 11), gridspec_kw={"hspace": 0.45,
                                                                 "wspace": 0.35})

    def categories(ax, points, order, labels):
        for i, key in enumerate(order):
            values = points.get(key, [])
            if values:
                ax.errorbar(i, np.mean(values), yerr=np.std(values, ddof=1) if len(values) > 1
                            else 0, fmt="o", color=ARRAY, ms=9, lw=THIN, capsize=6)
        ax.set_xticks(range(len(order)), labels)
        ax.set_xlim(-0.6, len(order) - 0.4)

    reset = r.sweep(f"P3-{d}-reset", "reservoir.params.reset_noise")
    order = sorted(k for k in reset if k is not None)
    categories(axes[0, 0], reset, order, [f"{100 * k:g}" for k in order])
    axes[0, 0].set_xlabel("Imperfect reset (CV of the start state, %)")

    warmup = r.sweep(f"P3-{d}-warmup", "reservoir.params.warmup")
    order = sorted(warmup)
    categories(axes[0, 1], warmup, order, [str(k) for k in order])
    axes[0, 1].set_xlabel("Warm-up before each beat (min)")
    axes[0, 1].tick_params(axis="x", labelrotation=45)

    rest = r.sweep(f"P3-{d}-warmup-input", "reservoir.params.warmup_input")
    categories(axes[0, 2], rest, ["zero", "rest"], ["None (design)", "Resting level"])
    axes[0, 2].set_xlabel("Inducer during the warm-up")

    levels = r.sweep(f"P3-{d}-input-levels", "reservoir.params.input_levels")
    order = sorted(levels, key=lambda k: (isinstance(k, str), k))
    categories(axes[0, 3], levels, order, [str(k) for k in order])
    axes[0, 3].set_xlabel("Inducer levels (design: continuous)")

    layout = r.sweep(f"P6-{d}-gaussian", "reservoir.params.interaction_diameters")
    order = sorted(layout)
    categories(axes[1, 0], layout, order, [f"{k:g}" for k in order])
    headline = r.seeds(f"P5-balanced-{d}-headline")
    if headline:
        axes[1, 0].axhline(np.mean(headline), color="#999999", ls="--", lw=THIN,
                           label=f"Lattice (design, {np.mean(headline):.3f})")
        axes[1, 0].legend(loc="lower right", fontsize=SMALL)
    axes[1, 0].set_xlabel("Gaussian layout: reach (trap spacings)")

    ax = axes[1, 1]
    rows = r.trials(f"P6-{d}-units-images")
    sizes = sorted({row["reservoir.params.units"] for row in rows})
    for agg, label, colour, marker, shift in ((ONE, "1 image", OKABE_ITO[5], "v", -0.15),
                                              (HEADLINE, "2 images", ARRAY, "o", 0.0),
                                              (FOREST, "4 images", OKABE_ITO[3], "D", 0.15)):
        points = r.sweep(f"P6-{d}-units-images", "reservoir.params.units",
                         features__aggregation=agg)
        x, m, s = ordered(points)
        ax.errorbar([sizes.index(v) + shift for v in x], m, yerr=s, fmt=marker + "-",
                    color=colour, lw=THIN, ms=8, capsize=4, label=label)
    ax.set_xticks(range(len(sizes)), [f"{v:,}" for v in sizes])
    ax.set_xlabel("Traps in the array (3 seeds)")
    ax.legend(loc="lower right", fontsize=SMALL)

    ax = axes[1, 2]
    times, final = image_times(r, d)
    order = sorted(times)
    categories(ax, times, order, [str(t) for t in order])
    if final:
        ax.axhline(np.mean(final), color=OKABE_ITO[5], ls="--", lw=THIN,
                   label=f"187-min image alone ({np.mean(final):.3f})")
        ax.legend(loc="lower right", fontsize=SMALL)
    ax.set_xlabel("Time of the earlier image (min; the later at 187)")
    axes[1, 3].axis("off")
    for ax, letter in zip(list(axes.flat)[:7], "ABCDEFG", strict=True):
        tidy(ax)
        ax.set_ylabel("Macro F1")
        panel(ax, letter)
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
                 "si_confusion_tasks": figure_confusion_tasks(r, d),
                 "si_class_images": figure_class_images(r, d),
                 "si_projections": figure_projections(r, d),
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
