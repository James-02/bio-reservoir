"""Convergence of the colony solver at the paper's settings (SI figure).

A. One colony with the design's flow and cell density (no variability), driven from its
   basal state by a constant inducer for 200 min: the largest error in LuxI over the run,
   relative to its range, against a method-of-steps reference (SciPy DOP853, tolerance
   1e-12).
B. The design itself (576 traps, coupled, kinetic variability, warm-up) and one beat: the
   error of what the readout receives, each trap's LuxI at 140 and 187 min over its
   reference image, against the same run at 128 substeps; the measurement noise (1%) for
   scale.

    python scripts/paper/convergence.py results/figures/paper --formats png,pdf

Writes ``convergence.csv`` and ``si_convergence`` to the folder, and prints the errors. Run
from the repository root (B reads ``experiments/paper/models/array.yaml``).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402

from utils.visualisation import OKABE_ITO, STYLE, add_formats_option, save_figure  # noqa: E402

SUBSTEPS = (8, 16, 32, 64, 128, 256)  # fewer than 8 are unstable at the design's density
ARRAY_SUBSTEPS, ARRAY_REFERENCE = (16, 32, 64, 128), 256  # the coupled array needs >= 16
PRODUCTION = 32
STEPS, INDUCER, DELAY = 200, 0.02, 10.0
LW, THIN, SMALL = 3.2, 2.7, 13


def rhs(y, delayed_hi, u, p):
    """The colony equations as implemented (Danino et al. 2010; the inducer enters with
    the influx of AHL)."""
    A, I, Hi, He = y  # noqa: E741 (LuxI, as in the equations)
    promoter = (p["del_"] + p["alpha"] * delayed_hi ** 2) / (1.0 + p["k1"] * delayed_hi ** 2)
    slow = 1.0 + p["f"] * (A + I)
    return np.array([
        p["CA"] * p["decay"] * promoter - p["gammaA"] * A / slow,
        p["CI"] * p["decay"] * promoter - p["gammaI"] * I / slow,
        p["b"] * I / (1.0 + p["k"] * I) - p["gammaH"] * A * Hi / (1.0 + p["g"] * A)
        + p["D"] * (He + u - Hi),
        -p["d"] / (1.0 - p["d"]) * p["D"] * (He - Hi) - p["mu"] * He,
    ])


def reference(y0, hi_history, inputs, p):
    """LuxI at the end of each input step, by the method of steps with DOP853; the history
    of Hi before t = 0 is linear between its stored points."""
    t_hist = np.arange(-(len(hi_history) - 1), 1.0)
    segments, luxi = [], []

    def delayed(t):
        s = t - p["delay"]
        if s <= 0:
            return np.interp(s, t_hist, hi_history)
        for a, b, sol in segments:
            if a <= s <= b:
                return sol.sol(s)[2]
        raise AssertionError(f"no solution stored at t = {s}")

    y = np.asarray(y0, dtype=float)
    for step, u in enumerate(inputs):
        sol = solve_ivp(lambda t, y, u=u: rhs(y, delayed(t), u, p), (step, step + 1), y,
                        method="DOP853", rtol=1e-12, atol=1e-14, dense_output=True)
        segments.append((step, step + 1, sol))
        y = sol.y[:, -1]
        luxi.append(y[1])
    return np.array(luxi)


def colony_errors(flow: float, density: float) -> list[dict]:
    from bioreservoir import BioReservoir
    from bioreservoir.dde import get_dde_params

    p = get_dde_params(delay=DELAY, mu=flow, d=density)
    inputs = np.full(STEPS, INDUCER)
    rows, expected = [], None
    for substeps in SUBSTEPS:
        node = BioReservoir(units=1, warmup=0, rk4_substeps=substeps, seed=3, input_dim=1,
                            output_variables="He", parallel=False, cell_coupling=1.0,
                            dde_kwargs={"delay": DELAY, "mu": flow, "d": density})
        y0, history = node.states[-1, 0].copy(), node.states[:, 0, 2].copy()
        luxi = np.array([node._step(np.full(1, u))[0, 1] for u in inputs])
        if expected is None:
            expected = reference(y0, history, inputs, p)
        error = np.abs(luxi - expected).max() / np.ptp(expected)
        rows.append({"part": "colony", "substeps": substeps, "error": error,
                     "error_max": error})
    return rows


def array_errors(design: Path) -> list[dict]:
    from training.figures import VARIABLES, array_states, build, run_config
    from training.run import load_task
    from utils.preprocessing import standardize_data

    cfg = run_config(design)
    X, Y, _ = load_task(cfg["task"])
    X, _ = standardize_data(X, X[:1], scaler_type=str(cfg["task"]["scaler"]))
    X = np.asarray(X, dtype=float).reshape(len(X), -1)
    labels = np.argmax(np.asarray(Y).reshape(len(Y), -1), axis=1)
    members = np.flatnonzero(labels == 0)
    index = int(members[np.argmin(((X[members] - X[members].mean(axis=0)) ** 2).sum(1))])
    beat = X[index].reshape(-1, 1)
    lux = VARIABLES.index("I")

    def ratios(substeps):
        cfg["reservoir"]["params"]["rk4_substeps"] = substeps
        start = np.asarray(build(cfg).start_state(input_dim=1)).reshape(-1, 4)[:, lux]
        states = array_states(cfg, beat, index)[0][:, :, lux]
        return np.concatenate([states[139] / start, states[186] / start])

    expected = ratios(ARRAY_REFERENCE)
    rows = []
    for substeps in ARRAY_SUBSTEPS:
        error = np.abs(ratios(substeps) / expected - 1.0)
        rows.append({"part": "array", "substeps": substeps,
                     "error": float(np.median(error)),
                     "error_max": float(np.percentile(error, 90))})
    return rows


def figure(rows: list[dict]):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"wspace": 0.3})
    ax = axes[0]
    part = [r for r in rows if r["part"] == "colony"]
    n = np.array([r["substeps"] for r in part], dtype=float)
    ax.loglog(n, [r["error"] for r in part], marker="o", color=OKABE_ITO[0], lw=LW, ms=8,
              label="Solver")
    if part:
        ax.loglog(n, part[-1]["error"] * (n[-1] / n) ** 2, color="#999999", ls=":", lw=THIN,
                  label="Second order")
    ax.set_xlabel("Substeps per minute")
    ax.set_ylabel("Error in LuxI (fraction of its range)")
    ax.legend(loc="lower left", fontsize=SMALL)

    ax = axes[1]
    part = [r for r in rows if r["part"] == "array"]
    ax.loglog([r["substeps"] for r in part], [r["error"] for r in part], marker="o",
              color=OKABE_ITO[0], lw=LW, ms=8, label="Median trap")
    ax.loglog([r["substeps"] for r in part], [r["error_max"] for r in part], marker="D",
              ls=":", color=OKABE_ITO[3], lw=LW, ms=8, label="90th percentile")
    ax.axhline(0.01, color="black", ls="--", lw=THIN, label="Measurement noise (1%)")
    ax.set_xlabel("Substeps per minute")
    ax.set_ylabel("Relative error of the image ratios")
    ax.legend(loc="lower left", fontsize=SMALL)
    for ax, letter in zip(axes, "AB", strict=True):
        ax.axvline(PRODUCTION, color="#bbbbbb", lw=THIN)
        ax.set_xticks(SUBSTEPS, [str(n) for n in SUBSTEPS])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.grid(True, alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(letter, loc="left", fontweight="bold", fontsize=20)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=Path, help="Output folder.")
    add_formats_option(parser)
    args = parser.parse_args()
    from training.run import load_config, resolve

    design = Path("experiments/paper/models/array.yaml")
    params = resolve(load_config(design))["reservoir"]["params"]
    rows = colony_errors(float(params["washout_rate"]), float(params["cell_density"]))
    rows += array_errors(design)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "convergence.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['part']:7} {row['substeps']:4d}  {row['error']:.2e}  "
              f"(largest {row['error_max']:.2e})")
    with plt.style.context(STYLE):
        for path in save_figure(figure(rows), "si_convergence", folder=args.out,
                                formats=args.formats):
            print(path)


if __name__ == "__main__":
    main()
