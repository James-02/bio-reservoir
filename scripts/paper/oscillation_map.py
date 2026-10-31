"""Do the colonies oscillate, and how fast, over the flow x cell-density grid of the F1 map
(Fig. 2b and c)?

Each array is the design's (its gain, coupling rate, kinetic variability and size) at one AHL
washout rate (flow) and cell density, built and warmed up as a trial builds it, then run on at
the warm-up drive for 300 steps, as ``flat_share.py`` does. A colony oscillates when its LuxI
trace over the last 200 steps has max / min of at least 1.5; its period is the median spacing
of its LuxI peaks (one step is one minute).

    python scripts/paper/oscillation_map.py results/figures/paper/oscillation_map.csv \\
        --design p576 --processes N

Writes one CSV row per flow, density and seed; ``paper_figures.py --oscillation`` reads it.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

FLOWS = (0.3, 0.4, 0.6, 0.9, 1.2)
DENSITIES = (0.6, 0.7, 0.75, 0.8, 0.85)
SEEDS = (6348, 6349, 6350, 6351, 6352)
STEPS, WINDOW, FLAT_RATIO = 300, 200, 1.5


def one(job: tuple[str, float, float, int]) -> dict:
    import reservoirpy
    from scipy.signal import find_peaks

    from bioreservoir.diagnostics import STATE_NAMES
    from optimization.reservoirs import build_reservoir
    from training.run import load_config, resolve

    reservoirpy.verbosity(0)
    design, flow, density, seed = job
    cfg = resolve(load_config(Path("experiments/v0.2/designs") / f"{design}.yaml"))
    params = dict(cfg["reservoir"]["params"], washout_rate=flow, cell_density=density,
                  reservoir_type="bioreservoir")
    node, _ = build_reservoir(params, seed=seed)
    node.initialize(np.zeros((1, 1)))
    work = node.copy()
    work.states = work._warmup_states.copy()
    if getattr(work, "_warmup_ring", None) is not None:
        work._ring, work._ring_pos = work._warmup_ring[0].copy(), work._warmup_ring[1]
    index = STATE_NAMES.index("I")
    trace = np.array([work._step(work._warmup_drive)[:, index] for _ in range(STEPS)])[-WINDOW:]
    with np.errstate(all="ignore"):
        ratio = trace.max(axis=0) / trace.min(axis=0)
    oscillating = ratio >= FLAT_RATIO
    periods = []
    for col in np.flatnonzero(oscillating):
        x = trace[:, col]
        peaks, _ = find_peaks(x, prominence=0.3 * (x.max() - x.min()))
        if len(peaks) >= 2:
            periods.append(np.median(np.diff(peaks)))
    return {"design": design, "flow": flow, "density": density, "seed": seed,
            "units": trace.shape[1], "oscillating": float(np.mean(oscillating)),
            "period": float(np.median(periods)) if periods else float("nan"),
            "ratio": float(np.nanmedian(ratio))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--design", default="p576",
                        help="Design file in experiments/v0.2/designs/ (default: %(default)s).")
    parser.add_argument("--processes", type=int, default=1)
    args = parser.parse_args()
    jobs = [(args.design, f, d, s) for f in FLOWS for d in DENSITIES for s in SEEDS]
    with ProcessPoolExecutor(args.processes) as pool:
        rows = list(pool.map(one, jobs))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("density \\ flow  " + "  ".join(f"{f:>12g}" for f in FLOWS))
    for d in DENSITIES:
        cells = []
        for f in FLOWS:
            part = [r for r in rows if r["flow"] == f and r["density"] == d]
            cells.append(f"{np.mean([r['oscillating'] for r in part]):4.0%} "
                         f"{np.nanmedian([r['period'] for r in part]):5.0f} min")
        print(f"{d:<14g}  " + "  ".join(f"{c:>12}" for c in cells))


if __name__ == "__main__":
    main()
