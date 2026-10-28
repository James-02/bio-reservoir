"""Share of colonies that do not oscillate, for each kinetic variability (CV) of the
robustness grid (P3-<design>-heterogeneity and -64), each design and reservoir seed.

Each array is built and warmed up as a trial builds it, then run on without input for 300
steps. A colony is flat when its LuxI trace over the last 200 of those steps has max / min
below 1.5 (the "explainer" definition of investigation I14), or, by the stricter definition,
when (max - min) / mean over the 300 steps is at most 0.1
(:func:`bioreservoir.diagnostics.oscillation_amplitudes`). CV 0.3 and above use 64 substeps,
as the -64 studies do.

    python scripts/paper/flat_share.py results/figures/paper/flat_share.csv --processes N

Writes one CSV row per design, CV and seed; ``paper_figures.py`` reads it for Fig. 5d.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

CVS = (0.0, 0.01, 0.018675014409681404, 0.03, 0.05, 0.065, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
SEEDS = (6348, 6349, 6350, 6351, 6352)
DESIGNS = ("p1024", "p1600")
STEPS, WINDOW, FLAT_RATIO, FLAT_REL = 300, 200, 1.5, 0.1


def one(job: tuple[str, float, int]) -> dict:
    import reservoirpy

    from bioreservoir.diagnostics import STATE_NAMES, oscillation_amplitudes
    from optimization.reservoirs import build_reservoir
    from training.run import load_config, resolve

    reservoirpy.verbosity(0)
    design, cv, seed = job
    cfg = resolve(load_config(Path("experiments/v0.2/designs") / f"{design}.yaml"))
    params = dict(cfg["reservoir"]["params"], param_heterogeneity_cv=cv,
                  rk4_substeps=64 if cv >= 0.3 else 32, reservoir_type="bioreservoir")
    node, _ = build_reservoir(params, seed=seed)
    node.initialize(np.zeros((1, 1)))
    relative = oscillation_amplitudes(node, steps=STEPS)
    work = node.copy()
    work.states = work._warmup_states.copy()
    if getattr(work, "_warmup_ring", None) is not None:
        work._ring, work._ring_pos = work._warmup_ring[0].copy(), work._warmup_ring[1]
    zero = np.zeros(work.states.shape[1])
    index = STATE_NAMES.index("I")
    trace = np.array([work._step(zero)[:, index] for _ in range(STEPS)])[-WINDOW:]
    with np.errstate(all="ignore"):
        ratio = trace.max(axis=0) / trace.min(axis=0)
    return {"design": design, "cv": cv, "seed": seed, "units": len(relative),
            "flat": float(np.mean(~(ratio >= FLAT_RATIO))),
            "flat_strict": float(np.mean(~(relative > FLAT_REL)))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--processes", type=int, default=1)
    args = parser.parse_args()
    jobs = [(d, cv, s) for d in DESIGNS for cv in CVS for s in SEEDS]
    with ProcessPoolExecutor(args.processes) as pool:
        rows = list(pool.map(one, jobs))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for d in DESIGNS:
        for cv in CVS:
            part = [r for r in rows if r["design"] == d and r["cv"] == cv]
            print(f"{d} CV {cv:<6.3g} flat {np.mean([r['flat'] for r in part]):6.1%}  "
                  f"strict {np.mean([r['flat_strict'] for r in part]):6.1%}")


if __name__ == "__main__":
    main()
