#!/usr/bin/env python3
"""Drive and state ranges of a run config's reservoir over beats from its pool (B15).

Builds the reservoir of a run config, runs a few beats of each class from the config's
beat pool, and reports how often the drive x_t exceeds the bound stated in the model
description (5e-3 by default) and how low the states go.

Usage
-----
    python scripts/analysis/drive_bounds.py experiments/benchmarks/v0.1/balanced.yaml
    python scripts/analysis/drive_bounds.py CONFIG --beats 20 --bound 5e-3 --json out.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from bioreservoir.diagnostics import drive_and_state_ranges
from optimization.reservoirs import build_reservoir
from training import run


def beats_per_class(Y: np.ndarray, n: int) -> np.ndarray:
    """Indices of the first ``n`` beats of each class, in pool order."""
    labels = np.argmax(Y.reshape(len(Y), -1), axis=1)
    return np.sort(np.concatenate([np.flatnonzero(labels == c)[:n]
                                   for c in np.unique(labels)]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", help="Run config (YAML).")
    parser.add_argument("--beats", type=int, default=10, help="Beats per class.")
    parser.add_argument("--bound", type=float, default=5e-3, help="Stated bound on |x_t|.")
    parser.add_argument("--json", default=None, help="Write the summary to this file.")
    args = parser.parse_args()

    cfg = run.resolve(run.load_config(args.config))
    if cfg["reservoir"].get("type") != "bioreservoir":
        parser.error("The config's reservoir is not a BioReservoir.")
    X, Y, _ = run.load_task(cfg["task"])
    idx = beats_per_class(Y, args.beats)
    params = dict(cfg["reservoir"]["params"], reservoir_type="bioreservoir")
    node, _ = build_reservoir(params, seed=cfg["reservoir"]["seed"])
    summary = drive_and_state_ranges(node, X[idx], bound=args.bound)
    summary.update(config=args.config, beats=len(idx))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=1)
            f.write("\n")
    print(f"{cfg['name']}: {len(idx)} beats, {summary['oscillator_steps']} oscillator-steps")
    print(f"  |x_t| max {summary['drive_abs_max']:.3g}; above {args.bound:g} in "
          f"{100 * summary['drive_fraction_above']:.1f}% of oscillator-steps")
    for name in summary["state_min"]:
        print(f"  {name:2s} min {summary['state_min'][name]:+.3g}  max "
              f"{summary['state_max'][name]:.3g}  negative in "
              f"{100 * summary['state_negative_fraction'][name]:.1f}%")


if __name__ == "__main__":
    main()
