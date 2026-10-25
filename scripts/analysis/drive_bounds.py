#!/usr/bin/env python3
"""Drive and state ranges of a run config's reservoir over beats from its pool.

Builds the reservoir of a run config, runs a few beats of each class from the config's
beat pool, and reports how often the drive x_t exceeds the bound stated in the model
description (5e-3 by default), how low the states go, and the share of the drive that
comes from the input and from each coupled variable (abstract coupling only).
``--oscillation STEPS`` also continues the warm-up without input for STEPS steps and
reports how many colonies still oscillate (``bioreservoir.diagnostics.oscillation_amplitudes``).
``--set PATH=VALUE`` changes a config value first, at a dotted path as in a study search
(``reservoir.params.heterogeneity_density_cv=0.3``); a study config gives its base values.

Usage
-----
    python scripts/analysis/drive_bounds.py experiments/benchmarks/v0.1/balanced.yaml
    python scripts/analysis/drive_bounds.py CONFIG --beats 20 --bound 5e-3 --json out.json
    python scripts/analysis/drive_bounds.py CONFIG --oscillation 300 \\
        --set reservoir.params.reset_noise=0.3 --set reservoir.seed=6338
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import yaml

from bioreservoir.diagnostics import drive_and_state_ranges, drive_shares, oscillation_amplitudes
from optimization.reservoirs import build_reservoir
from training import run
from training.study import apply

# A colony oscillates if its relative peak-to-peak amplitude, (max - min) / mean, is above this.
OSCILLATING = 0.1


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
    parser.add_argument("--set", action="append", default=[], metavar="PATH=VALUE",
                        help="Set a config value at a dotted path (VALUE is read as YAML).")
    parser.add_argument("--oscillation", type=int, default=0, metavar="STEPS",
                        help="Also report the oscillation of each colony over STEPS steps "
                             "without input after the warm-up.")
    args = parser.parse_args()

    values = {}
    for item in args.set:
        path, sep, value = item.partition("=")
        if not sep:
            parser.error(f"--set needs PATH=VALUE, got {item!r}.")
        values[path] = yaml.safe_load(value)
    cfg = run.resolve(apply(run.load_config(args.config), values))
    if cfg["reservoir"].get("type") != "bioreservoir":
        parser.error("The config's reservoir is not a BioReservoir.")
    X, Y, _ = run.load_task(cfg["task"])
    idx = beats_per_class(Y, args.beats)
    params = dict(cfg["reservoir"]["params"], reservoir_type="bioreservoir")
    node, _ = build_reservoir(params, seed=cfg["reservoir"]["seed"])
    summary = drive_and_state_ranges(node, X[idx], bound=args.bound)
    summary.update(config=args.config, settings=values, beats=len(idx))
    if node.coupling == "abstract":
        summary["drive_shares"] = drive_shares(node, X[idx])
    if args.oscillation:
        amplitude = oscillation_amplitudes(node, args.oscillation)
        finite = amplitude[np.isfinite(amplitude)]
        summary["oscillation"] = {
            "steps": args.oscillation, "colonies": int(len(amplitude)),
            "not_finite": int(len(amplitude) - len(finite)),
            "fraction_oscillating": float(np.mean(finite > OSCILLATING)) if len(finite) else 0.0,
            "amplitude_quantiles": dict(zip(("min", "q10", "median", "q90", "max"),
                                            np.quantile(finite, [0, 0.1, 0.5, 0.9, 1]).tolist()
                                            if len(finite) else [np.nan] * 5, strict=True))}
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=1)
            f.write("\n")
    settings = " ".join(f"{k}={v}" for k, v in values.items())
    print(f"{cfg['name']} {settings}: {len(idx)} beats, "
          f"{summary['oscillator_steps']} oscillator-steps")
    print(f"  |x_t| max {summary['drive_abs_max']:.3g}; above {args.bound:g} in "
          f"{100 * summary['drive_fraction_above']:.1f}% of oscillator-steps")
    for name in summary["state_min"]:
        print(f"  {name:2s} min {summary['state_min'][name]:+.3g}  max "
              f"{summary['state_max'][name]:.3g}  negative in "
              f"{100 * summary['state_negative_fraction'][name]:.1f}%")
    if "drive_shares" in summary:
        print("  drive shares: " + ", ".join(f"{name} {share:.3g}"
                                            for name, share in summary["drive_shares"].items()))
    if "oscillation" in summary:
        osc = summary["oscillation"]
        q = osc["amplitude_quantiles"]
        print(f"  oscillation over {osc['steps']} steps without input: "
              f"{100 * osc['fraction_oscillating']:.1f}% of {osc['colonies']} colonies above "
              f"{OSCILLATING:g} ({osc['not_finite']} not finite); relative amplitude min "
              f"{q['min']:.3g}, q10 {q['q10']:.3g}, median {q['median']:.3g}, q90 "
              f"{q['q90']:.3g}, max {q['max']:.3g}")


if __name__ == "__main__":
    main()
