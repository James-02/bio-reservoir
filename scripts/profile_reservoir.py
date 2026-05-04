"""Profile running the BioReservoir on a range of ECG training instances.

This is intended as a lightweight profiling harness (similar to `test.py`) that:
  1) loads the ECG dataset,
  2) builds a `BioReservoir`,
  3) runs `.run(...)` repeatedly over a configurable number of training instances,
  4) prints a cProfile summary sorted by cumulative time.

Notes:
- This script does not save plots by default (profiling runs should stay lightweight).
- Use `--instances` to control how many sequences are run through the reservoir.
"""

import argparse
import cProfile
import pstats
import logging

import reservoirpy as rpy

from utils.logger import setup_logging
from utils.preprocessing import load_ecg_data
from bioreservoir import BioReservoir
from bioreservoir import distance_matrix


def build_reservoir(
    seed: int,
    nodes: int,
    warmup: int,
    rk4_substeps: int,
    sr: float = None,
) -> BioReservoir:
    return BioReservoir(
        units=nodes,
        warmup=warmup,
        input_scaling=1e-2,
        rc_scaling=5e-4,
        rk4_substeps=rk4_substeps,
        cell_coupling=0.7,
        noise_rc=0.1,
        noise_in=0.1,
        input_bias=False,
        W=distance_matrix(
            spatial_dim=2,
            interaction_diameters=8.0,
            fade_alpha=0.01,
            decay_mode="exp",
            decay_power=2.0,
            include_self=False,
            normalize="spectral",
            sr=sr,
            sparsity=0.3,
            seed=seed,
            p_excite=0.6,
            signed=True,
            heterogeneous_gain=True,
            gain=1.0,
        ),
        parallel=True,
        seed=seed,
    )


def run_profile(
    X_train,
    reservoir: BioReservoir,
    n_sequences: int,
    reset_each: bool,
) -> None:
    # Ensure we don't exceed available data
    n = min(int(n_sequences), len(X_train))
    for i in range(n):
        reservoir.run(X_train[i], reset=reset_each)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_rows", type=int, default=20000, help="Training rows cap passed to load_ecg_data (training-only cap).")
    parser.add_argument("--max_per_class", type=int, default=3000)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--binary", action="store_true")
    parser.add_argument("--noise_rate", type=float, default=0.0)
    parser.add_argument("--noise_ratio", type=float, default=0.0)
    parser.add_argument("--scaler_type", type=str, default="sequence_zscore", choices=["sequence_zscore", "minmax", "standard", "none"])
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--nodes", type=int, default=250)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--rk4_substeps", type=int, default=12)
    parser.add_argument(
        "--sr",
        type=float,
        default=None,
        help="Optional precomputed spectral radius used to normalize W (skips eigen computation).",
    )

    parser.add_argument("--instances", type=int, default=50, help="How many training sequences to run through the reservoir.")
    parser.add_argument("--reset_each", action="store_true", help="Reset reservoir before each sequence.")

    parser.add_argument("--profile_sort", type=str, default="cumtime", choices=["cumtime", "tottime"], help="pstats sort key.")
    parser.add_argument("--top", type=int, default=40, help="Number of lines of profile output to print.")
    args = parser.parse_args()

    setup_logging(level=logging.INFO)
    rpy.verbosity(1)

    X_train, _, _, _ = load_ecg_data(
        rows=args.train_rows,
        max_per_class=args.max_per_class,
        balance_classes=True,
        test_ratio=args.test_ratio,
        encode_labels=True,
        binary=args.binary,
        noise_rate=args.noise_rate,
        noise_ratio=args.noise_ratio,
        standardize=True,
        scaler_type=args.scaler_type,
        seed=args.seed,
    )

    reservoir = build_reservoir(
        seed=args.seed,
        nodes=args.nodes,
        warmup=args.warmup,
        rk4_substeps=args.rk4_substeps,
        sr=args.sr,
    )

    profiler = cProfile.Profile()
    profiler.enable()
    run_profile(X_train, reservoir, n_sequences=args.instances, reset_each=args.reset_each)
    profiler.disable()

    stats = pstats.Stats(profiler).sort_stats(args.profile_sort)
    stats.print_stats(args.top)


if __name__ == "__main__":
    main()
