#!/usr/bin/env python3
"""Run heterogeneity robustness study with RF readout and 5-fold CV.

Parameters are hardcoded from the r1a-refined best trial (trial 416)
and ro1-readout-fixed best trial (trial 833, RandomForestClassifier).

Each (CV, seed) grid point runs 5-fold stratified cross-validation on
the balanced five-class ECG task (4015 instances) with a 1000-node
distance-topology reservoir and macro-F1 as the primary metric.

Usage
-----
    # Single grid point:
    python scripts/run_heterogeneity.py --cv 0.1 --het_seed 100

    # Full grid (8 CV levels × 5 seeds = 40 points):
    python scripts/run_heterogeneity.py --all --processes 5

    # Resume (skip existing results):
    python scripts/run_heterogeneity.py --all --processes 5 --resume
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import sys

import numpy as np
import reservoirpy as rpy
from reservoirpy.nodes import ScikitLearnNode
from sklearn.ensemble import RandomForestClassifier

from bioreservoir import BioReservoir, distance_matrix
from training import cross_validate
from utils.logger import setup_logging
from utils.preprocessing import load_ecg_data

SEED = 6337

# ---------------------------------------------------------------------------
# r1a-refined best trial (trial 416) — resolved parameters
# ---------------------------------------------------------------------------
RESERVOIR_PARAMS = dict(
    units=1000,
    warmup=50,
    rk4_substeps=64,
    rc_scaling=0.007036187544739917,
    input_scaling=0.02769874909436638,
    dde_scaling=0.0006439480590749527,
    cell_coupling=12.74579065885262,
    noise_in=0.01442,
    noise_rc=0.2397,
    input_connectivity=0.5281956378922816,
    output_variables="IHe",
    bias_scaling=0.1,
    input_bias=False,
    # Topology
    interaction_diameters=8.2,
    fade_alpha=2.80278092396274e-05,
    sparsity=0.5759952026896107,
    p_excite=0.8354274565239865,
    signed=True,
)

# ---------------------------------------------------------------------------
# ro1-readout-fixed best trial (trial 833) — RF classifier
# ---------------------------------------------------------------------------
RF_HYPERS = dict(
    n_estimators=326,
    criterion="log_loss",
    oob_score=True,
    max_features="sqrt",
    max_depth=23,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=SEED,
)

# ---------------------------------------------------------------------------
# Data / CV settings
# ---------------------------------------------------------------------------
INSTANCES = 4015
FOLDS = 5
SCALER_TYPE = "sequence_zscore"
NOISE_RATE = 0.2999
NOISE_RATIO = 0.2862
READOUT_AGGREGATION = "signal_mean"
PRIMARY_AVERAGE = "macro"

# ---------------------------------------------------------------------------
# Heterogeneity grid
# ---------------------------------------------------------------------------
CV_LEVELS = [0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
HET_SEEDS = [100, 200, 300, 400, 500]


def _build_reservoir(
    het_cv: float,
    het_seed: int,
    seed: int = SEED,
) -> BioReservoir:
    """Build the reservoir with per-node kinetic heterogeneity."""
    W = distance_matrix(
        spatial_dim=2,
        interaction_diameters=RESERVOIR_PARAMS["interaction_diameters"],
        fade_alpha=RESERVOIR_PARAMS["fade_alpha"],
        sparsity=RESERVOIR_PARAMS["sparsity"],
        signed=RESERVOIR_PARAMS["signed"],
        p_excite=RESERVOIR_PARAMS["p_excite"],
        cutoff_radius="auto",
        seed=seed,
    )

    return BioReservoir(
        units=RESERVOIR_PARAMS["units"],
        warmup=RESERVOIR_PARAMS["warmup"],
        rk4_substeps=RESERVOIR_PARAMS["rk4_substeps"],
        rc_scaling=RESERVOIR_PARAMS["rc_scaling"],
        input_scaling=RESERVOIR_PARAMS["input_scaling"],
        dde_scaling=RESERVOIR_PARAMS["dde_scaling"],
        cell_coupling=RESERVOIR_PARAMS["cell_coupling"],
        noise_in=RESERVOIR_PARAMS["noise_in"],
        noise_rc=RESERVOIR_PARAMS["noise_rc"],
        input_connectivity=RESERVOIR_PARAMS["input_connectivity"],
        output_variables=RESERVOIR_PARAMS["output_variables"],
        bias_scaling=RESERVOIR_PARAMS["bias_scaling"],
        input_bias=RESERVOIR_PARAMS["input_bias"],
        param_heterogeneity_cv=het_cv,
        param_heterogeneity_seed=het_seed,
        W=W,
        parallel=True,
        seed=seed,
    )


def run_one(
    het_cv: float,
    het_seed: int,
    processes: int,
    seed: int = SEED,
) -> dict:
    """Run a single heterogeneity grid point with 5-fold CV."""
    log = logging.getLogger(__name__)
    trial_name = f"heterogeneity-cv{het_cv:.3f}-seed{het_seed}"

    log.info("=== Heterogeneity CV=%.4f, seed=%d ===", het_cv, het_seed)

    # ---- Load data ----------------------------------------------------------
    X_train, Y_train, X_test, Y_test = load_ecg_data(
        rows=INSTANCES,
        balance_classes=True,
        binary=False,
        scaler_type=SCALER_TYPE,
        seed=seed,
    )
    X = np.concatenate((X_train, X_test), axis=0)
    Y = np.concatenate((Y_train, Y_test), axis=0)

    if len(X) > INSTANCES:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.permutation(len(X))[:INSTANCES])
        X, Y = X[idx], Y[idx]

    log.info("Dataset: %d instances, X shape %s", len(X), X.shape)

    # ---- Build reservoir & readout ------------------------------------------
    reservoir = _build_reservoir(het_cv, het_seed, seed=seed)
    readout = ScikitLearnNode(RandomForestClassifier, model_hypers=RF_HYPERS)

    log.info("Reservoir: %d nodes, het_cv=%.4f, het_seed=%d",
             RESERVOIR_PARAMS["units"], het_cv, het_seed)
    log.info("Readout  : RandomForestClassifier (%d estimators)",
             RF_HYPERS["n_estimators"])

    # ---- Cross-validate -----------------------------------------------------
    fold_backend = "loky" if processes > 1 else "sequential"
    fold_workers = processes if processes > 1 else None

    metrics = cross_validate(
        reservoir, readout, X, Y,
        folds=FOLDS,
        trial_name=trial_name,
        save_threshold=0.0,
        save_states=False,
        noise_rate=NOISE_RATE,
        noise_ratio=NOISE_RATIO,
        fold_backend=fold_backend,
        fold_workers=fold_workers,
        seed=seed,
        scaler_type=SCALER_TYPE,
        readout_aggregation=READOUT_AGGREGATION,
        readout_window=1,
        primary_average=PRIMARY_AVERAGE,
    )

    f1 = float(metrics.get("f1", 0))
    log.info("  CV=%.4f seed=%d → F1=%.4f", het_cv, het_seed, f1)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run heterogeneity robustness study with RF readout (5-fold CV).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true",
                       help="Run full grid (8 CV × 5 seeds = 40 points).")
    group.add_argument("--cv", type=float,
                       help="Single CV level to run (requires --het_seed).")
    parser.add_argument("--het_seed", type=int, default=None,
                        help="Single heterogeneity seed (required with --cv).")
    parser.add_argument("--processes", type=int, default=5,
                        help="Parallel fold workers.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--resume", action="store_true",
                        help="Skip grid points with existing results.")
    args = parser.parse_args()

    setup_logging(level=logging.INFO)
    log = logging.getLogger(__name__)
    rpy.set_seed(args.seed)
    rpy.verbosity(0)
    np.random.seed(args.seed)

    if args.all:
        grid = list(itertools.product(CV_LEVELS, HET_SEEDS))
        log.info("Full heterogeneity grid: %d points", len(grid))

        for i, (cv_val, het_seed) in enumerate(grid, 1):
            trial_name = f"heterogeneity-cv{cv_val:.3f}-seed{het_seed}"
            results_dir = f"results/runs/{trial_name}"

            if args.resume and os.path.isdir(results_dir):
                npz_files = [f for f in os.listdir(results_dir) if f.endswith(".npz")]
                if len(npz_files) >= FOLDS:
                    log.info("[%d/%d] SKIP %s (already done)", i, len(grid), trial_name)
                    continue

            log.info("[%d/%d] Running %s", i, len(grid), trial_name)
            run_one(cv_val, het_seed, args.processes, seed=args.seed)
    else:
        if args.het_seed is None:
            parser.error("--het_seed is required when using --cv")
        run_one(args.cv, args.het_seed, args.processes, seed=args.seed)

    log.info("=== Heterogeneity study complete ===")


if __name__ == "__main__":
    main()
