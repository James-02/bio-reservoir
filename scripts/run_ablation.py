#!/usr/bin/env python3
"""Run ablation studies with RF readout matching the headline results.

Parameters are hardcoded from the r1a-refined best trial (trial 416)
and ro1-readout-fixed best trial (trial 833, RandomForestClassifier).

Each condition runs 5-fold stratified cross-validation on the balanced
five-class ECG task (4015 instances) with a 1000-node distance-topology
reservoir and macro-F1 as the primary metric.

Usage
-----
    python scripts/run_ablation.py --condition baseline
    python scripts/run_ablation.py --condition no-coupling
    python scripts/run_ablation.py --condition no-delay
    python scripts/run_ablation.py --condition scramble

All four on one host (sequential):
    for c in baseline no-coupling no-delay scramble; do
        python scripts/run_ablation.py --condition $c --processes 32
    done
"""

from __future__ import annotations

import argparse
import logging
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
# Data / CV settings (matching rerun-best r1a-refined)
# ---------------------------------------------------------------------------
INSTANCES = 4015
FOLDS = 5
SCALER_TYPE = "sequence_zscore"
NOISE_RATE = 0.2999
NOISE_RATIO = 0.2862
READOUT_AGGREGATION = "signal_mean"
PRIMARY_AVERAGE = "macro"


def _build_reservoir(condition: str, seed: int = SEED) -> BioReservoir:
    """Build the reservoir with condition-specific overrides."""
    dde_kwargs: dict = {}
    rc_scaling = RESERVOIR_PARAMS["rc_scaling"]

    if condition == "no-coupling":
        rc_scaling = 0.0
    elif condition == "no-delay":
        dde_kwargs = {"delay": 0}

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
        rc_scaling=rc_scaling,
        input_scaling=RESERVOIR_PARAMS["input_scaling"],
        dde_scaling=RESERVOIR_PARAMS["dde_scaling"],
        cell_coupling=RESERVOIR_PARAMS["cell_coupling"],
        noise_in=RESERVOIR_PARAMS["noise_in"],
        noise_rc=RESERVOIR_PARAMS["noise_rc"],
        input_connectivity=RESERVOIR_PARAMS["input_connectivity"],
        output_variables=RESERVOIR_PARAMS["output_variables"],
        bias_scaling=RESERVOIR_PARAMS["bias_scaling"],
        input_bias=RESERVOIR_PARAMS["input_bias"],
        dde_kwargs=dde_kwargs,
        W=W,
        parallel=True,
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ablation studies with RF readout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--condition", required=True,
        choices=["baseline", "no-coupling", "no-delay", "scramble"],
        help="Ablation condition to run.",
    )
    parser.add_argument("--processes", type=int, default=5,
                        help="Parallel fold workers.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--noise_rate", type=float, default=None,
                        help="Override NOISE_RATE (default: module constant).")
    parser.add_argument("--noise_ratio", type=float, default=None,
                        help="Override NOISE_RATIO (default: module constant).")
    args = parser.parse_args()

    setup_logging(level=logging.INFO)
    log = logging.getLogger(__name__)
    rpy.set_seed(args.seed)
    rpy.verbosity(0)
    np.random.seed(args.seed)

    # Allow CLI overrides for augmentation values.
    noise_rate = args.noise_rate if args.noise_rate is not None else NOISE_RATE
    noise_ratio = args.noise_ratio if args.noise_ratio is not None else NOISE_RATIO

    condition = args.condition
    trial_name = f"ablation-{condition}-fixed"

    log.info("Ablation condition : %s", condition)
    log.info("Trial name         : %s", trial_name)
    log.info("Processes          : %d", args.processes)

    # ---- Load data ----------------------------------------------------------
    X_train, Y_train, X_test, Y_test = load_ecg_data(
        rows=INSTANCES,
        balance_classes=True,
        binary=False,
        scaler_type=SCALER_TYPE,
        seed=args.seed,
    )
    X = np.concatenate((X_train, X_test), axis=0)
    Y = np.concatenate((Y_train, Y_test), axis=0)

    if len(X) > INSTANCES:
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.permutation(len(X))[:INSTANCES])
        X, Y = X[idx], Y[idx]

    log.info("Dataset: %d instances, X shape %s", len(X), X.shape)

    # ---- Scramble: permute timesteps within each segment --------------------
    if condition == "scramble":
        log.info("Scrambling temporal order of each ECG segment")
        rng = np.random.default_rng(args.seed)
        for i in range(len(X)):
            perm = rng.permutation(X[i].shape[0])
            X[i] = X[i][perm]

    # ---- Build reservoir & readout ------------------------------------------
    reservoir = _build_reservoir(condition, seed=args.seed)
    readout = ScikitLearnNode(RandomForestClassifier, model_hypers=RF_HYPERS)

    log.info("Reservoir: %d nodes, output_dim=%d", RESERVOIR_PARAMS["units"],
             reservoir.output_dim)
    log.info("Readout  : RandomForestClassifier (%d estimators)", RF_HYPERS["n_estimators"])

    # ---- Cross-validate -----------------------------------------------------
    fold_backend = "loky" if args.processes > 1 else "sequential"
    fold_workers = args.processes if args.processes > 1 else None

    metrics = cross_validate(
        reservoir, readout, X, Y,
        folds=FOLDS,
        trial_name=trial_name,
        save_threshold=0.0,
        save_states=False,
        noise_rate=noise_rate,
        noise_ratio=noise_ratio,
        fold_backend=fold_backend,
        fold_workers=fold_workers,
        seed=args.seed,
        scaler_type=SCALER_TYPE,
        readout_aggregation=READOUT_AGGREGATION,
        readout_window=1,
        primary_average=PRIMARY_AVERAGE,
    )

    # ---- Report -------------------------------------------------------------
    log.info("=== %s results ===", condition)
    log.info("  F1       = %.4f", float(metrics.get("f1", 0)))
    log.info("  Accuracy = %.4f", float(metrics.get("accuracy", 0)))
    log.info("  MCC      = %.4f", float(metrics.get("mcc", 0)))
    log.info("  Kappa    = %.4f", float(metrics.get("kappa", 0)))
    log.info("Fold artefacts saved to results/runs/%s/", trial_name)


if __name__ == "__main__":
    main()
