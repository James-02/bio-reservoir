"""Run an ECG classification job with a BioReservoir.

All hyperparameters are exposed as CLI flags so that the script can be
driven from a shell, a SLURM job array, or a notebook without touching
source code.

Examples
--------
Quick smoke-test (10 nodes, 20 instances, 2 folds):

    python scripts/classify.py --nodes 10 --rows 20 --folds 2 \\
        --fold_workers 1 --trial_name test-run

Full binary classification with KNN readout:

    python scripts/classify.py \\
        --nodes 1000 --rows 50000 --binary \\
        --readout knn --knn_k 2 --knn_p 1.4 \\
        --folds 5 --fold_workers 5 \\
        --trial_name binary-knn

Ridge readout (fast, good baseline):

    python scripts/classify.py \\
        --nodes 500 --rows 20000 \\
        --readout ridge --ridge_regularization 1e-5 \\
        --folds 5 --trial_name ridge-run
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import reservoirpy as rpy
from reservoirpy.nodes import Ridge, ScikitLearnNode
from sklearn.neighbors import KNeighborsClassifier as KNN

from bioreservoir import BioReservoir, distance_matrix
from training import classify, cross_validate
from utils.logger import setup_logging
from utils.preprocessing import load_ecg_data


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BioReservoir ECG classification runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Data ---------------------------------------------------------------
    data = p.add_argument_group("data")
    data.add_argument("--rows", type=int, default=50_000,
                      help="Total rows loaded (training + test combined).")
    data.add_argument("--max_per_class", type=int, default=None,
                      help="Cap per class after loading (None = no cap).")
    data.add_argument("--test_ratio", type=float, default=0.2,
                      help="Fraction of data held out as test split.")
    data.add_argument("--binary", action="store_true",
                      help="Collapse to binary (Normal vs. Abnormal).")
    data.add_argument("--scaler_type", type=str, default="sequence_zscore",
                      choices=["sequence_zscore", "minmax", "standard", "none"],
                      help="Per-sequence normalisation applied before CV.")
    data.add_argument("--preserve_split", action="store_true",
                      help="Keep the original Kachuee train/test CSV split intact instead of pooling and resplitting.")

    # ---- Reservoir ----------------------------------------------------------
    res = p.add_argument_group("reservoir")
    res.add_argument("--nodes", type=int, default=1000,
                     help="Number of reservoir units.")
    res.add_argument("--warmup", type=int, default=100,
                     help="Warmup steps (input fed but states discarded).")
    res.add_argument("--rk4_substeps", type=int, default=32,
                     help="RK4 substeps per ECG timestep.")
    res.add_argument("--input_scaling", type=float, default=1e-3,
                     help="Input weight scaling factor.")
    res.add_argument("--rc_scaling", type=float, default=1e-3,
                     help="Recurrent weight scaling factor.")
    res.add_argument("--dde_scaling", type=float, default=5e-3,
                     help="DDE coupling scaling factor.")
    res.add_argument("--noise_in", type=float, default=0.05,
                     help="Input noise standard deviation.")
    res.add_argument("--noise_rc", type=float, default=0.15,
                     help="Recurrent noise standard deviation.")
    res.add_argument("--rk4_substep_aggregation_window", type=int, default=1,
                     help="Sub-step aggregation window for state extraction.")

    # ---- Topology -----------------------------------------------------------
    topo = p.add_argument_group("topology")
    topo.add_argument("--spatial_dim", type=int, default=2,
                      help="Spatial dimension for distance matrix (1, 2, or 3).")
    topo.add_argument("--interaction_diameters", type=float, default=8.2,
                      help="Coupling radius in cell-diameter units.")
    topo.add_argument("--fade_alpha", type=float, default=0.001,
                      help="Weight at the coupling radius boundary.")
    topo.add_argument("--sparsity", type=float, default=0.75,
                      help="Fraction of connections set to zero.")
    topo.add_argument("--cutoff_radius", type=str, default="auto",
                      help="Hard cutoff radius ('auto' or a float).")

    # ---- Readout ------------------------------------------------------------
    rdout = p.add_argument_group("readout")
    rdout.add_argument("--readout", type=str, default="knn",
                       choices=["knn", "ridge"],
                       help="Readout model type.")
    # KNN
    rdout.add_argument("--knn_k", type=int, default=2,
                       help="[KNN] Number of neighbours.")
    rdout.add_argument("--knn_weights", type=str, default="distance",
                       choices=["uniform", "distance"],
                       help="[KNN] Weight function.")
    rdout.add_argument("--knn_algorithm", type=str, default="auto",
                       choices=["auto", "ball_tree", "kd_tree", "brute"],
                       help="[KNN] Neighbour-finding algorithm.")
    rdout.add_argument("--knn_leaf_size", type=int, default=34,
                       help="[KNN] Leaf size for tree algorithms.")
    rdout.add_argument("--knn_p", type=float, default=1.4,
                       help="[KNN] Minkowski distance power (1=Manhattan, 2=Euclidean).")
    # Ridge
    rdout.add_argument("--ridge_regularization", type=float, default=1e-5,
                       help="[Ridge] L2 regularisation strength.")

    # ---- Cross-validation ---------------------------------------------------
    cv = p.add_argument_group("cross-validation")
    cv.add_argument("--folds", type=int, default=5,
                    help="Number of CV folds.")
    cv.add_argument("--noise_rate", type=float, default=0.1,
                    help="Noise augmentation rate applied to training split.")
    cv.add_argument("--noise_ratio", type=float, default=0.1,
                    help="Noise augmentation amplitude scale.")
    cv.add_argument("--fold_backend", type=str, default="loky",
                    choices=["loky", "sequential"],
                    help="Joblib backend for fold execution.")
    cv.add_argument("--fold_workers", type=int, default=5,
                    help="Number of parallel fold workers (1 = sequential).")
    cv.add_argument("--profile", action="store_true",
                    help="Collect and print per-section timing information.")
    cv.add_argument("--primary_average", type=str, default="weighted",
                    choices=["weighted", "macro"],
                    help="Primary average used for f1/precision/recall reporting.")

    # ---- Output -------------------------------------------------------------
    out = p.add_argument_group("output")
    out.add_argument("--trial_name", type=str, default=None,
                     help="Trial name for saving fold artefacts to results/runs/<trial_name>/. None = don't save.")
    out.add_argument("--save_threshold", type=float, default=None,
                     help="Min F1 required to save fold artefacts (None = always save when trial_name is set).")
    out.add_argument("--save_states", action="store_true",
                     help="Save reservoir states in fold artefacts. Required for readout re-optimisation "
                          "and state visualisations (PCA / t-SNE).  Disabled by default — states are "
                          "large arrays (~N_samples × N_units) and rarely needed at inference time.")

    # ---- Misc ---------------------------------------------------------------
    misc = p.add_argument_group("misc")
    misc.add_argument("--seed", type=int, default=6337)
    misc.add_argument("--log_level", type=str, default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    misc.add_argument("--verbosity", type=int, default=0, choices=[0, 1, 2],
                      help="ReservoirPy verbosity.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    setup_logging(level=getattr(logging, args.log_level))
    log = logging.getLogger(__name__)
    rpy.set_seed(args.seed)
    rpy.verbosity(args.verbosity)
    np.random.seed(args.seed)

    # ---- Load data ----------------------------------------------------------
    X_train, Y_train, X_test, Y_test = load_ecg_data(
        rows=args.rows,
        max_per_class=args.max_per_class,
        test_ratio=args.test_ratio,
        encode_labels=True,
        scaler_type=args.scaler_type,
        binary=args.binary,
        # Augmentation is applied per-fold in CV mode; for preserved splits we
        # apply it in the loader to the training partition only.
        noise_rate=args.noise_rate if args.preserve_split else 0,
        noise_ratio=args.noise_ratio if args.preserve_split else 0,
        preserve_split=args.preserve_split,
    )

    # ---- Build reservoir ----------------------------------------------------
    cutoff: float | str = args.cutoff_radius
    try:
        cutoff = float(cutoff)
    except (ValueError, TypeError):
        pass  # keep as string "auto"

    reservoir = BioReservoir(
        units=args.nodes,
        warmup=args.warmup,
        rc_scaling=args.rc_scaling,
        input_scaling=args.input_scaling,
        dde_scaling=args.dde_scaling,
        noise_in=args.noise_in,
        noise_rc=args.noise_rc,
        rk4_substeps=args.rk4_substeps,
        input_connectivity=1.0,
        rk4_substep_aggregation_window=args.rk4_substep_aggregation_window,
        W=distance_matrix(
            spatial_dim=args.spatial_dim,
            interaction_diameters=args.interaction_diameters,
            fade_alpha=args.fade_alpha,
            sparsity=args.sparsity,
            cutoff_radius=cutoff,
            seed=args.seed,
        ),
        parallel=True,
        seed=args.seed,
    )

    # ---- Build readout ------------------------------------------------------
    if args.readout == "knn":
        readout = ScikitLearnNode(KNN, model_hypers={
            "n_neighbors": args.knn_k,
            "weights": args.knn_weights,
            "algorithm": args.knn_algorithm,
            "leaf_size": args.knn_leaf_size,
            "p": args.knn_p,
        })
    else:
        readout = Ridge(ridge=args.ridge_regularization)

    # ---- Evaluation ---------------------------------------------------------
    if args.preserve_split:
        if args.folds != 1:
            raise SystemExit("--preserve_split requires --folds 1 to keep the original train/test partition intact.")

        metrics = classify(
            reservoir,
            readout,
            X_train,
            Y_train,
            X_test,
            Y_test,
            folds=1,
            trial_name=args.trial_name,
            save_threshold=args.save_threshold,
            save_states=args.save_states,
            noise_rate=0,
            noise_ratio=0,
            primary_average=args.primary_average,
        )
    else:
        X = np.concatenate((X_train, X_test), axis=0)
        Y = np.concatenate((Y_train, Y_test), axis=0)

        # --rows is the total dataset ceiling fed into cross-validation.
        # load_ecg_data applies it to training only; enforce it on the combined
        # array so that --rows 20 actually means "use 20 samples total".
        if args.rows is not None and len(X) > args.rows:
            rng_cv = np.random.default_rng(args.seed)
            idx = rng_cv.permutation(len(X))[: args.rows]
            idx.sort()
            X = X[idx]
            Y = Y[idx]
            log.debug("Dataset capped to %d samples after train/test reassembly.", len(X))

        metrics = cross_validate(
            reservoir,
            readout,
            X,
            Y,
            folds=args.folds,
            trial_name=args.trial_name,
            save_threshold=args.save_threshold,
            save_states=args.save_states,
            noise_rate=args.noise_rate,
            noise_ratio=args.noise_ratio,
            fold_backend=args.fold_backend,
            fold_workers=args.fold_workers,
            seed=args.seed,
            scaler_type=args.scaler_type,
            primary_average=args.primary_average,
            profile=args.profile,
        )

    # ---- Print results ------------------------------------------------------
    log.info("=== Classification results ===")
    for k, v in metrics.items():
        if k in ("profiling", "folds", "confusion_matrix"):
            continue
        log.info("  %s: %s", k, v)

    if args.profile and "profiling" in metrics:
        log.info("=== Profiling ===")
        for section, stats in metrics["profiling"].items():
            log.info("  %s: %s", section, stats)


if __name__ == "__main__":
    main()
