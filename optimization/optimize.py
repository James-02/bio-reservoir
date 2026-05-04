"""Unified CLI for BioReservoir hyperparameter optimisation.

Usage
-----
# List all named studies:
    python -m optimization.optimize list-studies

# Run a reservoir/training optimisation study:
    python -m optimization.optimize research --type training --study g1-scaling --processes 64

# Run a readout optimisation study:
    python -m optimization.optimize research --type readout --study_name ro1-readout \\
        --trial_name r1a-refined-best --classifiers KNN RF Ridge --processes 32

# Run a no-reservoir baseline study (classifiers on raw ECG features):
    python -m optimization.optimize research --type baseline \\
        --study_name b1-baseline --trials 500 --processes 32

# Evaluate an existing study:
    python -m optimization.optimize evaluate --type training --study g1-scaling \
        --show params,metrics,hypers --plots slice,importance

# Re-run best trial with saved states (training only):
    python -m optimization.optimize rerun-best --study r1a-refined --folds 5

# Apply best readout classifier to fold NPZ files (readout only):
    python -m optimization.optimize apply-best --study_name ro1-readout \\
        --trial_name r1a-refined-best --processes 4
"""

from __future__ import annotations

import argparse
import logging

import reservoirpy as rpy

from optimization.base import (
    DEFAULT_SEED,
    SHOW_DEFAULT,
    add_common_args,
)
from utils.logger import setup_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m optimization.optimize",
        description="BioReservoir hyperparameter optimisation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- list-studies ----------------------------------------------------
    sub.add_parser("list-studies", help="Print all named studies.")

    # ---- research --------------------------------------------------------
    p_r = sub.add_parser("research", help="Run an Optuna study.")
    add_common_args(p_r)
    p_r.add_argument(
        "--type", type=str, required=True,
        choices=["training", "readout", "baseline"],
        help="Optimisation type: 'training' (reservoir), 'readout', or 'baseline' (no reservoir).",
    )
    # Shared
    p_r.add_argument("--study", type=str, default=None,
                     help="Named study from optimization/studies.py (training type).")
    p_r.add_argument("--trials",    type=int, default=None)
    p_r.add_argument("--processes", type=int, default=1)
    p_r.add_argument("--seed",      type=int, default=DEFAULT_SEED)
    p_r.add_argument("--job_id",    type=int, default=0,
                     help="Server/worker ID stored for provenance.")
    # Training-specific
    p_r.add_argument("--instances", type=int, default=None)
    p_r.add_argument("--folds",     type=int, default=None)
    p_r.add_argument("--binary",    action="store_true")
    p_r.add_argument("--sampler",   type=str, default=None,
                     choices=["tpe", "random", "grid"])
    p_r.add_argument("--save_threshold", type=float, default=0.0)
    p_r.add_argument("--save_states",    action="store_true")
    p_r.add_argument("--fold_workers",   type=int, default=1)
    p_r.add_argument("--profile",        action="store_true")
    p_r.add_argument("--preserve_split", action="store_true",
                     help="Keep the original Kachuee train/test split unchanged (training only).")
    p_r.add_argument("--primary_average", type=str, default=None,
                     choices=["macro", "weighted"],
                     help="Override the primary precision/recall/F1 averaging mode (training only).")
    # Readout- and baseline-specific
    p_r.add_argument("--trial_name", type=str, default=None,
                     help="Trial name for readout fold artefacts.")
    p_r.add_argument("--classifiers", nargs="+",
                     default=["KNN", "RF", "Ridge", "LR", "SVM",
                              "MLP", "Perceptron", "DT", "GB"])
    # Baseline-specific
    p_r.add_argument("--noise_rate", type=float, default=None,
                     help="Per-fold noise augmentation rate (baseline only).")
    p_r.add_argument("--noise_ratio", type=float, default=None,
                     help="Per-fold noise augmentation ratio (baseline only).")
    p_r.add_argument("--max_per_class", type=int, default=None,
                     help="Cap per-class samples (baseline unbalanced).")

    # ---- evaluate --------------------------------------------------------
    p_e = sub.add_parser("evaluate", help="Analyse an existing study DB.")
    add_common_args(p_e)
    p_e.add_argument(
        "--type", type=str, required=True,
        choices=["training", "readout", "baseline"],
    )
    p_e.add_argument("--study", type=str, default=None)
    p_e.add_argument("--top_n", type=int, default=10)
    p_e.add_argument(
        "--show", type=str, default=None, metavar="SECTIONS",
        help=(
            "Comma-separated: params, metrics, hypers, timings, resources, env. "
            f"Default: {SHOW_DEFAULT}."
        ),
    )
    p_e.add_argument(
        "--plots", type=str, default=None,
        help=(
            "Comma-separated overview plot types. Defaults: training/classification "
            "use slice,importance; readout uses slice."
        ),
    )
    p_e.add_argument("--max_params", type=int, default=8)
    p_e.add_argument(
        "--classifier_plots", type=str, default=None,
        help=(
            "Readout only: comma-separated per-classifier plot types. "
            "Default: none."
        ),
    )
    p_e.add_argument(
        "--classifiers", nargs="+", default=None, metavar="CLF",
        help="Show per-classifier detail only for these classifiers "
             "(default: all). E.g. --classifiers KNN RF",
    )

    # ---- rerun-best (training only) --------------------------------------
    p_rb = sub.add_parser(
        "rerun-best",
        help="Re-run the best trial with save_states=True (training only).",
    )
    add_common_args(p_rb)
    p_rb.add_argument("--study",      type=str, default=None)
    p_rb.add_argument("--trial_name", type=str, default=None)
    p_rb.add_argument("--folds",      type=int, default=None)
    p_rb.add_argument("--instances",  type=int, default=None)
    p_rb.add_argument("--binary",     action="store_true")
    p_rb.add_argument("--seed",       type=int, default=DEFAULT_SEED)
    p_rb.add_argument("--processes",  type=int, default=1)
    p_rb.add_argument("--profile",    action="store_true",
                      help="Collect per-section pipeline timings.")
    p_rb.add_argument("--preserve_split", action="store_true",
                      help="Keep the original Kachuee train/test split unchanged.")
    p_rb.add_argument("--primary_average", type=str, default=None,
                      choices=["macro", "weighted"],
                      help="Override the primary precision/recall/F1 averaging mode.")
    p_rb.add_argument("--readout_study", type=str, default=None,
                      help="Load best readout from this Optuna study "
                           "instead of the default KNN.")
    p_rb.add_argument("--readout_trial", type=int, default=None,
                      help="Use this trial number from --readout_study "
                           "instead of the study's best trial.")
    p_rb.add_argument("--noise_rate", type=float, default=None,
                      help="Override the trial's noise_rate for data augmentation.")
    p_rb.add_argument("--noise_ratio", type=float, default=None,
                      help="Override the trial's noise_ratio for data augmentation.")

    # ---- apply-best (readout only) ---------------------------------------
    p_ab = sub.add_parser(
        "apply-best",
        help="Apply the best readout trial to fold NPZ files (readout only).",
    )
    add_common_args(p_ab)
    p_ab.add_argument("--trial_name", type=str, required=True)
    p_ab.add_argument("--processes",  type=int, default=1)
    p_ab.add_argument("--readout_trial", type=int, default=None,
                      help="Use this trial number instead of the study's best trial.")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    setup_logging(level=getattr(logging, getattr(args, "log_level", "INFO")))
    rpy.verbosity(getattr(args, "verbosity", 0))

    cmd = args.command

    if cmd == "list-studies":
        from optimization.studies import list_studies
        list_studies()
        return

    if cmd == "research":
        if args.type == "training":
            from optimization.training import cmd_research
            cmd_research(args)
        elif args.type == "baseline":
            from optimization.baseline import cmd_research
            cmd_research(args)
        else:
            from optimization.readout import cmd_research
            cmd_research(args)
        return

    if cmd == "evaluate":
        if args.type == "training":
            from optimization.training import cmd_evaluate
            cmd_evaluate(args)
        elif args.type == "baseline":
            from optimization.baseline import cmd_evaluate
            cmd_evaluate(args)
        else:
            from optimization.readout import cmd_evaluate
            cmd_evaluate(args)
        return

    if cmd == "rerun-best":
        from optimization.training import cmd_rerun_best
        cmd_rerun_best(args)
        return

    if cmd == "apply-best":
        from optimization.readout import cmd_apply_best
        cmd_apply_best(args)
        return

    raise SystemExit(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
