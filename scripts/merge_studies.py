"""Merge two Optuna studies that ran the same search space.

Trials with an identical parameter set (compared key-for-key and value-for-value)
are considered duplicates and skipped.  All other COMPLETE trials from the *source*
study are copied into the *target* study.

Usage
-----
# Dry-run – prints what would be added without touching the database:
    python scripts/merge_studies.py \
        --target  ro2-readout-unbalanced \
        --source  ro2-readout-unbalanced-2 \
        --dry-run

# Live merge:
    python scripts/merge_studies.py \
        --target ro2-readout-unbalanced \
        --source ro2-readout-unbalanced-2

Optionally supply explicit storage URLs if the databases are not in the
default logs/ directory:
    python scripts/merge_studies.py \
        --target ro2-readout-unbalanced \
        --source ro2-readout-unbalanced-2 \
        --target-storage sqlite:///logs/optuna-ro2-readout-unbalanced.db \
        --source-storage sqlite:///logs/optuna-ro2-readout-unbalanced-2.db
"""

import argparse
import sys
import copy
from typing import Optional, Tuple

import optuna
from optuna.trial import FrozenTrial, TrialState


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _default_storage(study_name: str) -> str:
    return f"sqlite:///logs/optuna-{study_name}.db?timeout=60"


def _param_fingerprint(trial: FrozenTrial) -> frozenset:
    """Return a hashable, order-independent fingerprint of a trial's parameters."""
    return frozenset(trial.params.items())


def _copy_trial(src_trial: FrozenTrial) -> FrozenTrial:
    """Return a deep copy suitable for insertion into another study.

    The trial number is set to -1 so Optuna assigns a fresh one.
    """
    t = copy.deepcopy(src_trial)
    # _trial_id and number are managed by the storage backend; reset them so
    # add_trial() does not try to reuse the source study's identifiers.
    t._trial_id = -1
    t.number = -1
    return t


# --------------------------------------------------------------------------- #
# Core merge logic                                                             #
# --------------------------------------------------------------------------- #

def _resolve_study_name(storage: str, hint: str) -> str:
    """Return the actual internal study name stored in *storage*.

    If *hint* is present in the database it is used directly.  Otherwise,
    if the database contains exactly one study its name is returned
    automatically.  This handles the common case where both databases were
    created with the same internal study name.
    """
    summaries = optuna.get_all_study_summaries(storage=storage)
    names = [s.study_name for s in summaries]
    if hint in names:
        return hint
    if len(names) == 1:
        print(f"  (resolved study name '{names[0]}' from {storage})")
        return names[0]
    raise ValueError(
        f"Cannot resolve study name '{hint}' in {storage}. "
        f"Available: {names}. Use --target-study-name / --source-study-name to specify."
    )


def merge_studies(
    target_name: str,
    source_name: str,
    target_storage: str,
    source_storage: str,
    *,
    target_study_name: Optional[str] = None,
    source_study_name: Optional[str] = None,
    dry_run: bool = False,
    states: tuple = (TrialState.COMPLETE,),
) -> None:
    """Copy unique trials from *source* into *target*.

    Parameters
    ----------
    target_name:        db-filename basename used to derive the storage URL
    source_name:        db-filename basename used to derive the storage URL
    target_storage:     SQLAlchemy URL for the target database
    source_storage:     SQLAlchemy URL for the source database
    target_study_name:  internal study name in target db (auto-detected if None)
    source_study_name:  internal study name in source db (auto-detected if None)
    dry_run:            if True, print actions without modifying the database
    states:             only consider trials whose TrialState is in this iterable
    """
    t_internal = _resolve_study_name(target_storage, target_study_name or target_name)
    s_internal = _resolve_study_name(source_storage, source_study_name or source_name)

    print(f"Loading target study '{t_internal}' from {target_storage}")
    target = optuna.load_study(study_name=t_internal, storage=target_storage)

    print(f"Loading source study '{s_internal}' from {source_storage}")
    source = optuna.load_study(study_name=s_internal, storage=source_storage)

    # Build a set of fingerprints from every trial in the target (any state).
    target_fingerprints: set = {
        _param_fingerprint(t) for t in target.trials
    }

    source_trials = [t for t in source.trials if t.state in states]
    print(f"\nTarget : {len(target.trials)} trials total "
          f"({sum(1 for t in target.trials if t.state == TrialState.COMPLETE)} complete)")
    print(f"Source : {len(source.trials)} trials total "
          f"({len(source_trials)} with requested state(s))")

    added = 0
    skipped_duplicate = 0
    skipped_other = 0

    for trial in source.trials:
        if trial.state not in states:
            skipped_other += 1
            continue

        fp = _param_fingerprint(trial)
        if fp in target_fingerprints:
            skipped_duplicate += 1
            continue

        # Unique trial – copy it into the target.
        new_trial = _copy_trial(trial)
        if dry_run:
            print(f"  [DRY-RUN] Would add trial with params: {trial.params}  "
                  f"(value={trial.value:.6f})")
        else:
            target.add_trial(new_trial)
            target_fingerprints.add(fp)  # avoid adding it again if source has dupes
        added += 1

    print(f"\n--- Result ---")
    print(f"  Unique trials added : {added}")
    print(f"  Duplicates skipped  : {skipped_duplicate}")
    print(f"  Non-target-state    : {skipped_other}")
    if dry_run:
        print("\n  (dry-run – no changes written)")
    else:
        print(f"\n  Target study now has "
              f"{sum(1 for t in target.trials if t.state == TrialState.COMPLETE)} "
              f"complete trials.")


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge two Optuna studies, skipping duplicate parameter sets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--target", required=True, metavar="STUDY_NAME",
        help="Name of the target study (trials are written here).",
    )
    parser.add_argument(
        "--source", required=True, metavar="STUDY_NAME",
        help="Name of the source study (read-only).",
    )
    parser.add_argument(
        "--target-storage", default=None, metavar="URL",
        help="SQLAlchemy storage URL for the target study. "
             "Defaults to logs/optuna-<target>.db",
    )
    parser.add_argument(
        "--source-storage", default=None, metavar="URL",
        help="SQLAlchemy storage URL for the source study. "
             "Defaults to logs/optuna-<source>.db",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be copied without modifying the database.",
    )
    parser.add_argument(
        "--target-study-name", default=None, metavar="NAME",
        help="Internal study name inside the target database. "
             "Auto-detected when the database contains a single study.",
    )
    parser.add_argument(
        "--source-study-name", default=None, metavar="NAME",
        help="Internal study name inside the source database. "
             "Auto-detected when the database contains a single study.",
    )
    parser.add_argument(
        "--include-pruned", action="store_true",
        help="Also copy PRUNED trials (default: COMPLETE only).",
    )

    args = parser.parse_args()

    target_storage = args.target_storage or _default_storage(args.target)
    source_storage = args.source_storage or _default_storage(args.source)

    states = [TrialState.COMPLETE]
    if args.include_pruned:
        states.append(TrialState.PRUNED)

    merge_studies(
        target_name=args.target,
        source_name=args.source,
        target_storage=target_storage,
        source_storage=source_storage,
        target_study_name=getattr(args, "target_study_name", None),
        source_study_name=getattr(args, "source_study_name", None),
        dry_run=args.dry_run,
        states=tuple(states),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
