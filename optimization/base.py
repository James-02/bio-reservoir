from __future__ import annotations

import argparse
from typing import Any, Callable, Dict, List, Optional, Set
import os
from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import logging

import optuna
import reservoirpy as rpy
from optuna.study import Study

from utils.logger import setup_logging

logger = logging.getLogger(__name__)

#: Default seed used across all optimisation pipelines.
DEFAULT_SEED: int = 6337

def _optimize_study(study: Study, trials: int, objective_func: Callable, **kwargs) -> Study:
    """Optimizes the study with a given objective function.

    Args:
        study (Study): The Optuna study object.
        trials (int): Number of trials to optimize.
        objective_func (Callable): The objective function to optimize.
        **kwargs: Additional keyword arguments for the objective function.

    Returns:
        Study: The optimized Optuna study object.
    """
    # Ensure each joblib worker process configures logging.
    # In the parent process this will be a no-op (handlers already exist), but in
    # spawned workers we need to (re)configure from scratch.
    setup_logging(level=kwargs.get("log_level", logging.DEBUG), log_file=kwargs.get("log_file"))

    rpy.verbosity(kwargs.get('verbosity', 0))

    # Optimize your study using the wrapped objective function
    study.optimize(lambda trial: objective_func(trial, **kwargs), n_trials=trials, catch=(Exception,))
    return study

def research(study: Study, trials: int, objective_func: Callable, processes: Optional[int] = None, **kwargs) -> List[Study]:
    """Conduct research by optimizing the study through hyperparameter sampling.

    Args:
        study (Study): The Optuna study object.
        trials (int): Number of trials to optimize.
        objective_func (Callable): The objective function to optimize.
        processes (int, optional): Number of processes for parallel optimization. Defaults to None.
        **kwargs: Additional keyword arguments for the optimization function.

    Returns:
        List[Study]: List of optimized Optuna study objects.
    """
    if processes <= 1:
        return [_optimize_study(study, trials, objective_func, **kwargs)]

    logger.info("Optimizing with %s processes", processes)
    return joblib.Parallel(n_jobs=processes)(
        joblib.delayed(_optimize_study)(study, trials // processes, objective_func, **kwargs) for _ in range(processes))

def evaluate_study(
    study: Study,
    objective_str: str = "Score",
    top_n: int = 1,
    *,
    use_print: bool = True,
) -> None:
    """Displays the top-N trials with their parameters and objective values.

    In addition to listing the individual top-N trials, this also prints a
    small aggregate summary (mean, std, min, max) of their objective values,
    which is helpful when comparing groups of trials (e.g. per classifier
    family in the readout study).

    Args:
        study (Study): The Optuna study object or its trials DataFrame.
        objective_str (str, optional): The name of the objective being optimized. Defaults to "Score".
        top_n (int, optional): Number of top trials to print. Defaults to 1.
    """
    # Allow passing either a Study or a DataFrame of trials
    if isinstance(study, Study):
        df = study.trials_dataframe()
    else:
        df = study

    # Sort trials by objective value (descending) and take top_n
    df_sorted = df.sort_values("value", ascending=False).head(top_n)

    # Precompute pretty alignment across all shown params.
    param_cols = [c for c in df_sorted.columns if c.startswith("params_")]
    param_names = [c[len("params_"):] for c in param_cols]
    name_width = max([len(n) for n in param_names], default=0)

    def emit(line: str = "") -> None:
        if use_print:
            print(line)
        else:
            logger.info(line)

    for rank, (_, trial_row) in enumerate(df_sorted.iterrows(), start=1):
        emit(f"=== Trial Rank {rank} ===")

        # Params in stable alphabetical order.
        for col in sorted(trial_row.index):
            if col.startswith("params_") and not pd.isna(trial_row[col]):
                param_value = trial_row[col]
                param_name = col[len("params_"):]
                emit(f"  {param_name:<{name_width}} : {param_value}")

        # Print seed and runtime if present as user attributes
        seed_col = "user_attrs_seed"
        runtime_col = "user_attrs_runtime"

        if seed_col in trial_row.index and not pd.isna(trial_row[seed_col]):
            emit(f"  {'Seed':<{name_width}} : {trial_row[seed_col]}")
        if runtime_col in trial_row.index and not pd.isna(trial_row[runtime_col]):
            emit(f"  {'Runtime':<{name_width}} : {trial_row[runtime_col]}")

        emit(f"  {'Trial':<{name_width}} : {trial_row['number']}")
        emit(f"  {objective_str:<{name_width}} : {trial_row['value']}")
        emit()

    # Simple aggregate stats over the top-N objective values to help compare
    # groups (e.g. different readout algorithms).
    if not df_sorted.empty:
        values = df_sorted["value"].astype(float)
        mean_v = values.mean()
        std_v = values.std(ddof=0)
        min_v = values.min()
        max_v = values.max()
        emit(f"Summary over top {len(values)} trials:")
        emit(f"  {objective_str} mean : {mean_v:.4f}")
        emit(f"  {objective_str} std  : {std_v:.4f}")
        emit(f"  {objective_str} min  : {min_v:.4f}")
        emit(f"  {objective_str} max  : {max_v:.4f}")
        emit()

    # Also show aggregate stats over *all* trials in this group (not just the
    # top-N). This is useful for understanding the overall quality and
    # stability of a classifier family across the whole search.
    if not df.empty and "value" in df.columns:
        all_vals = df["value"].astype(float).dropna()
        if not all_vals.empty:
            mean_all = all_vals.mean()
            std_all = all_vals.std(ddof=0)
            min_all = all_vals.min()
            max_all = all_vals.max()
            emit(f"Summary over ALL {len(all_vals)} trials:")
            emit(f"  {objective_str} mean : {mean_all:.4f}")
            emit(f"  {objective_str} std  : {std_all:.4f}")
            emit(f"  {objective_str} min  : {min_all:.4f}")
            emit(f"  {objective_str} max  : {max_all:.4f}")
            emit()



# =============================================================================
# Shared evaluation helpers
# =============================================================================
#
# User-attribute prefixes stored by the objective functions.  The evaluate
# commands use these to route columns into human-readable sections.
#
# Section keys map to one or more column prefixes in the Optuna
# trials_dataframe().
#
_ATTR_PREFIXES: Dict[str, tuple] = {
    "metrics":   ("user_attrs_metric_",),
    "hypers":    ("user_attrs_res_", "user_attrs_topo_",
                  "user_attrs_readout_", "user_attrs_dataset_"),
    "timings":   ("user_attrs_profile_", "user_attrs_fold_runtime_"),
    "resources": ("user_attrs_runtime_",),
    "env":       ("user_attrs_env_",),
}

SHOW_CHOICES = ("params", "metrics", "hypers", "timings", "resources", "env")
SHOW_DEFAULT = "params"

# Aliases so users can type friendlier names on the CLI (e.g. --show environment).
_SECTION_ALIASES: Dict[str, str] = {
    "environment": "env",
    "folds":       "metrics",   # per-fold view is part of the metrics section
}


def _pretty_label(col: str) -> str:
    """Strip known ``user_attrs_*`` / ``params_*`` prefixes for display."""
    for prefix in (
        "user_attrs_metric_", "user_attrs_res_", "user_attrs_topo_",
        "user_attrs_readout_", "user_attrs_dataset_", "user_attrs_env_",
        "user_attrs_profile_", "user_attrs_runtime_", "user_attrs_",
        "params_",
    ):
        if col.startswith(prefix):
            return col[len(prefix):]
    return col


def _cols_for_section(df_columns, section: str) -> List[str]:
    """Return sorted columns from *df_columns* that belong to *section*."""
    prefixes = _ATTR_PREFIXES.get(section, ())
    return sorted(c for c in df_columns if any(c.startswith(p) for p in prefixes))


def _get_float(row, key: str, default: float = 0.0) -> float:
    """Safely extract a float from a pandas Series row, returning *default* on NaN/missing."""
    v = row.get(key, default)
    if isinstance(v, float) and pd.isna(v):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _print_timings_table(row, df_cols) -> None:
    """Print pipeline timing statistics as an aligned table.

    Columns: Stage | Total (s) | Mean/fold | Std/fold | Calls/fold

    Sections are discovered dynamically from ``user_attrs_profile_*_total_s``
    columns.  Phase sections (train_reservoir, fit_readout, val_reservoir,
    predict) are shown first; per-instance sections (instance/*) follow
    separated by a rule.  Fold-level runtime summary is printed last.

    Prints a "(none)" hint when no profiling attrs are present.
    """
    import re

    PHASE_ORDER = ["train_reservoir", "fit_readout", "val_reservoir", "predict"]

    # Discover sections from *_total_s columns
    total_re = re.compile(r"^user_attrs_profile_(.+)_total_s$")
    sections: Dict[str, str] = {}  # safe_name -> display_name
    for col in df_cols:
        m = total_re.match(col)
        if m:
            safe = m.group(1)
            display = safe.replace("instance_", "instance/", 1) if safe.startswith("instance_") else safe
            sections[safe] = display

    # Fold-level runtime summary
    rt_mean_val = row.get("user_attrs_fold_runtime_mean_s")
    has_fold_rt = rt_mean_val is not None and not (isinstance(rt_mean_val, float) and pd.isna(rt_mean_val))

    if not sections and not has_fold_rt:
        print("  [pipeline timings]  (none \u2014 re-run research with --profile)")
        return

    phase = {k: v for k, v in sections.items() if not k.startswith("instance_")}
    inst  = {k: v for k, v in sections.items() if k.startswith("instance_")}

    stage_w = 28
    if sections:
        stage_w = max(stage_w, max(len(d) for d in sections.values()) + 2)

    hdr_fmt = f"  {{:<{stage_w}}} \u2502 {{:>10}} \u2502 {{:>10}} \u2502 {{:>10}} \u2502 {{:>12}}"
    row_fmt = f"  {{:<{stage_w}}} \u2502 {{:>10.4f}} \u2502 {{:>10.4f}} \u2502 {{:>10.4f}} \u2502 {{:>12}}"
    bar = "  " + "\u2500" * (stage_w + 51)

    print("  [pipeline timings]")
    print(bar)
    print(hdr_fmt.format("Stage", "Total (s)", "Mean/fold", "Std/fold", "Calls/fold"))
    print(bar)

    ordered = [k for k in PHASE_ORDER if k in phase]
    ordered += sorted(k for k in phase if k not in PHASE_ORDER)
    for safe in ordered:
        total = _get_float(row, f"user_attrs_profile_{safe}_total_s")
        mean  = _get_float(row, f"user_attrs_profile_{safe}_mean_s")
        std   = _get_float(row, f"user_attrs_profile_{safe}_std_s")
        cpf   = row.get(f"user_attrs_profile_{safe}_calls_per_fold", "?")
        try:
            cpf = int(round(float(cpf)))
        except (TypeError, ValueError):
            pass
        print(row_fmt.format(sections[safe], total, mean, std, str(cpf)))

    if inst:
        print(bar)
        for safe in sorted(inst):
            total = _get_float(row, f"user_attrs_profile_{safe}_total_s")
            mean  = _get_float(row, f"user_attrs_profile_{safe}_mean_s")
            std   = _get_float(row, f"user_attrs_profile_{safe}_std_s")
            cpf   = row.get(f"user_attrs_profile_{safe}_calls_per_fold", "?")
            try:
                cpf = int(round(float(cpf)))
            except (TypeError, ValueError):
                pass
            print(row_fmt.format(sections[safe], total, mean, std, str(cpf)))

    print(bar)

    if has_fold_rt:
        rt_mean = _get_float(row, "user_attrs_fold_runtime_mean_s")
        rt_std  = _get_float(row, "user_attrs_fold_runtime_std_s")
        rt_min  = _get_float(row, "user_attrs_fold_runtime_min_s")
        rt_max  = _get_float(row, "user_attrs_fold_runtime_max_s")
        per_fold: List[str] = []
        for i in range(32):  # support up to 32 folds
            col = f"user_attrs_fold_runtime_fold_{i}_s"
            if col not in row.index:
                break
            v = row.get(col)
            if not (isinstance(v, float) and pd.isna(v)):
                per_fold.append(f"{float(v):.3f}s")
        pf_str = "  [" + ", ".join(per_fold) + "]" if per_fold else ""
        print(f"  Fold runtime:  mean={rt_mean:.4f}s  std={rt_std:.4f}s  "
              f"min={rt_min:.4f}s  max={rt_max:.4f}s{pf_str}")


def _print_timings_cross_trial_summary(df_top) -> None:
    """Print a cross-trial summary of pipeline timing statistics.

    For each profiled stage shows mean and std of the per-fold mean time
    across the top-N trials.  This highlights which stages vary most with
    the choice of hyperparameters.  Only emits output when profiling columns
    are actually present in *df_top*.
    """
    import re

    PHASE_ORDER = ["train_reservoir", "fit_readout", "val_reservoir", "predict"]

    total_re = re.compile(r"^user_attrs_profile_(.+)_total_s$")
    sections: Dict[str, str] = {}
    for col in df_top.columns:
        m = total_re.match(col)
        if m:
            safe = m.group(1)
            display = safe.replace("instance_", "instance/", 1) if safe.startswith("instance_") else safe
            sections[safe] = display

    has_rt = "user_attrs_fold_runtime_mean_s" in df_top.columns and \
             df_top["user_attrs_fold_runtime_mean_s"].notna().any()

    if not sections and not has_rt:
        return

    n = len(df_top)
    phase = {k: v for k, v in sections.items() if not k.startswith("instance_")}
    inst  = {k: v for k, v in sections.items() if k.startswith("instance_")}

    stage_w = 28
    if sections:
        stage_w = max(stage_w, max(len(d) for d in sections.values()) + 2)

    hdr_fmt = f"  {{:<{stage_w}}} \u2502 {{:>12}} \u2502 {{:>12}}"
    row_fmt = f"  {{:<{stage_w}}} \u2502 {{:>12.4f}} \u2502 {{:>12.4f}}"
    bar = "  " + "\u2500" * (stage_w + 29)

    print(f"  [pipeline timings \u2014 cross-trial summary ({n} trials)]")
    print(bar)
    print(hdr_fmt.format("Stage", "\u03bc Mean/fold", "\u03c3 Mean/fold"))
    print(bar)

    ordered = [k for k in PHASE_ORDER if k in phase]
    ordered += sorted(k for k in phase if k not in PHASE_ORDER)
    for safe in ordered:
        col = f"user_attrs_profile_{safe}_mean_s"
        series = df_top[col].dropna().astype(float) if col in df_top.columns else pd.Series(dtype=float)
        mu  = series.mean() if not series.empty else 0.0
        sig = series.std(ddof=0) if len(series) > 1 else 0.0
        print(row_fmt.format(sections[safe], mu, sig))

    if inst:
        print(bar)
        for safe in sorted(inst):
            col = f"user_attrs_profile_{safe}_mean_s"
            series = df_top[col].dropna().astype(float) if col in df_top.columns else pd.Series(dtype=float)
            mu  = series.mean() if not series.empty else 0.0
            sig = series.std(ddof=0) if len(series) > 1 else 0.0
            print(row_fmt.format(sections[safe], mu, sig))

    print(bar)

    if has_rt:
        rt_series = df_top["user_attrs_fold_runtime_mean_s"].dropna().astype(float)
        rt_mu  = rt_series.mean() if not rt_series.empty else 0.0
        rt_sig = rt_series.std(ddof=0) if len(rt_series) > 1 else 0.0
        print(f"  Fold runtime (mean/fold):  \u03bc={rt_mu:.4f}s  \u03c3={rt_sig:.4f}s  across {n} trials")


def _print_kv_block(row, cols: List[str], header: str) -> None:
    """Print a headed key-value block from a pandas Series row.

    Silently skips NaN values so optional user_attrs don't clutter output.
    """
    pairs = []
    for c in cols:
        if c not in row.index:
            continue
        v = row[c]
        if isinstance(v, float) and pd.isna(v):
            continue
        pairs.append((_pretty_label(c), v))
    if not pairs:
        return
    maxw = max(len(k) for k, _ in pairs)
    print(f"  [{header}]")
    for k, v in pairs:
        fmt = f"{v:.6g}" if isinstance(v, float) else str(v)
        print(f"    {k:<{maxw}} : {fmt}")


def _print_fold_table(row, df_cols) -> None:
    """Print a per-fold metrics table discovered from ``fold_{i}_*`` user attrs.

    Auto-discovers columns matching ``user_attrs_fold_<int>_<metric>`` and
    formats them as an aligned table with one row per fold.
    """
    import re
    fold_re = re.compile(r"^user_attrs_fold_(\d+)_(.+)$")

    folds_data: Dict[int, Dict[str, float]] = {}
    for col in df_cols:
        m = fold_re.match(col)
        if not m:
            continue
        fold_idx = int(m.group(1))
        metric_name = m.group(2)
        val = row.get(col)
        if val is not None and not (isinstance(val, float) and pd.isna(val)):
            folds_data.setdefault(fold_idx, {})[metric_name] = float(val)

    if not folds_data:
        return

    # Canonical column order; anything else appended alphabetically.
    _ORDER = ["f1", "accuracy", "precision", "recall", "runtime_s"]
    all_metrics = sorted(set().union(*folds_data.values()))
    ordered = [m for m in _ORDER if m in all_metrics]
    ordered += [m for m in all_metrics if m not in ordered]

    col_w = 10
    fold_w = 6
    bar = "  " + "\u2500" * (fold_w + 3 + (col_w + 3) * len(ordered))

    print("  [per-fold metrics]")
    print(bar)
    hdr = f"  {'Fold':<{fold_w}}"
    for m in ordered:
        hdr += f" \u2502 {m:>{col_w}}"
    print(hdr)
    print(bar)
    for fi in sorted(folds_data):
        line = f"  {fi:<{fold_w}}"
        for m in ordered:
            v = folds_data[fi].get(m, float("nan"))
            line += f" \u2502 {v:>{col_w}.4f}"
        print(line)

    # Mean / std summary row
    if len(folds_data) > 1:
        import numpy as _np
        print(bar)
        mean_line = f"  {'mean':<{fold_w}}"
        std_line  = f"  {'std':<{fold_w}}"
        for m in ordered:
            vals = [folds_data[fi].get(m, float("nan")) for fi in sorted(folds_data)]
            arr = _np.array(vals)
            mean_line += f" \u2502 {_np.nanmean(arr):>{col_w}.4f}"
            std_line  += f" \u2502 {_np.nanstd(arr, ddof=0):>{col_w}.4f}"
        print(mean_line)
        print(std_line)
    print(bar)


def _print_class_table(row, df_cols) -> None:
    """Print per-class precision / recall / F1 / support from user attrs.

    Auto-discovers columns matching ``user_attrs_metric_class_<cls>_<metric>``
    and ``user_attrs_metric_macro_<metric>`` / ``user_attrs_metric_weighted_<metric>``.

    Handles both legacy naming (``metric_class_0_f1``) and new naming with
    explicit mean/std suffixes (``metric_class_0_f1_mean``,
    ``metric_class_0_f1_std``).
    """
    import re
    cls_re = re.compile(r"^user_attrs_metric_class_(\d+)_(.+)$")

    classes_data: Dict[int, Dict[str, float]] = {}
    for col in df_cols:
        m = cls_re.match(col)
        if not m:
            continue
        cls_idx = int(m.group(1))
        metric_name = m.group(2)
        val = row.get(col)
        if val is not None and not (isinstance(val, float) and pd.isna(val)):
            classes_data.setdefault(cls_idx, {})[metric_name] = float(val)

    if not classes_data:
        return

    # Detect new-style (mean/std) vs legacy naming.
    all_keys = set().union(*classes_data.values())
    has_mean_std = any(k.endswith("_mean") for k in all_keys)

    if has_mean_std:
        _BASE_ORDER = ["precision", "recall", "f1", "support"]
        bases = sorted({k.rsplit("_", 1)[0] for k in all_keys if k.endswith("_mean")})
        ordered_bases = [b for b in _BASE_ORDER if b in bases]
        ordered_bases += [b for b in bases if b not in ordered_bases]

        col_w = 16  # wider to fit "mean ± std"
        cls_w = 7
        bar = "  " + "\u2500" * (cls_w + 3 + (col_w + 3) * len(ordered_bases))

        print("  [per-class metrics (mean \u00b1 std across folds)]")
        print(bar)
        hdr = f"  {'Class':<{cls_w}}"
        for b in ordered_bases:
            hdr += f" \u2502 {b:>{col_w}}"
        print(hdr)
        print(bar)
        for ci in sorted(classes_data):
            line = f"  {ci:<{cls_w}}"
            for b in ordered_bases:
                mv = classes_data[ci].get(f"{b}_mean", float("nan"))
                sv = classes_data[ci].get(f"{b}_std", 0.0)
                if b == "support":
                    line += f" \u2502 {mv:>{col_w}.0f}"
                else:
                    line += f" \u2502 {mv:.4f}\u00b1{sv:.4f}".rjust(col_w + 3)
            print(line)
        print(bar)
    else:
        # Legacy display (no std available)
        _ORDER = ["precision", "recall", "f1", "support"]
        ordered = [m for m in _ORDER if m in all_keys]
        ordered += [m for m in sorted(all_keys) if m not in ordered]

        col_w = 10
        cls_w = 7
        bar = "  " + "\u2500" * (cls_w + 3 + (col_w + 3) * len(ordered))

        print("  [per-class metrics]")
        print(bar)
        hdr = f"  {'Class':<{cls_w}}"
        for m in ordered:
            hdr += f" \u2502 {m:>{col_w}}"
        print(hdr)
        print(bar)
        for ci in sorted(classes_data):
            line = f"  {ci:<{cls_w}}"
            for m in ordered:
                v = classes_data[ci].get(m, float("nan"))
                if m == "support":
                    line += f" \u2502 {v:>{col_w}.0f}"
                else:
                    line += f" \u2502 {v:>{col_w}.4f}"
            print(line)

        if len(classes_data) > 1:
            import numpy as _np
            print(bar)
            mean_line = f"  {'mean':<{cls_w}}"
            std_line  = f"  {'std':<{cls_w}}"
            for m in ordered:
                vals_arr = [classes_data[ci].get(m, float("nan")) for ci in sorted(classes_data)]
                arr = _np.array(vals_arr)
                if m == "support":
                    mean_line += f" \u2502 {_np.nanmean(arr):>{col_w}.0f}"
                    std_line  += f" \u2502 {_np.nanstd(arr, ddof=0):>{col_w}.0f}"
                else:
                    mean_line += f" \u2502 {_np.nanmean(arr):>{col_w}.4f}"
                    std_line  += f" \u2502 {_np.nanstd(arr, ddof=0):>{col_w}.4f}"
            print(mean_line)
            print(std_line)
        print(bar)

    # Macro / weighted avg rows
    _ORDER_FALLBACK = ["precision", "recall", "f1", "support"]
    _agg_bar_printed = False
    for avg_label in ("macro", "weighted"):
        vals: Dict[str, float] = {}
        for m_name in (ordered_bases if has_mean_std else _ORDER_FALLBACK):
            col = f"user_attrs_metric_{avg_label}_{m_name}"
            v = row.get(col)
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                vals[m_name] = float(v)
        if vals:
            if not _agg_bar_printed:
                print(bar)
                _agg_bar_printed = True
            _cw = col_w if has_mean_std else 10
            line = f"  {avg_label:<{cls_w}}"
            for m_name in (ordered_bases if has_mean_std else _ORDER_FALLBACK):
                v = vals.get(m_name, float("nan"))
                if m_name == "support":
                    line += f" \u2502 {v:>{_cw}.0f}"
                else:
                    line += f" \u2502 {v:>{_cw}.4f}"
            print(line)
    if _agg_bar_printed:
        print(bar)


def print_topn_trials(
    df,
    top_n: int,
    title: str,
    *,
    group_col: Optional[str] = None,
    sections: Optional[Set[str]] = None,
    strip_prefix: Optional[str] = None,
    best_only_detail: bool = False,
) -> None:
    """Print a human-readable evaluation report for the top-N trials.

    Sections are controlled by the *sections* argument (populated from
    ``--show`` on the CLI).  The ``params`` section and aggregate summary
    are always included.

    Parameters
    ----------
    df : pandas.DataFrame
        ``study.trials_dataframe()`` from Optuna.
    top_n : int
        Number of top-scoring trials to include.
    title : str
        Header label for the report block.
    group_col : str, optional
        Column to always retain in the params table even when it has only one
        unique value (e.g. ``"params_classifier"`` in per-classifier tables).
    strip_prefix : str, optional
        When set, strip this prefix (case-insensitive, with trailing ``_``)
        from displayed parameter names.  For example, ``strip_prefix="MLP"``
        turns ``mlp_alpha`` into ``alpha``.
    sections : set of str, optional
        Subset of :data:`SHOW_CHOICES` to render.  Defaults to
        ``{"params", "metrics"}``.

        ``"params"``     — Optuna trial parameters table (always shown).
        ``"metrics"``    — Mean F1/accuracy/precision/recall/runtime columns
                           saved as ``user_attrs_metric_*``.
        ``"hypers"``     — Full reservoir / topology / readout / dataset
                           user_attrs, grouped by sub-category.
        ``"timings"``    — Pipeline section timings (``profile_*`` user_attrs;
                           only populated when research was run with
                           ``--profile``).
        ``"resources"``  — CPU % and RSS MB statistics sampled during the
                           trial (``runtime_*`` user_attrs).
        ``"env"``        — Runtime environment snapshot (Python version, CPU
                           model, package versions, git commit, …) printed
                           once from the best trial.
    """
    if sections is None:
        sections = {"params", "metrics"}
    if "value" not in df.columns or df.empty:
        return

    df_top = df.sort_values("value", ascending=False).head(int(top_n))
    if df_top.empty:
        return

    # ── params (+ optional metrics) table ───────────────────────────────────
    param_cols = [c for c in df_top.columns if c.startswith("params_")]
    active_params: List[str] = []
    for c in sorted(param_cols):
        non_nan = df_top[c].dropna()
        if not non_nan.empty and non_nan.nunique() > 0:
            if c != group_col:
                active_params.append(c)

    cols_to_show = ["value", "number"]
    if group_col and group_col in df_top.columns:
        cols_to_show.append(group_col)
    cols_to_show.extend(active_params)

    if "metrics" in sections:
        # Only aggregate metrics in the summary table; per-class / per-fold
        # details are printed separately in the detail blocks below.
        _DETAIL_INFIXES = ("_class_", "_fold_", "_macro_", "_weighted_")
        _EXCLUDE_SUFFIXES = ("_confusion_matrix",)
        metric_cols = sorted(
            c for c in df_top.columns
            if c.startswith("user_attrs_metric_")
            and not any(inf in c for inf in _DETAIL_INFIXES)
            and not any(c.endswith(s) for s in _EXCLUDE_SUFFIXES)
        )
        cols_to_show.extend(metric_cols)
    else:
        metric_cols = []

    available = [c for c in cols_to_show if c in df_top.columns]
    tab = df_top[available].copy()

    renamed: Dict[str, str] = {"number": "trial", "value": "f1"}
    if group_col:
        renamed[group_col] = _pretty_label(group_col)
    for c in active_params:
        label = _pretty_label(c)
        # Strip classifier prefix for per-classifier tables (e.g. mlp_alpha -> alpha).
        if strip_prefix:
            pfx = strip_prefix.lower().replace(" ", "_") + "_"
            if label.lower().startswith(pfx):
                label = label[len(pfx):]
        renamed[c] = label
    for c in metric_cols:
        renamed[c] = _pretty_label(c)
    tab = tab.rename(columns=renamed)

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    with pd.option_context("display.width", None, "display.max_colwidth", 24):
        print(tab.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    # ── per-trial detail blocks ──────────────────────────────────────────────
    detail_sections = [s for s in ("metrics", "hypers", "timings", "resources") if s in sections]
    if detail_sections:
        for rank, (_, row) in enumerate(df_top.iterrows()):
            trial_num = int(row.get("number", "?"))
            f1_val = float(row.get("value", float("nan")))
            print(f"\n  ── Trial {trial_num}  (f1={f1_val:.4f}) ──")

            if "metrics" in sections:
                # Per-fold / per-class tables only for the best trial
                # (rank 0) when best_only_detail is set.
                if not best_only_detail or rank == 0:
                    _print_fold_table(row, df_top.columns)
                    _print_class_table(row, df_top.columns)

            if "hypers" in sections:
                hyper_cols = _cols_for_section(df_top.columns, "hypers")
                res_topo = [c for c in hyper_cols
                            if c.startswith(("user_attrs_res_", "user_attrs_topo_"))]
                readout  = [c for c in hyper_cols if c.startswith("user_attrs_readout_")]
                dataset  = [c for c in hyper_cols if c.startswith("user_attrs_dataset_")]
                _print_kv_block(row, res_topo, "reservoir + topology")
                _print_kv_block(row, readout,  "readout")
                _print_kv_block(row, dataset,  "dataset")

            if "timings" in sections:
                tc = _cols_for_section(df_top.columns, "timings")
                if tc:
                    _print_timings_table(row, df_top.columns)
                else:
                    print("  [pipeline timings]  (none — re-run research with --profile)")

            if "resources" in sections:
                rc = _cols_for_section(df_top.columns, "resources")
                if rc:
                    _print_kv_block(row, rc, "resource usage (CPU % / RSS MB)")
                else:
                    print("  [resource usage]  (none — psutil may not be installed)")

    # ── cross-trial timings summary (only meaningful when top_n > 1) ────────
    if "timings" in sections and len(df_top) > 1:
        _print_timings_cross_trial_summary(df_top)

    # ── env block (once, from best trial) ───────────────────────────────────
    if "env" in sections:
        best_row = df_top.iloc[0]
        env_cols = _cols_for_section(df_top.columns, "env")
        if env_cols:
            trial_label = int(best_row.get("number", "?"))
            print(f"\n  [environment — trial {trial_label}]")
            maxw = max(len(_pretty_label(c)) for c in env_cols)
            for col in env_cols:
                val = best_row.get(col)
                if isinstance(val, float) and pd.isna(val):
                    continue
                print(f"    {_pretty_label(col):<{maxw}} : {val}")

    # ── aggregate stats ──────────────────────────────────────────────────────
    vals     = df_top["value"].astype(float)
    all_vals = df["value"].astype(float).dropna()
    print(f"\n  Summary — top {len(vals):>4}: "
          f"mean={vals.mean():.4f}  std={vals.std(ddof=0):.4f}  "
          f"min={vals.min():.4f}  max={vals.max():.4f}")
    if len(all_vals) > len(vals):
        print(f"  Summary — all  {len(all_vals):>4}: "
              f"mean={all_vals.mean():.4f}  std={all_vals.std(ddof=0):.4f}  "
              f"min={all_vals.min():.4f}  max={all_vals.max():.4f}")

    print()


def parse_show_sections(show_str: str) -> Set[str]:
    """Parse a ``--show`` CLI string into a validated set of section names.

    Unknown section names are silently dropped with a warning so users are
    informed without crashing.  Aliases like ``environment`` → ``env`` are
    resolved first.
    """
    parts = {s.strip().lower() for s in show_str.split(",") if s.strip()}
    # Resolve aliases before validation.
    parts = {_SECTION_ALIASES.get(p, p) for p in parts}
    unknown = parts - set(SHOW_CHOICES)
    for u in unknown:
        logger.warning("Unknown --show section %r (valid: %s)", u, ", ".join(SHOW_CHOICES))
    return parts - unknown


def _call_plot_func(
    study: Study,
    plot_func: Callable,
    params: Optional[list] = None,
    target_name: str = "Macro F1",
) -> bool:
    """Call an Optuna plot function, handling signature differences.

    Returns *True* on success, *False* if the plot was skipped.
    """
    # Build kwargs that the function *might* accept.
    kw: Dict[str, Any] = {"study": study}
    if params is not None:
        kw["params"] = params
    kw["target_name"] = target_name

    try:
        try:
            plot_func(**kw)
        except TypeError:
            # Some Optuna functions don't accept params / target_name.
            kw.pop("params", None)
            kw.pop("target_name", None)
            plot_func(**kw)
    except ValueError as exc:
        logger.warning(
            "Skipping plot %s: %s  (tip: need >1 completed trial for %s)",
            getattr(plot_func, "__name__", plot_func),
            exc,
            "param_importances/contour",
        )
        return False
    return True


def _postprocess_optuna_figure(fig: plt.Figure) -> None:
    """Strip Optuna auto-titles, fix legends, and rotate long tick labels."""
    import matplotlib.text as mtext

    for ax in fig.get_axes():
        # 1. Strip any title set via ax.set_title().
        ax.set_title("")

        # 2. Strip title Text objects that Optuna injects as direct children
        #    (e.g. "Hyperparameter Importances", "Slice Plot").
        for child in ax.get_children():
            if isinstance(child, mtext.Text):
                txt = child.get_text()
                if any(kw in txt for kw in ("Importances", "Slice")):
                    child.set_text("")

        # 3. Fix the importance-plot legend: Optuna labels the bars
        #    "Objective Value"; replace with the algorithm name "fANOVA".
        legend = ax.get_legend()
        if legend is not None:
            for lt in legend.get_texts():
                if lt.get_text() == "Objective Value":
                    lt.set_text("fANOVA")

        # 4. Rotate long x-tick labels to prevent overlap.
        labels = [t.get_text() for t in ax.get_xticklabels()]
        if any(len(lbl) > 4 for lbl in labels if lbl):
            ax.tick_params(axis="x", rotation=45)
            for label in ax.get_xticklabels():
                label.set_ha("right")

    # Strip any figure-level suptitle.
    fig.suptitle("")


def _save_and_close(fig: plt.Figure, filename: str, *, also_pdf: bool = False) -> None:
    """Save a figure to results/optimization/ and close it."""
    _project_root = Path(__file__).resolve().parent.parent
    results_dir = _project_root / "results" / "optimization"
    results_dir.mkdir(parents=True, exist_ok=True)

    png_path = results_dir / filename
    fig.savefig(png_path, bbox_inches="tight", dpi=800)
    logger.info("Saved %s", png_path)

    if also_pdf:
        pdf_path = png_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        logger.info("Saved %s", pdf_path)


def plot_results(
    study: Study,
    plot_func: Callable,
    params: Optional[Dict] = None,
    filename: Optional[str] = None,
    *,
    target_name: str = "Macro F1",
    also_pdf: bool = False,
    max_params_per_plot: int = 8,
) -> None:
    """Plots the results of the Optuna study.

    Args:
        study: The Optuna study object.
        plot_func: The Optuna visualisation function (e.g. ``plot_slice``).
        params: Parameter list for plots that support it.
        filename: The filename to save the plot (PNG).
        target_name: Y-axis / objective label passed to Optuna.
        also_pdf: If *True*, save a PDF copy alongside the PNG.
        max_params_per_plot: For ``plot_slice``, if *params* has more entries
            than this the figure is split into two halves saved as
            ``<stem>-A.png`` / ``<stem>-B.png``.
    """
    from optuna.visualization.matplotlib import plot_slice as _optuna_plot_slice

    # --- Strip fixed_* params (single-value conditional placeholders) ----
    if params is not None:
        params = [p for p in params if not p.startswith("fixed_")]
        if not params:
            params = None

    # --- Split wide slice plots into A/B halves --------------------------
    is_slice = (plot_func is _optuna_plot_slice)
    if is_slice and params is not None and len(params) > max_params_per_plot:
        mid = (len(params) + 1) // 2
        stem = Path(filename).stem if filename else "plot"
        suffix = Path(filename).suffix if filename else ".png"
        for tag, subset in [("A", params[:mid]), ("B", params[mid:])]:
            sub_file = f"{stem}-{tag}{suffix}" if filename else None
            plot_results(
                study, plot_func, params=subset, filename=sub_file,
                target_name=target_name, also_pdf=also_pdf,
                max_params_per_plot=max_params_per_plot,
            )
        return

    # --- Generate the plot -----------------------------------------------
    if not _call_plot_func(study, plot_func, params=params,
                           target_name=target_name):
        return

    fig = plt.gcf()
    _postprocess_optuna_figure(fig)

    if filename:
        _save_and_close(fig, filename, also_pdf=also_pdf)

    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Shared Optuna helpers (used by both classification and readout modules)
# ---------------------------------------------------------------------------

def get_default_storage(study_name: str, *, must_exist: bool = False) -> str:
    """Return the default SQLite Optuna storage URL for *study_name*.

    Parameters
    ----------
    study_name : str
        Logical study name used to derive the DB filename.
    must_exist : bool
        If *True*, raise :class:`SystemExit` when the DB file is missing
        instead of letting SQLAlchemy silently create an empty one.
    """
    db_path = f"logs/optuna-{study_name}.db"
    if must_exist and not os.path.exists(db_path):
        raise SystemExit(
            f"Study database not found: {db_path}\n"
            f"Run a 'research' command first to create it."
        )
    os.makedirs("logs", exist_ok=True)
    return f"sqlite:///{db_path}?timeout=60"


def build_study(
    *,
    study_name: str,
    storage: str,
    seed: int = DEFAULT_SEED,
    study_config: Optional[Dict[str, Any]] = None,
    sampler_override: Optional[str] = None,
) -> optuna.study.Study:
    """Create (or load) an Optuna study with the appropriate sampler.

    Parameters
    ----------
    study_name : str
        Name stored in the Optuna DB.
    storage : str
        Optuna storage URL (typically SQLite).
    seed : int
        RNG seed forwarded to TPE / Random samplers.
    study_config : dict, optional
        Study configuration from :mod:`optimization.studies`.  When provided,
        the sampler type and grid spec are read from this dict.
    sampler_override : str, optional
        Override the sampler type (``"tpe"``, ``"random"``, ``"grid"``).
    """
    sampler_name = sampler_override or (study_config or {}).get("sampler", "tpe")
    if sampler_name == "tpe":
        sampler: optuna.samplers.BaseSampler = optuna.samplers.TPESampler(
            constant_liar=True,
        )
    elif sampler_name == "random":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    elif sampler_name == "grid":
        grid = (study_config or {}).get("_grid")
        if grid is None:
            raise ValueError(
                f"Study {study_name!r} uses grid sampler but has no '_grid' key."
            )
        sampler = optuna.samplers.GridSampler(grid)
    else:
        raise ValueError(f"Unknown sampler {sampler_name!r}.")
    return optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        sampler=sampler,
        load_if_exists=True,
    )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments shared by all optimisation subcommands."""
    parser.add_argument(
        "--study_name", type=str, default=None,
        help="Optuna DB study name.  Defaults to the --study value.",
    )
    parser.add_argument(
        "--storage", type=str, default=None,
        help="Optuna storage URL.  Defaults to sqlite:///logs/optuna-<study>.db?timeout=60",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
    )
    parser.add_argument("--verbosity", type=int, default=0, choices=[0, 1, 2])
