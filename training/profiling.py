"""training.profiling — lightweight instrumentation for the training pipeline.

Contains two independent tools:

Profiler
    Section-based wall-time timer.  Zero overhead when ``enabled=False``.
    Used inside :mod:`training.pipeline` and :mod:`training.cross_validate` to
    decompose runtime into ``train_reservoir``, ``fit_readout``, ``predict``,
    and per-instance ``instance/reservoir_run`` / ``instance/state_aggregation``
    phases.

ResourceMonitor
    Background daemon thread that samples per-process CPU % and RSS MB via
    psutil at a configurable interval.  Used in the Optuna objective to collect
    resource usage across a full trial.  Falls back gracefully when psutil is
    not installed.

Both tools are fully standalone — no imports from other project modules.
"""

from __future__ import annotations

import contextlib
import time
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------

class Profiler:
    """Lightweight section timer for profiling the training pipeline.

    When ``enabled=False``, all :meth:`section` calls are pure no-ops —
    zero overhead on the hot path.

    Sections are named by convention: use ``"phase/sub-phase"`` to create a
    two-level hierarchy that :func:`_merge_profiles` can average across folds.

    Usage::

        profiler = Profiler(enabled=True)

        with profiler.section("train_reservoir"):
            train_states = train(reservoir, X_train, profiler=profiler)
        with profiler.section("fit_readout"):
            fit_readout(readout, train_states, Y_train)

        profiler.summary()
        # {"train_reservoir": {"total_s": 4.2, "calls": 1, "mean_s": 4.2},
        #  "instance/reservoir_run": {"total_s": 4.0, "calls": 800, "mean_s": 0.005}}

        profiler.flat_summary(prefix="profile_")
        # {"profile_train_reservoir_total_s": 4.2, ...}  — ready for Optuna user_attrs
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._data: Dict[str, Dict[str, float]] = {}

    @contextlib.contextmanager
    def section(self, name: str):
        """Context manager that times the enclosed block under *name*."""
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            entry = self._data.setdefault(name, {"total_s": 0.0, "calls": 0})
            entry["total_s"] += elapsed
            entry["calls"] += 1

    def summary(self) -> Dict[str, Any]:
        """Return ``{section: {total_s, calls, mean_s}}`` for all timed sections."""
        out: Dict[str, Any] = {}
        for name, d in self._data.items():
            calls = d["calls"]
            total = d["total_s"]
            out[name] = {
                "total_s": round(total, 6),
                "calls": calls,
                "mean_s": round(total / calls, 6) if calls > 0 else 0.0,
            }
        return out

    def flat_summary(self, prefix: str = "") -> Dict[str, float]:
        """Flat ``{prefix}{section}_{stat}`` → value dict.

        ``/`` in section names is replaced with ``_`` so the keys are valid
        Python identifiers and safe as Optuna ``user_attr`` keys.
        """
        out: Dict[str, float] = {}
        for name, stats in self.summary().items():
            safe = name.replace("/", "_")
            for k, v in stats.items():
                out[f"{prefix}{safe}_{k}"] = v
        return out


def _merge_profiles(profiles: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """Combine :meth:`Profiler.summary` dicts across multiple folds.

    Returns per-section statistics that describe the whole trial:

    * ``total_s``       — wall time in this section across **all folds** combined.
    * ``mean_s``        — per-fold average (``total_s / n_folds``).
    * ``std_s``         — population std of per-fold totals (0.0 for 1 fold).
    * ``calls_total``   — total invocation count across all folds.
    * ``calls_per_fold``— average calls per fold (= N_instances for inner sections).
    * ``n_folds``       — number of folds merged.

    Example with 2 folds, 10 instances each::

        train_reservoir:       total_s=2.3  mean_s=1.15  std_s=0.03  calls_per_fold=1
        instance/reservoir_run: total_s=3.3  mean_s=1.65  std_s=0.05  calls_per_fold=20
    """
    import math
    valid = [p for p in profiles if p]
    if not valid:
        return {}
    n_folds = len(valid)
    # Collect per-fold values for each section
    fold_totals: Dict[str, List[float]] = {}
    fold_calls:  Dict[str, List[int]]   = {}
    for prof in valid:
        for section, stats in prof.items():
            fold_totals.setdefault(section, []).append(stats["total_s"])
            fold_calls.setdefault(section,  []).append(stats["calls"])
    result: Dict[str, Any] = {}
    for section in fold_totals:
        totals = fold_totals[section]
        calls  = fold_calls[section]
        total_s      = sum(totals)
        calls_total  = sum(calls)
        mean_s       = total_s / n_folds
        std_s        = math.sqrt(
            sum((t - mean_s) ** 2 for t in totals) / n_folds
        ) if n_folds > 1 else 0.0
        result[section] = {
            "total_s":        round(total_s, 6),
            "mean_s":         round(mean_s, 6),
            "std_s":          round(std_s, 6),
            "calls_total":    int(calls_total),
            "calls_per_fold": round(calls_total / n_folds, 1),
            "n_folds":        n_folds,
        }
    return result


# ---------------------------------------------------------------------------
# ResourceMonitor
# ---------------------------------------------------------------------------

class ResourceMonitor:
    """Background thread that samples CPU and RSS memory during a trial.

    Samples are collected from the **current process only** (not system-wide)
    using :mod:`psutil`.  Because the process may spawn numba / OpenMP threads
    internally, ``cpu_percent`` can exceed 100 % — it is the sum across all
    logical cores used by this process.  On a 20-core machine a fully parallel
    DDE run will typically read ~2000 %.

    When ``fold_backend="sequential"`` (the default for ``objective``), all
    fold work happens inside this process so the monitor captures everything.
    If ``fold_backend="loky"`` were used, child worker processes would **not**
    be included; in that case consider wrapping each worker with its own
    monitor instead.

    The first ``cpu_percent`` sample is discarded if it is zero — psutil
    always returns 0.0 immediately after resetting the measurement window.

    Falls back gracefully when psutil is not installed.

    Usage::

        monitor = ResourceMonitor(interval=0.5).start()
        # ... run trial ...
        stats = monitor.stop()
        # {"runtime_cpu_pct_min": 42.1, "runtime_cpu_pct_mean": 78.3, ...
        #  "runtime_rss_mb_min":  310.2, ...}
    """

    def __init__(self, interval: float = 0.5) -> None:
        self.interval = interval
        self._cpu: List[float] = []
        self._rss_mb: List[float] = []
        self._thread: Optional[Any] = None
        self._stop_event: Optional[Any] = None
        try:
            import psutil as _ps
            self._psutil = _ps
            self._proc = _ps.Process()
        except ImportError:
            import logging
            logging.getLogger(__name__).warning(
                "psutil not installed; runtime resource monitoring disabled."
            )
            self._psutil = None
            self._proc = None

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():  # type: ignore[union-attr]
            try:
                if self._proc is not None:
                    self._cpu.append(self._proc.cpu_percent(interval=None))
                    self._rss_mb.append(self._proc.memory_info().rss / 1024 ** 2)
            except Exception:
                pass
            self._stop_event.wait(self.interval)  # type: ignore[union-attr]

    def start(self) -> "ResourceMonitor":
        """Start the background sampling thread and return ``self``."""
        if self._psutil is None:
            return self
        import threading
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        # Warm up cpu_percent (first call always returns 0.0)
        if self._proc is not None:
            self._proc.cpu_percent(interval=None)
        self._thread.start()
        return self

    def stop(self) -> Dict[str, Any]:
        """Stop sampling and return ``{metric_min, metric_mean, metric_max}`` stats."""
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

        import numpy as np

        def _stats(vals: List[float], key: str) -> Dict[str, Any]:
            if not vals:
                return {f"{key}_min": "N/A", f"{key}_mean": "N/A", f"{key}_max": "N/A"}
            arr = np.asarray(vals, dtype=float)
            # psutil cpu_percent() always returns 0.0 on the very first call after
            # the warmup reset; drop leading zeros so they don't distort min/mean.
            nonzero = arr[arr > 0.0]
            effective = nonzero if len(nonzero) > 0 else arr
            return {
                f"{key}_min":  round(float(effective.min()),  2),
                f"{key}_mean": round(float(effective.mean()), 2),
                f"{key}_max":  round(float(effective.max()),  2),
            }

        result: Dict[str, Any] = {}
        result.update(_stats(self._cpu,    "runtime_cpu_pct"))
        result.update(_stats(self._rss_mb, "runtime_rss_mb"))
        return result
