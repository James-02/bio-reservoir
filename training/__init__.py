"""training — reservoir training pipeline, cross-validation, and profiling.

Public API
----------
train, fit_readout, predict
    Per-instance pipeline functions.

cross_validate, run_fold, classify
    Fold-level and study-level runners.

Profiler, ResourceMonitor
    Optional timing / resource instrumentation.
"""

from training.cross_validate import classify, cross_validate, run_fold
from training.pipeline import fit_readout, predict, train
from training.profiling import Profiler, ResourceMonitor

__all__ = [
    "train",
    "fit_readout",
    "predict",
    "cross_validate",
    "run_fold",
    "classify",
    "Profiler",
    "ResourceMonitor",
]
