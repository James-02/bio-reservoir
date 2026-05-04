"""utils.results — typed container for CV fold artefacts.

Provides :class:`FoldArtifact`, a single source of truth for saving and loading
the outputs of a cross-validation run.  All other project modules that need to
persist or restore fold data import from here.

Directory layout
----------------
New (default)::

    results/runs/<trial_name>/fold-0.npz
    results/runs/<trial_name>/fold-1.npz
    ...

Legacy flat layout (read-only support for existing files)::

    results/runs/<trial_name>-fold-0.npz
    results/runs/<trial_name>-fold-1.npz
    ...

Usage
-----
Saving (inside ``training.cross_validate``)::

    artifact = FoldArtifact(
        trial_name="classification-42",
        fold_index=fold.fold_index,
        train_states=train_states,
        Y_train=Y_train,
        test_states=test_states,
        Y_test=Y_test,
        Y_pred=Y_pred,
        metrics=fold_metrics,
        model_hypers={**reservoir.hypers, **readout.hypers},
    )
    artifact.save()

Loading a full trial (by name)::

    folds = FoldArtifact.load_trial("classification-42")
    for fa in folds:
        print(fa.fold_index, fa.metrics["f1"])

Listing all saved trials::

    for name in FoldArtifact.list_trials():
        folds = FoldArtifact.load_trial(name)
        print(name, len(folds), "folds")
"""

from __future__ import annotations

import glob as _glob
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from utils.analysis import Metrics

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_RESULTS_DIR = "results/runs"
"""Default root directory for trial artefacts.

Can be overridden project-wide by setting the ``BIORESERVOIR_RESULTS_DIR``
environment variable, or per-call via the ``results_dir`` parameter.
"""


def _default_results_dir() -> str:
    return os.environ.get("BIORESERVOIR_RESULTS_DIR", DEFAULT_RESULTS_DIR)


# ---------------------------------------------------------------------------
# FoldArtifact
# ---------------------------------------------------------------------------

@dataclass
class FoldArtifact:
    """All artefacts produced by a single CV fold or single-split run.

    Attributes
    ----------
    trial_name : str
        Human-readable identifier for the parent trial, e.g.
        ``"classification-42"`` or ``"my-run"``.  Used as the sub-directory
        name under ``results_dir``.
    fold_index : int
        Zero-based fold index within the trial.  Use ``0`` for single-split
        (non-CV) runs.
    Y_train : np.ndarray
        Training labels (one-hot or class-index).  Always persisted.
    Y_test : np.ndarray
        Validation/test labels.  Always persisted.
    Y_pred : np.ndarray
        Readout predictions on the validation/test split.  Always persisted.
    metrics : Metrics
        Typed performance metrics for this fold.
    train_states : np.ndarray, optional
        Reservoir states extracted from the training split, shape
        ``(n_train, n_features)``.  Set ``save_states=True`` in :meth:`save`
        to write these to disk.  Required for:

        * PCA / t-SNE visualisations of the reservoir state space
          (``scripts/analysis/analyse_fold_results.py``, ``analyse_fold_diagnostics.py``)
        * Re-optimising the readout layer without re-running the reservoir
          (``optimization/readout.py``)

        Omitted by default because state matrices can be very large
        (e.g. 50 000 samples × 1 000 nodes × 4 variables ≈ 1.6 GB per fold).
    test_states : np.ndarray, optional
        Reservoir states for the validation/test split.  Same conditions as
        ``train_states``.
    model_hypers : dict
        Hyperparameters of the reservoir + readout used for this fold.
    profiling : dict, optional
        Per-phase wall-clock timing from :class:`training.profiling.Profiler`.
        **Not persisted to disk.**  Populated by ``run_fold`` when
        ``profile=True`` and discarded after averaging across folds in
        ``cross_validate``.
    path : str
        File path this artefact was loaded from (empty string if not yet
        saved or constructed in-memory).
    """

    trial_name: str
    fold_index: int
    Y_train: np.ndarray
    Y_test: np.ndarray
    Y_pred: np.ndarray
    metrics: Metrics
    # Optional — see docstring for when to enable
    train_states: Optional[np.ndarray] = None
    test_states: Optional[np.ndarray] = None
    model_hypers: Dict[str, Any] = field(default_factory=dict)
    # Ephemeral: timing data from Profiler; never written to disk
    profiling: Optional[Dict[str, Any]] = None
    path: str = ""

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------

    def save(self, results_dir: Optional[str] = None, save_states: bool = False) -> str:
        """Save this fold to ``{results_dir}/{trial_name}/fold-{fold_index}.npz``.

        Creates intermediate directories as needed.  Writes atomically via a
        temporary file so readers never observe a partially written NPZ.

        Parameters
        ----------
        results_dir : str, optional
            Root directory for trial artefacts.  Defaults to
            :data:`DEFAULT_RESULTS_DIR` (or ``BIORESERVOIR_RESULTS_DIR`` env var).
        save_states : bool, default False
            Whether to include ``train_states`` and ``test_states`` in the
            saved file.  Set to ``True`` when you intend to:

            * Run PCA or t-SNE visualisations on reservoir state trajectories
              (``analyse_fold_results.py``, ``analyse_fold_diagnostics.py``)
            * Re-optimise the readout layer without re-running the reservoir
              (``optimization/readout.py``)

            Omitted by default because state matrices can be very large
            (e.g. 50 000 samples × 1 000 nodes × 4 variables ≈ 1.6 GB per fold)
            and are not needed for comparing or reporting classification results.

        Returns
        -------
        str
            Absolute path of the written file.
        """
        if not self.trial_name:
            raise ValueError("trial_name must be set before calling save()")

        root = results_dir or _default_results_dir()
        out_dir = os.path.join(root, self.trial_name)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"fold-{self.fold_index}.npz")

        arrays: Dict[str, Any] = dict(
            trial_name  = np.bytes_(self.trial_name),
            fold_index  = np.array(self.fold_index, dtype=np.int32),
            Y_train     = self.Y_train,
            Y_test      = self.Y_test,
            Y_pred      = self.Y_pred,
            metrics     = np.array(self.metrics.to_dict(), dtype=object),
            model_hypers= np.array(self.model_hypers, dtype=object),
        )
        if save_states and self.train_states is not None:
            arrays["train_states"] = self.train_states
        if save_states and self.test_states is not None:
            arrays["test_states"] = self.test_states

        _atomic_npz_save(path, **arrays)
        self.path = path
        return path

    # -----------------------------------------------------------------------
    # Load — single file
    # -----------------------------------------------------------------------

    @classmethod
    def load(cls, path: str) -> "FoldArtifact":
        """Load a single fold artefact from an NPZ file.

        Supports both the new format (keys: ``trial_name``, ``fold_index``,
        ``model_hypers``) and the legacy format (key: ``model-hypers``).

        Parameters
        ----------
        path : str
            Path to the ``.npz`` file.

        Returns
        -------
        FoldArtifact
        """
        data = np.load(path, allow_pickle=True)

        metrics_dict = _extract_dict(data, ("metrics",))
        metrics = Metrics.from_dict(metrics_dict)
        model_hypers = _extract_dict(data, ("model_hypers", "model-hypers"))

        fold_index = int(data["fold_index"]) if "fold_index" in data.files else 0

        if "trial_name" in data.files:
            raw = data["trial_name"]
            raw_val = raw.item() if isinstance(raw, np.ndarray) else raw
            trial_name = raw_val.decode("utf-8") if isinstance(raw_val, bytes) else str(raw_val)
        else:
            stem = os.path.splitext(os.path.basename(path))[0]
            idx = stem.rfind("-fold-")
            trial_name = stem[:idx] if idx >= 0 else stem

        train_states = data["train_states"] if "train_states" in data.files else None
        test_states  = data["test_states"]  if "test_states"  in data.files else None

        return cls(
            trial_name  = trial_name,
            fold_index  = fold_index,
            Y_train     = data["Y_train"],
            Y_test      = data["Y_test"],
            Y_pred      = data["Y_pred"],
            metrics     = metrics,
            train_states= train_states,
            test_states = test_states,
            model_hypers= model_hypers,
            path        = path,
        )

    # -----------------------------------------------------------------------
    # Load — full trial
    # -----------------------------------------------------------------------

    @classmethod
    def load_trial(
        cls,
        trial_name: str,
        results_dir: Optional[str] = None,
    ) -> List["FoldArtifact"]:
        """Load all fold artefacts for a named trial.

        Searches for fold files in this order:

        1. ``{results_dir}/{trial_name}/fold-*.npz``   (new subdir layout)
        2. ``{results_dir}/{trial_name}-fold-*.npz``   (legacy flat layout)

        Parameters
        ----------
        trial_name : str
            The trial name string used when saving (e.g. ``"classification-42"``).
        results_dir : str, optional
            Root directory.  Defaults to :data:`DEFAULT_RESULTS_DIR`.

        Returns
        -------
        List[FoldArtifact]
            Artefacts sorted by ``fold_index``.

        Raises
        ------
        FileNotFoundError
            If no fold files are found in either layout.
        """
        root = results_dir or _default_results_dir()

        # New subdir layout
        subdir = os.path.join(root, trial_name)
        paths = sorted(_glob.glob(os.path.join(subdir, "fold-*.npz")))

        # Legacy flat layout
        if not paths:
            paths = sorted(_glob.glob(os.path.join(root, f"{trial_name}-fold-*.npz")))

        if not paths:
            raise FileNotFoundError(
                f"No fold files found for trial '{trial_name}'.\n"
                f"  Checked (new):    {os.path.join(subdir, 'fold-*.npz')}\n"
                f"  Checked (legacy): {os.path.join(root, trial_name + '-fold-*.npz')}"
            )

        artifacts = [cls.load(p) for p in paths]
        artifacts.sort(key=lambda a: a.fold_index)
        return artifacts

    # -----------------------------------------------------------------------
    # Discovery
    # -----------------------------------------------------------------------

    @classmethod
    def list_trials(cls, results_dir: Optional[str] = None) -> List[str]:
        """Return the names of all trials that have saved fold artefacts.

        Discovers both the new subdir layout and the legacy flat layout.

        Parameters
        ----------
        results_dir : str, optional
            Root directory.  Defaults to :data:`DEFAULT_RESULTS_DIR`.

        Returns
        -------
        List[str]
            Sorted list of trial names.
        """
        root = results_dir or _default_results_dir()
        names: set = set()

        if not os.path.isdir(root):
            return []

        # New subdir layout: directories containing at least one fold-*.npz
        for entry in os.scandir(root):
            if entry.is_dir():
                if _glob.glob(os.path.join(entry.path, "fold-*.npz")):
                    names.add(entry.name)

        # Legacy flat layout: infer trial name from *-fold-N.npz filenames
        for p in _glob.glob(os.path.join(root, "*-fold-*.npz")):
            stem = os.path.splitext(os.path.basename(p))[0]
            idx = stem.rfind("-fold-")
            if idx >= 0:
                names.add(stem[:idx])

        return sorted(names)

    # -----------------------------------------------------------------------
    # Convenience properties
    # -----------------------------------------------------------------------

    @property
    def f1(self) -> float:
        """F1 score from ``metrics``."""
        return self.metrics.f1

    @property
    def accuracy(self) -> float:
        """Accuracy from ``metrics``."""
        return self.metrics.accuracy

    @property
    def runtime(self) -> float:
        """Runtime in seconds from ``metrics``."""
        return self.metrics.runtime

    def __repr__(self) -> str:
        return (
            f"FoldArtifact(trial='{self.trial_name}', fold={self.fold_index}, "
            f"f1={self.f1:.4f}, path='{self.path}')"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _atomic_npz_save(path: str, **arrays: Any) -> None:
    """Write arrays to *path* atomically via a temp file + rename."""
    tmp = path + f".tmp-{random.randint(0, 99999)}"
    np.savez(tmp, **arrays)
    # np.savez appends .npz if not present
    src = tmp if os.path.exists(tmp) else tmp + ".npz"
    os.replace(src, path)


def _extract_dict(data: Any, keys: Sequence[str]) -> Dict[str, Any]:
    """Extract a dict stored as a 0-d object array under one of the given keys."""
    for k in keys:
        if k in data.files:
            v = data[k]
            if isinstance(v, np.ndarray) and v.shape == ():
                return v.item()
            return dict(v)
    return {}
