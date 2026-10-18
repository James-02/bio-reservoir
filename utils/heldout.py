"""Held-out splits: validation and test beats that no v0.1 optimisation study used.

The v0.1 studies drew every beat from one balanced pool of 4,015 beats.  The held-out
splits take N, S, V and Q beats (not F) from the remaining beats of
``data/ecg/ecg_data.npz``, in three seeded draws:

* ``validation-1`` and ``validation-2`` (draws 1 and 2, 803 beats per class each) have
  been evaluated during model development;
* ``test`` (draw 3) is the final test set, evaluated once at the end.  It is balanced,
  with as many beats per class as the limiting class allows, and was recorded before any
  evaluation.

A split is a list of row indices (beat IDs) into ``ecg_data.npz``, recorded with checksums
in ``data/splits/heldout.json`` (tracked in git, unlike the data).  :func:`load_split`
returns a split's beats, scaled like the training pool, and a run config evaluates on one
with ``test: {split: <name>}`` (see ``training.run``).  ``--check`` rebuilds the draws from
the data and verifies them against the record.  Never tune anything on a held-out split::

    python -m utils.heldout --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from utils import config
from utils.preprocessing import dataset_sha256, load_ecg_data, standardize_data

SEED = config.SEED
POOL_INSTANCES = 4015
CLASSES = {0: "N", 1: "S", 2: "V", 4: "Q"}
PER_CLASS = (803, 803, None)
SPLITS = {1: "validation-1", 2: "validation-2", 3: "test"}
STATUS = {1: "evaluated during model development", 2: "evaluated during model development",
          3: "final test set, not yet evaluated"}
SPLITS_FILE = config.PROJECT_ROOT / "data" / "splits" / "heldout.json"
# Scalers that act on each beat alone, so a split is scaled exactly like the pool.
PER_BEAT_SCALERS = ("sequence_zscore", "sequence_minmax", "sequence_minmax_pos", "none")


def _sha1(ids: np.ndarray) -> str:
    """SHA-1 fingerprint of the beat IDs, recorded next to the SHA-256 (not a security use)."""
    data = np.asarray(ids, dtype=np.int64).tobytes()
    return hashlib.sha1(data, usedforsecurity=False).hexdigest()  # guardrails-disable-line


def dataset_path() -> Path:
    return config.data_dir() / "ecg" / "ecg_data.npz"


def load_split(name: str, scaler_type: str = "sequence_zscore"
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The beats of a held-out split.

    Parameters
    ----------
    name : str
        ``"validation-1"``, ``"validation-2"`` or ``"test"``.
    scaler_type : str
        Per-beat scaler, as for the training pool (``utils.preprocessing``).

    Returns
    -------
    X : np.ndarray, shape (n, steps, 1)
        The scaled beats.
    Y : np.ndarray, shape (n, 1, n_classes)
        One-hot labels over the dataset's five classes.
    beat_ids : np.ndarray, shape (n,)
        Row indices of the beats in ``ecg_data.npz``.
    """
    if scaler_type not in PER_BEAT_SCALERS:
        raise ValueError(f"Held-out splits need a per-beat scaler "
                         f"({', '.join(PER_BEAT_SCALERS)}); got {scaler_type!r}.")
    record = json.loads(SPLITS_FILE.read_text())
    draws = {draw["split"]: draw for draw in record["draws"].values()}
    if name not in draws:
        raise ValueError(f"Unknown held-out split {name!r}; choose from {sorted(draws)}.")
    path = dataset_path()
    if dataset_sha256(path) != record["dataset"]["content_sha256"]:
        raise ValueError(f"{path} is not the dataset the held-out splits were drawn from.")
    ids = np.asarray(draws[name]["beat_ids"], dtype=np.int64)
    with np.load(path, allow_pickle=True) as data:
        X = np.asarray(data["X"], dtype=np.float64)
        y = np.asarray(data["Y"]).reshape(-1).astype(int)
    beats = X[ids].reshape(len(ids), -1, 1)
    _, beats = standardize_data(beats, beats, scaler_type=scaler_type)
    labels = np.eye(len(np.unique(y)))[y[ids]][:, None, :]
    return beats, labels, ids


def overlap(pool: np.ndarray, beats: np.ndarray, decimals: int = 6) -> int:
    """How many of *beats* also occur in *pool* (values compared after rounding)."""
    keys = {np.round(np.ravel(x), decimals).tobytes() for x in pool}
    return sum(np.round(np.ravel(x), decimals).tobytes() in keys for x in beats)


def load_pool(instances: int = POOL_INSTANCES, seed: int = SEED) -> np.ndarray:
    """Return the v0.1 balanced CV pool (as ``optimization.training`` builds it)."""
    X_tr, _, X_te, _ = load_ecg_data(
        rows=instances, balance_classes=True, max_per_class=None, binary=False,
        noise_rate=0.0, noise_ratio=0.0, scaler_type="sequence_zscore", seed=seed,
    )
    X = np.concatenate((X_tr, X_te))
    if len(X) > instances:
        X = X[np.sort(np.random.default_rng(seed).permutation(len(X))[:instances])]
    return np.asarray(X, dtype=np.float64)


def build_draws(dataset: Path, per_class: Sequence[int | None] = PER_CLASS,
                seed: int = SEED) -> tuple[dict[int, np.ndarray], np.ndarray, int]:
    """Return the draws (beat IDs), the labels and the number of pool beats matched.

    Beats are matched to the pool by their per-beat z-scored values rounded to 6
    decimals.  Draw *k* uses ``default_rng(seed + k)``; a per-class size of ``None``
    takes as many beats per class as the limiting class has left.
    """
    full = np.load(dataset, allow_pickle=True)
    X = np.asarray(full["X"], dtype=np.float64)
    y = np.asarray(full["Y"]).reshape(-1).astype(int)
    sd = X.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    Z = (X - X.mean(axis=1, keepdims=True)) / sd
    pool_keys = {np.round(x.ravel(), 6).tobytes() for x in load_pool()}
    used = np.array([np.round(z, 6).tobytes() in pool_keys for z in Z])
    matched = int(used.sum())
    draws = {}
    for k, size in enumerate(per_class, 1):
        available = {c: np.flatnonzero((y == c) & ~used) for c in CLASSES}
        n = size if size is not None else min(len(ids) for ids in available.values())
        rng = np.random.default_rng(seed + k)
        draws[k] = np.concatenate([rng.choice(available[c], min(n, len(available[c])),
                                              replace=False) for c in CLASSES])
        used[draws[k]] = True
    return draws, y, matched


def describe(draws: dict[int, np.ndarray], y: np.ndarray, matched: int, dataset: Path) -> dict:
    record = {
        "description": "Held-out beats unused by every v0.1 optimisation study "
                       "(utils/heldout.py)",
        "dataset": {"path": "data/ecg/ecg_data.npz", "content_sha256": dataset_sha256(dataset),
                    "beats": int(len(y))},
        "pool": {"instances": POOL_INSTANCES, "seed": SEED, "matched_beats": matched},
        "classes": {str(c): name for c, name in CLASSES.items()},
        "draws": {},
    }
    for k, ids in draws.items():
        ids = np.asarray(ids, dtype=np.int64)
        record["draws"][str(k)] = {
            "split": SPLITS.get(k, ""),
            "seed": SEED + k,
            "status": STATUS.get(k, ""),
            "counts": {CLASSES[c]: int((y[ids] == c).sum()) for c in CLASSES},
            "sha1": _sha1(ids),
            "sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
            "beat_ids": ids.tolist(),
        }
    return record


def dumps(record: dict) -> str:
    """JSON with one line per beat-ID list."""
    record = json.loads(json.dumps(record))
    lists = {}
    for k, draw in record["draws"].items():
        lists[f'"@beat_ids_{k}@"'] = json.dumps(draw["beat_ids"])
        draw["beat_ids"] = f"@beat_ids_{k}@"
    text = json.dumps(record, indent=1)
    for placeholder, value in lists.items():
        text = text.replace(placeholder, value)
    return text + "\n"


def check(record: dict, draws: dict[int, np.ndarray]) -> list[str]:
    """Return the rebuilt draws whose SHA-1 differs from the recorded draws."""
    problems = []
    for k, ids in draws.items():
        sha1 = _sha1(ids)
        if sha1 != record["draws"][str(k)]["sha1"]:
            problems.append(f"draw {k}: SHA-1 {sha1} differs from {SPLITS_FILE.name}")
    return problems


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m utils.heldout", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="Build the draws and record them.")
    mode.add_argument("--check", action="store_true", help="Rebuild and verify the draws.")
    args = parser.parse_args(argv)

    draws, y, matched = build_draws(dataset_path())
    recorded = json.loads(SPLITS_FILE.read_text()) if SPLITS_FILE.is_file() else {}
    if args.write:
        if recorded:
            raise SystemExit(f"{SPLITS_FILE} exists; recorded splits are never rewritten")
        SPLITS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SPLITS_FILE.write_text(dumps(describe(draws, y, matched, dataset_path())))
        print(f"wrote {SPLITS_FILE}")
    else:
        if not recorded:
            raise SystemExit(f"{SPLITS_FILE} not found")
        problems = check(recorded, draws)
        if dataset_sha256(dataset_path()) != recorded["dataset"]["content_sha256"]:
            problems.append(f"{dataset_path()} differs from the recorded dataset")
        if problems:
            raise SystemExit("\n".join(problems))
        print("held-out splits verified")
    for k, ids in draws.items():
        counts = {CLASSES[c]: int((y[ids] == c).sum()) for c in CLASSES}
        print(f"  {SPLITS[k]} (draw {k}): {len(ids)} beats {counts}  ({STATUS[k]})")


if __name__ == "__main__":
    main()
