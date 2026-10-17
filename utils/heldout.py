"""Held-out beat draws: beats that no v0.1 optimisation phase used.

The v0.1 studies drew every beat from one balanced pool of 4,015 beats.  Held-out draws
take N, S, V and Q beats (not F) from the remaining beats of ``data/ecg/ecg_data.npz``:

* draws 1 and 2 (803 beats per class each) were consulted during the revision pilots;
* draw 3 is the confirmatory set (revision plan X6).  It is balanced, with as many beats
  per class as the limiting class allows, and is built and sealed before any evaluation.

A draw is a list of row indices (beat IDs) into ``ecg_data.npz``.  ``--write`` records the
IDs with checksums in ``experiments/heldout/draws.json``; ``--check`` rebuilds the draws
and verifies them against that file.  Never tune anything on a held-out draw::

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
from utils.preprocessing import dataset_sha256, load_ecg_data

SEED = config.SEED
POOL_INSTANCES = 4015
CLASSES = {0: "N", 1: "S", 2: "V", 4: "Q"}
PER_CLASS = (803, 803, None)
PILOT_SHA1 = {1: "14b37f4db9e4a2ae262f9b869ca696e7ad19c397",
              2: "c2a6e3bb52c3e8916618c87cf83f83e875294e77"}
STATUS = {1: "consulted during the revision pilots", 2: "consulted during the revision pilots",
          3: "sealed: confirmatory, not evaluated"}
DRAWS_FILE = config.PROJECT_ROOT / "experiments" / "heldout" / "draws.json"


def dataset_path() -> Path:
    return config.data_dir() / "ecg" / "ecg_data.npz"


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
        "description": "Held-out beats unused by every v0.1 optimisation phase "
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
            "seed": SEED + k,
            "status": STATUS.get(k, ""),
            "counts": {CLASSES[c]: int((y[ids] == c).sum()) for c in CLASSES},
            "sha1": hashlib.sha1(ids.tobytes()).hexdigest(),
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
    """Return mismatches between rebuilt draws, the recorded draws and the pilot SHA-1s."""
    problems = []
    for k, ids in draws.items():
        sha1 = hashlib.sha1(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()
        if k in PILOT_SHA1 and sha1 != PILOT_SHA1[k]:
            problems.append(f"draw {k}: SHA-1 {sha1} differs from the pilot draw {PILOT_SHA1[k]}")
        if record and sha1 != record["draws"][str(k)]["sha1"]:
            problems.append(f"draw {k}: SHA-1 {sha1} differs from {DRAWS_FILE.name}")
    return problems


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m utils.heldout", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="Build the draws and record them.")
    mode.add_argument("--check", action="store_true", help="Rebuild and verify the draws.")
    args = parser.parse_args(argv)

    draws, y, matched = build_draws(dataset_path())
    recorded = json.loads(DRAWS_FILE.read_text()) if DRAWS_FILE.is_file() else {}
    if args.write:
        if recorded:
            raise SystemExit(f"{DRAWS_FILE} exists; draws are sealed and never rewritten")
        problems = check({}, draws)
        if problems:
            raise SystemExit("\n".join(problems))
        DRAWS_FILE.parent.mkdir(parents=True, exist_ok=True)
        DRAWS_FILE.write_text(dumps(describe(draws, y, matched, dataset_path())))
        print(f"wrote {DRAWS_FILE}")
    else:
        if not recorded:
            raise SystemExit(f"{DRAWS_FILE} not found")
        problems = check(recorded, draws)
        if dataset_sha256(dataset_path()) != recorded["dataset"]["content_sha256"]:
            problems.append(f"{dataset_path()} differs from the recorded dataset")
        if problems:
            raise SystemExit("\n".join(problems))
        print("held-out draws verified")
    for k, ids in draws.items():
        counts = {CLASSES[c]: int((y[ids] == c).sum()) for c in CLASSES}
        print(f"  draw {k}: {len(ids)} beats {counts}  ({STATUS[k]})")


if __name__ == "__main__":
    main()
