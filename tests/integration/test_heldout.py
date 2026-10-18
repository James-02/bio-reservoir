"""Rebuilding the held-out splits from the data gives the recorded splits, and each split
loads without any beat of the v0.1 pool (needs data/ecg)."""

import json

import numpy as np
import pytest

from utils import heldout
from utils.preprocessing import dataset_sha256

DATA = heldout.dataset_path()

pytestmark = pytest.mark.skipif(not DATA.is_file(), reason="needs data/ecg/ecg_data.npz")


def test_rebuilt_draws_match_the_recorded_file():
    record = json.loads(heldout.SPLITS_FILE.read_text())
    assert dataset_sha256(DATA) == record["dataset"]["content_sha256"]
    try:
        draws, _, matched = heldout.build_draws(DATA)
    except FileNotFoundError as exc:
        pytest.skip(f"source CSVs not available: {exc}")
    assert matched == record["pool"]["matched_beats"]
    assert heldout.check(record, draws) == []


def test_each_split_loads_without_beats_of_the_v0_1_pool():
    record = json.loads(heldout.SPLITS_FILE.read_text())
    try:
        pool = heldout.load_pool()
    except FileNotFoundError as exc:
        pytest.skip(f"source CSVs not available: {exc}")
    for draw in record["draws"].values():
        X, Y, ids = heldout.load_split(draw["split"])
        counts = np.bincount(Y[:, 0, :].argmax(axis=1), minlength=5)
        assert {name: int(counts[int(c)]) for c, name in record["classes"].items()} == \
            draw["counts"]
        assert ids.tolist() == draw["beat_ids"]
        assert heldout.overlap(pool, X) == 0
