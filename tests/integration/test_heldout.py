"""Rebuilding the held-out draws from the data gives the recorded draws (needs data/ecg)."""

import json

import pytest

from utils import heldout
from utils.preprocessing import dataset_sha256

DATA = heldout.dataset_path()

pytestmark = pytest.mark.skipif(not DATA.is_file(), reason="needs data/ecg/ecg_data.npz")


def test_rebuilt_draws_match_the_recorded_file():
    record = json.loads(heldout.DRAWS_FILE.read_text())
    assert dataset_sha256(DATA) == record["dataset"]["content_sha256"]
    try:
        draws, _, matched = heldout.build_draws(DATA)
    except FileNotFoundError as exc:
        pytest.skip(f"source CSVs not available: {exc}")
    assert matched == record["pool"]["matched_beats"]
    assert heldout.check(record, draws) == []
