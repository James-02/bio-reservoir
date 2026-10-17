"""Rebuilding the held-out draws from the data gives the sealed draws (needs data/ecg)."""

import json

import pytest

from utils import config, heldout

DATA = config.PROJECT_ROOT / heldout.DATASET

pytestmark = pytest.mark.skipif(not DATA.is_file(), reason="needs data/ecg/ecg_data.npz")


def test_rebuilt_draws_match_sealed_file():
    record = json.loads(heldout.DRAWS_FILE.read_text())
    assert heldout.sha256_file(DATA) == record["dataset"]["sha256"]
    try:
        draws, _, matched = heldout.build_draws(DATA)
    except FileNotFoundError as exc:
        pytest.skip(f"source CSVs not available: {exc}")
    assert matched == record["pool"]["matched_beats"]
    assert heldout.check(record, draws) == []
