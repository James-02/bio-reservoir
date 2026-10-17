"""The sealed held-out draws file is complete and self-consistent (no data needed)."""

import hashlib
import json

import numpy as np
import pytest

from utils import heldout


@pytest.fixture(scope="module")
def record():
    return json.loads(heldout.DRAWS_FILE.read_text())


def test_checksums_match_recorded_beat_ids(record):
    for draw in record["draws"].values():
        ids = np.asarray(draw["beat_ids"], dtype=np.int64)
        assert hashlib.sha1(ids.tobytes()).hexdigest() == draw["sha1"]
        assert hashlib.sha256(ids.tobytes()).hexdigest() == draw["sha256"]
        assert sum(draw["counts"].values()) == len(ids)


def test_pilot_draws_are_unchanged(record):
    for k, sha1 in heldout.PILOT_SHA1.items():
        assert record["draws"][str(k)]["sha1"] == sha1


def test_draws_are_disjoint(record):
    sets = [set(draw["beat_ids"]) for draw in record["draws"].values()]
    assert sum(len(s) for s in sets) == len(set().union(*sets))


def test_draw_3_is_balanced_and_sealed(record):
    draw = record["draws"]["3"]
    assert len(set(draw["counts"].values())) == 1
    assert draw["status"].startswith("sealed")
