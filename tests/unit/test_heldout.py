"""Held-out draws: the recorded file is self-consistent, and the draw logic on synthetic data."""

import functools
import hashlib
import json

import numpy as np
import pytest

from utils import heldout
from utils.preprocessing import dataset_sha256

# Beats per class (0 N, 1 S, 2 V, 3 F, 4 Q); the first two of each class form the pool.
CLASS_SIZES = {0: 12, 1: 9, 2: 12, 3: 4, 4: 12}
PER_CLASS = (3, 3, None)


@pytest.fixture(scope="module")
def record():
    return json.loads(heldout.DRAWS_FILE.read_text())


def test_checksums_match_recorded_beat_ids(record):
    for draw in record["draws"].values():
        ids = np.asarray(draw["beat_ids"], dtype=np.int64)
        assert hashlib.sha1(ids.tobytes()).hexdigest() == draw["sha1"]
        assert hashlib.sha256(ids.tobytes()).hexdigest() == draw["sha256"]
        assert sum(draw["counts"].values()) == len(ids)


def test_draws_are_disjoint(record):
    sets = [set(draw["beat_ids"]) for draw in record["draws"].values()]
    assert sum(len(s) for s in sets) == len(set().union(*sets))


def test_draw_3_is_balanced_and_not_yet_evaluated(record):
    draw = record["draws"]["3"]
    assert len(set(draw["counts"].values())) == 1
    assert draw["status"] == heldout.STATUS[3]


def _zscore(X):
    sd = X.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - X.mean(axis=1, keepdims=True)) / sd


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    """A small ``ecg/ecg_data.npz`` with its pool, and a temporary draws file."""
    rng = np.random.default_rng(0)
    y = np.concatenate([np.full(n, c) for c, n in CLASS_SIZES.items()])
    X = rng.normal(size=(len(y), 8))
    pool = np.concatenate([np.flatnonzero(y == c)[:2] for c in CLASS_SIZES])
    path = tmp_path / "ecg" / "ecg_data.npz"
    path.parent.mkdir()
    np.savez(path, X=X, Y=y.reshape(-1, 1).astype(float))
    monkeypatch.setenv("BIORESERVOIR_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(heldout, "load_pool", lambda: _zscore(X[pool])[:, :, None])
    monkeypatch.setattr(heldout, "DRAWS_FILE", tmp_path / "heldout" / "draws.json")
    return path, y, pool


def test_draws_exclude_the_pool_and_class_f(dataset):
    path, y, pool = dataset
    draws, labels, matched = heldout.build_draws(path, PER_CLASS)
    ids = np.concatenate(list(draws.values()))
    assert matched == len(pool)
    assert np.array_equal(labels, y)
    assert len(set(ids)) == len(ids)
    assert not set(ids) & set(pool)
    assert not (y[ids] == 3).any()


def test_draw_sizes_follow_the_rule(dataset):
    path, y, _ = dataset
    draws, _, _ = heldout.build_draws(path, PER_CLASS)
    counts = {k: {c: int((y[ids] == c).sum()) for c in heldout.CLASSES}
              for k, ids in draws.items()}
    assert counts[1] == counts[2] == {0: 3, 1: 3, 2: 3, 4: 3}
    # Draw 3 takes what the limiting class has left: S has 9 - 2 (pool) - 6 = 1.
    assert counts[3] == {0: 1, 1: 1, 2: 1, 4: 1}


def test_draws_are_reproducible_from_the_seed(dataset):
    path, _, _ = dataset
    first, _, _ = heldout.build_draws(path, PER_CLASS)
    again, _, _ = heldout.build_draws(path, PER_CLASS)
    other, _, _ = heldout.build_draws(path, PER_CLASS, seed=heldout.SEED + 100)
    assert all(np.array_equal(first[k], again[k]) for k in first)
    assert not np.array_equal(first[1], other[1])


def test_record_round_trips_with_one_line_per_draw(dataset):
    path, _, _ = dataset
    draws, labels, matched = heldout.build_draws(path, PER_CLASS)
    record = heldout.describe(draws, labels, matched, path)
    assert record["dataset"]["content_sha256"] == dataset_sha256(path)
    assert record["pool"]["matched_beats"] == matched
    for k, ids in draws.items():
        draw = record["draws"][str(k)]
        assert draw["beat_ids"] == ids.tolist()
        assert draw["sha256"] == hashlib.sha256(ids.astype(np.int64).tobytes()).hexdigest()
        assert draw["status"] == heldout.STATUS[k]
    text = heldout.dumps(record)
    assert json.loads(text) == record
    for draw in record["draws"].values():
        assert f'"beat_ids": {json.dumps(draw["beat_ids"])}' in text


def test_check_reports_draws_that_differ_from_the_record(dataset):
    path, _, _ = dataset
    draws, labels, matched = heldout.build_draws(path, PER_CLASS)
    record = heldout.describe(draws, labels, matched, path)
    assert heldout.check(record, draws) == []
    record["draws"]["2"]["sha1"] = "0" * 40
    problems = heldout.check(record, draws)
    assert len(problems) == 1 and problems[0].startswith("draw 2:")


def test_load_pool_uses_the_v0_1_settings(monkeypatch):
    X = np.arange(36, dtype=float).reshape(12, 3, 1)
    calls = {}

    def fake_load(**kwargs):
        calls.update(kwargs)
        return X[:8], None, X[8:], None

    monkeypatch.setattr(heldout, "load_ecg_data", fake_load)
    pool = heldout.load_pool(instances=10, seed=1)
    assert (calls["rows"], calls["balance_classes"], calls["binary"]) == (10, True, False)
    assert (calls["scaler_type"], calls["seed"]) == ("sequence_zscore", 1)
    assert pool.dtype == np.float64
    assert np.array_equal(pool, X[np.sort(np.random.default_rng(1).permutation(12)[:10])])
    assert np.array_equal(heldout.load_pool(instances=12, seed=1), X)


@pytest.fixture
def cli(dataset, monkeypatch):
    monkeypatch.setattr(heldout, "build_draws",
                        functools.partial(heldout.build_draws, per_class=PER_CLASS))
    return dataset


def test_cli_writes_once_then_verifies(cli, capsys):
    path, _, _ = cli
    with pytest.raises(SystemExit, match="not found"):
        heldout.main(["--check"])
    heldout.main(["--write"])
    written = json.loads(heldout.DRAWS_FILE.read_text())
    assert written["dataset"]["content_sha256"] == dataset_sha256(path)
    with pytest.raises(SystemExit, match="never rewritten"):
        heldout.main(["--write"])
    heldout.main(["--check"])
    assert "held-out draws verified" in capsys.readouterr().out


def test_cli_check_fails_when_the_dataset_changes(cli):
    path, _, _ = cli
    heldout.main(["--write"])
    with np.load(path) as data:
        X, Y = data["X"].copy(), data["Y"]
    X[-1, 0] += 1.0
    np.savez(path, X=X, Y=Y)
    with pytest.raises(SystemExit, match="recorded dataset"):
        heldout.main(["--check"])
