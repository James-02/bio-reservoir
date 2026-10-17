"""Statistical sentinel for the v0.1 model (N = 200, 600 real beats, 2 folds): macro-F1
stays within 0.02 of the stored reference.  Needs data/ecg (skipped without it); minutes."""

import json

import golden_master as gm
import pytest

from utils import config

DATA = config.PROJECT_ROOT / "data" / "ecg" / "ecg_data.npz"

pytestmark = pytest.mark.skipif(not DATA.is_file() or not gm.SENTINEL_FILE.is_file(),
                                reason="needs data/ecg/ecg_data.npz and the sentinel reference")


def test_sentinel_macro_f1(tmp_path, monkeypatch):
    monkeypatch.setenv("BIORESERVOIR_RESULTS_DIR", str(tmp_path))
    reference = json.loads(gm.SENTINEL_FILE.read_text())
    metrics = gm.sentinel_metrics("P2-distance")
    expected = reference["P2-distance"]["f1_macro"]
    assert abs(metrics["f1_macro"] - expected) <= reference["tolerance"], (metrics, expected)
