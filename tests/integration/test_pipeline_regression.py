"""Pipeline regression test: 2-fold CV of the P2-distance configuration (N = 200, 600 real
beats, KNN readout) stays within 0.02 macro-F1 of v0.1. Needs data/ecg; takes minutes.
See tests/regression_reference.py."""

import json

import pytest
import regression_reference as ref

from utils import config

DATA = config.data_dir() / "ecg" / "ecg_data.npz"

pytestmark = pytest.mark.skipif(not DATA.is_file() or not ref.PIPELINE_REFERENCE.is_file(),
                                reason="needs data/ecg/ecg_data.npz and the pipeline reference")


def test_pipeline_macro_f1_matches_v0_1(tmp_path, monkeypatch):
    monkeypatch.setenv("BIORESERVOIR_RESULTS_DIR", str(tmp_path))
    reference = json.loads(ref.PIPELINE_REFERENCE.read_text())
    metrics = ref.pipeline_metrics("P2-distance")
    expected = reference["P2-distance"]["f1_macro"]
    assert abs(metrics["f1_macro"] - expected) <= reference["tolerance"], (metrics, expected)
