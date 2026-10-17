"""Golden master: the v0.1 model reproduces its stored outputs (see tests/golden_master.py)."""

import golden_master as gm
import numpy as np
import pytest


@pytest.fixture(scope="module")
def reference():
    if not gm.GOLDEN_FILE.is_file():
        pytest.fail(f"missing {gm.GOLDEN_FILE}; run: python tests/golden_master.py --write")
    return np.load(gm.GOLDEN_FILE)


def test_presets_match_archived_best_trials():
    assert gm.PRESETS["P2-distance"]["cell_coupling"] == 12.74579065885262
    assert gm.PRESETS["P2-random"]["topology_type"] == "random"


@pytest.mark.parametrize("name", sorted(gm.PRESETS))
def test_matches_golden_master(name, reference):
    for key, value in gm.golden_outputs(name).items():
        expected = reference[f"{name}__{key}"]
        assert value.shape == expected.shape, key
        np.testing.assert_allclose(value, expected, rtol=1e-6, atol=1e-9, err_msg=f"{name}: {key}")
