"""Model regression test: with fixed synthetic beats, the reservoir (solver, node, topology,
noise, state aggregation) reproduces the v0.1 outputs. See tests/regression_reference.py."""

import numpy as np
import pytest
import regression_reference as ref


@pytest.fixture(scope="module")
def reference():
    if not ref.MODEL_REFERENCE.is_file():
        pytest.fail(f"missing {ref.MODEL_REFERENCE}; run tests/regression_reference.py --write")
    return np.load(ref.MODEL_REFERENCE)


def test_presets_match_archived_best_trials():
    assert ref.PRESETS["P2-distance"]["cell_coupling"] == 12.74579065885262
    assert ref.PRESETS["P2-random"]["topology_type"] == "random"


@pytest.mark.parametrize("name", sorted(ref.PRESETS))
def test_model_outputs_match_v0_1(name, reference):
    for key, value in ref.model_outputs(name).items():
        expected = reference[f"{name}__{key}"]
        assert value.shape == expected.shape, key
        np.testing.assert_allclose(value, expected, rtol=1e-6, atol=1e-9, err_msg=f"{name}: {key}")
