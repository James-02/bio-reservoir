"""P-centre regression test: with the realism options absent or at zero, the P centre
reproduces the outputs of the code before them. See tests/realism_reference.py."""

import numpy as np
import pytest
import realism_reference as ref

ZERO = {"heterogeneity_density_cv": 0.0, "heterogeneity_dilution_cv": 0.0, "input_levels": 1,
        "reset_noise": 0.0}


@pytest.fixture(scope="module")
def reference():
    if not ref.REFERENCE.is_file():
        pytest.fail(f"missing {ref.REFERENCE}; run tests/realism_reference.py --write")
    return np.load(ref.REFERENCE)


@pytest.mark.parametrize("options", [{}, ZERO], ids=["absent", "zero"])
def test_p_centre_outputs_match_the_code_before_the_realism_options(options, reference):
    for key, value in ref.p_centre_outputs(**options).items():
        expected = reference[key]
        assert value.shape == expected.shape, key
        np.testing.assert_allclose(value, expected, rtol=1e-6, atol=1e-9, err_msg=key)


def test_the_realism_options_change_the_p_centre_outputs(reference):
    changed = ref.p_centre_outputs(heterogeneity_density_cv=0.1, reset_noise=0.01)
    assert not np.allclose(changed["features"], reference["features"], rtol=1e-6, atol=1e-9)
