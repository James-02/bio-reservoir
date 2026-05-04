"""Integration tests for the cross-validation pipeline.

These tests run a full (miniaturized) cross-validation to confirm that
the pipeline assembles correctly end-to-end. They are slower than unit
tests and are kept in the `integration/` sub-package so they can be
run selectively with::

    pytest tests/integration/
"""

import pytest
import numpy as np


@pytest.mark.skip(reason="not yet implemented")
def test_cross_validate_two_folds():
    """cross_validate() should complete on a tiny dataset with n_folds=2."""
    from training.cross_validate import cross_validate

    # TODO:
    # 1. Build a minimal labelled dataset (e.g. 20 samples, 2 classes, short series)
    # 2. Construct a BioReservoir with a very small number of units
    # 3. Call cross_validate(..., n_folds=2)
    # 4. Assert returned metrics dict contains expected keys (accuracy, f1, …)


@pytest.mark.skip(reason="not yet implemented")
def test_cross_validate_profile_flag():
    """cross_validate(profile=True) should populate profiling information."""
    from training.cross_validate import cross_validate

    # TODO: same tiny dataset, assert returned profile keys are present and > 0
