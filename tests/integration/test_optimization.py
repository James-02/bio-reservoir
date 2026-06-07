"""Integration tests for the Optuna-based hyperparameter optimisation.

Run with::

    pytest tests/integration/ -m optuna
"""

import pytest


@pytest.mark.skip(reason="not yet implemented")
def test_single_optuna_trial_populates_user_attrs():
    """A single Optuna trial should finish and populate profiling user_attrs."""
    import optuna
    from optimization.training import objective  # adjust name as needed

    # TODO:
    # 1. Create an in-memory Optuna study
    # 2. Build a minimal dataset and pass it via closure / objective factory
    # 3. Run study.optimize(..., n_trials=1)
    # 4. Assert trial.user_attrs contains expected profiling keys


@pytest.mark.skip(reason="not yet implemented")
def test_optuna_pruning_integration():
    """Pruned trials should not raise unhandled exceptions."""
    # TODO: use a SuccessiveHalvingPruner, assert trial state is PRUNED not FAIL
