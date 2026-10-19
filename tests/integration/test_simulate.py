"""Beats simulated once in worker processes get the features of one process."""

import numpy as np

from optimization.reservoirs import build_reservoir
from training.simulate import simulate_beats

PARAMS = {
    "reservoir_type": "bioreservoir", "units": 4, "warmup": 3, "rk4_substeps": 16,
    "input_scaling": 1.0, "input_connectivity": 1.0, "input_bias": False, "bias_scaling": 0.0,
    "rc_scaling": 0.1, "dde_scaling": 0.01, "cell_coupling": 1.0, "noise_in": 0.2,
    "noise_rc": 0.2, "output_variables": "IHe", "topology_type": "distance",
    "interaction_diameters": 3.0, "fade_alpha": 0.05, "sparsity": 1.0, "signed": False,
    "p_excite": 1.0,
}


def test_worker_processes_give_the_features_of_one_process():
    X = np.random.default_rng(0).normal(size=(9, 10, 1))
    reservoir = build_reservoir(PARAMS, seed=2)[0]
    here = simulate_beats(reservoir, X, noise_seed=5, aggregation="mean")
    workers = simulate_beats(reservoir, X, noise_seed=5, aggregation="mean", processes=3)
    np.testing.assert_array_equal(here, workers)
