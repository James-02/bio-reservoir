"""With the realism options on, beats simulated in worker processes get the features of one
process: each beat draws its reset perturbation from its own seed."""

import numpy as np
import pytest

from optimization.reservoirs import build_reservoir
from training.simulate import simulate_beats

PARAMS = {
    "reservoir_type": "bioreservoir", "units": 16, "warmup": 5, "rk4_substeps": 16,
    "input_scaling": 0.155113, "input_connectivity": 0.66, "input_bias": False,
    "bias_scaling": 0.1, "dde_scaling": 0.01, "cell_coupling": 12.75, "noise_in": 0.01442,
    "output_variables": "IHe", "topology_type": "lattice", "coupling": "diffusion",
    "diffusion_rate": 0.6, "input_encoding": "nonnegative", "input_sign": "positive",
    "param_heterogeneity_cv": 0.065, "param_heterogeneity_seed": 7,
}


@pytest.mark.parametrize("options", [
    {"reset_noise": 0.1},
    {"reset_noise": 0.03, "heterogeneity_density_cv": 0.2, "heterogeneity_dilution_cv": 0.2,
     "input_levels": "continuous"},
], ids=["reset", "all"])
def test_worker_processes_give_the_features_of_one_process_with_reset_noise(options):
    X = np.random.default_rng(0).normal(size=(9, 10, 1))
    reservoir = build_reservoir(dict(PARAMS, **options), seed=2)[0]
    here = simulate_beats(reservoir, X, noise_seed=5, aggregation="mean")
    workers = simulate_beats(reservoir, X, noise_seed=5, aggregation="mean", processes=3)
    np.testing.assert_array_equal(here, workers)
    plain = simulate_beats(build_reservoir(dict(PARAMS), seed=2)[0], X, noise_seed=5,
                           aggregation="mean")
    assert not np.allclose(here, plain)
