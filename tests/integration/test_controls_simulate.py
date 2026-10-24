"""The digital controls give the same features in worker processes as in one process."""

import numpy as np
import pytest

from optimization.reservoirs import build_reservoir
from training.simulate import simulate_beats


@pytest.mark.parametrize("params", [
    {"reservoir_type": "random_features", "units": 16, "length": 10},
    {"reservoir_type": "random_features", "units": 16, "length": 10, "activation": "linear"},
    {"reservoir_type": "leaky_bank", "units": 6, "tau_max": 32.0},
    {"reservoir_type": "leaky_bank", "units": 6, "expansion": 20},
])
def test_worker_processes_give_the_features_of_one_process(params):
    X = np.random.default_rng(0).normal(size=(9, 10, 1))
    control = build_reservoir(params, seed=2)[0]
    here = simulate_beats(control, X, noise_seed=5, aggregation="images:3")
    workers = simulate_beats(control, X, noise_seed=5, aggregation="images:3", processes=3)
    np.testing.assert_array_equal(here, workers)
    # The controls draw no noise: a fresh build gives the same features with any noise seed.
    again = simulate_beats(build_reservoir(params, seed=2)[0], X, noise_seed=6,
                           aggregation="images:3")
    np.testing.assert_array_equal(here, again)
