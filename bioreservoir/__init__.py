"""bioreservoir — genetic oscillator reservoir computing library.

This package implements a biological reservoir computer based on the
Danino et al. (2010) quorum-sensing oscillator model.

Primary public API
------------------
BioReservoir
    The main ReservoirPy Node subclass. Instantiate with ``units``,
    ``warmup``, ``input_scaling``, ``dde_scaling``, etc.

distance_matrix
    Topology generator: 2-D spatial layout with exponential distance-decay
    weights and configurable sparsity / spectral normalisation.
    This is the biologically motivated topology used in all paper results.

random_gaussian_weights
    Gaussian sparse recurrent weight matrix for the ESN ablation baseline
    (``topology_type='random'``).  ReservoirPy-compatible callable pattern;
    supports ``signed``, ``p_excite``, ``connectivity``, and spectral-radius
    normalisation natively in this library.

solve_dde_rk4
    Low-level RK4 Method-of-Steps DDE solver (Numba-compiled).

get_dde_params
    Return the full DDE parameter dict (with defaults filled in).
"""

from bioreservoir.node import BioReservoir
from bioreservoir.topology import distance_matrix, random_gaussian_weights
from bioreservoir.dde import solve_dde_rk4, get_dde_params, generate_node_params

__all__ = [
    "BioReservoir",
    "distance_matrix",
    "solve_dde_rk4",
    "get_dde_params",
    "generate_node_params",
]
