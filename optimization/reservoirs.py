"""Reservoir builders shared by optimisation and rerun workflows.

This module keeps reservoir construction separate from the Optuna objective so
alternative reservoirs can reuse the same data, CV, readout, and deployment
pipeline. The default remains the biological DDE reservoir, but a standard
ReservoirPy leaky ESN can now be evaluated through the same workflow.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from reservoirpy.node import Node
from reservoirpy.nodes import Reservoir

from bioreservoir import BioReservoir, distance_matrix, random_gaussian_weights


def normalize_reservoir_type(value: str | None) -> str:
    """Normalize reservoir type aliases to canonical names."""
    raw = str(value or "bioreservoir").strip().lower()
    aliases = {
        "bio": "bioreservoir",
        "biological": "bioreservoir",
        "genetic": "bioreservoir",
        "bioreservoir": "bioreservoir",
        "esn": "esn",
        "echo-state-network": "esn",
        "echo_state_network": "esn",
    }
    if raw not in aliases:
        raise ValueError(
            f"Unknown reservoir_type {value!r}. Choose 'bioreservoir' or 'esn'."
        )
    return aliases[raw]


def _build_bioreservoir_topology(params: Dict[str, Any], seed: int):
    topo = params.get("topology_type", "distance")
    if topo == "distance":
        return distance_matrix(
            spatial_dim=2,
            interaction_diameters=float(params["interaction_diameters"]),
            fade_alpha=float(params["fade_alpha"]),
            sparsity=float(params["sparsity"]),
            signed=bool(params["signed"]),
            p_excite=float(params["p_excite"]),
            cutoff_radius="auto",
            seed=seed,
        )
    if topo == "random":
        rc_conn = params.get("rc_connectivity")
        connectivity = float(rc_conn) if rc_conn is not None else 1.0 - float(
            params.get("sparsity", 0.5)
        )
        return random_gaussian_weights(
            connectivity=max(1e-3, connectivity),
            sr=1.0,
            signed=bool(params["signed"]),
            p_excite=float(params.get("p_excite", 0.5)),
            seed=seed,
        )
    raise ValueError(
        f"Unknown topology_type {topo!r}. Choose 'distance' or 'random'."
    )


def build_reservoir(params: Dict[str, Any], seed: int) -> Tuple[Node, Dict[str, Any]]:
    """Build the requested reservoir and return reproducibility metadata."""
    reservoir_type = normalize_reservoir_type(params.get("reservoir_type", "bioreservoir"))
    units = int(params["units"])

    if reservoir_type == "bioreservoir":
        W = _build_bioreservoir_topology(params, seed)
        het_cv = float(params.get("param_heterogeneity_cv", 0.0))
        het_seed_raw = params.get("param_heterogeneity_seed")
        het_seed = int(het_seed_raw) if het_seed_raw is not None else None
        node = BioReservoir(
            units=units,
            warmup=int(params["warmup"]),
            rk4_substeps=int(params["rk4_substeps"]),
            rc_scaling=float(params["rc_scaling"]),
            input_scaling=float(params["input_scaling"]),
            input_connectivity=float(params["input_connectivity"]),
            input_bias=bool(params["input_bias"]),
            bias_scaling=float(params["bias_scaling"]),
            cell_coupling=float(params["cell_coupling"]),
            dde_scaling=float(params["dde_scaling"]),
            noise_in=float(params["noise_in"]),
            noise_rc=float(params["noise_rc"]),
            rk4_substep_aggregation_window=int(params["rk4_substep_aggregation"]),
            output_variables=str(params.get("output_variables", "all")),
            param_heterogeneity_cv=het_cv,
            param_heterogeneity_seed=het_seed,
            W=W,
            seed=seed,
        )
        attrs = {
            "reservoir_type": "bioreservoir",
            "res_units": units,
            "res_warmup": params["warmup"],
            "res_rk4_substeps": params["rk4_substeps"],
            "res_rk4_substep_agg": params["rk4_substep_aggregation"],
            "res_input_scaling": params["input_scaling"],
            "res_rc_scaling": params["rc_scaling"],
            "res_dde_scaling": params["dde_scaling"],
            "res_cell_coupling": params["cell_coupling"],
            "res_noise_in": params["noise_in"],
            "res_noise_rc": params["noise_rc"],
            "res_input_bias": params["input_bias"],
            "res_bias_scaling": params["bias_scaling"],
            "res_input_connectivity": params["input_connectivity"],
            "res_output_variables": params.get("output_variables", "all"),
            "res_heterogeneity_cv": het_cv,
            "res_heterogeneity_seed": het_seed,
            "topo_type": params["topology_type"],
            "topo_sparsity": params.get("sparsity"),
            "topo_rc_connectivity": params.get("rc_connectivity"),
            "topo_signed": params["signed"],
            "topo_p_excite": params["p_excite"],
            "topo_interaction_diameters": params.get("interaction_diameters"),
            "topo_fade_alpha": params.get("fade_alpha"),
            "topo_cutoff_radius": "auto",
            "topo_seed": seed,
            "esn_spectral_radius": None,
            "esn_leak_rate": None,
            "esn_activation": None,
        }
        return node, attrs

    node = Reservoir(
        units=units,
        lr=float(params.get("esn_leak_rate", 1.0)),
        sr=float(params.get("esn_spectral_radius", 0.95)),
        input_bias=bool(params["input_bias"]),
        noise_rc=float(params["noise_rc"]),
        noise_in=float(params["noise_in"]),
        input_scaling=float(params["input_scaling"]),
        bias_scaling=float(params["bias_scaling"]),
        input_connectivity=float(params["input_connectivity"]),
        rc_connectivity=float(params.get("rc_connectivity", 0.1)),
        activation=str(params.get("esn_activation", "tanh")),
        seed=seed,
    )
    attrs = {
        "reservoir_type": "esn",
        "res_units": units,
        "res_warmup": None,
        "res_rk4_substeps": None,
        "res_rk4_substep_agg": None,
        "res_input_scaling": params["input_scaling"],
        "res_rc_scaling": None,
        "res_dde_scaling": None,
        "res_cell_coupling": None,
        "res_noise_in": params["noise_in"],
        "res_noise_rc": params["noise_rc"],
        "res_input_bias": params["input_bias"],
        "res_bias_scaling": params["bias_scaling"],
        "res_input_connectivity": params["input_connectivity"],
        "res_output_variables": "state",
        "res_heterogeneity_cv": None,
        "res_heterogeneity_seed": None,
        "topo_type": "esn",
        "topo_sparsity": None,
        "topo_rc_connectivity": params.get("rc_connectivity", 0.1),
        "topo_signed": None,
        "topo_p_excite": None,
        "topo_interaction_diameters": None,
        "topo_fade_alpha": None,
        "topo_cutoff_radius": None,
        "topo_seed": seed,
        "esn_spectral_radius": params.get("esn_spectral_radius", 0.95),
        "esn_leak_rate": params.get("esn_leak_rate", 1.0),
        "esn_activation": params.get("esn_activation", "tanh"),
    }
    return node, attrs