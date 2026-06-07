"""Convenience script to generate topology visualisations from a reservoir.

This script assumes you can construct your reservoir topology via
``generate_distance_weights`` from ``reservoir.topology``. It will generate a
set of useful plots for understanding the geometry and distance-based weights
of the reservoir.

Run from the project root, e.g.:

    python visualise_topology.py
"""

import argparse

from bioreservoir import distance_matrix
from utils.visualisation import (
    plot_distance_histograms,
    plot_weight_vs_distance_from_matrix,
    plot_spatial_topology,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise distance-based reservoir topology")
    parser.add_argument("--units", type=int, default=250, help="Number of reservoir nodes")
    parser.add_argument("--spatial-dim", type=int, default=2, help="Spatial dimensionality")
    parser.add_argument("--decay-mode", type=str, default="exp", help="Decay mode for weights")
    parser.add_argument("--decay-power", type=float, default=2.0, help="Decay power for weights")
    parser.add_argument("--include-self", action="store_true", help="Include self-connections")
    parser.add_argument("--sparsity", type=float, default=1.0, help="Connection sparsity")
    parser.add_argument("--gain", type=float, default=1.0, help="Global gain on weights")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--diameters", type=float, default=8.2,
                        help="Interaction distance in units of mean NN diameters")
    parser.add_argument("--alpha", type=float, default=0.01,
                        help="Target kernel weight at interaction distance (exp(-(R/ell)^2) = alpha)")

    args = parser.parse_args()

    W, coords = distance_matrix(
    # 1) Build one topology instance using the same logic as distance_matrix
        m=args.units,
        n=args.units,
        spatial_dim=args.spatial_dim,
        decay_mode=args.decay_mode,
        decay_power=args.decay_power,
        include_self=False,
        normalize="none",  # important: inspect raw kernel
        jitter=0.0,
        signed=False,
        cutoff_radius=None,
        sparsity=args.sparsity,
        seed=args.seed,
        gain=args.gain,
        interaction_diameters=args.diameters,
        fade_alpha=args.alpha,
        return_coords=True,
    )

    # 2) Pure distance statistics (independent of W)
    _ = plot_distance_histograms(
        units=args.units,
        seed=args.seed,
        spatial_dim=args.spatial_dim,
        filename="distance_histograms.png",
        show=False,
    )

    # 3) Weight vs distance from this particular matrix
    _ = plot_weight_vs_distance_from_matrix(
        W=W,
        coords=coords,
        diameters=args.diameters,
        filename="weight_vs_distance_matrix.png",
        show=False,
    )

    # 6) Spatial view with strongest edges
    if args.spatial_dim == 2:
        plot_spatial_topology(
            W=W,
            coords=coords,
            max_edges_per_node=5,
            thresh_quantile=0.9,
            filename="spatial_topology.png",
            show=False,
        )


if __name__ == "__main__":
    main()
