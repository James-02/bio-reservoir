"""Convenience script to generate topology visualisations from a reservoir.

This script assumes you can construct your reservoir topology via
``generate_distance_weights`` from ``reservoir.topology``. It will generate a
set of useful plots for understanding the geometry and distance-based weights
of the reservoir.

Run from the project root, e.g.:

    python visualise_topology.py
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from bioreservoir import distance_matrix
from utils.visualisation import _save_figure, use_v0_1_style


def plot_distance_histograms(units: int = 250,
                             seed: int = 0,
                             spatial_dim: int = 2,
                             bins: int = 50,
                             filename: str = "distance_histograms.png",
                             show: bool = True) -> dict[str, float]:
    """Visualise the distribution of pairwise and nearest-neighbour distances.

    This helps understand the geometry of the randomly generated node cloud
    before applying any kernel.

    Args:
        units: Number of nodes.
        seed: Random seed.
        spatial_dim: Spatial dimension for the embedding.
        bins: Number of histogram bins.
        filename: File name (relative to ``results/``) used for saving.
        show: Whether to display the plot interactively.

    Returns:
        Basic statistics of the nearest-neighbour distribution as a dict with
        keys ``mean``, ``std``, ``min`` and ``max``.
    """

    rng = np.random.RandomState(seed)
    coords = rng.uniform(0.0, 1.0, size=(units, spatial_dim))

    diff = coords[:, None, :] - coords[None, :, :]
    D = np.linalg.norm(diff, axis=-1)
    D_sorted = np.sort(D, axis=1)
    d1 = D_sorted[:, 1]

    # Stats for nearest-neighbour distances
    stats = {
        "mean": float(d1.mean()),
        "std": float(d1.std()),
        "min": float(d1.min()),
        "max": float(d1.max()),
    }

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.hist(D.flatten(), bins=bins, alpha=0.7)
    plt.xlabel("Pairwise distance")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    plt.hist(d1, bins=bins, alpha=0.7, color="tab:orange")
    plt.xlabel("Nearest-neighbour distance")
    plt.ylabel("Count")
    plt.axvline(stats["mean"], color="k", linestyle="--", label="mean")
    plt.legend()

    plt.tight_layout()
    _save_figure(filename=os.path.join("topology/", filename))
    if show:
        plt.show()

    return stats


def _compute_distance_matrices_from_coords(coords: np.ndarray) -> dict[str, np.ndarray]:
    """Utility: compute full and nearest-neighbour distance matrices from coords."""

    diff = coords[:, None, :] - coords[None, :, :]
    D = np.linalg.norm(diff, axis=-1)
    D_sorted = np.sort(D, axis=1)
    d1 = D_sorted[:, 1]
    return {"D": D, "d1": d1}


def plot_weight_vs_distance_from_matrix(
    W: np.ndarray,
    coords: np.ndarray,
    diameters: float = 8.2,
    filename: str = "weight_vs_distance.png",
    show: bool = True,
 ) -> dict[str, float]:
    """Scatter plot of distance vs. raw kernel weight from an existing matrix.

    This variant does *not* generate weights; it assumes you already have a
    distance-based weight matrix ``W`` and corresponding node coordinates
    ``coords``.

    Args:
        W: Weight matrix of shape ``(N, N)``.
        coords: Coordinates of shape ``(N, D)`` for the same nodes as in ``W``.
        diameters: Number of mean nearest-neighbour distances used to define
            the reference distance ``R = diameters * mean_nn``.
        filename: File name (relative to ``results/``) used for saving.
        show: Whether to display the plot interactively.

    Returns:
        Dictionary with basic distance statistics: ``mean_nn`` (mean nearest
        neighbour distance), ``R`` (reference distance) and ``diameters``.
    """

    mats = _compute_distance_matrices_from_coords(coords)
    D, d1 = mats["D"], mats["d1"]

    mean_nn = float(d1.mean())
    R = float(diameters * mean_nn)

    # Flatten and drop self-distances (diagonal). Support both dense and
    # scipy.sparse csr_matrix inputs for W.
    N = D.shape[0]
    off_diag = ~np.eye(N, dtype=bool)
    d_flat = D[off_diag]
    if hasattr(W, "toarray"):
        # Sparse matrix: convert to dense array for this diagnostic plot.
        W_dense = W.toarray()
        w_flat = W_dense[off_diag]
    else:
        w_flat = np.asarray(W)[off_diag]

    plt.figure(figsize=(8, 6))
    plt.scatter(d_flat, w_flat, s=3, alpha=0.2)
    plt.axvline(mean_nn, color="g", linestyle="dotted", label="~1 diameter (mean NN)")
    plt.axvline(R, color="r", linestyle="dashed", label=f"~{diameters:.1f} diameters")
    plt.xlabel("Distance")
    plt.ylabel("Weight (raw)")
    plt.legend()
    plt.tight_layout()

    _save_figure(filename=os.path.join("topology/", filename))
    if show:
        plt.show()

    return {"mean_nn": mean_nn, "R": R, "diameters": float(diameters)}


def plot_spatial_topology(
    W: np.ndarray,
    coords: np.ndarray,
    max_edges_per_node: int = 5,
    thresh_quantile: float = 0.9,
    filename: str = "spatial_topology.png",
    show: bool = True,
) -> None:
    """Plot nodes in space and overlay a subset of strongest edges.

    Edges are filtered by taking, for each node, up to ``max_edges_per_node``
    outgoing edges above the ``thresh_quantile`` of absolute weight.

    Args:
        W: Weight matrix of shape ``(N, N)``.
        coords: Coordinates of shape ``(N, 2)`` (2D plot).
        max_edges_per_node: Maximum number of strongest edges drawn per node.
        thresh_quantile: Quantile of absolute weights used as global threshold.
        filename: File name (relative to ``results/``) used for saving.
        show: Whether to display the plot interactively.
    """

    assert coords.shape[1] == 2, "plot_spatial_topology currently supports 2D coords only"

    N = W.shape[0]
    if hasattr(W, "toarray"):
        W_dense = W.toarray()
    else:
        W_dense = W

    abs_W = np.abs(W_dense)

    # Exclude diagonal when computing threshold
    abs_offdiag = abs_W.copy()
    np.fill_diagonal(abs_offdiag, 0.0)
    thresh = np.quantile(abs_offdiag[abs_offdiag > 0], thresh_quantile) if np.any(abs_offdiag > 0) else 0.0

    plt.figure(figsize=(6, 6))

    # Draw edges
    for i in range(N):
        row = np.array(abs_W[i, :]).ravel().copy()

        # Indices of edges above threshold
        candidates = np.where(row >= thresh)[0]
        # Keep up to max_edges_per_node strongest
        if len(candidates) > max_edges_per_node:
            top_idx = np.argsort(row[candidates])[-max_edges_per_node:]
            candidates = candidates[top_idx]

        for j in candidates:
            x = [coords[i, 0], coords[j, 0]]
            y = [coords[i, 1], coords[j, 1]]
            plt.plot(x, y, color="lightgray", linewidth=0.5)

    # Draw nodes on top
    plt.scatter(coords[:, 0], coords[:, 1], s=10, c="tab:blue")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Spatial topology (strong edges)")
    plt.tight_layout()

    _save_figure(filename=os.path.join("topology/", filename))
    if show:
        plt.show()


def main() -> None:
    use_v0_1_style()
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
