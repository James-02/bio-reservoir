"""Topology utilities for spatially-structured reservoirs.

This module contains helpers to build distance-based recurrent matrices
for biologically-inspired diffusion / signaling between nodes.

Public API:
    generate_distance_weights(...): returns (W, coords)
    spectral_radius(W): largest absolute eigenvalue
"""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from functools import partial
from scipy.sparse import csr_matrix

__all__ = ["generate_distance_weights", "spectral_radius", "distance_matrix", "random_gaussian_weights"]

def spectral_radius(mat: np.ndarray) -> float:
    """Return largest absolute eigenvalue (spectral radius)."""
    if mat.size == 0:
        return 0.0
    # Use scipy sparse eigsh for sparse matrices if available, otherwise numpy
    if hasattr(mat, 'toarray'):
        try:
            from scipy.sparse.linalg import eigs
            # k=1, which='LM' finds the eigenvalue with the largest magnitude
            vals = eigs(mat, k=1, which='LM', return_eigenvectors=False)
            return float(np.abs(vals[0]))
        except ImportError:
            # Fallback to dense if scipy.sparse.linalg is not available
            vals = np.linalg.eigvals(mat.toarray())
    else:
        vals = np.linalg.eigvals(mat)
        
    return float(np.max(np.abs(vals)))

def generate_distance_weights(
    units: int,
    *,
    spatial_dim: int = 2,
    length_scale: float = 0.1,
    decay_mode: str = "exp",
    decay_power: float = 2.0,
    include_self: bool = False,
    normalize: str = "spectral",
    sr: Optional[float] = None,
    jitter: float = 0.0,
    sparsity: float = 1.0,
    seed: Optional[int] = None,
    gain: float = 1.0,  # Final multiplicative factor applied to the matrix (useful to bake in rc scaling).
    cutoff_radius: Optional[float] = None,
    signed: bool = False,
    p_excite: float = 0.5,
    heterogeneous_gain: bool = False,
) -> Tuple[np.ndarray | csr_matrix, np.ndarray]:
    """Generate a distance-decay weight matrix and node coordinates.

    Returns
    -------
    W : numpy.ndarray or scipy.sparse.csr_matrix
        Weight matrix optionally sparsified.
    coords : numpy.ndarray of shape (units, spatial_dim)
        Node coordinates in [0,1]^spatial_dim.
    """
    rng = np.random.RandomState(seed)
    coords = rng.uniform(0.0, 1.0, size=(units, spatial_dim))

    diff = coords[:, None, :] - coords[None, :, :]
    D = np.linalg.norm(diff, axis=-1)

    if jitter > 0.0:
        D += rng.normal(0.0, jitter, size=D.shape)
        np.clip(D, 0.0, None, out=D)

    scaled = (D / max(length_scale, 1e-12)) ** decay_power

    if decay_mode == "exp":
        W = np.exp(-scaled)
    elif decay_mode == "inv":
        W = 1.0 / (1.0 + scaled)
    else:
        raise ValueError(f"Unknown decay_mode '{decay_mode}'.")

    if not include_self:
        np.fill_diagonal(W, 0.0)

    # Optional geometric cutoff to enforce locality and sparsity: zero
    # connections beyond a given interaction radius. This is useful when the
    # biological model suggests that effects are negligible beyond a certain
    # distance (e.g. a fixed number of cell diameters).
    if cutoff_radius is not None:
        mask_geo = D <= cutoff_radius
        W *= mask_geo.astype(W.dtype)

    if normalize == "spectral":
        # Computing spectral radius via eigendecomposition is expensive.
        # Allow callers to pass in a precomputed value when reusing the same
        # topology configuration repeatedly (e.g., Optuna trials).
        sr = float(sr) if sr is not None else spectral_radius(W)
        if sr > 0:
            W /= sr
    elif normalize == "row":
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        W /= row_sums
    elif normalize != "none":
        raise ValueError(f"Unknown normalize option '{normalize}'.")

    # Optional sign diversity: multiply by random sign matrix.
    # This preserves locality and distance-based magnitudes but introduces
    # a mix of excitatory/inhibitory interactions.
    if signed:
        # Draw signs with P(+1) = p_excite, P(-1) = 1 - p_excite
        sign_mask = (rng.rand(*W.shape) < p_excite).astype(W.dtype)
        sign_mask[sign_mask == 0] = -1.0
        W *= sign_mask

    # Optional heterogeneous per-node gain: multiply on the left by a
    # diagonal matrix diag(g), where g_i are node-specific gains.
    if heterogeneous_gain:
        g = rng.lognormal(mean=0.0, sigma=0.5, size=units).astype(W.dtype)
        W = (g[:, None] * W)

    if sparsity < 1.0:
        mask = rng.rand(*W.shape) < sparsity
        W *= mask

    # Convert to sparse if we introduced any structural sparsity (either via
    # geometric cutoff or random sparsity). This keeps matrix-vector products
    # efficient in large reservoirs.
    if cutoff_radius is not None or sparsity < 1.0:
        W = csr_matrix(W)

    if gain != 1.0:
        W *= gain

    return W, coords


def random_gaussian_weights(
    m: int = None,
    n: int = None,
    *,
    connectivity: float = 0.1,
    sr: float = 1.0,
    signed: bool = True,
    p_excite: float = 0.5,
    include_self: bool = False,
    seed: Optional[int] = None,
    **kwargs,
) -> np.ndarray | partial:
    """Gaussian sparse recurrent weight matrix, ReservoirPy-compatible.

    Draws weights from N(0, 1), applies sparsity, optionally strips negative
    weights (``signed=False``) or applies a random sign mask with
    ``p_excite`` fraction of excitatory connections (``signed=True``), then
    normalises to spectral radius ``sr``.

    When called without ``m``/``n`` (the standard ReservoirPy initialiser
    pattern), returns a :func:`functools.partial` that will receive
    ``(m, n)`` from :func:`reservoirpy.nodes.reservoirs.base.initialize`
    once the reservoir size is known.  This avoids the silent
    ``output_dim`` override that occurs when a pre-built matrix is passed
    directly to :class:`~bioreservoir.node.BioReservoir`.

    Parameters
    ----------
    m, n : int, optional
        Output shape.  Must be equal (square recurrent matrix).  If omitted,
        a deferred partial is returned.
    connectivity : float, default=0.1
        Fraction of non-zero entries (1 - sparsity).
    sr : float, default=1.0
        Target spectral radius after normalisation.
    signed : bool, default=True
        If True, draw signs independently with P(+1) = ``p_excite``.
        If False, all weights are made non-negative (excitatory only).
    p_excite : float, default=0.5
        Fraction of excitatory (+) connections when ``signed=True``.
    include_self : bool, default=False
        If True, allow non-zero diagonal (self-connections).
    seed : int, optional
        RNG seed for reproducibility.
    """
    if m is None and n is None:
        return partial(
            random_gaussian_weights,
            connectivity=connectivity,
            sr=sr,
            signed=signed,
            p_excite=p_excite,
            include_self=include_self,
            seed=seed,
        )

    if m != n:
        raise ValueError(f"Recurrent weight matrix must be square (got {m}x{n}).")

    rng = np.random.RandomState(seed)

    # Draw magnitudes from half-normal (|N(0,1)|), assign connectivity mask.
    magnitudes = np.abs(rng.randn(m, m))
    mask = rng.rand(m, m) < connectivity
    W = magnitudes * mask

    if not include_self:
        np.fill_diagonal(W, 0.0)

    if signed:
        # Each connection independently excitatory (+) or inhibitory (-).
        sign_mask = (rng.rand(m, m) < p_excite).astype(W.dtype)
        sign_mask[sign_mask == 0] = -1.0
        W = W * sign_mask
    # signed=False: W already non-negative from abs above.

    # Spectral-radius normalisation.
    # reservoirpy's initialize() always passes sr=<its own sr kwarg> when calling
    # the W initialiser, which may be None (BioReservoir doesn't set an sr on the
    # node level — it uses rc_scaling instead).  That explicit None overrides the
    # partial's captured sr=1.0, so we must guard here.
    sr = float(sr) if sr is not None else 1.0
    current_sr = spectral_radius(W)
    if current_sr > 0:
        W *= sr / current_sr

    return W


def distance_matrix(
    m: int = None,
    n: int = None,
    *,
    spatial_dim: int = 2,
    length_scale: float = 0.2,
    decay_mode: str = "exp",
    decay_power: float = 2.0,
    include_self: bool = False,
    normalize: str = "spectral",
    sr: Optional[float] = None,
    jitter: float = 0.0,
    sparsity: float = 0.1,
    seed: Optional[int] = None,
    return_coords: bool = False,
    gain: float = 1.0,
    # High-level biological parametrization: interaction range expressed in
    # "cell diameters" and an associated fade level. If provided, these will
    # be used to automatically calibrate ``length_scale`` and a sensible
    # geometric cutoff radius once the number of units (m) is known.
    interaction_diameters: Optional[float] = 8.2,
    fade_alpha: float = 0.01,
    cutoff_radius: Optional[float] | str = 'auto',
    signed: bool = True,
    p_excite: float = 0.5,
    heterogeneous_gain: bool = False,
    **kwargs,
) -> np.ndarray | csr_matrix | partial:
    """ReservoirPy-compatible initializer for distance-based recurrent weights.

    If ``m`` and ``n`` are not specified, returns a partial function for deferred
    matrix creation, which is the standard ReservoirPy initializer pattern.

    If you need the generated coordinates as well, either set ``return_coords=True``
    or call :func:`generate_distance_weights` directly.

    Parameters
    ----------
    m, n : int, optional
        Shape of the matrix. If not provided, returns a partial function.
        For recurrent matrices, ``m`` must equal ``n``.
    spatial_dim : int, default=2
        Dimensionality of the space where nodes are placed (e.g., 2 for a plane).
    length_scale : float, default=0.2
        Characteristic length controlling interaction decay. Smaller is more local.
    decay_mode : {'exp', 'inv'}, default='exp'
        Decay function type. 'exp' is Gaussian-like, 'inv' has a heavier tail.
    decay_power : float, default=2.0
        Exponent on the scaled distance, e.g., 2.0 for squared distance.
    include_self : bool, default=False
        If True, allows self-connections (non-zero diagonal).
    normalize : {'spectral', 'row', 'none'}, default='spectral'
        Normalization applied to the raw weight matrix.
    jitter : float, default=0.0
        Standard deviation of Gaussian noise added to distances for heterogeneity.
    sparsity : float, default=1.0
        If < 1.0, randomly set weights to zero to achieve this density.
    seed : int, optional
        Seed for all random number generation to ensure reproducibility.

    Returns
    -------
    np.ndarray or scipy.sparse.csr_matrix or functools.partial
        Weight matrix with a ``.coordinates`` attribute, or a partial function.
    """
    if m is None and n is None:
        # Standard ReservoirPy-style initializer: return a partially
        # configured function that will later receive (m, n) once the
        # reservoir size is known. We include all configuration parameters,
        # including the high-level interaction_diameters description if
        # provided.
        return partial(
            distance_matrix,
            spatial_dim=spatial_dim,
            length_scale=length_scale,
            decay_mode=decay_mode,
            decay_power=decay_power,
            include_self=include_self,
            normalize=normalize,
            sr=sr,
            jitter=jitter,
            sparsity=sparsity,
            seed=seed,
            gain=gain,
            interaction_diameters=interaction_diameters,
            fade_alpha=fade_alpha,
            cutoff_radius=cutoff_radius,
            signed=signed,
            p_excite=p_excite,
            heterogeneous_gain=heterogeneous_gain,
        )

    if m != n:
        raise ValueError(f"Distance matrix must be square (got {m}x{n}).")
    # If the user provided a biologically meaningful interaction range in
    # terms of cell diameters, calibrate the underlying length_scale and a
    # sensible cutoff radius for this specific reservoir size (m units).
    if interaction_diameters is not None:
        # Approximate one "diameter" as the mean nearest-neighbor distance in
        # the unit hypercube where nodes are placed. We re-sample coordinates
        # here for the sake of calibration; since this is only used to set
        # scales, small discrepancies with the final coordinates are
        # acceptable.
        rng = np.random.RandomState(seed)
        coords_cal = rng.uniform(0.0, 1.0, size=(m, spatial_dim))
        diff_cal = coords_cal[:, None, :] - coords_cal[None, :, :]
        D_cal = np.linalg.norm(diff_cal, axis=-1)
        D_sorted = np.sort(D_cal, axis=1)
        d1_mean = D_sorted[:, 1].mean()

        R = interaction_diameters * d1_mean
        # Ensure fade_alpha is in (0,1) to avoid numerical issues.
        eps = 1e-12
        alpha = np.clip(fade_alpha, eps, 1 - eps)
        length_scale = R / np.sqrt(-np.log(alpha))

        # Geometric cutoff behaviour:
        #   - cutoff_radius is None: no geometric clipping is applied.
        #   - cutoff_radius is 'auto': use the calibrated interaction
        #     distance R as cutoff.
        #   - cutoff_radius is a float: use that explicit radius.
        if isinstance(cutoff_radius, str):
            if cutoff_radius == "auto":
                cutoff_radius = R
            else:
                raise ValueError(
                    f"Unknown cutoff_radius string value '{cutoff_radius}'; "
                    "expected 'auto' or a float or None."
                )

    W, coords = generate_distance_weights(
        units=m,
        spatial_dim=spatial_dim,
        length_scale=length_scale,
        decay_mode=decay_mode,
        decay_power=decay_power,
        include_self=include_self,
        normalize=normalize,
        sr=sr,
        jitter=jitter,
        sparsity=sparsity,
        seed=seed,
        gain=gain,
        cutoff_radius=cutoff_radius,
        signed=signed,
        p_excite=p_excite,
        heterogeneous_gain=heterogeneous_gain,
    )

    if return_coords:
        return W, coords
    return W
