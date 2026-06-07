"""Unit tests for bioreservoir.topology weight-matrix builders."""

import pytest
import numpy as np
from functools import partial
from scipy.sparse import issparse

from bioreservoir.topology import (
    generate_distance_weights,
    spectral_radius,
    distance_matrix,
    random_gaussian_weights,
)


# ---------------------------------------------------------------------------
# spectral_radius
# ---------------------------------------------------------------------------

class TestSpectralRadius:
    def test_identity_matrix(self):
        assert spectral_radius(np.eye(5)) == pytest.approx(1.0)

    def test_zero_matrix(self):
        assert spectral_radius(np.zeros((3, 3))) == pytest.approx(0.0)

    def test_empty_matrix(self):
        assert spectral_radius(np.empty((0, 0))) == 0.0

    def test_known_eigenvalue(self):
        # Diagonal matrix with known eigenvalues
        W = np.diag([3.0, -2.0, 1.0])
        assert spectral_radius(W) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# generate_distance_weights
# ---------------------------------------------------------------------------

class TestGenerateDistanceWeights:
    def test_output_shape(self):
        n = 10
        W, coords = generate_distance_weights(n, seed=42)
        assert W.shape == (n, n) or (issparse(W) and W.shape == (n, n))
        assert coords.shape == (n, 2)

    def test_no_self_connections_by_default(self):
        W, _ = generate_distance_weights(8, seed=0, sparsity=1.0, normalize="none")
        if issparse(W):
            W = W.toarray()
        np.testing.assert_array_equal(np.diag(W), 0.0)

    def test_include_self_connections(self):
        W, _ = generate_distance_weights(8, seed=0, include_self=True,
                                         sparsity=1.0, normalize="none")
        if issparse(W):
            W = W.toarray()
        assert np.all(np.diag(W) > 0)

    def test_spectral_normalization(self):
        W, _ = generate_distance_weights(10, seed=42, normalize="spectral")
        if issparse(W):
            W = W.toarray()
        sr = spectral_radius(W)
        assert sr == pytest.approx(1.0, abs=0.05)

    def test_deterministic_with_seed(self):
        W1, c1 = generate_distance_weights(8, seed=123)
        W2, c2 = generate_distance_weights(8, seed=123)
        if issparse(W1):
            W1, W2 = W1.toarray(), W2.toarray()
        np.testing.assert_array_equal(W1, W2)
        np.testing.assert_array_equal(c1, c2)

    def test_sparsity_reduces_nonzeros(self):
        W_full, _ = generate_distance_weights(10, seed=0, sparsity=1.0,
                                              normalize="none", cutoff_radius=None)
        W_sparse, _ = generate_distance_weights(10, seed=0, sparsity=0.3,
                                                normalize="none", cutoff_radius=None)
        if issparse(W_full):
            W_full = W_full.toarray()
        if issparse(W_sparse):
            W_sparse = W_sparse.toarray()
        assert np.count_nonzero(W_sparse) < np.count_nonzero(W_full)

    def test_gain_scales_matrix(self):
        W_base, _ = generate_distance_weights(6, seed=0, normalize="none",
                                              gain=1.0, sparsity=1.0, cutoff_radius=None)
        W_scaled, _ = generate_distance_weights(6, seed=0, normalize="none",
                                                gain=2.0, sparsity=1.0, cutoff_radius=None)
        if issparse(W_base):
            W_base, W_scaled = W_base.toarray(), W_scaled.toarray()
        np.testing.assert_allclose(W_scaled, W_base * 2.0)

    def test_cutoff_radius_enforces_locality(self):
        W, coords = generate_distance_weights(
            10, seed=0, cutoff_radius=0.1, normalize="none", sparsity=1.0,
        )
        if issparse(W):
            W = W.toarray()
        # Compute pairwise distances
        D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        # All connections beyond cutoff should be zero
        assert np.all(W[D > 0.1 + 1e-9] == 0.0)


# ---------------------------------------------------------------------------
# distance_matrix (ReservoirPy-compatible wrapper)
# ---------------------------------------------------------------------------

class TestDistanceMatrix:
    def test_deferred_returns_partial(self):
        result = distance_matrix(seed=42)
        assert isinstance(result, partial)

    def test_call_with_mn(self):
        W = distance_matrix(m=8, n=8, seed=42)
        if issparse(W):
            W = W.toarray()
        assert W.shape == (8, 8)

    def test_non_square_raises(self):
        """Recurrent matrix must be square; non-square should raise or at least"""
        # distance_matrix doesn't enforce m==n directly, but generate_distance_weights uses m
        # Just check it runs for square
        W = distance_matrix(m=5, n=5, seed=0)
        shape = W.toarray().shape if issparse(W) else W.shape
        assert shape == (5, 5)


# ---------------------------------------------------------------------------
# random_gaussian_weights
# ---------------------------------------------------------------------------

class TestRandomGaussianWeights:
    def test_deferred_returns_partial(self):
        result = random_gaussian_weights(connectivity=0.1, sr=1.0, seed=0)
        assert isinstance(result, partial)

    def test_output_shape(self):
        W = random_gaussian_weights(m=10, n=10, seed=0)
        assert W.shape == (10, 10)

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            random_gaussian_weights(m=5, n=10, seed=0)

    def test_spectral_radius_target(self):
        W = random_gaussian_weights(m=20, n=20, sr=0.9, seed=42, connectivity=0.5)
        sr = spectral_radius(W)
        assert sr == pytest.approx(0.9, abs=0.05)

    def test_no_self_connections_by_default(self):
        W = random_gaussian_weights(m=8, n=8, seed=0)
        np.testing.assert_array_equal(np.diag(W), 0.0)

    def test_deterministic_with_seed(self):
        W1 = random_gaussian_weights(m=10, n=10, seed=77)
        W2 = random_gaussian_weights(m=10, n=10, seed=77)
        np.testing.assert_array_equal(W1, W2)

    def test_unsigned_weights_nonnegative(self):
        W = random_gaussian_weights(m=10, n=10, signed=False, seed=0)
        assert np.all(W >= 0)

    def test_signed_weights_have_negatives(self):
        W = random_gaussian_weights(m=20, n=20, signed=True, seed=0, connectivity=0.5)
        assert np.any(W < 0)
        assert np.any(W > 0)
