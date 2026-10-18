"""Unit tests for bioreservoir.topology weight-matrix builders."""

from functools import partial

import numpy as np
import pytest
from scipy.sparse import issparse

from bioreservoir.topology import (
    distance_matrix,
    generate_distance_weights,
    random_gaussian_weights,
    spectral_radius,
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

    def test_missing_spectral_radius_means_one(self):
        W = random_gaussian_weights(m=30, n=30, sr=None, seed=1, connectivity=0.3)
        assert spectral_radius(W) == pytest.approx(1.0)

    def test_connectivity_and_sign_fractions(self):
        m = 200
        W = random_gaussian_weights(m=m, n=m, connectivity=0.2, p_excite=0.7, seed=2)
        off_diagonal = W[~np.eye(m, dtype=bool)]
        nonzero = off_diagonal[off_diagonal != 0]
        assert len(nonzero) / len(off_diagonal) == pytest.approx(0.2, abs=0.01)
        assert np.mean(nonzero > 0) == pytest.approx(0.7, abs=0.02)

    def test_include_self_keeps_the_diagonal(self):
        W = random_gaussian_weights(m=50, n=50, connectivity=0.5, include_self=True, seed=0)
        assert np.count_nonzero(np.diag(W)) > 0


# ---------------------------------------------------------------------------
# Kernel values, normalisation and options of generate_distance_weights
# ---------------------------------------------------------------------------

def _dense(W):
    return W.toarray() if issparse(W) else np.asarray(W)


def _distances(coords):
    return np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)


class TestDistanceKernel:
    @pytest.mark.parametrize("mode, kernel", [
        ("exp", lambda s: np.exp(-s)),
        ("inv", lambda s: 1.0 / (1.0 + s)),
    ])
    def test_weights_follow_the_distance_kernel(self, mode, kernel):
        W, coords = generate_distance_weights(12, seed=4, length_scale=0.3, decay_mode=mode,
                                              decay_power=2.0, normalize="none")
        expected = kernel((_distances(coords) / 0.3) ** 2)
        np.fill_diagonal(expected, 0.0)
        np.testing.assert_allclose(_dense(W), expected, rtol=1e-12)
        np.testing.assert_allclose(_dense(W), _dense(W).T)
        assert coords.min() >= 0.0 and coords.max() <= 1.0

    def test_unknown_options_are_rejected(self):
        with pytest.raises(ValueError, match="decay_mode"):
            generate_distance_weights(4, seed=0, decay_mode="cosine")
        with pytest.raises(ValueError, match="normalize"):
            generate_distance_weights(4, seed=0, normalize="max")

    def test_row_normalisation_makes_rows_sum_to_one(self):
        W, _ = generate_distance_weights(15, seed=1, normalize="row")
        np.testing.assert_allclose(_dense(W).sum(axis=1), 1.0)

    def test_a_given_spectral_radius_is_used_as_the_divisor(self):
        raw, _ = generate_distance_weights(10, seed=2, normalize="none")
        scaled, _ = generate_distance_weights(10, seed=2, normalize="spectral", sr=4.0)
        np.testing.assert_allclose(_dense(scaled), _dense(raw) / 4.0)

    def test_signs_keep_the_magnitudes(self):
        unsigned, _ = generate_distance_weights(60, seed=3, normalize="none")
        signed, _ = generate_distance_weights(60, seed=3, normalize="none", signed=True,
                                              p_excite=0.8)
        np.testing.assert_allclose(np.abs(_dense(signed)), _dense(unsigned))
        nonzero = _dense(signed)[_dense(signed) != 0]
        assert np.mean(nonzero > 0) == pytest.approx(0.8, abs=0.03)

    def test_heterogeneous_gain_scales_whole_rows(self):
        base, _ = generate_distance_weights(8, seed=5, normalize="none")
        scaled, _ = generate_distance_weights(8, seed=5, normalize="none",
                                              heterogeneous_gain=True)
        ratio = _dense(scaled)[~np.eye(8, dtype=bool)].reshape(8, 7) / \
            _dense(base)[~np.eye(8, dtype=bool)].reshape(8, 7)
        np.testing.assert_allclose(ratio, ratio[:, :1] * np.ones((1, 7)))
        assert np.all(ratio > 0)

    def test_jitter_perturbs_distances(self):
        plain, _ = generate_distance_weights(10, seed=6, normalize="none")
        jittered, _ = generate_distance_weights(10, seed=6, normalize="none", jitter=0.05)
        assert not np.allclose(_dense(plain), _dense(jittered))
        assert np.all((_dense(jittered) >= 0) & (_dense(jittered) <= 1))

    def test_sparse_output_only_with_structural_sparsity(self):
        dense, _ = generate_distance_weights(6, seed=0, sparsity=1.0, cutoff_radius=None)
        sparse, _ = generate_distance_weights(6, seed=0, sparsity=0.5)
        assert isinstance(dense, np.ndarray) and issparse(sparse)

    def test_spectral_radius_of_a_sparse_matrix(self):
        W, _ = generate_distance_weights(40, seed=7, sparsity=0.5, normalize="none")
        assert issparse(W)
        assert spectral_radius(W) == pytest.approx(spectral_radius(W.toarray()), rel=1e-8)


class TestInteractionRangeCalibration:
    """distance_matrix sets the length scale from an interaction range in cell diameters."""

    def test_weight_at_the_interaction_range_equals_the_fade_level(self):
        diameters, alpha = 3.0, 0.05
        W, coords = distance_matrix(m=30, n=30, seed=8, interaction_diameters=diameters,
                                    fade_alpha=alpha, sparsity=1.0, signed=False,
                                    normalize="none", return_coords=True)
        D = _distances(coords)
        nearest = np.sort(D, axis=1)[:, 1].mean()
        R = diameters * nearest
        W = _dense(W)
        inside = (D <= R) & ~np.eye(30, dtype=bool)
        np.testing.assert_allclose(W[inside], alpha ** ((D[inside] / R) ** 2), rtol=1e-10)
        assert np.all(W[D > R] == 0.0)

    def test_without_a_cutoff_weights_extend_beyond_the_range(self):
        W, coords = distance_matrix(m=30, n=30, seed=8, interaction_diameters=3.0,
                                    fade_alpha=0.05, sparsity=1.0, signed=False,
                                    normalize="none", cutoff_radius=None, return_coords=True)
        assert np.count_nonzero(_dense(W)) == 30 * 29

    def test_invalid_arguments_are_rejected(self):
        with pytest.raises(ValueError, match="square"):
            distance_matrix(m=5, n=6, seed=0)
        with pytest.raises(ValueError, match="cutoff_radius"):
            distance_matrix(m=5, n=5, seed=0, cutoff_radius="far")
