"""Simulate every beat once: the features of a run, shared by all its folds.

A run builds one reservoir, and every beat starts from the state the reservoir reaches
after its warm-up, so the features of a beat depend only on the beat and on the noise
drawn while it runs. Each beat is therefore simulated once and every fold fits its readout
on the cached rows (:func:`training.cross_validate.cross_validate_features`): the same
cross-validation as simulating the beats of every fold, for the cost of one pass over the
pool. The beats run in worker processes with one thread each, which keeps every core busy,
and each beat draws its noise from its own seed, so the features do not depend on how the
beats are shared between the processes.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from typing import Any

import numpy as np

# Beat scalings that use only the beat itself; scalings fitted on a training fold would
# make the features depend on the fold.
PER_BEAT_SCALERS = ("sequence_zscore", "sequence_minmax", "sequence_minmax_pos", "none")

_THREAD_SETTINGS = ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS")

_cached: dict[str, Any] = {}


def beat_seed(seed: int, index: int) -> int:
    """The noise seed of the beat at ``index``: a function of the run's seed and the beat."""
    return int(np.random.SeedSequence([int(seed), int(index)]).generate_state(1)[0])


def reseed_noise(node, seed: int) -> None:
    """Give ``node`` a new noise generator seeded with ``seed`` (same distribution settings).

    BioReservoir and ReservoirPy's reservoirs draw their noise from the
    ``noise_generator`` hyperparameter; nodes without one are left unchanged.
    """
    generator = node.hypers.get("noise_generator")
    if generator is None:
        return
    from reservoirpy.utils.random import rand_generator

    keywords = {k: v for k, v in generator.keywords.items() if k != "rng"}
    node.hypers["noise_generator"] = partial(generator.func, rng=rand_generator(seed),
                                             **keywords)


def _reservoir(params: dict[str, Any], seed: int):
    from optimization.reservoirs import build_reservoir

    key = repr((sorted(params.items()), seed))
    if _cached.get("key") != key:
        _cached.clear()
        _cached.update(key=key, node=build_reservoir(params, seed=seed)[0])
    return _cached["node"]


def _simulate_chunk(job) -> tuple[np.ndarray, np.ndarray]:
    from training.pipeline import _aggregate_states

    params, seed, noise_seed, mode, beats, indices = job
    node = _reservoir(params, seed)
    rows = []
    for beat, index in zip(beats, indices, strict=True):
        reseed_noise(node, beat_seed(noise_seed, index))
        x = np.asarray(beat, dtype=float).reshape(len(beat), -1)
        rows.append(_aggregate_states(np.asarray(node.run(x, reset=True)), mode, x).ravel())
    return indices, np.asarray(rows)


@contextmanager
def _one_thread_per_process() -> Iterator[None]:
    """Worker processes started inside this block use one numba, OpenMP and BLAS thread."""
    saved = {name: os.environ.get(name) for name in _THREAD_SETTINGS}
    os.environ.update(dict.fromkeys(_THREAD_SETTINGS, "1"))
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def simulate_beats(params: dict[str, Any], seed: int, X: np.ndarray, *, noise_seed: int,
                   aggregation: str = "final", window: int = 1, processes: int = 1,
                   first_index: int = 0) -> np.ndarray:
    """Features of every beat in ``X``, each simulated once from the warm-up state.

    Parameters
    ----------
    params, seed : the reservoir, as for :func:`optimization.reservoirs.build_reservoir`
        (with ``reservoir_type``).
    X : np.ndarray, shape (n, T) or (n, T, 1)
        Scaled beats.
    noise_seed : int
        Beat ``i`` of ``X`` draws its noise from ``beat_seed(noise_seed, first_index + i)``.
    aggregation, window : as for :func:`training.pipeline.train`.
    processes : int
        Worker processes (started fresh, one thread each); 1 simulates here.

    Returns
    -------
    np.ndarray, shape (n, n_features)
    """
    mode = f"last_N:{window}" if window > 1 else aggregation
    X = np.asarray(X)
    indices = np.arange(first_index, first_index + len(X))
    if processes <= 1 or len(X) < 2:
        return _simulate_chunk((params, seed, noise_seed, mode, X, indices))[1]
    chunks = [c for c in np.array_split(np.arange(len(X)), min(len(X), 4 * processes))
              if len(c)]
    jobs = [(params, seed, noise_seed, mode, X[c], indices[c]) for c in chunks]
    features = None
    with _one_thread_per_process():
        pool = mp.get_context("spawn").Pool(processes)
    try:
        for done, rows in pool.imap_unordered(_simulate_chunk, jobs):
            if features is None:
                features = np.empty((len(X), rows.shape[1]))
            features[done - first_index] = rows
    finally:
        pool.close()
        pool.join()
    return features
