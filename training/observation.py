"""What a measurement of the reservoir sees: some of its variables, with measurement noise
and calibration error.

In a device the colonies' fluorescent reporter (LuxI) is imaged, so the readout sees one
variable of each colony rather than its whole state (``features.variables``), at one or a
few times (``features.aggregation``: ``images:K`` evenly spaced, or ``images_at:t1,t2,...`` at
chosen times). Each measured value is multiplied by its own
1 + sigma N(0, 1) (``features.measurement_noise``), drawn from a generator of the beat's own.
The noise applies to the feature row, after aggregation: an image is one noisy snapshot,
and an average over the beat (``mean``, ``last_N:K``) one noisy measurement of the average.
The colonies of a test fold are read with gains calibrated to a relative error
(``features.calibration_error``), which the readout never sees during training.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Third seed word of each stream, so its draws are independent of the beats' noise seeds.
_MEASUREMENT, _CALIBRATION = 1, 2


def observed_columns(output_variables: Sequence[str], variables: Sequence[str],
                     units: int) -> np.ndarray:
    """State columns of ``variables``, colony by colony.

    The reservoir outputs its ``output_variables`` colony by colony (colony 0's variables,
    then colony 1's, ...).
    """
    output_variables = list(output_variables)
    missing = [v for v in variables if v not in output_variables]
    if missing:
        raise ValueError(f"features.variables {missing} are not among the reservoir's output "
                         f"variables {output_variables}.")
    positions = np.array([output_variables.index(v) for v in variables])
    return (np.arange(units)[:, None] * len(output_variables) + positions).ravel()


def measure(features: np.ndarray, noise: float, seed: int, first_index: int = 0) -> np.ndarray:
    """``features`` measured with multiplicative noise of relative SD ``noise``.

    Row ``i`` holds beat ``first_index + i`` and is measured with a generator of its own, so
    a beat's measurement does not depend on the other beats or on the folds.
    """
    if not noise:
        return features
    measured = np.empty(features.shape)
    for row, values in enumerate(features):
        rng = np.random.default_rng([int(seed), first_index + row, _MEASUREMENT])
        measured[row] = values * (1.0 + noise * rng.standard_normal(values.shape))
    return measured


def calibration_gains(channels: int, images: int, error: float, seed: int,
                      fold: int) -> np.ndarray:
    """Gains of the ``channels`` (colony and variable) of a test fold, calibrated with relative
    error ``error``; a channel has the same gain in each of the ``images``."""
    rng = np.random.default_rng([int(seed), int(fold), _CALIBRATION])
    return np.tile(1.0 + error * rng.standard_normal(channels), images)


def image_times(aggregation: str) -> list[int]:
    """Times of the images of ``images_at:t1,t2,...``, in input steps after the beat starts
    (one step is one minute): positive integers in increasing order.

    The image at time t is the state after t steps.
    """
    text = aggregation.split(":", 1)[1] if aggregation.startswith("images_at:") else ""
    parts = text.split(",")
    if not all(p.strip().isdigit() for p in parts):
        raise ValueError(f"Invalid images_at mode {aggregation!r}; expected "
                         "'images_at:t1,t2,...' with whole minutes, e.g. 'images_at:155,187'.")
    times = [int(p) for p in parts]
    if times[0] < 1 or any(b <= a for a, b in zip(times[:-1], times[1:], strict=True)):
        raise ValueError(f"{aggregation!r}: the image times must be 1 or more and strictly "
                         "increasing.")
    return times


def image_count(aggregation: str) -> int:
    """Number of images in a feature row of ``aggregation``."""
    if aggregation.startswith("images_at:"):
        return len(image_times(aggregation))
    return int(aggregation.split(":")[1]) if aggregation.startswith("images:") else 1
