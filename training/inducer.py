"""Designed inducer-pattern tasks for slow sensing.

Each sample is an inducer-concentration trace of 240 input steps (one step is one minute,
so four hours) with a class label. The class lies in the timing, order or shape of the
pulses rather than in the mean input, so a time average of the input cannot solve the
tasks. Patterns (balanced classes, ``n_per_class`` traces each):

- ``freq`` (5 classes): pulse trains with periods of 30, 45, 60, 90 and 120 min, a 25% duty
  cycle, a random phase and amplitudes of 0.8-1.2.
- ``ramp`` (2): an asymmetric triangle, slow rise and fast fall or the reverse, then at
  least 40 min of baseline.
- ``order`` (2): pulse A (30 min at 0.5) then pulse B (10 min at 1.0), or B then A; both
  classes get the same two pulses, so the same dose.
- ``shape`` (4): a square, triangle, sawtooth or Gaussian pulse of amplitude 0.3-1.5.
- ``xor`` and ``parity3`` (2): two or three slots, each with an A or a B pulse; the class is
  the parity of the number of B pulses.
- ``xor_tr`` and ``parity3_tr`` (2): as ``xor`` and ``parity3``, with a rising or a falling
  20-min sawtooth in each slot (equal area, energy and peak), so that no statistic of the
  input values separates the classes.

Conditions: ``ideal`` adds N(0, 0.05) noise to the trace; ``device`` replays it as a syringe
pump would: the noisy trace held for 2 min at a time, on 16 levels up to 1.6, then mixed in
the chamber with a time constant of 3 min.

The traces are divided by the SD of the pool (``scale: sd``) or multiplied by a number.
``image_delay`` appends that many minutes of baseline (no inducer, delivered like the rest of
the trace) after the pattern, so the image is taken that long after the pattern ends: the
memory of the array. The SD is that of the patterns, so the scale does not change with the
image delay. A run config selects the task with ``task: {type: inducer, pattern: freq}`` (and
``condition``, ``n_per_class``, ``scale``, ``image_delay``, ``seed``); the beats need no
further scaling (``scaler: none``). Feed them to a non-negative reservoir input with
``input_range`` from 0, so that the zero baseline stays zero.
"""

from __future__ import annotations

from typing import Any

import numpy as np

LENGTH = 240
PATTERNS = ("freq", "ramp", "order", "shape", "xor", "parity3", "xor_tr", "parity3_tr")
#: The name of each class of each pattern, in label order (B: the short pulse, or the rising
#: sawtooth of the time-reversed pair).
CLASS_NAMES = {
    "freq": ("30 min", "45 min", "60 min", "90 min", "120 min"),
    "ramp": ("Slow rise", "Fast rise"),
    "order": ("A then B", "B then A"),
    "shape": ("Square", "Triangle", "Sawtooth", "Gaussian"),
    **dict.fromkeys(("xor", "parity3", "xor_tr", "parity3_tr"), ("Even B", "Odd B")),
}
CONDITIONS = ("ideal", "device")
TASK_DEFAULTS = {"condition": "ideal", "n_per_class": 400, "scale": "sd", "image_delay": 0}


def _place(x: np.ndarray, t: int, segment: np.ndarray) -> None:
    end = min(len(x), t + len(segment))
    x[t:end] += segment[: end - t]


def _pulse_a(amplitude: float) -> np.ndarray:
    return np.full(30, 0.5 * amplitude)


def _pulse_b(amplitude: float) -> np.ndarray:
    return np.full(10, 1.0 * amplitude)


def _triangle(rise: int, fall: int, amplitude: float) -> np.ndarray:
    return np.concatenate([np.linspace(0, amplitude, rise, endpoint=False),
                           np.linspace(amplitude, 0, fall + 1)])


def _freq(rng, n):
    traces, labels = [], []
    for label, period in enumerate((30, 45, 60, 90, 120)):
        width = int(round(period / 4))
        for _ in range(n):
            x = np.zeros(LENGTH)
            for start in range(-int(rng.integers(0, period)), LENGTH, period):
                a, b = max(start, 0), min(start + width, LENGTH)
                if b > a:
                    x[a:b] = rng.uniform(0.8, 1.2)
            traces.append(x)
            labels.append(label)
    return traces, labels


def _ramp(rng, n):
    traces, labels = [], []
    for label in (0, 1):
        for _ in range(n):
            slow, fast = int(rng.integers(40, 81)), int(rng.integers(3, 9))
            amplitude = rng.uniform(0.7, 1.3)
            segment = (_triangle(slow, fast, amplitude) if label == 0
                       else _triangle(fast, slow, amplitude))
            x = np.zeros(LENGTH)
            _place(x, int(rng.integers(10, LENGTH - 40 - len(segment))), segment)
            traces.append(x)
            labels.append(label)
    return traces, labels


def _order(rng, n):
    traces, labels = [], []
    for label in (0, 1):
        for _ in range(n):
            a, b = _pulse_a(rng.uniform(0.85, 1.15)), _pulse_b(rng.uniform(0.85, 1.15))
            first, second = (a, b) if label == 0 else (b, a)
            x = np.zeros(LENGTH)
            t1 = int(rng.integers(10, 51))
            t2 = t1 + len(first) + int(rng.integers(20, 61))
            _place(x, t1, first)
            _place(x, t2, second)
            traces.append(x)
            labels.append(label)
    return traces, labels


def _shape(rng, n):
    traces, labels = [], []
    for label in range(4):
        for _ in range(n):
            duration = int(rng.integers(40, 81))
            amplitude = rng.uniform(0.3, 1.5)
            t = np.arange(duration)
            if label == 0:
                segment = np.full(duration, amplitude)
            elif label == 1:
                segment = amplitude * (1 - np.abs(2 * t / (duration - 1) - 1))
            elif label == 2:
                segment = amplitude * t / (duration - 1)
            else:
                segment = amplitude * np.exp(-0.5 * ((t - (duration - 1) / 2)
                                                     / (duration / 6)) ** 2)
            x = np.zeros(LENGTH)
            _place(x, int(rng.integers(10, LENGTH - duration - 30)), segment)
            traces.append(x)
            labels.append(label)
    return traces, labels


def _slots(rng, n, centres, reversed_pair=False):
    traces, labels = [], []
    count = {0: 0, 1: 0}
    while min(count.values()) < n:
        bits = rng.integers(0, 2, len(centres))
        label = int(bits.sum() % 2)
        if count[label] >= n:
            continue
        x = np.zeros(LENGTH)
        for bit, centre in zip(bits, centres, strict=True):
            if reversed_pair:
                ramp = np.linspace(0.0, 1.0, 20) * rng.uniform(0.85, 1.15)
                segment = ramp if bit else ramp[::-1]
            else:
                segment = (_pulse_b(rng.uniform(0.85, 1.15)) if bit
                           else _pulse_a(rng.uniform(0.85, 1.15)))
            _place(x, int(centre + rng.integers(-10, 11) - len(segment) // 2), segment)
        traces.append(x)
        labels.append(label)
        count[label] += 1
    return traces, labels


GENERATORS = {
    "freq": _freq, "ramp": _ramp, "order": _order, "shape": _shape,
    "xor": lambda rng, n: _slots(rng, n, (45, 125)),
    "parity3": lambda rng, n: _slots(rng, n, (40, 100, 160)),
    "xor_tr": lambda rng, n: _slots(rng, n, (45, 125), reversed_pair=True),
    "parity3_tr": lambda rng, n: _slots(rng, n, (40, 100, 160), reversed_pair=True),
}


def ideal(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The designed trace with N(0, 0.05) noise."""
    return x + rng.normal(0, 0.05, len(x))


def device(x: np.ndarray, rng: np.random.Generator, hold: int = 2, levels: int = 16,
           tau_mix: float = 3.0, vmax: float = 1.6) -> np.ndarray:
    """The trace as a syringe pump delivers it: noisy, held for ``hold`` steps, on
    ``levels`` levels up to ``vmax``, and mixed in the chamber with time constant
    ``tau_mix``."""
    x = x + rng.normal(0, 0.05, len(x))
    x = np.repeat(x[::hold], hold)[: len(x)]
    x = np.round(np.clip(x, 0, vmax) / vmax * (levels - 1)) / (levels - 1) * vmax
    mixed = np.zeros_like(x)
    for t in range(1, len(x)):
        mixed[t] = mixed[t - 1] + (x[t] - mixed[t - 1]) / tau_mix
    return mixed


def resolve_task(task: dict[str, Any], seed: int) -> dict[str, Any]:
    """The explicit task section of an inducer task; refuses unknown settings."""
    unknown = sorted(set(task) - {"type", "pattern", "seed", *TASK_DEFAULTS})
    if unknown:
        raise ValueError(f"An inducer task has no setting {', '.join(unknown)}; it takes "
                         f"pattern, {', '.join(TASK_DEFAULTS)} and seed.")
    resolved = {"type": "inducer", **TASK_DEFAULTS, **task, "scaler": "none"}
    resolved.setdefault("seed", seed)
    if resolved.get("pattern") not in PATTERNS:
        raise ValueError(f"task.pattern must be one of {', '.join(PATTERNS)}.")
    if resolved["condition"] not in CONDITIONS:
        raise ValueError(f"task.condition must be one of {', '.join(CONDITIONS)}.")
    if int(resolved["n_per_class"]) < 5:
        raise ValueError("task.n_per_class must be at least 5 (five folds).")
    if resolved["scale"] != "sd" and not float(resolved["scale"]) > 0:
        raise ValueError("task.scale must be 'sd' or a positive number.")
    image_delay = resolved["image_delay"]
    if int(image_delay) != image_delay or int(image_delay) < 0:
        raise ValueError("task.image_delay must be a whole number of minutes, at least 0.")
    return resolved


def load(task: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list]:
    """The traces (n, 240 + image_delay, 1) and one-hot labels (n, 1, classes) of an inducer task.

    The pattern draws come from ``(seed, pattern)`` and the condition's noise from
    ``(seed, pattern, condition)``, so each task is the same in every run with its seed.
    """
    seed, pattern = int(task["seed"]), str(task["pattern"])
    key = PATTERNS.index(pattern)
    traces, labels = GENERATORS[pattern](np.random.default_rng([seed, key]),
                                         int(task["n_per_class"]))
    baseline = np.zeros(int(task.get("image_delay", 0)))
    noise = np.random.default_rng([seed, key, 1 + CONDITIONS.index(task["condition"])])
    deliver = ideal if task["condition"] == "ideal" else device
    X = np.stack([deliver(np.concatenate([x, baseline]), noise) for x in traces])
    X = X / X[:, :LENGTH].std() if task["scale"] == "sd" else X * float(task["scale"])
    labels = np.asarray(labels)
    Y = np.eye(labels.max() + 1)[labels][:, None, :]
    return X[..., None], Y, []
