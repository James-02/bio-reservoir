"""Digital controls for the colony reservoir: random features of the beat and a bank of leaky
integrators.

The controls ask whether the colony array does more than a
generic nonlinear feature map of the beat (:class:`RandomFeatures`) or a bank of timescales
(:class:`LeakyBank`). Both are ReservoirPy nodes with one state per input step, so they run
through the same pipeline as the colony reservoir: every beat from an empty history, the same
folds, observation (``features.aggregation``, ``window``, ``measurement_noise``,
``calibration_error``), readouts and statistics. A run config selects them with
``reservoir.type: random_features`` or ``leaky_bank`` (:func:`build_control`); the readout
then sees their states where it would see the colonies' reporter levels.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from reservoirpy.node import Node

CONTROL_TYPES = ("random_features", "leaky_bank")

# Length of an MIT-BIH beat, in samples.
BEAT_LENGTH = 187

ACTIVATIONS = {"tanh": np.tanh, "linear": lambda a: a}

# Settings of each control besides units; a resolved run config lists them all.
CONTROL_DEFAULTS: dict[str, dict[str, Any]] = {
    "random_features": {"length": BEAT_LENGTH, "activation": "tanh", "input_scaling": 1.0,
                        "bias_scaling": 1.0},
    "leaky_bank": {"tau_min": 1.0, "tau_max": 256.0, "band_pass": True, "expansion": None,
                   "input_range": [-3.0, 8.0]},
}


class _SequenceMemory(Node):
    """A node whose state at step t depends on the inputs up to t through a memory of its own.

    Within a sequence, ReservoirPy resets a node to its own current state object before every
    step (and when ``run`` continues without ``reset``); any other reset (``run(...,
    reset=True)``, ``initialize``, a start from another state) begins a new sequence, with an
    empty memory. The test is on the object, not its values: a state can be exactly zero
    within a sequence.
    """

    def _clear_memory(self) -> None:
        raise NotImplementedError

    def reset(self, to_state: np.ndarray | None = None) -> Node:
        if to_state is None or to_state is not self.__dict__.get("_state"):
            self._clear_memory()
        return super().reset(to_state=to_state)


def _random_features_initialize(node: RandomFeatures, x: np.ndarray | None = None,
                                y: np.ndarray | None = None) -> None:
    if x is None:
        return
    node.set_input_dim(x.shape[1])
    inputs = int(node.hypers["length"]) * x.shape[1]
    rng = np.random.default_rng(node.seed)
    W = rng.normal(0.0, float(node.hypers["input_scaling"]) / np.sqrt(inputs),
                   size=(inputs, int(node.hypers["units"])))
    bias_scaling = float(node.hypers["bias_scaling"])
    bias = rng.uniform(-bias_scaling, bias_scaling, size=int(node.hypers["units"]))
    node.set_param("W", W.astype(node.dtype))
    node.set_param("bias", bias.astype(node.dtype).reshape(1, -1))


def _random_features_forward(node: RandomFeatures, x: np.ndarray) -> np.ndarray:
    length, step = int(node.hypers["length"]), node._step
    if step >= length:
        raise ValueError(f"RandomFeatures holds {length} input steps (length); the sequence "
                         "is longer. Set length to the sequence length.")
    if node._drive is None:
        node._drive = np.zeros((1, int(node.hypers["units"])), dtype=node.dtype)
    d = x.shape[1]
    # W h_t = W h_(t-1) + (rows of W for step t) x_t: the history enters only through W h_t.
    node._drive = node._drive + x.reshape(1, d) @ node.W[step * d:(step + 1) * d]
    node._step = step + 1
    return ACTIVATIONS[node.hypers["activation"]](node._drive + node.bias)


class RandomFeatures(_SequenceMemory):
    r"""Random features of the input seen so far (an extreme learning machine of the beat).

    At input step t the node holds the inputs seen so far, :math:`h_t = (x_1, \dots, x_t, 0,
    \dots, 0)`, zero-padded to ``length`` steps, and outputs

    .. math:: s_t = f(W h_t + b), \quad W_{ij} \sim N(0, g^2 / (L d)), \quad
              b_i \sim U(-\beta, \beta),

    with :math:`L` = ``length``, :math:`d` the input dimension, :math:`g` = ``input_scaling``,
    :math:`\beta` = ``bias_scaling`` and :math:`f` = ``activation``. At the last step of a beat
    of ``length`` samples, :math:`s_L = f(W z + b)` is the random feature map of the whole beat
    :math:`z`; earlier steps see only the beat so far, so images over the beat are causal.

    Parameters
    ----------
    units : int
        Number of random features.
    length : int, default 187
        Input steps the node holds (the beat length, in samples). A longer sequence raises.
    activation : {"tanh", "linear"}
        ``"linear"`` gives a random linear map of the beat.
    input_scaling : float, default 1.0
        :math:`g`: the SD of each pre-activation is :math:`g` times the RMS of the inputs.
    bias_scaling : float, default 1.0
        Half-width :math:`\beta` of the uniform bias.
    seed : int, optional
        Seed of :math:`W` and :math:`b` (drawn in this order from ``numpy.random.default_rng``).
    """

    def __init__(self, units: int, length: int = BEAT_LENGTH, activation: str = "tanh",
                 input_scaling: float = 1.0, bias_scaling: float = 1.0,
                 seed: int | None = None, input_dim: int | None = None,
                 name: str | None = None):
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation {activation!r}; choose from "
                             f"{sorted(ACTIVATIONS)}.")
        if int(units) < 1 or int(length) < 1:
            raise ValueError("RandomFeatures needs units >= 1 and length >= 1.")
        self.seed = seed
        self._drive: np.ndarray | None = None
        self._step = 0
        super().__init__(
            params={"W": None, "bias": None},
            hypers={"units": int(units), "length": int(length), "activation": activation,
                    "input_scaling": float(input_scaling), "bias_scaling": float(bias_scaling),
                    "output_variables": "state"},
            forward=_random_features_forward,
            initializer=_random_features_initialize,
            output_dim=int(units), input_dim=input_dim, name=name)

    def _clear_memory(self) -> None:
        self._drive, self._step = None, 0


def leaky_bank_size(units: int, band_pass: bool = True) -> int:
    """Features of a bank of ``units`` integrators of one input, before any expansion."""
    return 2 * units - 1 if band_pass else units


def _leaky_bank_initialize(node: LeakyBank, x: np.ndarray | None = None,
                           y: np.ndarray | None = None) -> None:
    if x is None:
        return
    node.set_input_dim(x.shape[1])
    hypers = node.hypers
    bank = x.shape[1] * leaky_bank_size(int(hypers["units"]), bool(hypers["band_pass"]))
    taus = np.geomspace(float(hypers["tau_min"]), float(hypers["tau_max"]), int(hypers["units"]))
    node.set_param("rate", (1.0 / taus).astype(node.dtype).reshape(1, -1))
    expansion = hypers["expansion"]
    if expansion is None:
        node.set_output_dim(bank)
        return
    rng = np.random.default_rng(node.seed)
    W = rng.normal(0.0, 1.0 / np.sqrt(bank), size=(bank, int(expansion)))
    node.set_param("W", W.astype(node.dtype))
    node.set_output_dim(int(expansion))


def _leaky_bank_forward(node: LeakyBank, x: np.ndarray) -> np.ndarray:
    low, high = node.hypers["input_range"]
    u = np.clip(x.reshape(-1, 1), low, high) - low                 # (d, 1)
    if node._memory is None:
        node._memory = np.zeros((u.shape[0], node.rate.shape[1]), dtype=node.dtype)
    node._memory = node._memory + node.rate * (u - node._memory)   # (d, units)
    y = node._memory
    if node.hypers["band_pass"]:
        y = np.concatenate([y, y[:, 1:] - y[:, :-1]], axis=1)
    state = y.reshape(1, -1)
    return state if node.W is None else state @ node.W


class LeakyBank(_SequenceMemory):
    r"""A bank of leaky integrators of the encoded input, with their band-pass differences.

    The input is encoded as for the colony reservoir with ``input_encoding: nonnegative``:
    :math:`u_t = \mathrm{clip}(x_t, l, h) - l`, with :math:`(l, h)` = ``input_range``. Each of
    the ``units`` integrators, with time constants :math:`\tau_k` spaced geometrically from
    ``tau_min`` to ``tau_max`` samples, follows

    .. math:: y_k(t) = y_k(t-1) + (u_t - y_k(t-1)) / \tau_k, \quad y_k(0) = 0,

    and with ``band_pass`` the state adds the differences of neighbouring integrators,
    :math:`y_{k+1}(t) - y_k(t)`: :math:`2 K - 1` features per input for :math:`K` integrators.
    With ``expansion`` the state is mapped by a fixed random matrix
    :math:`M_{ij} \sim N(0, 1/n)` (:math:`n` the bank's features) to ``expansion`` features, to
    match the feature count of another model.

    Parameters
    ----------
    units : int
        Number of integrators per input.
    tau_min, tau_max : float, default 1 and 256
        Shortest and longest time constants, in input steps (at least 1).
    band_pass : bool, default True
        Add the differences of neighbouring integrators.
    expansion : int, optional
        Number of features of the random linear expansion; none by default.
    input_range : (float, float), default (-3, 8)
        Clipping range :math:`(l, h)` of the input encoding.
    seed : int, optional
        Seed of the expansion matrix.
    """

    def __init__(self, units: int, tau_min: float = 1.0, tau_max: float = 256.0,
                 band_pass: bool = True, expansion: int | None = None,
                 input_range: tuple[float, float] = (-3.0, 8.0), seed: int | None = None,
                 input_dim: int | None = None, name: str | None = None):
        if int(units) < 1 or (band_pass and int(units) < 2):
            raise ValueError("LeakyBank needs units >= 1 (>= 2 with band_pass).")
        if not 1.0 <= float(tau_min) <= float(tau_max):
            raise ValueError("LeakyBank needs 1 <= tau_min <= tau_max (in input steps).")
        if expansion is not None and int(expansion) < 1:
            raise ValueError("LeakyBank expansion must be at least 1.")
        low, high = (float(v) for v in input_range)
        if not low < high:
            raise ValueError("input_range must be (low, high) with low < high.")
        self.seed = seed
        self._memory: np.ndarray | None = None
        output_dim = None
        if expansion is not None:
            output_dim = int(expansion)
        elif input_dim is not None:
            output_dim = int(input_dim) * leaky_bank_size(int(units), bool(band_pass))
        super().__init__(
            params={"rate": None, "W": None},
            hypers={"units": int(units), "tau_min": float(tau_min), "tau_max": float(tau_max),
                    "band_pass": bool(band_pass),
                    "expansion": None if expansion is None else int(expansion),
                    "input_range": (low, high), "output_variables": "state"},
            forward=_leaky_bank_forward,
            initializer=_leaky_bank_initialize,
            output_dim=output_dim, input_dim=input_dim, name=name)

    def _clear_memory(self) -> None:
        self._memory = None


def build_control(kind: str, params: dict[str, Any], seed: int) -> tuple[Node, dict[str, Any]]:
    """Build a control node from run-config parameters; return it and its settings.

    ``kind`` is ``"random_features"`` or ``"leaky_bank"``; ``params`` holds the keyword
    arguments of :class:`RandomFeatures` or :class:`LeakyBank` (``units`` required).
    """
    if kind not in CONTROL_TYPES:
        raise ValueError(f"Unknown control {kind!r}; choose from {CONTROL_TYPES}.")
    units = int(params["units"])
    p = {**CONTROL_DEFAULTS[kind], **{k: v for k, v in params.items()
                                      if k in CONTROL_DEFAULTS[kind]}}
    if kind == "random_features":
        node = RandomFeatures(
            units=units, length=int(p["length"]), activation=str(p["activation"]),
            input_scaling=float(p["input_scaling"]), bias_scaling=float(p["bias_scaling"]),
            seed=seed)
    else:
        node = LeakyBank(
            units=units, tau_min=float(p["tau_min"]), tau_max=float(p["tau_max"]),
            band_pass=bool(p["band_pass"]),
            expansion=None if p["expansion"] is None else int(p["expansion"]),
            input_range=tuple(p["input_range"]), seed=seed)
    settings = {k: v for k, v in node.hypers.items() if k != "output_variables"}
    attrs = {"reservoir_type": kind, "res_units": units, "topo_seed": seed,
             **{f"control_{k}": list(v) if isinstance(v, tuple) else v
                for k, v in settings.items() if k != "units"}}
    return node, attrs
