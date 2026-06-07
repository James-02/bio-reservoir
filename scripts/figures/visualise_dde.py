"""Visualise ECG-driven DDE dynamics for the genetic oscillator reservoir.

This script generates a single *Methods-style* figure that shows how an
external ECG drive perturbs a single oscillator node.

What it does
------------
- Simulates one BioReservoir node with an explicit 0-input warmup.
- Builds a baseline trajectory with **0 input throughout**.
- Builds one ECG input trajectory, repeated across multiple input scales.
- Plots luxI + aiiA (only) as small-multiple rows:
	- top row: baseline (0 input)
	- subsequent rows: ECG-driven at each input scale
	- bottom row: the ECG input trace (shown once)

Outputs are saved under ``results/states/``.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Dict, List

import numpy as np

from bioreservoir import BioReservoir
from utils.preprocessing import load_ecg_data
from utils.visualisation import plot_dde_ecg_multiscale



def _simulate_step_only(*, reservoir: BioReservoir, input_series: np.ndarray) -> np.ndarray:
	"""Simulate by calling ``BioReservoir._step`` for each timestep.

	Returns array of shape (T, units, 4) for variables (A, I, Hi, He).
	"""
	inp = np.asarray(input_series, dtype=np.float64)
	if inp.ndim == 1:
		inp = np.repeat(inp.reshape(-1, 1), reservoir.units, axis=1)
	elif inp.ndim == 2:
		if inp.shape[1] != reservoir.units:
			raise ValueError(f"input_series second dim must be units={reservoir.units}; got {inp.shape}")
	else:
		raise ValueError(f"input_series must be 1D or 2D; got shape {inp.shape}")

	T = int(inp.shape[0])
	out = np.empty((T, reservoir.units, 4), dtype=np.float64)
	for t in range(T):
		out[t] = reservoir._step(inp[t])
	return out


def _make_reservoir(*, units: int, seed: int) -> BioReservoir:
	return BioReservoir(
		units=int(units),
		warmup=0,  # warmup plotted explicitly
		output_variables=("A", "I", "Hi", "He"),
		seed=int(seed),
	)


def _parse_scales(s: str) -> List[float]:
	vals: List[float] = []
	for tok in str(s).split(","):
		tok = tok.strip()
		if not tok:
			continue
		vals.append(float(tok))
	vals = sorted(set(vals))
	if not vals:
		raise ValueError("--scales must contain at least one value")
	if any(v < 0 for v in vals):
		raise ValueError("--scales values must be >= 0")
	return vals


def _build_ecg_drive(*, ecg: np.ndarray, timesteps: int, warmup: int, scale: float) -> np.ndarray:
	"""Resample ECG to driven length, z-score it, scale it, prepend warmup zeros."""
	ecg = np.asarray(ecg, dtype=np.float64).reshape(-1)
	ecgt = np.interp(
		np.linspace(0, 1, int(timesteps)),
		np.linspace(0, 1, int(ecg.shape[0])),
		ecg,
	).astype(np.float64)
	ecgt = (ecgt - float(np.mean(ecgt))) / (float(np.std(ecgt)) + 1e-8)
	ecgt = float(scale) * ecgt
	return np.concatenate([np.zeros(int(warmup), dtype=np.float64), ecgt])


def _build_ecg_drive_unscaled(*, ecg: np.ndarray, timesteps: int, warmup: int) -> np.ndarray:
	"""Resample ECG to driven length, z-score it, prepend warmup zeros (no extra scaling)."""
	ecg = np.asarray(ecg, dtype=np.float64).reshape(-1)
	ecgt = np.interp(
		np.linspace(0, 1, int(timesteps)),
		np.linspace(0, 1, int(ecg.shape[0])),
		ecg,
	).astype(np.float64)
	ecgt = (ecgt - float(np.mean(ecgt))) / (float(np.std(ecgt)) + 1e-8)
	return np.concatenate([np.zeros(int(warmup), dtype=np.float64), ecgt])


def _states_to_vars(states_all: np.ndarray) -> Dict[str, np.ndarray]:
	# states_all: (T, units=1, 4) in order (A, I, Hi, He)
	return {
		"A": states_all[:, 0, 0],
		"I": states_all[:, 0, 1],
		"Hi": states_all[:, 0, 2],
		"He": states_all[:, 0, 3],
	}

def main() -> None:
	parser = argparse.ArgumentParser(description="ECG-driven DDE multiscale figure")
	parser.add_argument("--timesteps", type=int, default=1000, help="Driven segment length")
	parser.add_argument("--warmup", type=int, default=100, help="0-input warmup steps (drawn dashed)")
	parser.add_argument("--seed", type=int, default=1234, help="Seed for reservoir initialisation")
	parser.add_argument("--ecg-index", type=int, default=0, help="ECG instance index (into training set)")
	parser.add_argument(
		"--scales",
		type=str,
		default="1e-5,1e-4,1e-3,1e-2",
		help="Comma-separated ECG input scales (applied after per-sequence z-score)",
	)
	parser.add_argument("--rows", type=int, default=2000, help="How many training rows to load (for balancing)")
	parser.add_argument("--debug", action="store_true", help="Enable detailed debug logging")
	parser.add_argument("--show", action="store_true", help="Show the figure interactively")

	args = parser.parse_args()

	logging.basicConfig(
		level=(logging.DEBUG if bool(args.debug) else logging.INFO),
		format="%(levelname)s:%(name)s:%(message)s",
	)
	logger = logging.getLogger(__name__)
	logging.getLogger("matplotlib").setLevel(logging.WARNING)
	logging.getLogger("numba").setLevel(logging.WARNING)

	timesteps = int(args.timesteps)
	warmup = int(args.warmup)
	seed = int(args.seed)
	scales = _parse_scales(str(args.scales))
	if timesteps <= 0:
		raise ValueError("--timesteps must be > 0")
	if warmup < 0:
		raise ValueError("--warmup must be >= 0")

	T = warmup + timesteps
	t_axis = np.arange(T, dtype=np.int64)

	logger.debug("Loading ECG data for state figure (rows=%s)", int(args.rows))
	X_train, Y_train, _, _ = load_ecg_data(
		rows=int(args.rows),
		test_ratio=0.2,
		encode_labels=True,
		standardize=True,
		repeat_targets=False,
		shuffle=True,
		binary=False,
		noise_rate=0.0,
		noise_ratio=0.0,
		scaler_type="sequence_zscore",
		balance_classes=True,
		max_per_class=None,
	)
	idx = int(args.ecg_index)
	idx = max(0, min(idx, len(X_train) - 1))
	ecg = np.asarray(X_train[idx]).reshape(-1).astype(np.float64)
	# Y_train is one-hot (N, 1, C) when encode_labels=True.
	try:
		y_idx = int(np.argmax(np.asarray(Y_train[idx]).reshape(-1)))
		# Keep this mapping local so we don't pull in the full plotting module here.
		class_names = ["Normal", "Supraventricular", "Ventricular", "Fusion", "Unknown"]
		ecg_label = class_names[y_idx] if 0 <= y_idx < len(class_names) else f"class {y_idx}"
	except Exception:
		ecg_label = None
	logger.debug(
		"Selected ECG idx=%s shape=%s min=%.6g max=%.6g mean=%.6g std=%.6g neg_frac=%.4f",
		idx,
		ecg.shape,
		float(np.min(ecg)),
		float(np.max(ecg)),
		float(np.mean(ecg)),
		float(np.std(ecg)),
		float(np.mean(ecg < 0)),
	)

	# Baseline: 0 input for the full simulation.
	reservoir_base = _make_reservoir(units=1, seed=seed)
	u0 = np.zeros(T, dtype=np.float64)
	states0 = _simulate_step_only(reservoir=reservoir_base, input_series=u0)
	baseline_states = _states_to_vars(states0)

	# Driven: ECG input at multiple scales.
	driven_states_by_scale: Dict[float, Dict[str, np.ndarray]] = {}
	driven_input_by_scale: Dict[float, np.ndarray] = {}
	input_unscaled = _build_ecg_drive_unscaled(ecg=ecg, timesteps=timesteps, warmup=warmup)
	for s in scales:
		reservoir = _make_reservoir(units=1, seed=seed)
		u = _build_ecg_drive(ecg=ecg, timesteps=timesteps, warmup=warmup, scale=float(s))
		states = _simulate_step_only(reservoir=reservoir, input_series=u)
		driven_states_by_scale[float(s)] = _states_to_vars(states)
		driven_input_by_scale[float(s)] = u
		logger.debug(
			"u(t) scale=%s min=%.6g max=%.6g mean=%.6g std=%.6g neg_frac=%.4f",
			float(s),
			float(np.min(u)),
			float(np.max(u)),
			float(np.mean(u)),
			float(np.std(u)),
			float(np.mean(u < 0)),
		)

	plot_dde_ecg_multiscale(
		t_axis=t_axis,
		warmup=warmup,
		baseline_states_by_var=baseline_states,
		driven_states_by_scale=driven_states_by_scale,
		driven_input_by_scale=driven_input_by_scale,
		input_trace_unscaled=input_unscaled,
		ecg_class_label=ecg_label,
		filename=os.path.join("states/", "dde_ecg_multiscale.png"),
		show=bool(args.show),
	)


if __name__ == "__main__":
	main()

