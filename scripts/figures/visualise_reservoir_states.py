"""Visualise reservoir node states on real ECG inputs.

Foundation script (intentionally minimal):

- Loads the ECG dataset using :func:`utils.preprocessing.load_ecg_data`.
- Builds an :class:`reservoir.reservoir.BioReservoir` with a configurable
  number of nodes (default 250).
- Runs the reservoir on one ECG instance using the standard `.run(...)` API.
- Extracts a single DDE variable (default LuxI == "I") for each node over time.
- Plots a heatmap (time x node) so we can see inter-node differences.

This gives us a solid base to later add:
- node-difference metrics (variance across nodes, correlation structure, etc.)
- multiple instances/classes overlays
- drive decomposition diagnostics from `_compute_input`

Outputs are saved under `results/states/`.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from bioreservoir import BioReservoir
from bioreservoir import distance_matrix
from utils.preprocessing import load_ecg_data
from utils.visualisation import _save_figure, gene_full_name, classes as ECG_CLASSES


_VAR_TO_INDEX = {"A": 0, "I": 1, "Hi": 2, "He": 3}


def _parse_var(name: str) -> str:
	name = str(name)
	if name not in _VAR_TO_INDEX:
		raise ValueError(f"Unknown variable '{name}'. Choose from: {sorted(_VAR_TO_INDEX)}")
	return name


def _plot_state_heatmap(
	*,
	state_matrix: np.ndarray,
	var: str,
	filename: str,
	show: bool,
	vmin: float | None = None,
	vmax: float | None = None,
) -> None:
	"""Plot time x node heatmap for a single variable.

	Parameters
	----------
	state_matrix:
		Shape (T, units). Each column is a node.
	"""
	plt.figure(figsize=(14, 6))
	plt.imshow(
		state_matrix.T,
		aspect="auto",
		origin="lower",
		interpolation="nearest",
		cmap="viridis",
		vmin=vmin,
		vmax=vmax,
	)
	plt.colorbar(label=f"{gene_full_name(var)} concentration")
	plt.xlabel("Timestep")
	plt.ylabel("Node index")
	plt.tight_layout()
	_save_figure(filename=os.path.join("states/", filename))
	if show:
		plt.show()


def _as_dense(W):
	"""Convert scipy sparse matrices to dense numpy arrays if needed."""
	return W.toarray() if hasattr(W, "toarray") else np.asarray(W)


def _plot_topology(
	*,
	coords: np.ndarray,
	W: np.ndarray,
	filename: str,
	show: bool,
	edges_per_node: int = 3,
) -> None:
	"""Plot spatial topology: node positions + strongest outgoing edges."""
	coords = np.asarray(coords)
	if coords.ndim != 2 or coords.shape[1] != 2:
		raise ValueError(f"Expected coords shape (units, 2); got {coords.shape}")

	W = _as_dense(W)
	units = coords.shape[0]
	if W.shape[0] != units or W.shape[1] != units:
		raise ValueError(f"Expected W shape ({units}, {units}); got {W.shape}")

	# Pick top-k outgoing edges per node by |weight| (ignore self).
	k = max(0, int(edges_per_node))
	plt.figure(figsize=(7.5, 7))
	plt.scatter(coords[:, 0], coords[:, 1], s=14, alpha=0.9, c="black")

	if k > 0:
		absW = np.abs(W).copy()
		np.fill_diagonal(absW, 0.0)
		for i in range(units):
			idx = np.argsort(absW[i])[-k:]
			x0, y0 = coords[i]
			for j in idx:
				w = W[i, j]
				if w == 0:
					continue
				x1, y1 = coords[j]
				# Color by sign (excite/inhibit) if using signed matrix.
				color = "tab:blue" if w >= 0 else "tab:red"
				linestyle = "solid" if w >= 0 else "dashed"
				plt.plot([x0, x1], [y0, y1], color=color, alpha=0.25, linestyle=linestyle, linewidth=0.8)

	# Legend explaining edge colours.
	handles = [
		Line2D([0], [0], marker="o", color="w", markerfacecolor="black", markersize=6, label="node"),
		Line2D([0], [0], color="tab:blue", linewidth=2, label="excitation (+)"),
		Line2D([0], [0], color="tab:red", linewidth=2, label="inhibition (-)", linestyle="dashed"),
	]
	plt.legend(handles=handles, loc="upper right", frameon=True, fontsize=9)
	plt.xlabel("x")
	plt.ylabel("y")
	plt.axis("equal")
	plt.tight_layout()
	_save_figure(filename=os.path.join("topology/", filename))
	if show:
		plt.show()


def _compute_input_diagnostics(
	*,
	reservoir: BioReservoir,
	x: np.ndarray,
) -> dict[str, np.ndarray]:
	"""Compute per-timestep diagnostics: input vs recurrent contribution.

	We intentionally mirror the math in `reservoir.reservoir._compute_input`, but we
	only compute summary statistics (std) rather than the full per-node vectors.

	Returns arrays of shape (T,): inp_std, rec_std, drive_std, pre_state_std, ratio.
	"""
	x = np.asarray(x)
	if x.ndim != 2 or x.shape[1] != 1:
		raise ValueError(f"Expected input sequence shape (T, 1); got {x.shape}")

	T = x.shape[0]
	inp_std = np.zeros(T, dtype=np.float64)
	rec_std = np.zeros(T, dtype=np.float64)
	drive_std = np.zeros(T, dtype=np.float64)
	pre_state_std = np.zeros(T, dtype=np.float64)
	ratio = np.zeros(T, dtype=np.float64)
	inp_mean_abs = np.zeros(T, dtype=np.float64)
	rec_mean_abs = np.zeros(T, dtype=np.float64)
	drive_mean_abs = np.zeros(T, dtype=np.float64)

	for t in range(T):
		u = x[t].reshape(-1, 1)  # (1, 1)
		r = reservoir.state().T  # (units, 1)

		# Input term (no noise for interpretability).
		inp_term = reservoir.Win @ u
		inp_term = inp_term + reservoir.bias

		# Recurrent term (no noise for interpretability).
		rec_term = reservoir.W @ r
		rec_term = reservoir.rc_scaling * rec_term

		drive = inp_term + rec_term

		inp_std[t] = float(np.std(inp_term))
		rec_std[t] = float(np.std(rec_term))
		drive_std[t] = float(np.std(drive))
		ratio[t] = inp_std[t] / rec_std[t] if rec_std[t] > 0 else np.inf
		inp_mean_abs[t] = float(np.mean(np.abs(inp_term)))
		rec_mean_abs[t] = float(np.mean(np.abs(rec_term)))
		drive_mean_abs[t] = float(np.mean(np.abs(drive)))

		# Advance reservoir using the standard forward path by calling run-step logic.
		# We reuse `reservoir._step` via the Node forward by running one step:
		reservoir(x[t].reshape(1, 1))
		# Effective DDE drive after clipping/normalisation inside the reservoir.
		# The reservoir caches this as `_last_final_input` when diagnostics are enabled.
		final_input = getattr(reservoir, "_last_final_input", None)
		if final_input is None:
			# Backwards/alternate cache key (older experiments).
			final_input = getattr(reservoir, "_last_pre_state", None)
		if final_input is None:
			pre_state_std[t] = np.nan
		else:
			pre_state_std[t] = float(np.std(np.asarray(final_input)))

	return {
		"inp_std": inp_std,
		"rec_std": rec_std,
		"drive_std": drive_std,
		"pre_state_std": pre_state_std,
		"inp_over_rec": ratio,
		"inp_mean_abs": inp_mean_abs,
		"rec_mean_abs": rec_mean_abs,
		"drive_mean_abs": drive_mean_abs,
	}

def _plot_input_vs_recurrent(
	*,
	diag: dict[str, np.ndarray],
	filename: str,
	show: bool,
	x_ecg: np.ndarray | None = None,
	input_scaling: float | None = None,
	rc_scaling: float | None = None,
	ecg_class_label: str | None = None,
) -> None:
	"""Two-panel drive decomposition figure for the methods section.

	Top panel: per-timestep std across reservoir nodes for the input term,
	recurrent term, and their sum (drive), on a log scale.
	Bottom panel (optional): raw ECG signal providing temporal context,
	matching the style of dde_ecg_multiscale.png.

	Accessibility: each series uses a distinct Okabe-Ito colour *and* a
	distinct linestyle so the figure is readable in greyscale and by
	colour-blind readers.
	"""
	# Okabe-Ito palette (colour-blind safe).
	C_INP = "#E69F00"  # orange  – input term
	C_REC = "#56B4E9"  # sky-blue – recurrent term
	C_DRV = "#009E73"  # green    – combined drive

	T = len(diag["inp_std"])
	t = np.arange(T)
	has_ecg = x_ecg is not None

	# Panels: mean(|·|), [ECG].
	height_ratios = [2.2, 1.2] if has_ecg else [2.2]
	n_panels = 2 if has_ecg else 1
	fig, axes = plt.subplots(
		n_panels, 1,
		figsize=(14, 6),
		gridspec_kw={"height_ratios": height_ratios, "hspace": 0.10},
		sharex=True,
		constrained_layout=True,
	)

	# ---- Panel 0: mean absolute magnitude (directly comparable across terms) ----
	ax_mean = axes[0]
	ax_mean.plot(
		t, diag["inp_mean_abs"],
		label=r"Input term: $\alpha_u \mathbf{W}_{in} u_t + b$",
		color=C_INP, linewidth=1.8, linestyle="solid",
	)
	ax_mean.plot(
		t, diag["rec_mean_abs"],
		label=r"Recurrent term: $\alpha_r \mathbf{W} r_{t-1}$",
		color=C_REC, linewidth=1.8, linestyle="dashed",
	)
	ax_mean.plot(
		t, diag["drive_mean_abs"],
		label=r"Drive: input $+$ recurrent",
		color=C_DRV, linewidth=1.4, linestyle="dashdot", alpha=0.9,
	)
	ax_mean.set_yscale("log")
	ax_mean.set_ylabel(r"Mean per node (log)")
	ax_mean.grid(True, alpha=0.25)
	ax_mean.spines["top"].set_visible(False)
	ax_mean.spines["right"].set_visible(False)
	ax_mean.legend(ncol=3, fontsize=9, frameon=True, loc="upper right")

	# Scaling annotation on the mean-abs panel.
	param_parts: list[str] = []
	if input_scaling is not None:
		exp = int(round(np.log10(input_scaling)))
		param_parts.append(rf"$\alpha_u = 10^{{{exp}}}$")
	if rc_scaling is not None:
		exp = int(round(np.log10(rc_scaling)))
		param_parts.append(rf"$\alpha_r = 10^{{{exp}}}$")
	if param_parts:
		ax_mean.annotate(
			",\u2002".join(param_parts),
			xy=(0.02, 0.05),
			xycoords="axes fraction",
			fontsize=9,
			bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="grey", alpha=0.85),
		)

	# ---- Panel 1 (or 0 if no ECG): raw ECG context strip ----
	if has_ecg:
		ax_ecg = axes[1]
		ecg_seg = np.asarray(x_ecg).ravel()[:T]
		if ecg_class_label is not None and str(ecg_class_label).strip():
			ecg_legend_txt = f"ECG: {ecg_class_label}"
		else:
			ecg_legend_txt = "ECG signal"
		ax_ecg.plot(
			t, ecg_seg,
			color="#CC0000", linewidth=1.6, label=ecg_legend_txt,
		)

		ax_ecg.set_ylabel("ECG input")
		ax_ecg.set_xlabel("Timestep")
		ax_ecg.grid(True, alpha=0.18)
		ax_ecg.spines["top"].set_visible(False)
		ax_ecg.spines["right"].set_visible(False)
		ax_ecg.legend(loc="upper right", fontsize=8, frameon=True)
	else:
		ax_mean.set_xlabel("Timestep")

	_save_figure(filename=os.path.join("states/", filename))
	if show:
		plt.show()
	plt.close(fig)

def main() -> None:
	parser = argparse.ArgumentParser(description="Visualise BioReservoir node states on ECG data")
	parser.add_argument("--nodes", type=int, default=250, help="Number of reservoir nodes")
	parser.add_argument("--instance", type=int, default=0, help="Which ECG training instance to visualise")
	parser.add_argument("--timesteps", type=int, default=None, help="Optionally truncate the ECG sequence to first N timesteps")
	parser.add_argument("--var", type=str, default="I", help="Which DDE state to visualise: A, I, Hi, He")
	parser.add_argument("--warmup", type=int, default=40, help="Reservoir warmup parameter passed to BioReservoir")
	parser.add_argument("--seed", type=int, default=1337, help="Random seed for reproducible initialization")
	parser.add_argument("--plot-topology", action="store_true", help="Plot 2D node positions and strongest recurrent edges")
	parser.add_argument("--topology-edges", type=int, default=3, help="Edges per node to draw in topology view")
	parser.add_argument(
		"--plot-drive",
		action="store_true",
		help="Plot input-vs-recurrent contribution over time (diagnostic; runs forward step-by-step)",
	)
	parser.add_argument(
		"--plot-node-drive",
		action="store_true",
		help="Plot time×node heatmaps for input term, recurrent term, and their ratio",
	)
	parser.add_argument(
		"--node-drive-var",
		type=str,
		default=None,
		help=(
			"Optional: compute input/recurrent terms as if the reservoir state were a single variable. "
			"Choose from A, I, Hi, He. If omitted, uses the reservoir's full state vector."
		),
	)
	parser.add_argument("--show", action="store_true")
	args = parser.parse_args()

	nodes = int(args.nodes)
	instance = int(args.instance)
	warmup = int(args.warmup)
	seed = int(args.seed)
	var = _parse_var(args.var)

	# Load ECG dataset (mirrors test.py defaults, but keep it lightweight).
	X_train, Y_train, _, _ = load_ecg_data(
		rows=2000,
		balance_classes=True,
		encode_labels=True,
		shuffle=False,
		noise_rate=0,
		noise_ratio=0,
		seed=seed,
	)

	if instance < 0 or instance >= len(X_train):
		raise ValueError(f"--instance out of range: {instance} (train size={len(X_train)})")

	x = np.asarray(X_train[instance])
	if x.ndim != 2 or x.shape[1] != 1:
		# load_ecg_data returns sequences shaped (T, 1)
		raise ValueError(f"Unexpected ECG instance shape {x.shape}; expected (T, 1)")
	if args.timesteps is not None:
		x = x[: int(args.timesteps)]

	reservoir = BioReservoir(
		units=nodes,
		warmup=warmup,
		input_scaling=1e-1,
		input_connectivity=1.0,
		rc_connectivity=0.1,
		rc_scaling=1e-3,
		dde_scaling=3e-5,
		W=distance_matrix(interaction_diameters=8.2, sparsity=1.0, seed=seed),
		seed=seed,
		diagnostics=bool(args.plot_drive or args.plot_node_drive),
		# Important: ensure I is included in the observable reservoir state.
		output_variables="all",
	)

	# Optional: topology plot (coordinates + strongest edges). We re-generate the
	# recurrent matrix with coordinates so the picture matches the actual W used.
	if bool(args.plot_topology):
		W_topo, coords = distance_matrix(m=nodes, n=nodes, seed=seed, sparsity=1.0, return_coords=True)
		_plot_topology(
			coords=coords,
			W=W_topo,
			edges_per_node=int(args.topology_edges),
			filename=f"reservoir_topology_nodes{nodes}.png",
			show=bool(args.show),
		)

	states = reservoir.run(x, reset=True)
	# `states` is (T, units * n_vars_selected) when using reservoirpy Node outputs.
	# With output_variables="all", that is (T, units*4) with order [A,I,Hi,He] per node.
	T = states.shape[0]
	if states.ndim != 2 or states.shape[1] != nodes * 4:
		raise ValueError(f"Unexpected reservoir.run output shape {states.shape}; expected (T, {nodes*4})")

	states_reshaped = states.reshape(T, nodes, 4)
	var_idx = _VAR_TO_INDEX[var]
	var_states = states_reshaped[:, :, var_idx]  # (T, nodes)

	_plot_state_heatmap(
		state_matrix=var_states,
		var=var,
		filename=f"reservoir_heatmap_{gene_full_name(var)}_nodes{nodes}_inst{instance}.png",
		show=bool(args.show),
	)

	# Optional: drive decomposition diagnostic.
	if bool(args.plot_drive):
		# Run diagnostics on the same instance. We reset first to align with the heatmap run.
		# Decode class label: Y_train is one-hot when encode_labels=True.
		y_inst = np.asarray(Y_train[instance]).ravel()
		class_idx = int(np.argmax(y_inst)) if len(y_inst) > 1 else int(y_inst[0])
		ecg_label = ECG_CLASSES[class_idx] if class_idx < len(ECG_CLASSES) else f"class {class_idx}"

		reservoir.reset()
		diag = _compute_input_diagnostics(reservoir=reservoir, x=x)
		_plot_input_vs_recurrent(
			diag=diag,
			filename=f"reservoir_drive_diag_nodes{nodes}_inst{instance}.png",
			show=bool(args.show),
			x_ecg=x.squeeze(),
			input_scaling=reservoir.input_scaling,
			rc_scaling=reservoir.rc_scaling,
			ecg_class_label=ecg_label,
		)

if __name__ == "__main__":
	main()
