from functools import partial
import time
from typing import Any, Optional, Union, Dict, List, Callable, Sequence

import numpy as np
from reservoirpy.mat_gen import bernoulli, normal
from reservoirpy.node import Node
from reservoirpy.type import Weights
from reservoirpy.utils.random import noise, rand_generator
from reservoirpy.nodes.reservoirs.base import initialize, initialize_feedback
from reservoirpy.utils.validation import is_array
from reservoirpy.activationsfunc import identity

from bioreservoir.dde import get_dde_params, log_dde_params, solve_dde_rk4, generate_node_params
from bioreservoir.topology import distance_matrix

import logging

logger = logging.getLogger(__name__)

class BioReservoir(Node):
    def __init__(
        self,
        units: int = 250,
        warmup: int = 60,
        rc_scaling: float = 1e-4,
        rk4_substeps: int = 24,
        cell_coupling: float = 1.0,
        dde_scaling: float = 5e-3,
        dde_kwargs: Optional[Dict] = None,
        sr: Optional[float] = None,
        input_bias: bool = False,
        noise_rc: float = 0.1,
        noise_in: float = 0.05,
        noise_fb: float = 0.1,
        noise_type: str = "normal",
        noise_kwargs: Dict = None,
        input_scaling: Union[float, Sequence] = 1e-2,
        bias_scaling: float = 1.0,
        fb_scaling: Union[float, Sequence] = 1e-4,
        input_connectivity: float = 1.0,
        rc_connectivity: float = 0.1,
        fb_connectivity: float = 0.1,
        fb_activation: Callable = identity,
        Win: Union[Weights, Callable] = bernoulli,
        W: Union[Weights, Callable] = normal,
        Wfb: Union[Weights, Callable] = bernoulli,
        bias: Union[Weights, Callable] = bernoulli,
        input_dim: Optional[int] = None,
        feedback_dim: Optional[int] = None,
        parallel: bool = True,
        seed: Optional[int] = None,
        output_variables: Sequence[str] = ("A", "I", "Hi", "He"),
        rk4_substep_aggregation_window: int = 1,
        diagnostics: bool = False,
        param_heterogeneity_cv: float = 0.0,
        param_heterogeneity_seed: Optional[int] = None,
        **kwargs,
    ):
        if units is None and not is_array(W):
            raise ValueError("'units' must be specified if 'W' is not a matrix.")

        rng = rand_generator(seed)
        noise_kwargs = noise_kwargs or {}
        dde_kwargs = dde_kwargs or {}

        # ---------------------------------------------------------------------------
        # Map output variable names to indices in [A, I, Hi, He].
        # Accepts either a tuple of variable names or a named preset string.
        # New presets can be added here without changing the objective function.
        # ---------------------------------------------------------------------------
        _OUTPUT_VAR_PRESETS = {
            "all":    ("A", "I", "Hi", "He"),  # all 4 DDE state variables
            "AI":     ("A", "I"),               # LuxI activator + intracellular signal
            "IHi":    ("I", "Hi"),              # intracellular dynamics pair
            "IHe":    ("I", "He"),              # intracellular + extracellular QS
            "AHe":    ("A", "He"),              # activator + QS communication signal
            "AHi":    ("A", "Hi"),              # activator + internal AiiA state
            "He":     ("He",),                  # extracellular quorum signal only
            "I":      ("I",),                   # intracellular signal only
            "A":      ("A",),                   # activator only
            "Hi":     ("Hi",),                  # internal state only
            "AIHi":   ("A", "I", "Hi"),         # all except extracellular
            "IHiHe":  ("I", "Hi", "He"),        # all except activator
        }
        var_index_map = {"A": 0, "I": 1, "Hi": 2, "He": 3}
        if not output_variables or output_variables == "all":
            output_variables = ("A", "I", "Hi", "He")
        elif isinstance(output_variables, str):
            if output_variables not in _OUTPUT_VAR_PRESETS:
                raise ValueError(
                    f"Unknown output_variables preset {output_variables!r}. "
                    f"Valid presets: {sorted(_OUTPUT_VAR_PRESETS)}. "
                    "Alternatively, pass a tuple of variable names from ('A','I','Hi','He')."
                )
            output_variables = _OUTPUT_VAR_PRESETS[output_variables]
        self.output_indices = np.array([var_index_map[v] for v in output_variables], dtype=np.int64)
        # Clamp aggregation window to at least 1
        self.rk4_substep_aggregation_window = max(1, int(rk4_substep_aggregation_window))
        output_dim = (len(self.output_indices) * units) if units is not None else None

        super(BioReservoir, self).__init__(
            fb_initializer=partial(
                initialize_feedback,
                Wfb_init=Wfb,
                fb_activation=fb_activation,
                fb_scaling=fb_scaling,
                fb_connectivity=fb_connectivity,
                seed=seed,
            ),
            params={
                "W": None,
                "Win": None,
                "Wfb": None,
                "bias": None,
                "internal_state": None
            },
            hypers={
                "warmup": warmup,
                "sr": sr,
                "dde_scaling": dde_scaling,
                "rc_scaling": rc_scaling,
                "cell_coupling": cell_coupling,
                "input_scaling": input_scaling,
                "bias_scaling": bias_scaling,
                "fb_scaling": fb_scaling,
                "rc_connectivity": rc_connectivity,
                "input_connectivity": input_connectivity,
                "fb_connectivity": fb_connectivity,
                "noise_in": noise_in,
                "noise_rc": noise_rc,
                "noise_out": noise_fb,
                "noise_type": noise_type,
                "units": units,
                "parallel": parallel,
                "param_heterogeneity_cv": param_heterogeneity_cv,
                "noise_generator": partial(noise, rng=rng, **noise_kwargs),
                "output_variables": output_variables,
            },
            forward=_forward,
            initializer=partial(
                initialize,
                sr=sr,
                input_scaling=input_scaling,
                bias_scaling=bias_scaling,
                input_connectivity=input_connectivity,
                rc_connectivity=rc_connectivity,
                W_init=W,
                Win_init=Win,
                bias_init=bias,
                input_bias=input_bias,
                seed=seed,
            ),
            # Output dimension is configurable subset of [A, I, Hi, He]
            output_dim=output_dim,
            feedback_dim=feedback_dim,
            input_dim=input_dim,
            **kwargs,
        )

        # Resolve to a *full* parameter dict (includes defaults + derived constants).
        # This ensures keys like 'delay' always exist for history buffer sizing.
        self.dde_kwargs = get_dde_params(**dde_kwargs)

        # Log DDE parameters only from the reservoir (library entrypoint).
        # The helper lives in dde.py, but *this* module decides when to emit.
        log_dde_params(self.dde_kwargs, level=logging.DEBUG)

        # Solver-specific configuration (not part of the biological model).
        self.rk4_substeps = int(rk4_substeps)
        self.rk4_timespan = np.linspace(0.0, 1.0, self.rk4_substeps + 1)

        # Store seed for reproducible log-normal initialization
        self.seed = seed

        self.states = self._initialize_states()

        # Per-node kinetic heterogeneity (cell-to-cell variability).
        self.param_heterogeneity_cv = float(param_heterogeneity_cv)
        self._het_seed = param_heterogeneity_seed if param_heterogeneity_seed is not None else self.seed
        if self.param_heterogeneity_cv > 0:
            self.node_params = generate_node_params(
                self.units, self.param_heterogeneity_cv,
                self.dde_kwargs, seed=self._het_seed,
            )
            logger.info(
                "Generated per-node kinetic heterogeneity (CV=%.3f) for %d nodes",
                self.param_heterogeneity_cv, self.units,
            )
        else:
            self.node_params = None

        logger.info(
            "Initialized BioReservoir(units=%s, warmup=%s, rk4_substeps=%s, output_variables=%s)",
            self.units,
            self.warmup,
            self.rk4_substeps,
            tuple(output_variables),
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("----- Reservoir Hyperparameters -----")
            for k, v in self.hypers.items():
                logger.debug("%s: %s", k, v)
            logger.debug("------------------------------------")
        self._warmup_nodes()
        self._warmup_states = self.states.copy()
        self.t = 0

        # Diagnostics are optional and can be expensive (extra allocations/copies).
        # When enabled, `_compute_input` will populate the *_last_* caches.
        self.diagnostics = bool(diagnostics)
        self._last_inp_term = None
        self._last_rec_term = None
        self._last_drive = None
        self._last_final_input = None

        # Per-timestep timing accumulator — only populated when diagnostics=True.
        # Lists grow as _forward() is called and are cleared on reset().
        # Call get_timing_summary() to obtain statistics after a run.
        self._timing: Dict[str, List[float]] = {
            "input_computation": [],
            "dde_solve": [],
        }

    def get_timing_summary(self) -> Dict[str, Any]:
        """Return per-component timing statistics accumulated since the last reset().

        Components (only populated when ``diagnostics=True``):

        * ``"input_computation"`` — Wall time spent inside :func:`_compute_input`
          per external timestep (Win @ u, W @ r, noise, scaling).
        * ``"dde_solve"``         — Wall time spent inside :func:`solve_dde_rk4`
          per external timestep (the RK4 integration across substeps).

        Each entry contains::

            {
                "total_s": float,   # total wall time across all timesteps
                "calls":   int,     # number of timesteps recorded
                "mean_s":  float,   # mean per-timestep time
                "min_s":   float,
                "max_s":   float,
            }

        Returns an empty dict when ``diagnostics=False`` or no steps have been
        taken since the last reset.
        """
        if not getattr(self, "diagnostics", False):
            return {}
        out: Dict[str, Any] = {}
        for key, vals in self._timing.items():
            if not vals:
                continue
            arr = np.asarray(vals, dtype=np.float64)
            out[key] = {
                "total_s": round(float(arr.sum()),  6),
                "calls":   len(arr),
                "mean_s":  round(float(arr.mean()), 9),
                "min_s":   round(float(arr.min()),  9),
                "max_s":   round(float(arr.max()),  9),
            }
        return out

    def _initialize_states(self) -> np.ndarray:
        """Initialize states with biologically realistic log-normal distributions.
        
        Biological justification:
        - Gene expression follows log-normal distribution (Elowitz et al. 2002, Science)
        - Parameters based on measured cell-to-cell variability (Ozbudak et al. 2002)
        - Cells start in exponential growth phase with low basal expression (Danino et al. 2010)
        
        Log-normal parameters:
        - mean: log-space mean (actual mean ≈ exp(mean + sigma²/2))
        - sigma: log-space std (CV ≈ sqrt(exp(sigma²) - 1))
        
        Initial concentrations:
        - A, I (genes): Low basal expression, CV ≈ 0.52 (typical for bacteria)
        - Hi (internal AHL): Very low, cells haven't activated yet
        - He (external AHL): Low baseline (NOT ZERO - always some ambient AHL)
        """
        history_len = int(np.ceil(self.dde_kwargs['delay'])) + 1
        states = np.zeros((history_len, self.units, 4))

        # Use numpy's default_rng for reproducibility
        rng = np.random.default_rng(self.seed if hasattr(self, 'seed') else None)
        
        # Generate random values for each history timestep
        # (Maintains temporal correlation in history buffer)
        for i in range(history_len):
            # aiiA (A): Low basal expression
            # mean=-2.0 → exp(-2) ≈ 0.135, sigma=0.5 → CV ≈ 0.52
            init_A = rng.lognormal(mean=-2.0, sigma=0.5, size=self.units)
            
            # luxI (I): Low basal expression (same distribution as A)
            init_I = rng.lognormal(mean=-2.0, sigma=0.5, size=self.units)
            
            # Internal AHL (Hi): Very low (cells haven't produced much yet)
            # mean=-4.0 → exp(-4) ≈ 0.018, sigma=0.3 → CV ≈ 0.30
            init_Hi = rng.lognormal(mean=-4.0, sigma=0.3, size=self.units)
            
            # External AHL (He): Low baseline
            # mean=-3.0 → exp(-3) ≈ 0.050, sigma=0.3 → CV ≈ 0.30
            # There's always some ambient AHL in the environment
            init_He = rng.lognormal(mean=-3.0, sigma=0.3, size=self.units)
            
            states[i, :, 0] = init_A
            states[i, :, 1] = init_I
            states[i, :, 2] = init_Hi
            states[i, :, 3] = init_He  # Previously was zero - now has basal level

        return states

    def _update_history(self, new_state: np.ndarray) -> None:
        self.states[:-1] = self.states[1:]
        self.states[-1] = new_state

    def _warmup_nodes(self) -> None:
        zero_input = np.zeros(self.units)
        logger.info("Warmup: running %s steps of zero input", self.warmup)
        for _ in range(self.warmup):
            self._step(zero_input)

    def _step(self, x: np.ndarray) -> np.ndarray:
        """Advance the DDE state by one input timestep.

        Notes
        -----
        - Always treats ``x`` as a 1D vector of length ``units`` (one drive per node).
        - Ensures the internal history buffer ``self.states`` keeps shape
          ``(history_len, units, 4)`` so the numba DDE solver sees consistent
          scalar A, I, Hi, He values per node.
        """

        # Normalize input shape to 1D (units,) for the DDE solver.
        x = np.asarray(x).reshape(self.units)

        # Scale the normalized reservoir drive by biological cell coupling strength.
        # This models how strongly the external AHL (He-like signal) influences each oscillator.
        # input_effect = self.cell_coupling * x
        # This input will be used in the DDE equations to combine with the external AHL.
        sol = solve_dde_rk4(
            self.states,
            self.rk4_timespan,
            input_vec=x * self.cell_coupling,
            parallel=self.parallel,
            node_params=self.node_params,
            **self.dde_kwargs,
        )

        # Use the final step of the RK4 solver as the output for this timestep.
        next_state = sol[-1]

        # Defensive: ensure we keep a (units, 4) tensor in the history buffer.
        if next_state.ndim == 3:
            # If a future change returns a full trajectory here, collapse it.
            next_state = next_state[-1]

        # Update internal history buffer for historical interpolation.
        self._update_history(next_state)
        return next_state

    def reset(self, to_state: np.ndarray = None) -> "BioReservoir":
        """Reset the reservoir to warmup state or a specified state."""
        # Clear per-timestep timing lists so each run's stats are independent.
        if getattr(self, "diagnostics", False) and hasattr(self, "_timing"):
            for lst in self._timing.values():
                lst.clear()
        # Check if warmup states exist (may be called during initialization)
        if hasattr(self, '_warmup_states') and self._warmup_states is not None:
            # 'to_state' will be None if the current state is None (first timestep)
            if to_state is None:
                # Reset internal history to warmup states
                self.states = self._warmup_states.copy()
                # Project full 4-variable state through output_indices so that
                # the Node state shape matches the current output_dim
                last_full = self._warmup_states[-1]                 # (units, 4)
                selected = last_full[:, self.output_indices]        # (units, n_selected_vars)
                to_state = selected.reshape(1, -1)                  # (1, units * n_selected_vars)
            else:
                if self.is_initialized and not np.array_equal(to_state, self._state):
                    # State is changing - reset our internal history buffer to warmup states
                    self.t = 0
                    self.states = self._warmup_states.copy()
                # else: state unchanged, don't reset history (to_state == current_state)

        # Call parent reset to properly set _state and _state_proxy
        super(BioReservoir, self).reset(to_state=to_state)


def _forward(reservoir: BioReservoir, x: np.ndarray) -> np.ndarray:
    # print("\n\n==========================================\nTimestep (t): ", reservoir.t)
    reservoir.t += 1

    # When diagnostics are on, time each component separately so callers can
    # inspect where wall time is spent: input matrix multiply vs DDE integration.
    if getattr(reservoir, "diagnostics", False) and hasattr(reservoir, "_timing"):
        _t0 = time.perf_counter()
        final_input = _compute_input(reservoir, x)
        _t1 = time.perf_counter()
        next_states = reservoir._step(final_input)
        _t2 = time.perf_counter()
        reservoir._timing["input_computation"].append(_t1 - _t0)
        reservoir._timing["dde_solve"].append(_t2 - _t1)
    else:
        final_input = _compute_input(reservoir, x)
        next_states = reservoir._step(final_input)

    # next_states may be either (units, 4) or (time, units, 4) depending on solver
    if next_states.ndim == 3:
        # Aggregate over last K RK4 substeps within this single external timestep.
        K = reservoir.rk4_substep_aggregation_window
        K = min(K, next_states.shape[0])
        window = next_states[-K:]              # (K, units, 4)
        last_state = window.mean(axis=0)       # (units, 4)
    else:
        last_state = next_states               # (units, 4)

    if last_state.ndim == 1:
        last_state = last_state.reshape(1, -1)

    selected = last_state[:, reservoir.output_indices]
    return selected.reshape(1, -1)


def _compute_input(reservoir: BioReservoir, x: np.ndarray) -> np.ndarray:
    # Raw input at this timestep, typically normalized ECG in [0, 1] or [-1, 1]
    u = x.reshape(-1, 1)          # shape (input_dim, 1)
    r = reservoir.state().T       # previous reservoir state, shape (units, 1)
    # print("Input u: ", u.T)
    # print("Previous state r (index 0): ", r[0])

    # ===== 1. Input term: Win @ u (+ noise, optional bias) =====
    # Avoid noise generation/allocation when gain is 0.
    if reservoir.noise_in and reservoir.noise_in != 0.0:
        noise_in = reservoir.noise_generator(
            dist=reservoir.noise_type,
            shape=u.shape,
            gain=reservoir.noise_in,
        )
        u_eff = u + noise_in
    else:
        u_eff = u

    inp_term = reservoir.Win @ u_eff    # (units, 1); already scaled by input_scaling in initializer
    # print("Input term before bias (index 0): ", inp_term[0])

    # Optional bias; recommend keeping it small or zero-centered.
    # If bias_scaling is used in initializer, bias is already scaled there.
    inp_term = inp_term + reservoir.bias
    # print("Input term after bias (index 0): ", inp_term[0])

    # ===== 2. Recurrent term: W @ r (+ noise) =====
    if reservoir.noise_rc and reservoir.noise_rc != 0.0:
        noise_rc = reservoir.noise_generator(
            dist=reservoir.noise_type,
            shape=r.shape,
            gain=reservoir.noise_rc,
        )
        r_eff = r + noise_rc
    else:
        r_eff = r

    rec_term = reservoir.W @ r_eff     # (units, 1)
    # print("Recurrent term before gain (index 0): ", rec_term[0])

    # Use `rc_scaling` as RECURRENT GAIN (pre-normalization)
    rec_term *= reservoir.rc_scaling
    # print("Recurrent term after applying rc_scaling gain (index 0): ", rec_term[0])

    # ===== 3. Combine into pre-drive =====
    # Combine into pre-drive
    drive = inp_term + rec_term   # (units, 1)

    # Cache for diagnostics/visualisation (optional).
    if getattr(reservoir, "diagnostics", False):
        reservoir._last_inp_term = inp_term
        reservoir._last_rec_term = rec_term
        reservoir._last_drive = drive
    # print("Combined drive before feedback (index 0): ", drive[0])

    # Optional feedback term added into the same pre-drive
    if reservoir.has_feedback:
        y = reservoir.feedback().reshape(-1, 1)
        if reservoir.noise_fb and reservoir.noise_fb != 0.0:
            noise_fb = reservoir.noise_generator(
                dist=reservoir.noise_type,
                shape=y.shape,
                gain=reservoir.noise_fb,
            )
            y_fb = reservoir.fb_activation(y) + noise_fb
        else:
            y_fb = reservoir.fb_activation(y)
        # Feedback treated like an additional input path
        drive = drive + (reservoir.Wfb @ y_fb)
        # print("Drive with feedback (index 0): ", drive[0])

    # ---- Diagnostic statistics across units (for tuning gains) ----
    # Hot-path: only compute/log per-step stats if diagnostics are enabled *and*
    # this module logger is set to DEBUG.
    if getattr(reservoir, "diagnostics", False) and logger.isEnabledFor(logging.DEBUG):
        inp_std = float(np.std(inp_term))
        rec_std = float(np.std(rec_term))
        drive_std = float(np.std(drive))
        ratio = inp_std / rec_std if rec_std > 0 else np.inf
        logger.debug(
            "t=%s std(inp)=%.3e std(rec)=%.3e std(drive)=%.3e inp/rec=%.3f",
            getattr(reservoir, "t", -1),
            inp_std,
            rec_std,
            drive_std,
            ratio,
        )

    # ===== 4. Map to DDE-safe range =====
    # No nonlinear bounding is applied. input_scaling, rc_scaling, and
    # dde_scaling are the sole parameters controlling the amplitude of x_t.
    # All nonlinearity is supplied internally by the DDE gene circuit.
    # dde_scaling should be set to keep |x_t| within the oscillatory regime
    # characterized for the bio-pixel model (~5e-3 default).
    final_input = reservoir.dde_scaling * drive

    # Cache the effective DDE drive (what is actually fed into the oscillator step).
    if getattr(reservoir, "diagnostics", False):
        reservoir._last_final_input = final_input
    # print("Final scaled final_input (index 0): ", final_input[0])

    return final_input
