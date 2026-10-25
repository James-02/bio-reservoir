import logging
import time
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import numpy as np
from reservoirpy.activationsfunc import identity
from reservoirpy.mat_gen import bernoulli, normal
from reservoirpy.mat_gen import zeros as rpy_zeros
from reservoirpy.node import Node
from reservoirpy.nodes.reservoirs.base import initialize, initialize_feedback
from reservoirpy.type import Weights
from reservoirpy.utils.random import noise, rand_generator
from reservoirpy.utils.validation import is_array
from scipy import sparse

from bioreservoir.dde import (
    _NP_D_COEFF,
    _NP_D_FRAC,
    _NP_MU,
    UnstableIntegrationError,
    generate_node_params,
    get_dde_params,
    log_dde_params,
    prepare_conductances,
    solve_dde_rk4,
    solve_dde_substep,
    substep_history,
)
from bioreservoir.realism import (
    RESET_STREAM,
    check_input_levels,
    level_weights,
    perturb_start,
    stream,
    vary_colonies,
)
from bioreservoir.topology import scale_conductances

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
        dde_kwargs: dict | None = None,
        sr: float | None = None,
        input_bias: bool = False,
        noise_rc: float = 0.1,
        noise_in: float = 0.05,
        noise_fb: float = 0.1,
        noise_type: str = "normal",
        noise_kwargs: dict = None,
        input_scaling: float | Sequence = 1e-2,
        bias_scaling: float = 1.0,
        fb_scaling: float | Sequence = 1e-4,
        input_connectivity: float = 1.0,
        rc_connectivity: float = 0.1,
        fb_connectivity: float = 0.1,
        fb_activation: Callable = identity,
        Win: Weights | Callable = bernoulli,
        W: Weights | Callable = normal,
        Wfb: Weights | Callable = bernoulli,
        bias: Weights | Callable = bernoulli,
        input_dim: int | None = None,
        feedback_dim: int | None = None,
        parallel: bool = True,
        seed: int | None = None,
        output_variables: Sequence[str] = ("A", "I", "Hi", "He"),
        diagnostics: bool = False,
        param_heterogeneity_cv: float = 0.0,
        param_heterogeneity_seed: int | None = None,
        param_heterogeneity_mean_preserving: bool = True,
        geometry: str = "per_node",
        couple_vars: Sequence[float] | None = None,
        delay_scheme: str = "substep",
        coupling: str = "abstract",
        diffusion_rate: float = 1.0,
        input_encoding: str = "signed",
        input_range: Sequence[float] = (-3.0, 8.0),
        input_sign: str = "mixed",
        heterogeneity_density_cv: float = 0.0,
        heterogeneity_dilution_cv: float = 0.0,
        input_levels: int | str = 1,
        reset_noise: float = 0.0,
        **kwargs,
    ):
        """Genetic-oscillator reservoir node (one DDE oscillator per unit).

        ``coupling`` sets how the oscillators (colonies) interact.

        * ``"abstract"`` (the v0.1 model): once per input step, each oscillator receives
          ``rc_scaling * W r`` from the observed states of the previous step, added to its
          input drive; ``W`` is a signed kernel.
        * ``"diffusion"``: colonies exchange extracellular AHL through the medium. At every
          RK4 stage, ``dHe_i/dt`` gains ``sum_j G_ij (He_j - He_i)``, a graph Laplacian
          over non-negative conductances ``G``. ``W`` gives the kernel (for example
          ``distance_matrix(signed=False)`` or ``lattice_matrix()``), scaled so that a
          colony's conductances sum to ``diffusion_rate`` on average (per model time
          unit); symmetric conductances conserve AHL. ``rc_scaling`` and ``noise_rc``
          are not used. The exchange is integrated explicitly, so an exchange too fast for
          the RK4 step (``1 / rk4_substeps``) is refused.
        * ``"none"``: independent oscillators.

        ``input_encoding="nonnegative"`` delivers the input as a non-negative inducer
        level: each sample is clipped to ``input_range`` and shifted by its lower end (a
        fixed, causal offset), and input noise cannot take it below 0;
        ``"signed"`` uses the samples as they are. ``input_sign="positive"`` makes the
        input weights (and bias) non-negative, so every colony that receives the input
        is induced by it; ``"mixed"`` keeps the random signs.

        ``geometry`` sets how the recurrent term is built when several variables are
        observed. ``"per_node"``: one ``units x units`` kernel over ``units``
        oscillators, applied to every observed variable of an oscillator with the weights
        ``couple_vars`` (default 1 each), so oscillator ``i`` is driven by
        ``sum_j K_ij sum_v w_v x_vj``. ``"v0.1"``: the v0.1 construction, in which
        ReservoirPy builds a kernel over ``units x n_observed`` points and only the first
        ``units`` drives reach the oscillators (kept to reproduce v0.1).

        ``delay_scheme`` sets how the delayed Hi is read within an input step.
        ``"substep"``: at every RK4 stage, from the Hi history at substep resolution
        (a delay of 0 is the ordinary differential equation). ``"frozen"``: once per input
        step, held for all its substeps (v0.1).

        ``param_heterogeneity_cv`` > 0 draws per-oscillator kinetic rates around the
        nominal values (log-normal, coefficient of variation ``param_heterogeneity_cv``),
        with mean equal to the nominal value unless
        ``param_heterogeneity_mean_preserving=False`` (v0.1, biased upwards).

        Realism options (:mod:`bioreservoir.realism`), off by default:
        ``heterogeneity_density_cv`` and ``heterogeneity_dilution_cv`` vary each colony's
        cell density ``d`` (with its growth factor) and washout ``mu`` (mean-preserving
        log-normal, from their own streams of the heterogeneity seed);
        ``input_levels`` (K >= 1 or ``"continuous"``) gives the input connections K levels
        in ``(0, 1] * input_scaling``, drawn with the reservoir seed (needs
        ``geometry="per_node"``); ``reset_noise`` multiplies the warm-up state that every
        input sequence starts from by log-normal factors of that coefficient of variation,
        drawn from a stream reseeded by :meth:`reseed_reset`.
        """
        if units is None and not is_array(W):
            raise ValueError("'units' must be specified if 'W' is not a matrix.")
        if geometry not in ("per_node", "v0.1"):
            raise ValueError(f"geometry must be 'per_node' or 'v0.1', got {geometry!r}.")
        if delay_scheme not in ("substep", "frozen"):
            raise ValueError(
                f"delay_scheme must be 'substep' or 'frozen', got {delay_scheme!r}.")
        if coupling not in ("abstract", "diffusion", "none"):
            raise ValueError(
                f"coupling must be 'abstract', 'diffusion' or 'none', got {coupling!r}.")
        if coupling == "diffusion" and (geometry, delay_scheme) != ("per_node", "substep"):
            raise ValueError("coupling='diffusion' needs geometry='per_node' and "
                             "delay_scheme='substep'.")
        if input_encoding not in ("signed", "nonnegative"):
            raise ValueError(f"input_encoding must be 'signed' or 'nonnegative', "
                             f"got {input_encoding!r}.")
        if input_sign not in ("mixed", "positive"):
            raise ValueError(f"input_sign must be 'mixed' or 'positive', got {input_sign!r}.")
        if input_sign == "positive" and geometry != "per_node":
            raise ValueError("input_sign='positive' needs geometry='per_node'.")
        low, high = (float(v) for v in input_range)
        if not low < high:
            raise ValueError(f"input_range must be (low, high) with low < high, got "
                             f"{tuple(input_range)}.")
        input_levels = check_input_levels(input_levels)
        if input_levels != 1 and geometry != "per_node":
            raise ValueError("input_levels other than 1 need geometry='per_node'.")
        if float(reset_noise) < 0:
            raise ValueError(f"reset_noise must be 0 or more, got {reset_noise}.")

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
        self.geometry = geometry
        self.delay_scheme = delay_scheme
        self.coupling = coupling
        self.diffusion_rate = float(diffusion_rate)
        self.input_encoding = input_encoding
        self.input_range = (low, high)
        self.input_sign = input_sign
        self.input_levels = input_levels
        if couple_vars is None:
            couple_vars = np.ones(len(self.output_indices))
        self.couple_vars = np.asarray(couple_vars, dtype=float).reshape(-1)
        if len(self.couple_vars) != len(self.output_indices):
            raise ValueError("couple_vars needs one weight per observed variable.")
        output_dim = (len(self.output_indices) * units) if units is not None else None

        super().__init__(
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
                initialize if geometry == "v0.1" else _initialize_per_node,
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
                mean_preserving=param_heterogeneity_mean_preserving,
            )
            logger.info(
                "Generated per-node kinetic heterogeneity (CV=%.3f) for %d nodes",
                self.param_heterogeneity_cv, self.units,
            )
        else:
            self.node_params = None
        self.heterogeneity_density_cv = float(heterogeneity_density_cv)
        self.heterogeneity_dilution_cv = float(heterogeneity_dilution_cv)
        if self.heterogeneity_density_cv or self.heterogeneity_dilution_cv:
            table = (self.node_params if self.node_params is not None
                     else generate_node_params(self.units, 0.0, self.dde_kwargs, seed=0))
            self.node_params = vary_colonies(
                table, self.dde_kwargs["d0"], self.heterogeneity_density_cv,
                self.heterogeneity_dilution_cv, seed=self._het_seed)

        # Imperfect reset: the stream is reseeded for every input sequence (reseed_reset).
        self.reset_noise = float(reset_noise)
        self._reset_rng = stream(self.seed, RESET_STREAM)

        # Per-node parameter table for the substep solver, and the Hi history at
        # substep resolution (the frozen v0.1 scheme reads the per-step history only).
        n_nodes = self.states.shape[1]
        self._param_table = (self.node_params if self.node_params is not None
                             else generate_node_params(n_nodes, 0.0, self.dde_kwargs, seed=0))
        if self.delay_scheme == "substep":
            self._ring, self._ring_pos = substep_history(
                self.states, self.dde_kwargs["delay"], self.rk4_substeps)

        # Conductances are built before the warm-up, so that the warm-up is coupled too.
        self.conductance = None
        self._conductances = None
        if self.coupling == "diffusion":
            kernel = W if is_array(W) else W(units, units, sr=sr, connectivity=rc_connectivity,
                                              dtype=self.dtype, seed=seed)
            self.conductance = scale_conductances(kernel, self.diffusion_rate)
            self._conductances = prepare_conductances(self.conductance)
        if self.delay_scheme == "substep":
            self._check_step()
        # A and I stay non-negative under any drive; Hi and He too under non-negative drive
        # (the abstract coupling's recurrent term is signed). Below that, the solver blew up.
        self._nonnegative = slice(0, 4) if (self.input_encoding == "nonnegative"
                                            and self.input_sign == "positive"
                                            and self.coupling != "abstract") else slice(0, 2)

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
        self._warmup_ring = ((self._ring.copy(), self._ring_pos)
                             if self.delay_scheme == "substep" else None)
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
        self._timing: dict[str, list[float]] = {
            "input_computation": [],
            "dde_solve": [],
        }

    def get_timing_summary(self) -> dict[str, Any]:
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
        out: dict[str, Any] = {}
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

        # Normalize input shape to 1D (units,) for the DDE solver. With the v0.1
        # geometry ReservoirPy produces one drive per observed value; only the first
        # `units` reach the oscillators (as in v0.1).
        n_nodes = self.states.shape[1]
        x = np.asarray(x).reshape(-1)[:n_nodes]

        if self.delay_scheme == "substep":
            try:
                next_state, self._ring_pos = solve_dde_substep(
                    self.states[-1],
                    self._ring,
                    self._ring_pos,
                    x * self.cell_coupling,
                    self.rk4_substeps,
                    parallel=self.parallel,
                    node_params=self._param_table,
                    conductance=self._conductances,
                    **self.dde_kwargs,
                )
            except ZeroDivisionError as exc:
                raise UnstableIntegrationError(self._unstable_message()) from exc
            self._check_state(next_state)
            self._update_history(next_state)
            return next_state

        # Scale the normalized reservoir drive by biological cell coupling strength.
        # This models how strongly the external AHL (He-like signal) influences each oscillator.
        # input_effect = self.cell_coupling * x
        # This input will be used in the DDE equations to combine with the external AHL.
        try:
            sol = solve_dde_rk4(
                self.states,
                self.rk4_timespan,
                input_vec=x * self.cell_coupling,
                parallel=self.parallel,
                node_params=self.node_params,
                **self.dde_kwargs,
            )
        except ZeroDivisionError as exc:
            raise UnstableIntegrationError(self._unstable_message()) from exc

        # The state at the end of the input step (the solver returns every substep).
        next_state = sol[-1]
        self._check_state(next_state)

        # Update internal history buffer for historical interpolation.
        self._update_history(next_state)
        return next_state

    def _check_step(self) -> None:
        """Refuse an RK4 step too long for a colony's fastest exchange of AHL.

        RK4 is stable on a mode decaying at rate r only for steps below about 2.78 / r. Hi
        and He relax towards each other at D / (1 - d) (He's washout mu adds to it), and the
        diffusive coupling adds at most twice the colony's total conductance (Gershgorin's
        bound on the graph Laplacian).
        """
        table = self._param_table
        D, d, mu = table[:, _NP_D_COEFF], table[:, _NP_D_FRAC], table[:, _NP_MU]
        rates = D / (1 - d) + mu
        if self._conductances is not None:
            rates = rates + 2 * self._conductances.row_sum
        fastest = float(np.max(rates))
        if fastest > 2.7 * self.rk4_substeps:
            raise UnstableIntegrationError(
                f"The AHL exchange is too fast for {self.rk4_substeps} RK4 substeps: its "
                f"fastest mode decays at up to {fastest:.3g} per time unit, so the explicit "
                f"solver needs at least {int(np.ceil(fastest / 2.7))} substeps (or a lower "
                "diffusion_rate or cell density).")

    def _check_state(self, state: np.ndarray) -> None:
        """Stop a run whose states blew up: non-finite, or negative where they cannot be."""
        if not np.isfinite(state).all() or state[:, self._nonnegative].min() < -1e-6:
            raise UnstableIntegrationError(self._unstable_message())

    def _unstable_message(self) -> str:
        return (f"The colony states blew up with {self.rk4_substeps} RK4 substeps per input "
                "step: the step is too long for the fastest rates of some colonies (such as "
                "fast degradation at large kinetic heterogeneity); use more substeps.")

    def _restore_warmup(self) -> None:
        """Return the state and delay histories to their values after warm-up.

        With ``reset_noise`` > 0, an initialized node starts each input sequence from the
        warm-up state perturbed by :func:`bioreservoir.realism.perturb_start`. The reset
        inside ``initialize()`` is left unperturbed, so a sequence draws one perturbation
        whether or not it is the node's first.
        """
        self.states = self._warmup_states.copy()
        if getattr(self, "_warmup_ring", None) is not None:
            self._ring = self._warmup_ring[0].copy()
            self._ring_pos = self._warmup_ring[1]
        if getattr(self, "reset_noise", 0.0) > 0 and self.is_initialized:
            ring = self._ring if self.delay_scheme == "substep" else None
            self.states, ring = perturb_start(self.states, ring, self.reset_noise,
                                              self._reset_rng)
            if ring is not None:
                self._ring = ring

    def reseed_reset(self, seed: int) -> None:
        """Draw the following reset perturbations (``reset_noise``) from ``seed``'s stream."""
        self._reset_rng = stream(seed, RESET_STREAM)

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
                self._restore_warmup()
                # Project full 4-variable state through output_indices so that
                # the Node state shape matches the current output_dim
                last_full = self._warmup_states[-1]                 # (units, 4)
                selected = last_full[:, self.output_indices]        # (units, n_selected_vars)
                to_state = selected.reshape(1, -1)                  # (1, units * n_selected_vars)
            else:
                if self.is_initialized and not np.array_equal(to_state, self._state):
                    # State is changing - reset our internal history buffer to warmup states
                    self.t = 0
                    self._restore_warmup()
                # else: state unchanged, don't reset history (to_state == current_state)

        # Call parent reset to properly set _state and _state_proxy
        super().reset(to_state=to_state)


def _initialize_per_node(
    reservoir,
    x=None,
    y=None,
    sr=None,
    input_scaling=None,
    bias_scaling=None,
    input_connectivity=None,
    rc_connectivity=None,
    W_init=None,
    Win_init=None,
    bias_init=None,
    input_bias=None,
    seed=None,
):
    """ReservoirPy initializer with one coordinate per oscillator.

    Builds the ``units x units`` kernel ``K``, the operator ``K (x) couple_vars`` from the
    observed state (variables of oscillator j adjacent) to the ``units`` drives, and
    ``Win`` and the bias with one row per oscillator. With diffusive coupling ``W`` is the
    conductance matrix instead (the drive has no recurrent term).
    """
    if x is None:
        return
    reservoir.set_input_dim(x.shape[1])
    dtype = reservoir.dtype
    n = int(reservoir.hypers["units"])
    if reservoir.coupling == "diffusion":
        W = reservoir.conductance
    else:
        K = W_init if is_array(W_init) else W_init(
            n, n, sr=sr, connectivity=rc_connectivity, dtype=dtype, seed=seed)
        if K.shape != (n, n):
            raise ValueError(f"The recurrent kernel must be {n} x {n} (one row per "
                             f"oscillator), got {K.shape}.")
        if len(reservoir.couple_vars) == 1 and reservoir.couple_vars[0] == 1.0:
            W = K  # identical to the v0.1 construction
        else:
            weights = sparse.csr_matrix(reservoir.couple_vars.reshape(1, -1))
            W = sparse.kron(sparse.csr_matrix(K), weights, format="csr")

    Win = Win_init if is_array(Win_init) else Win_init(
        n, x.shape[1], input_scaling=input_scaling, connectivity=input_connectivity,
        dtype=dtype, seed=seed)
    if Win.shape != (n, x.shape[1]):
        raise ValueError(f"Win must be {n} x {x.shape[1]}, got {Win.shape}.")
    if input_bias:
        bias = bias_init if is_array(bias_init) else bias_init(
            n, 1, input_scaling=bias_scaling, connectivity=input_connectivity, dtype=dtype,
            seed=seed)
    else:
        bias = rpy_zeros(n, 1, dtype=dtype)
    if reservoir.input_sign == "positive":
        Win, bias = abs(Win), abs(bias)
    Win = level_weights(Win, reservoir.input_levels, seed)

    reservoir.set_param("W", W.astype(dtype))
    reservoir.set_param("Win", Win.astype(dtype))
    reservoir.set_param("bias", bias.astype(dtype))
    reservoir.set_param("internal_state", reservoir.zero_state())


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

    # The observed variables at the end of the input step, oscillator by oscillator.
    selected = next_states[:, reservoir.output_indices]  # (units, n_observed)
    return selected.reshape(1, -1)


def _compute_input(reservoir: BioReservoir, x: np.ndarray) -> np.ndarray:
    # Raw input at this timestep, typically normalized ECG in [0, 1] or [-1, 1]
    u = x.reshape(-1, 1)          # shape (input_dim, 1)
    nonnegative = reservoir.input_encoding == "nonnegative"
    if nonnegative:
        low, high = reservoir.input_range
        u = np.clip(u, low, high) - low
    # print("Input u: ", u.T)

    # ===== 1. Input term: Win @ u (+ noise, optional bias) =====
    # Avoid noise generation/allocation when gain is 0.
    if reservoir.noise_in and reservoir.noise_in != 0.0:
        noise_in = reservoir.noise_generator(
            dist=reservoir.noise_type,
            shape=u.shape,
            gain=reservoir.noise_in,
        )
        u_eff = u + noise_in
        if nonnegative:
            u_eff = np.maximum(u_eff, 0.0)  # noise cannot make a concentration negative
    else:
        u_eff = u

    inp_term = reservoir.Win @ u_eff    # (units, 1); already scaled by input_scaling in initializer
    # print("Input term before bias (index 0): ", inp_term[0])

    # Optional bias; recommend keeping it small or zero-centered.
    # If bias_scaling is used in initializer, bias is already scaled there.
    inp_term = inp_term + reservoir.bias
    # print("Input term after bias (index 0): ", inp_term[0])

    # ===== 2. Recurrent term: W @ r (+ noise), the abstract coupling only =====
    if reservoir.coupling != "abstract":
        rec_term = np.zeros_like(inp_term)
    else:
        r = reservoir.state().T       # previous reservoir state, shape (units, 1)
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
        # Use `rc_scaling` as RECURRENT GAIN (pre-normalization)
        rec_term *= reservoir.rc_scaling

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
