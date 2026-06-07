"""Named hyperparameter study registry.

Each study is a self-contained description of which parameters to optimise
(with Optuna suggest specs) and which to hold fixed.  The objective function
in classification.py reads the active study config and dispatches suggest
calls accordingly — there is no commented-out code in the objective.

Structure
---------
BASELINE      : dict
    Default fixed values for every parameter.  Studies override a subset.
    These were established from preliminary iterative experiments and are
    validated by the dedicated studies (g1–g7).

STUDY_REGISTRY : dict[str, dict]
    Maps a study name (used on the CLI via --study) to a config dict with:
        "description"  str            Human-readable purpose.
        "sampler"      str            "tpe" | "random" | "grid".
        "n_trials"     int            Recommended trial budget.
        "folds"        int            Recommended CV folds (1 for exploration).
        "instances"    int            Dataset pool size.
        "binary"       bool           Use binary classification.
        "optimize"     dict[str,dict] Parameters to pass to trial.suggest_*.
        "fixed"        dict[str,Any]  Per-study overrides of BASELINE.

Suggest spec format
-------------------
Float:       {"type": "float",  "low": ..., "high": ..., "log": bool}
Int:         {"type": "int",    "low": ..., "high": ..., "step": int}
Categorical: {"type": "categorical", "choices": [...]}

Conditional parameters (e.g. interaction_diameters only when
topology_type == "distance") are listed in "conditional_on":
    {"type": "float", "low": ..., "high": ..., "conditional_on": "topology_type",
     "condition_value": "distance"}
When the condition is not met, the BASELINE value is used instead.

Phase notes
-----------
Phase 1 (g1–g7)   Broad single-fold exploratory studies.  Run in parallel
                   across servers; each study is independent.
Phase 2 (r1)       Joint refinement in narrow ranges from Phase 1 best values.
                   Update BASELINE below with Phase 1 results before running.
Phase 3 (ro1, f2, f3)
                   Final results:
                   ro1-readout  — readout optimisation on frozen r1a-refined-best
                                  states (handled by readout.py, not this file).
                                  Run on same host as r1a-refined via:
                                    python -m optimization.optimize rerun-best
                                      --study r1a-refined --folds 5
                                    python -m optimization.optimize research --type readout
                                      --study_name ro1-readout
                                      --trial_name r1a-refined-best
                   f2-unbalanced — grid over class-size multiples; no re-opt needed.
                   f3-binary     — single 5-fold CV with binary=True; no re-opt needed.
"""

from __future__ import annotations
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Baseline — best-known fixed values from preliminary iterative experiments.
# Update after evaluating each Phase and before launching the next Phase.
# ---------------------------------------------------------------------------
BASELINE: Dict[str, Any] = {
    "reservoir_type":          "bioreservoir",
    # -------------------------------------------------------------------------
    # Updated from Phase 1 results (March 2026).
    # Each value is the best-found from the corresponding g-study (1-fold, N=250).
    # These are held fixed in all Phase 1 studies except the one being explored.
    # -------------------------------------------------------------------------
    # --- Reservoir structure ---
    # units: g6 pending (run after Phase 2); keep 250 until r1a/r1b confirms.
    "units":                    250,
    "warmup":                   50,        # g2-dde Phase 1 best (was 40)
    "rk4_substeps":             60,        # g2-dde Phase 1 best (was 44); at range ceiling — r1 will confirm
    "rk4_substep_aggregation":  1,
    # --- Scaling  (g1-scaling Phase 1 best) ---
    # Note: g1 best is substantially different from preliminary values.
    # g7 coupling study confirms cell_coupling=8.862 in its own search.
    "input_scaling":            0.001031,  # g1 best (was 0.08011)
    "rc_scaling":               0.0005707, # g1 best (was 0.01949)
    "dde_scaling":              0.007459,  # g1 best (was 0.003703)
    "cell_coupling":            8.862,     # g7 best (was 16.28)
    "bias_scaling":             0.1,
    # --- Readout aggregation  (g5b-readout Phase 1 best) ---
    # last_N with window=36 outperforms "final" across all g5b trials.
    # readout_aggregation is active in all studies via params.get(..., "final") fallback.
    "readout_aggregation":      "last_N",  # g5b best (was "final")
    "readout_window":           36,        # g5b best (was 1); only active for "last_N"
    # --- Noise  (g4-noise-reservoir Phase 1 best) ---
    # Phase 1 noise studies showed modest F1 (~0.79–0.80), suggesting noise is a
    # weak lever; these are the best-found values but contribution is marginal.
    "noise_in":                 0.01442,   # g4-reservoir best (was 0.07871)
    "noise_rc":                 0.2397,    # g4-reservoir best (was 0.1211)
    # --- Dataset augmentation  (g4-noise-data Phase 1 best) ---
    "noise_rate":               0.2999,    # g4-data best (was 0.0825)
    "noise_ratio":              0.2862,    # g4-data best (was 0.2348)
    "preserve_split":           False,     # True = keep original Kachuee train/test split
    # --- Metric semantics ---
    # Primary optimization/reporting average used for f1/precision/recall.
    # Submission default: macro treats each rhythm class equally and is the
    # most defensible headline metric across balanced, unbalanced, and binary
    # tasks. Weighted averages are still stored for every run.
    "objective_average":        "macro",
    # --- Dataset class balance ---
    "balance_classes":          True,
    "max_per_class":            None,      # None = use minority class size (fully balanced)
    # --- Input layer  (g5-input Phase 1 best) ---
    "input_bias":               False,     # g5 confirmed (was False)
    "fixed_bias_scaling":       0.1,       # g5 best
    "input_connectivity":       0.3679,    # g5 best (was 0.19)
    # --- Topology  (g3a-topology-distance Phase 1 best) ---
    "topology_type":            "distance",  # g3a confirmed; r1a/r1b split will compare properly
    "sparsity":                 0.134,     # g3a best (was 0.5)
    "signed":                   True,      # g3a best (was False)
    "p_excite":                 0.66,      # g3a best (was 0.5)
    # interaction_diameters is fixed at the biologically-derived value — never optimised.
    "interaction_diameters":    8.2,
    "fade_alpha":               1.014e-06, # g3a best (was 1.813e-05)
    # --- Output variables  (g5-input Phase 1 best) ---
    "output_variables":         "He",      # g5 best (was "IHe"); single H state variable
    # --- Dataset scaling  (g0-scaler Phase 1 confirmed) ---
    "scaler_type":              "sequence_zscore",  # g0 confirmed (was default)
    # --- Biological heterogeneity ---
    "param_heterogeneity_cv":   0.0,     # CV of per-node kinetic parameter variation (0 = homogeneous)
    "param_heterogeneity_seed": None,    # separate seed for heterogeneity RNG (None = use reservoir seed)
}

# ---------------------------------------------------------------------------
# Phase 1 — Broad exploratory, 1 fold, N=250
# Each study fixes all parameters except the group under investigation.
# These studies can be deployed simultaneously across servers.
# ---------------------------------------------------------------------------

_PHASE1_COMMON_FIXED = {
    # Everything at baseline unless overridden by the study's own "fixed" key.
    # The objective merges: BASELINE <- study["fixed"] <- optimized params.
}

STUDY_REGISTRY: Dict[str, Dict[str, Any]] = {

    # -----------------------------------------------------------------------
    # g0-scaler
    # Purpose: ablation study justifying sequence_zscore as the dataset
    #   preprocessing choice (paper Methods section).
    #   - Six scalers are compared under identical reservoir conditions.
    #   - sequence_zscore: per-sequence zero-mean/unit-variance — ECG-appropriate
    #     because each heartbeat varies in absolute amplitude; normalising per
    #     beat preserves morphological shape while removing amplitude bias.
    #   - sequence_minmax: per-sequence [0, 1] — preserves shape, retains
    #     amplitude variation within a beat but loses cross-beat range info.
    #   - sequence_minmax_pos: per-sequence [0.1, 0.9] — fully positive drive;
    #     avoids zero-valued inputs that produce no reservoir response.
    #   - minmax: global per-time-index — collapses per-beat morphological
    #     differences; discards information the reservoir needs.
    #   - standard: global per-time-index z-score — same limitation as minmax.
    #   - none: raw unscaled signal — sanity check, expected worst performance.
    #   GridSampler: one trial per scaler per pass.  n_trials=30 gives 5 passes
    #   so each of the 6 scalers is evaluated 5 times for variance estimates.
    # -----------------------------------------------------------------------
    "g0-scaler": {
        "description": (
            "Dataset scaling ablation: 6 scalers compared under identical reservoir. "
            "Justifies sequence_zscore choice in Methods. Uses GridSampler "
            "(5 passes × 6 scalers = 30 trials) for per-scaler variance estimates."
        ),
        "sampler":   "grid",
        "n_trials":  30,   # 5 complete passes × 6 scalers
        "folds":     1,
        "instances": 4015,
        "binary":    False,
        "optimize": {
            "scaler_type": {
                "type": "categorical",
                "choices": [
                    "sequence_zscore",    # per-sequence zero-mean/unit-var (expected best)
                    "sequence_minmax",    # per-sequence [0, 1]
                    "sequence_minmax_pos",# per-sequence [0.1, 0.9] — fully positive
                    "minmax",            # global per time-index min-max
                    "standard",          # global per time-index z-score
                    "none",              # no scaling (sanity check)
                ],
            },
        },
        "fixed": {},
        "_grid": {
            "scaler_type": [
                "sequence_zscore",
                "sequence_minmax",
                "sequence_minmax_pos",
                "minmax",
                "standard",
                "none",
            ],
        },
    },

    # -----------------------------------------------------------------------
    # g1-scaling
    # Purpose: characterise the scaling parameter landscape.
    #   - Determines optimal input_scaling / rc_scaling ratio (Fig S4).
    #   - Confirms whether dde_scaling has independent effect or is absorbed.
    #   - ~100 trials is sufficient for 3 log-scale continuous params.
    # -----------------------------------------------------------------------
    "g1-scaling": {
        "description": (
            "Scaling parameter landscape: input_scaling, rc_scaling, dde_scaling. "
            "Confirms ratio alpha_u/alpha_r governs performance and whether "
            "dde_scaling has independent effect. "
            "Phase 1 best (0.8848): input=0.080, rc=0.019, dde=0.0037 — all interior, "
            "ranges extended to confirm ceilings."
        ),
        "sampler":   "tpe",
        "n_trials":  200,
        "folds":     1,
        "instances": 4015,
        "binary":    False,
        "optimize": {
            # Phase 1 best was 0.080; high end extended from 0.1 → 0.5 to confirm ceiling.
            "input_scaling": {"type": "float", "low": 1e-4, "high": 5e-1, "log": True},
            # Phase 1 best was 0.019; high end extended from 0.1 → 0.2 to confirm ceiling.
            "rc_scaling":    {"type": "float", "low": 1e-4, "high": 2e-1, "log": True},
            # Phase 1 best was 0.0037; extended to 0.02 (previously only up to 0.01).
            "dde_scaling":   {"type": "float", "low": 5e-5, "high": 2e-2, "log": True},
        },
        "fixed": {},
    },

    # -----------------------------------------------------------------------
    # g2-dde
    # Purpose: validate integration convergence and warmup adequacy.
    #   - Finds optimal warmup and rk4_substeps (Fig S2, Methods validation).
    #   - Phase 1 identified warmup=40 AT THE FLOOR of [40,160] — extended down
    #     to 10 to find the true minimum.
    #   - readout_window is REMOVED: Phase 1 showed only +0.007 F1 benefit
    #     (window=110 over window=1) and r1-refined confirms "final" timestep
    #     aggregation dominates (17/20 top trials). readout_window is now its
    #     own study (g5b-readout).
    # -----------------------------------------------------------------------
    "g2-dde": {
        "description": (
            "DDE integration parameters: warmup and rk4_substeps. "
            "Validates solver accuracy claims in Methods. "
            "Phase 1 best (0.8667): warmup=40 (was at floor — now extended down), "
            "rk4_substeps=44. readout_window moved to g5b-readout."
        ),
        "sampler":   "tpe",
        "n_trials":  120,
        "folds":     1,
        "instances": 4015,
        "binary":    False,
        "optimize": {
            # Phase 1 best was 40 — the FLOOR of [40,160]. Floor extended to 10.
            "warmup":       {"type": "int", "low": 10, "high": 80, "step": 5},
            "rk4_substeps": {"type": "int", "low": 12, "high": 64, "step": 4},
        },
        "fixed": {},
    },

    # -----------------------------------------------------------------------
    # g3-topology  ***DEPRECATED — reference only***
    # The joint study is invalid: random topology had 100% crash rate in Phase 1 
    # due to a code bug (fixed), and the shared BASELINE params were tuned for 
    # distance topology, making the comparison unfair.
    # See g3a-topology-distance and g3b-topology-random for the redesigned studies.
    # -----------------------------------------------------------------------
    "g3-topology": {
        "description": (
            "DEPRECATED — Joint topology comparison. Invalid: random topology crashed "
            "(100%) due to output_dim override bug (fixed). BASELINE params tuned for "
            "distance, making the comparison structurally unfair. "
            "Replaced by g3a-topology-distance + g3b-topology-random."
        ),
        "sampler":   "tpe",
        "n_trials":  0,   # Do not run; kept for historical reference only.
        "folds":     1,
        "instances": 4015,
        "binary":    False,
        "optimize": {
            "topology_type":    {"type": "categorical", "choices": ["distance", "random"]},
            "sparsity":         {"type": "float",       "low": 0.1,  "high": 1.0},
            "signed":           {"type": "categorical", "choices": [True, False]},
            "fade_alpha": {
                "type": "float", "low": 1e-5, "high": 0.1, "log": True,
                "conditional_on": "topology_type", "condition_value": "distance",
            },
        },
        "fixed": {},
    },

    # -----------------------------------------------------------------------
    # g3a-topology-distance  ***MOST IMPORTANT STUDY — PRIMARY CLAIM***
    # Purpose: optimise the biological distance-based topology independently.
    #   - topology_type fixed to "distance" — never competes with random.
    #   - sparsity: fraction of distance-decay connections retained.
    #   - fade_alpha: spatial decay rate (range extended down to 1e-7 vs
    #     Phase 1's 1e-5 floor; Phase 1 best was AT the floor).
    #   - signed / p_excite: E/I balance of the biological connectivity.
    #   - interaction_diameters is biologically fixed at 8.2 — not optimised.
    #   - 200 trials: enough to characterise this 4-parameter space with TPE.
    # This study directly supports the biological novelty claim (Fig S5/S6).
    # -----------------------------------------------------------------------
    "g3a-topology-distance": {
        "description": (
            "Distance-based topology optimisation (standalone). "
            "Optimises sparsity, fade_alpha, signed, p_excite independently of random. "
            "fade_alpha range extended to 1e-7 (Phase 1 best was at 1e-5 floor). "
            "Primary study supporting the biological distance-connectivity claim."
        ),
        "sampler":   "tpe",
        "n_trials":  200,
        "folds":     1,
        "instances": 4015,
        "binary":    False,
        "optimize": {
            # Fraction of the distance-decay matrix connections that survive the
            # random sparsity mask.  Low sparsity = sparser (fewer kept).
            "sparsity": {"type": "float", "low": 0.05, "high": 1.0},
            "signed":   {"type": "categorical", "choices": [True, False]},
            # E/I balance: only meaningful when signed=True.
            "p_excite": {
                "type": "float", "low": 0.3, "high": 0.7,
                "conditional_on": "signed", "condition_value": True,
            },
            # Phase 1 best=1.813e-5 was at its range FLOOR (1e-5); extend down.
            "fade_alpha": {"type": "float", "low": 1e-7, "high": 1e-2, "log": True},
        },
        "fixed": {
            "topology_type": "distance",
        },
    },

    # -----------------------------------------------------------------------
    # g3b-topology-random  ***BASELINE COMPARISON FOR g3a***
    # Purpose: establish a fair random-ESN baseline, optimised independently.
    #   - topology_type fixed to "random" — full, unbiased TPE search.
    #   - rc_connectivity: direct fraction of non-zero edges (NOT 1-sparsity).
    #     This avoids the sign-flip confusion in the joint g3 study.
    #   - signed / p_excite: same E/I options as g3a for a fair structural comparison.
    #   - fade_alpha NOT included (irrelevant for random topology).
    #   - 150 trials: random has fewer parameters, so fewer trials needed.
    # -----------------------------------------------------------------------
    "g3b-topology-random": {
        "description": (
            "Random topology optimisation (standalone ESN baseline). "
            "Uses rc_connectivity (direct edge density, 0–1) not inverted sparsity. "
            "Provides a fair baseline comparison for g3a biological topology. "
            "signed and p_excite included for structural symmetry with g3a."
        ),
        "sampler":   "tpe",
        "n_trials":  150,
        "folds":     1,
        "instances": 4015,
        "binary":    False,
        "optimize": {
            # Direct edge density: 0=no connections, 1=fully connected.
            # Distinct from g3a's 'sparsity' to avoid semantic sign-flip confusion.
            "rc_connectivity": {"type": "float", "low": 0.05, "high": 1.0},
            "signed": {"type": "categorical", "choices": [True, False]},
            # E/I balance: only meaningful when signed=True.
            "p_excite": {
                "type": "float", "low": 0.3, "high": 0.7,
                "conditional_on": "signed", "condition_value": True,
            },
        },
        "fixed": {
            "topology_type": "random",
        },
    },

    # -----------------------------------------------------------------------
    # g4-noise-reservoir
    # Purpose: characterise sensitivity to reservoir noise (Fig S7).
    #   - Not to "find optimal noise" but to show stability within a range.
    #   - Robustness demonstration supports biological plausibility argument.
    # -----------------------------------------------------------------------
    "g4-noise-reservoir": {
        "description": (
            "Reservoir noise sensitivity: noise_in and noise_rc. "
            "Characterises the stable operating range for Fig S7 (noise robustness)."
        ),
        "sampler":   "tpe",
        "n_trials":  100,
        "folds":     1,
        "instances": 4015,
        "binary":    False,
        "optimize": {
            "noise_in": {"type": "float", "low": 0.0, "high": 0.3},
            "noise_rc": {"type": "float", "low": 0.0, "high": 0.3},
        },
        "fixed": {},
    },

    # -----------------------------------------------------------------------
    # g4-noise-data
    # Purpose: characterise sensitivity to training data augmentation.
    #   - Finds whether noise augmentation helps or hurts and by how much.
    # -----------------------------------------------------------------------
    "g4-noise-data": {
        "description": (
            "Data augmentation sensitivity: noise_rate and noise_ratio. "
            "Quantifies contribution of Gaussian noise augmentation to generalisation."
        ),
        "sampler":   "tpe",
        "n_trials":  80,
        "folds":     1,
        "instances": 4015,
        "binary":    False,
        "optimize": {
            "noise_rate":  {"type": "float", "low": 0.0, "high": 0.3},
            "noise_ratio": {"type": "float", "low": 0.0, "high": 0.3},
        },
        "fixed": {},
    },

    # -----------------------------------------------------------------------
    # g4a-augmentation-grid
    # Purpose: re-validate data augmentation parameters with the optimized
    #   reservoir from Phase 2 (r1a-refined trial 416, N=1000).
    #   Phase 1 g4-noise-data found noise_rate=0.2999/noise_ratio=0.2862
    #   at N=250 with unoptimized reservoir; these values may be suboptimal
    #   at the final operating point.
    #   Grid includes 0.0 to confirm that augmentation is beneficial.
    #   5-fold CV for reliable variance estimates at each grid point.
    # -----------------------------------------------------------------------
    "g4a-augmentation-grid": {
        "description": (
            "Data augmentation re-validation grid: noise_rate × noise_ratio "
            "(7×7 = 49 points, 5-fold CV). Uses r1a-refined best (trial 416, "
            "N=1000) with default KNN readout. Includes 0.0 to test whether "
            "augmentation is beneficial at the final reservoir operating point."
        ),
        "sampler":   "grid",
        "n_trials":  49,
        "folds":     5,
        "instances": 4015,
        "binary":    False,
        "optimize": {
            "noise_rate":  {"type": "categorical", "choices": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]},
            "noise_ratio": {"type": "categorical", "choices": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]},
        },
        "fixed": {
            # r1a-refined best trial (trial 416, F1=0.9047)
            "topology_type":       "distance",
            "signed":              True,
            "input_bias":          False,
            "units":               1000,
            "warmup":              50,
            "rk4_substeps":        64,
            "input_scaling":       0.0277,
            "rc_scaling":          0.007036,
            "dde_scaling":         0.0006439,
            "cell_coupling":       12.75,
            "sparsity":            0.576,
            "fade_alpha":          2.803e-05,
            "p_excite":            0.8354,
            "input_connectivity":  0.5282,
            "output_variables":    "IHe",
            "readout_aggregation": "signal_mean",
            "noise_in":            0.01442,
            "noise_rc":            0.2397,
        },
        "_grid": {
            "noise_rate":  [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
            "noise_ratio": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        },
        # 49 grid points → 49 parallel Optuna workers, each trial does
        # 5 sequential folds. At N=1000 each trial needs ~3 GB RAM.
        "max_processes": 49,
    },

    # -----------------------------------------------------------------------
    # g5-input
    # Purpose: justify hardcoded input layer design choices.
    #   - input_connectivity = 1.0 (full connectivity, every node receives input)
    #   - input_bias = False (no bias term)
    #   - If bias is added, characterise optimal bias_scaling.
    # -----------------------------------------------------------------------
    "g5-input": {
        "description": (
            "Input/output layer sensitivity: input_connectivity, input_bias, "
            "bias_scaling, and output_variables (subset of DDE state vars). "
            "Validates fixed I/O layer choices. bias_scaling is conditional on "
            "input_bias=True; output_variables chooses which DDE state variables "
            "the readout observes."
        ),
        "sampler":   "tpe",
        "n_trials":  150,
        "folds":     1,
        "instances": 4015,
        "binary":    False,
        "optimize": {
            "input_connectivity": {"type": "float", "low": 0.1, "high": 1.0},
            "input_bias":         {"type": "categorical", "choices": [True, False]},
            # Conditional: only when input_bias == True
            "bias_scaling": {
                "type": "float", "low": 1e-4, "high": 1e-1, "log": True,
                "conditional_on": "input_bias", "condition_value": True,
            },
            # Which DDE state variables the readout layer observes.
            # "all"  = A, I, Hi, He  (4 x units features)
            # "AI"   = A, I          (2 x units — production-layer signals)
            # "IHi"  = I, Hi         (2 x units — intracellular dynamics)
            # "IHe"  = I, He         (2 x units — intra + extracellular QS signal)
            # "AHe"  = A, He         (2 x units — activation + QS communication)
            # "He"   = He            (1 x units — extracellular QS signal only)
            "output_variables": {
                "type": "categorical",
                "choices": ["all", "AI", "IHi", "IHe", "AHe", "He"],
            },
        },
        "fixed": {},
    },

    # -----------------------------------------------------------------------
    # g5b-readout
    # Purpose: standalone characterisation of readout aggregation strategy.
    #   - readout_aggregation determines HOW the reservoir state time-series is
    #     collapsed to a feature vector for the KNN readout.
    #   - readout_window only applies for "last_N" aggregation; conditional.
    #   - Phase 1 RESULT: "last_N" with window=36 wins (F1=0.7831).
    #     "final" did not appear in top-10. BASELINE updated to last_N/36.
    #     r1a/r1b will optimise window jointly with all other parameters.
    # -----------------------------------------------------------------------
    "g5b-readout": {
        "description": (
            "Readout aggregation strategy: final timestep vs signal_mean vs last_N window. "
            "Standalone Phase 1 study (moved from g2-dde). "
            "readout_window is conditional on readout_aggregation == 'last_N'. "
            "Phase 1 result: last_N/window=36 best (F1=0.7831); BASELINE updated accordingly."
        ),
        "sampler":   "tpe",
        "n_trials":  100,
        "folds":     1,
        "instances": 4015,
        "binary":    False,
        "optimize": {
            # Aggregation strategy for collapsing reservoir state → feature vector.
            # "final"       = last timestep only
            # "signal_mean" = mean over ECG signal region (non-zero-padded)
            # "last_N"      = mean over last readout_window timesteps (Phase 1 winner)
            "readout_aggregation": {
                "type": "categorical",
                "choices": ["final", "signal_mean", "last_N"],
            },
            # Only sampled when readout_aggregation == "last_N".
            # Phase 1 g2 best was 110; range covers [1, 200] to explore fully.
            "readout_window": {
                "type": "int", "low": 1, "high": 200,
                "conditional_on": "readout_aggregation", "condition_value": "last_N",
            },
        },
        "fixed": {},
    },

    # -----------------------------------------------------------------------
    # g6-units
    # Purpose: reservoir size scaling curve (Fig S3).
    #   - Grid sampler: exactly 1 trial per unit count (no repeats).
    #   - Run multiple times (via multiple processes hitting same DB) to get
    #     variance bars: with a fixed seed the grid will always revisit the
    #     same points, so use different job_ids (which vary the trial seed
    #     in objective if needed).
    # NOTE: with GridSampler and n_trials == len(choices), each grid point
    #       is trialled exactly once.  Use --processes 1 to avoid race
    #       conditions where concurrent workers duplicate the same point.
    # PHASE NOTE: Uses Phase 2 r1a-refined best-trial values as fixed baseline
    #   so that the scaling curve reflects performance at optimized parameters.
    #   Running with Phase 1 baseline produces a flat curve because Phase 1
    #   params were calibrated at N=250 and underperform at larger sizes.
    # -----------------------------------------------------------------------
    "g6-units": {
        "description": (
            "Reservoir size scaling: units from 50 to 1000. "
            "Generates Fig S3 (performance vs size). "
            "Fixed baseline from r1a-refined best trial (Phase 2). "
            "Uses GridSampler — one trial per grid point."
        ),
        "sampler":   "grid",
        "n_trials":  10,   # exactly 1 trial per grid point
        "folds":     1,
        "instances": 4015,
        "binary":    False,
        "optimize": {
            "units": {"type": "categorical", "choices": [50, 100, 150, 200, 250, 300, 400, 500, 750, 1000]},
        },
        "fixed": {
            # r1a-refined best trial (trial 416, F1=0.9047)
            "topology_type":       "distance",
            "signed":              True,
            "input_bias":          False,
            "warmup":              50,
            "rk4_substeps":        64,
            "input_scaling":       0.0277,
            "rc_scaling":          0.007036,
            "dde_scaling":         0.0006439,
            "cell_coupling":       12.75,
            "sparsity":            0.576,
            "fade_alpha":          2.803e-05,
            "p_excite":            0.8354,
            "input_connectivity":  0.5282,
            "output_variables":    "IHe",
            "readout_aggregation": "signal_mean",
            "noise_in":            0.01442,
            "noise_rc":            0.2397,
            "noise_rate":          0.2999,
            "noise_ratio":         0.2862,
        },
        "_grid": {
            "units": [50, 100, 150, 200, 250, 300, 400, 500, 750, 1000],
        },
    },

    # -----------------------------------------------------------------------
    # g7-coupling
    # Purpose: determine whether cell_coupling is a free parameter or
    #   redundant with input_scaling / rc_scaling.
    #   - If the importance plot shows cell_coupling ≈ 0 relative to scaling,
    #     it can be fixed to 1.0 with a quantitative justification.
    # -----------------------------------------------------------------------
    "g7-coupling": {
        "description": (
            "cell_coupling redundancy: optimize cell_coupling alongside scaling. "
            "If importance near zero, fixes cell_coupling=1.0 permanently."
        ),
        "sampler":   "tpe",
        "n_trials":  120,
        "folds":     1,
        "instances": 4015,
        "binary":    False,
        "optimize": {
            "cell_coupling": {"type": "float", "low": 1e-2, "high": 1e2, "log": True},
            "dde_scaling":   {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
            "input_scaling": {"type": "float", "low": 1e-4, "high": 1e-1, "log": True},
            "rc_scaling":    {"type": "float", "low": 1e-4, "high": 1e-1, "log": True},
        },
        "fixed": {},
    },

    # -----------------------------------------------------------------------
    # Phase 2 — Joint Refinement
    # Update fixed values below with the best results from Phase 1 before
    # running this study.  Narrow ranges (±0.5–1 order of magnitude around
    # Phase 1 best) allow TPE to find cross-parameter optima.
    # -----------------------------------------------------------------------
    # r1a-refined  (distance topology)
    # Joint refinement over all key parameters with topology_type fixed to
    # "distance" (biologically motivated). This is the primary paper result.
    # BASELINE updated from Phase 1 before this run.
    # r1a and r1b run simultaneously on separate hosts for fair comparison.
    "r1a-refined": {
        "description": (
            "Joint optimisation — distance topology. "
            "All key parameters searched together via TPE. Topology fixed=distance. "
            "BASELINE updated from Phase 1 (g1–g7) best values. "
            "Readout aggregation freed (last_N won Phase 1 g5b, window optimised jointly). "
            "units×output_variables interaction explored (He=1×units, IHe=2×, all=4×). "
            "Defines primary performance ceiling for biologically motivated topology."
        ),
        "sampler":   "tpe",
        "n_trials":  750,
        "folds":     1,
        "instances": 4015,
        "binary":    False,
        "optimize": {
            # ---- Scaling (g1 Phase 1 best: input=0.001, rc=0.0006, dde=0.0075) ----
            # Ranges centred on Phase 1 best; extended 2 log-decades each side.
            "input_scaling":  {"type": "float", "low": 1e-4,  "high": 0.1,  "log": True},
            "rc_scaling":     {"type": "float", "low": 5e-5,  "high": 0.05, "log": True},
            "dde_scaling":    {"type": "float", "low": 5e-4,  "high": 0.05, "log": True},
            # g7 Phase 1 best=8.862; range [3, 50] to catch interactions with scaling.
            "cell_coupling":  {"type": "float", "low": 3.0,  "high": 50.0, "log": True},
            # ---- DDE integration (g2 Phase 1 best: warmup=50, substeps=60) ----
            # substeps=60 was at the Phase 1 ceiling — extend up to 88 to confirm.
            "warmup":         {"type": "int", "low": 10, "high": 80,  "step": 10},
            "rk4_substeps":   {"type": "int", "low": 28, "high": 88,  "step": 4},
            # ---- Topology (g3a Phase 1 best: sparsity=0.134, fade_alpha=1e-6) ----
            # interaction_diameters is biologically fixed — not optimised.
            "sparsity":       {"type": "float", "low": 0.05, "high": 0.8},
            "fade_alpha":     {"type": "float", "low": 1e-7,  "high": 1e-3, "log": True},
            "p_excite":       {"type": "float", "low": 0.3,   "high": 0.9},
            # ---- Units ----
            # units×output_vars determines KNN feature dimensionality.
            # units=1000 + all(4 vars) = 4000d → curse of dimensionality for KNN(k=4).
            "units": {"type": "categorical", "choices": [250, 500, 750, 1000]},
            # ---- Input layer (g5 Phase 1 best: connectivity=0.368) ----
            "input_connectivity": {"type": "float", "low": 0.05, "high": 1.0},
            # ---- Output variables (g5 Phase 1 best: "He") ----
            "output_variables": {
                "type": "categorical",
                "choices": ["all", "AI", "IHi", "IHe", "AHe", "He"],
            },
            # ---- Readout aggregation (g5b Phase 1 best: last_N/window=36) ----
            # Freed here so window interacts properly with warmup/substeps.
            "readout_aggregation": {
                "type": "categorical",
                "choices": ["final", "signal_mean", "last_N"],
            },
            "readout_window": {
                "type": "int", "low": 2, "high": 200,
                "conditional_on": "readout_aggregation", "condition_value": "last_N",
            },
        },
        "fixed": {
            # Topology type fixed — unbiased comparison with r1b handled by separate run.
            "topology_type":  "distance",  # g3a confirmed; biologically motivated
            "signed":         True,         # g3a best
            "input_bias":     False,        # g5 confirmed
            # Noise params: Phase 1 best values (weak lever — modest F1 in g4 studies).
            "noise_in":       0.01442,      # g4-reservoir best
            "noise_rc":       0.2397,       # g4-reservoir best
            "noise_rate":     0.2999,       # g4-data best
            "noise_ratio":    0.2862,       # g4-data best
        },
        # Cap parallel workers: units up to 1000, ~3 GB RAM/trial.
        # 64 workers corresponds to ~192 GB peak at units=1000.
        "max_processes":  64,
    },

    # r1b-refined  (random/Gaussian topology)
    # Identical parameter budget and ranges as r1a, but topology fixed to
    # "random" (Gaussian weights). Deployed simultaneously on a separate host
    # so comparison is unbiased — each topology gets its own full joint optimisation.
    # The Phase 1 g3b comparison was confounded by a BASELINE tuned for distance;
    # r1b corrects this by freely optimising all parameters for random topology.
    "r1b-refined": {
        "description": (
            "Joint optimisation — random (Gaussian) topology. "
            "Mirror of r1a but topology_type fixed=random and sparsity→rc_connectivity. "
            "Run simultaneously with r1a on a separate host for unbiased topology comparison. "
            "Phase 1 g3b was confounded by a distance-tuned BASELINE; r1b corrects this. "
            "Provides the fair performance ceiling for Gaussian random ESN topology."
        ),
        "sampler":   "tpe",
        "n_trials":  750,
        "folds":     1,
        "instances": 4015,
        "binary":    False,
        "optimize": {
            # ---- Scaling (same ranges as r1a) ----
            "input_scaling":  {"type": "float", "low": 1e-4,  "high": 0.1,  "log": True},
            "rc_scaling":     {"type": "float", "low": 5e-5,  "high": 0.05, "log": True},
            "dde_scaling":    {"type": "float", "low": 5e-4,  "high": 0.05, "log": True},
            "cell_coupling":  {"type": "float", "low": 3.0,   "high": 50.0, "log": True},
            # ---- DDE integration (same ranges as r1a) ----
            "warmup":         {"type": "int", "low": 10, "high": 80, "step": 10},
            "rk4_substeps":   {"type": "int", "low": 28, "high": 88, "step": 4},
            # ---- Topology — random uses rc_connectivity, not sparsity/fade_alpha ----
            # rc_connectivity: fraction of non-zero entries in the random weight matrix.
            # p_excite: fraction of excitatory (positive) weights when signed=True.
            "rc_connectivity": {"type": "float", "low": 0.05, "high": 1.0},
            "p_excite":        {"type": "float", "low": 0.3,  "high": 0.9},
            # ---- Units ----
            "units": {"type": "categorical", "choices": [250, 500, 750, 1000]},
            # ---- Input layer (same ranges as r1a) ----
            "input_connectivity": {"type": "float", "low": 0.05, "high": 1.0},
            # ---- Output variables (same choices as r1a) ----
            "output_variables": {
                "type": "categorical",
                "choices": ["all", "AI", "IHi", "IHe", "AHe", "He"],
            },
            # ---- Readout aggregation (same as r1a) ----
            "readout_aggregation": {
                "type": "categorical",
                "choices": ["final", "signal_mean", "last_N"],
            },
            "readout_window": {
                "type": "int", "low": 2, "high": 200,
                "conditional_on": "readout_aggregation", "condition_value": "last_N",
            },
        },
        "fixed": {
            "topology_type":  "random",   # Gaussian random ESN — comparable to r1a
            "signed":         True,        # g3b best; allows inhibitory connections
            "input_bias":     False,       # g5 confirmed
            # Noise params: Phase 1 best values.
            "noise_in":       0.01442,
            "noise_rc":       0.2397,
            "noise_rate":     0.2999,
            "noise_ratio":    0.2862,
        },
        # 48 workers is a conservative upper bound for common workstation/server setups.
        "max_processes":  48,
    },

    # -----------------------------------------------------------------------
    # Phase 3 — Final results
    # -----------------------------------------------------------------------
    # ro1-readout is handled entirely by readout.py, not this registry.
    # See module docstring Phase notes for the exact command sequence.
    # Prerequisite: run `classification rerun-best --study r1a-refined --folds 5`
    # first to produce state-saved fold artefacts.
    #
    # f1-final-5class and f4-large removed: r1a-refined best trial with the
    # ro1-optimised readout IS the headline 5-class result.  There is nothing
    # further to optimise at the reservoir level.
    # -----------------------------------------------------------------------

    "f2-unbalanced": {
        "description": (
            "Class imbalance sensitivity: grid over max_per_class as a "
            "multiple of the minority-class count (803 samples for MIT-BIH). "
            "1x = fully balanced; 5x = close to full unbalanced dataset. "
            "Fixed reservoir params from r1a-refined best trial. "
            "UPDATE 'fixed' section with r1a-refined best-trial values before running."
        ),
        "sampler":   "grid",
        "n_trials":  5,   # 5 grid points: 1x–5x minority size
        "folds":     5,
        "instances": 50000,  # large pool so each max_per_class point is fully covered
        "binary":    False,
        "optimize": {
            # Each choice is a multiple of 803 (MIT-BIH minority class size).
            # 1x (803)  = perfectly balanced
            # 2x (1606) = 2× imbalance
            # 3x (2409), 4x (3212), 5x (4015 = full dataset per class max)
            "max_per_class": {
                "type": "categorical",
                "choices": [803, 1606, 2409, 3212, 4015],
            },
        },
        "fixed": {
            # balance_classes=True + max_per_class controls imbalance degree:
            # each class is capped at max_per_class samples, so larger classes
            # grow relative to the fixed minority class.
            "balance_classes": True,
            # r1a-refined best trial (trial 416, F1=0.9047)
            "topology_type":       "distance",
            "signed":              True,
            "input_bias":          False,
            "units":               1000,
            "warmup":              50,
            "rk4_substeps":        64,
            "input_scaling":       0.0277,
            "rc_scaling":          0.007036,
            "dde_scaling":         0.0006439,
            "cell_coupling":       12.75,
            "sparsity":            0.576,
            "fade_alpha":          2.803e-05,
            "p_excite":            0.8354,
            "input_connectivity":  0.5282,
            "output_variables":    "IHe",
            "readout_aggregation": "signal_mean",
            "noise_in":            0.01442,
            "noise_rc":            0.2397,
            "noise_rate":          0.2999,
            "noise_ratio":         0.2862,
        },
        "_grid": {
            "max_per_class": [803, 1606, 2409, 3212, 4015],
        },
        # 5 unique grid points → cap Optuna-level processes to 5 so each
        # worker gets exactly one unique grid point.  Folds run sequentially
        # within each trial (fold_workers=1) so each of the 5 processes gets
        # ~25 CPUs for numba — running 25 concurrent reservoir sims causes
        # severe OMP/numba thread oversubscription on a 128-core host.
        "max_processes": 5,
        "fold_workers":  1,
    },

    # -----------------------------------------------------------------------
    # f2-unbalanced-v4
    # Re-run of the class-imbalance grid search WITHOUT noise augmentation.
    # After the g4a-augmentation-grid study showed augmentation is negligible
    # at N=1000 (+0.08%), we drop it from all headline evaluations.
    # Uses 5-fold CV with macro F1 objective.
    # -----------------------------------------------------------------------
    "f2-unbalanced-v4": {
        "description": (
            "Class imbalance sensitivity (no augmentation): grid over "
            "max_per_class as a multiple of the minority-class count (803). "
            "5-fold CV with macro F1 objective. No noise augmentation."
        ),
        "sampler":   "grid",
        "n_trials":  5,
        "folds":     5,
        "instances": 50000,
        "binary":    False,
        "optimize": {
            "max_per_class": {
                "type": "categorical",
                "choices": [803, 1606, 2409, 3212, 4015],
            },
        },
        "fixed": {
            "balance_classes":     True,
            "objective_average":   "macro",
            # r1a-refined best trial (trial 416, F1=0.9047)
            "topology_type":       "distance",
            "signed":              True,
            "input_bias":          False,
            "units":               1000,
            "warmup":              50,
            "rk4_substeps":        64,
            "input_scaling":       0.0277,
            "rc_scaling":          0.007036,
            "dde_scaling":         0.0006439,
            "cell_coupling":       12.75,
            "sparsity":            0.576,
            "fade_alpha":          2.803e-05,
            "p_excite":            0.8354,
            "input_connectivity":  0.5282,
            "output_variables":    "IHe",
            "readout_aggregation": "signal_mean",
            "noise_in":            0.01442,
            "noise_rc":            0.2397,
            "noise_rate":          0.0,
            "noise_ratio":         0.0,
        },
        "_grid": {
            "max_per_class": [803, 1606, 2409, 3212, 4015],
        },
        "max_processes": 5,
        "fold_workers":  1,
    },

    # ro2-readout-unbalanced
    # Readout optimisation on the best f2-unbalanced grid point (highest F1
    # across the 5 max_per_class levels), using the same pipeline as ro1-readout
    # but with unbalanced training data.
    # Run after f2-unbalanced completes:
    #   python -m optimization.optimize rerun-best \
    #       --study f2-unbalanced --folds 5
    #   python -m optimization.optimize research --type readout \
    #       --study_name ro2-readout-unbalanced \
    #       --trial_name f2-unbalanced-best
    # This gives the headline "unbalanced" F1 cited in the paper.

    # -----------------------------------------------------------------------
    # f2-unbalanced-macro
    # Clean re-run of the class-imbalance grid search with macro F1 objective.
    # Single-fold: this study exists only to show the trend of F1 increasing
    # with max_per_class, not to produce headline metrics.
    # The existing f2-unbalanced study used weighted F1 due to a code bug
    # (readout.py defaulted to weighted; classification.py used weighted
    # before BASELINE["objective_average"] was added).
    # -----------------------------------------------------------------------
    "f2-unbalanced-macro": {
        "description": (
            "Class imbalance sensitivity (macro F1 objective): grid over "
            "max_per_class as a multiple of the minority-class count (803). "
            "Single-fold evaluation for trend analysis. "
            "Fixed reservoir params from r1a-refined best trial."
        ),
        "sampler":   "grid",
        "n_trials":  5,
        "folds":     1,
        "instances": 50000,
        "binary":    False,
        "optimize": {
            "max_per_class": {
                "type": "categorical",
                "choices": [803, 1606, 2409, 3212, 4015],
            },
        },
        "fixed": {
            "balance_classes": True,
            "objective_average": "macro",
            # r1a-refined best trial (trial 416, F1=0.9047)
            "topology_type":       "distance",
            "signed":              True,
            "input_bias":          False,
            "units":               1000,
            "warmup":              50,
            "rk4_substeps":        64,
            "input_scaling":       0.0277,
            "rc_scaling":          0.007036,
            "dde_scaling":         0.0006439,
            "cell_coupling":       12.75,
            "sparsity":            0.576,
            "fade_alpha":          2.803e-05,
            "p_excite":            0.8354,
            "input_connectivity":  0.5282,
            "output_variables":    "IHe",
            "readout_aggregation": "signal_mean",
            "noise_in":            0.01442,
            "noise_rc":            0.2397,
            "noise_rate":          0.2999,
            "noise_ratio":         0.2862,
        },
        "_grid": {
            "max_per_class": [803, 1606, 2409, 3212, 4015],
        },
        "max_processes": 5,
        "fold_workers":  1,
    },

    "f3-binary": {
        "description": (
            "Binary (Normal vs Arrhythmia) classification. "
            "Uses r1a-refined best reservoir params with binary=True. "
            "n_trials=1: nothing to optimise — report 5-fold CV metrics directly. "
            "Run via: python -m optimization.optimize rerun-best "
            "--study r1a-refined --folds 5 --binary "
            "--trial_name r1a-refined-best-binary"
        ),
        "sampler":   "tpe",
        "n_trials":  1,
        "folds":     5,
        "instances": 4015,
        "binary":    True,
        "optimize": {},  # nothing to optimise; all fixed from r1a-refined best
        "fixed": {
            # r1a-refined best trial (trial 416, F1=0.9047)
            "topology_type":       "distance",
            "signed":              True,
            "input_bias":          False,
            "units":               1000,
            "warmup":              50,
            "rk4_substeps":        64,
            "input_scaling":       0.0277,
            "rc_scaling":          0.007036,
            "dde_scaling":         0.0006439,
            "cell_coupling":       12.75,
            "sparsity":            0.576,
            "fade_alpha":          2.803e-05,
            "p_excite":            0.8354,
            "input_connectivity":  0.5282,
            "output_variables":    "IHe",
            "readout_aggregation": "signal_mean",
            "noise_in":            0.01442,
            "noise_rc":            0.2397,
            "noise_rate":          0.2999,
            "noise_ratio":         0.2862,
        },
    },
    # -----------------------------------------------------------------------
    # h1-heterogeneity
    # Purpose: Biological realism robustness study.  Measures classification
    #   performance under per-node kinetic parameter heterogeneity, simulating
    #   cell-to-cell variability in gene expression (Elowitz et al. 2002,
    #   Ozbudak et al. 2002).  Each node draws kinetic rates from log-normal
    #   distributions centred on the nominal Danino et al. (2010) values.
    #   Grid: 8 CV levels × 5 seeds = 40 trials.  CV=0.0 is the homogeneous
    #   baseline; seeds vary the random parameter realization while keeping
    #   topology, initial conditions, and data splits identical.
    #   Uses r1a-refined best reservoir params with single-fold evaluation
    #   (fold-0).  This is a robustness check, not model selection, so full
    #   cross-validation is unnecessary.
    # -----------------------------------------------------------------------
    "h1-heterogeneity": {
        "description": (
            "Biological realism: per-node kinetic heterogeneity sweep. "
            "Grid over CV (0.0–0.5) × 5 seeds.  Tests robustness to "
            "cell-to-cell gene expression noise (log-normal kinetic rates). "
            "Fixed reservoir params from r1a-refined best trial."
        ),
        "sampler":   "grid",
        "n_trials":  40,   # 8 CV levels × 5 seeds
        "folds":     1,
        "instances": 4015,
        "binary":    False,
        "optimize": {
            "param_heterogeneity_cv": {
                "type": "categorical",
                "choices": [0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5],
            },
            "param_heterogeneity_seed": {
                "type": "categorical",
                "choices": [100, 200, 300, 400, 500],
            },
        },
        "fixed": {
            # r1a-refined best trial (trial 416, F1=0.9047)
            "topology_type":       "distance",
            "signed":              True,
            "input_bias":          False,
            "units":               1000,
            "warmup":              50,
            "rk4_substeps":        64,
            "input_scaling":       0.0277,
            "rc_scaling":          0.007036,
            "dde_scaling":         0.0006439,
            "cell_coupling":       12.75,
            "sparsity":            0.576,
            "fade_alpha":          2.803e-05,
            "p_excite":            0.8354,
            "input_connectivity":  0.5282,
            "output_variables":    "IHe",
            "readout_aggregation": "signal_mean",
            "noise_in":            0.01442,
            "noise_rc":            0.2397,
            "noise_rate":          0.2999,
            "noise_ratio":         0.2862,
        },
        "_grid": {
            "param_heterogeneity_cv":   [0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5],
            "param_heterogeneity_seed": [100, 200, 300, 400, 500],
        },
        # 40 unique grid points; single-fold evaluation (~30–45 min/trial).
        # With 18 parallel workers most trials run concurrently.
        "max_processes": 18,
        "fold_workers":  1,
    },

    # -----------------------------------------------------------------------
    # esn-balanced
    # Standard ReservoirPy leaky ESN baseline on the balanced five-class task.
    # Kept materially cheaper than the BioReservoir search while using the same
    # data pipeline, CV protocol, frozen-state readout optimisation, and metric
    # semantics.
    # -----------------------------------------------------------------------
    "esn-balanced": {
        "description": (
            "ReservoirPy leaky ESN baseline on the balanced five-class task. "
            "Uses the same preprocessing, CV protocol, and macro-F1 objective "
            "as BioReservoir, with a moderate Optuna budget for a fair but "
            "cheaper baseline."
        ),
        "sampler": "tpe",
        "n_trials": 256,
        "folds": 1,
        "instances": 4015,
        "binary": False,
        "optimize": {
            "units": {"type": "int", "low": 100, "high": 1000, "step": 50},
            "input_scaling": {"type": "float", "low": 1e-3, "high": 2.0, "log": True},
            "input_connectivity": {"type": "float", "low": 0.05, "high": 1.0},
            "rc_connectivity": {"type": "float", "low": 0.01, "high": 1.0, "log": True},
            "noise_in": {"type": "float", "low": 0.0, "high": 0.05},
            "noise_rc": {"type": "float", "low": 0.0, "high": 0.05},
            "esn_spectral_radius": {"type": "float", "low": 1e-2, "high": 2.0, "log": True},
            "esn_leak_rate": {"type": "float", "low": 0.1, "high": 1.0},
            "input_bias": {"type": "categorical", "choices": [False, True]},
            "bias_scaling": {
                "type": "float", "low": 1e-3, "high": 1.0, "log": True,
                "conditional_on": "input_bias", "condition_value": True,
            },
            "esn_activation": {"type": "categorical", "choices": ["tanh", "sigmoid"]},
        },
        "fixed": {
            "reservoir_type": "esn",
            "objective_average": "macro",
            "preserve_split": False,
            "readout_aggregation": "signal_mean",
            "balance_classes": True,
            "max_per_class": None,
            "scaler_type": "sequence_zscore",
            "noise_rate": BASELINE["noise_rate"],
            "noise_ratio": BASELINE["noise_ratio"],
        },
    },

    # -----------------------------------------------------------------------
    # b1-baseline
    # No-reservoir baseline: classifiers trained directly on the raw z-scored
    # 187-dimensional ECG feature vectors, bypassing the reservoir entirely.
    # Uses the same classifier families, hyperparameter ranges, 5-fold
    # stratified CV protocol, per-fold augmentation/standardisation, and
    # macro-F1 objective as the reservoir readout studies (ro1-readout).
    # Handled by optimization/baseline.py, not classification.py.
    #
    # Run via:
    #   python -m optimization.optimize research --type baseline \
    #       --study_name b1-baseline --trials 500 --processes 32
    # -----------------------------------------------------------------------
    "b1-baseline": {
        "description": (
            "No-reservoir baseline: classifiers on raw z-scored 187-d ECG vectors. "
            "Same classifier families, hyperparameter ranges, 5-fold CV protocol, "
            "per-fold augmentation, and macro-F1 objective as the reservoir readout "
            "studies. Provides direct evidence of the reservoir's contribution."
        ),
        "type":      "baseline",
        "sampler":   "tpe",
        "n_trials":  500,
        "folds":     5,
        "instances": 4015,
        "binary":    False,
        "optimize": {},  # classifier hyperparameters are handled by baseline.py
        "fixed": {
            "objective_average": "macro",
            "balance_classes":   True,
            "max_per_class":     None,
            "scaler_type":       "sequence_zscore",
            "noise_rate":        BASELINE["noise_rate"],
            "noise_ratio":       BASELINE["noise_ratio"],
        },
    },

    # -----------------------------------------------------------------------
    # b2-baseline-unbalanced
    # No-reservoir baseline on the unbalanced 5x dataset (max_per_class=4015),
    # matching the f2-unbalanced 5x configuration.  Uses the same large pool
    # size so that each class contributes up to 4015 or its natural count.
    # Compares directly with the reservoir's headline unbalanced F1 = 0.920.
    # -----------------------------------------------------------------------
    "b2-baseline-unbalanced": {
        "description": (
            "No-reservoir baseline on 5x unbalanced dataset (max_per_class=4015). "
            "Matches f2-unbalanced 5x setting.  Compares with reservoir F1=0.920."
        ),
        "type":      "baseline",
        "sampler":   "tpe",
        "n_trials":  500,
        "folds":     5,
        "instances": 50000,
        "binary":    False,
        "optimize": {},
        "fixed": {
            "objective_average": "macro",
            "balance_classes":   True,
            "max_per_class":     4015,
            "scaler_type":       "sequence_zscore",
            "noise_rate":        BASELINE["noise_rate"],
            "noise_ratio":       BASELINE["noise_ratio"],
        },
    },

    # -----------------------------------------------------------------------
    # b3-baseline-binary
    # No-reservoir baseline on the binary (Normal vs Arrhythmia) task.
    # Uses the same balanced binary dataset as the reservoir binary study.
    # instances=4015 matches the readout-selection pool; the reservoir's
    # headline binary result (F1=0.961) used the full ~37k balanced dataset,
    # but readout selection was done on 4015 instances.
    # Compares directly with the reservoir's binary readout F1 = 0.933
    # (on 4015 instances) and the headline F1 = 0.961 (full dataset).
    # -----------------------------------------------------------------------
    "b3-baseline-binary": {
        "description": (
            "No-reservoir baseline on balanced binary (Normal vs Arrhythmia). "
            "instances=4015 for parity with reservoir readout selection."
        ),
        "type":      "baseline",
        "sampler":   "tpe",
        "n_trials":  500,
        "folds":     5,
        "instances": 4015,
        "binary":    True,
        "optimize": {},
        "fixed": {
            "objective_average": "macro",
            "balance_classes":   True,
            "max_per_class":     None,
            "scaler_type":       "sequence_zscore",
            "noise_rate":        BASELINE["noise_rate"],
            "noise_ratio":       BASELINE["noise_ratio"],
        },
    },
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_study(name: str) -> Dict[str, Any]:
    """Return study config for *name*, raising KeyError with helpful message."""
    if name not in STUDY_REGISTRY:
        available = ", ".join(sorted(STUDY_REGISTRY))
        raise KeyError(
            f"Unknown study {name!r}. Available: {available}"
        )
    return STUDY_REGISTRY[name]


def list_studies() -> None:
    """Print a human-readable summary of all registered studies."""
    print(f"{'Study':<25} {'Sampler':<8} {'Trials':>7} {'Folds':>5}  Description")
    print("-" * 90)
    for name, cfg in STUDY_REGISTRY.items():
        print(
            f"{name:<25} {cfg['sampler']:<8} {cfg['n_trials']:>7} "
            f"{cfg['folds']:>5}  {cfg['description'][:60]}"
        )
