# Changelog

Notable changes to this project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow SemVer.
Each entry is marked **[science]** if it changes a reported number, or
**[engineering]** if it does not.

## [Unreleased]

### Changed

- [engineering] `optimization/deploy.py` also copies `tests/` and installs the `dev` extra,
  so the test suite runs on a host like any other command
  (`--run "python -m pytest tests/unit tests/integration"`). Data files that `data/` links
  to, as in a worktree that uses the prepared dataset, are copied as files.
- [engineering] The figure scripts draw the `v0.1` figure designs from a run or study
  config: one colony (Fig. 3b), the array and its coupling (Fig. 4), the dataset (Fig. 5)
  and heterogeneity (Fig. S10, also for the realism studies).
- [engineering] Figures share the `v0.1` look (`utils.visualisation.STYLE`), save PNGs
  (`--formats pdf,svg` adds PDF and SVG) and warn if they would break the ACS figure rules
  at their printed size.
- [engineering] Run, comparison and fold-analysis figures name the classes and draw each
  fold's scores as dots with their mean and SD; `analyse_fold_results.py --plots` draws
  `confusion`, `metrics` and `pca`.
- [engineering] The `v0.2` bioreservoir studies run 32 RK4 substeps, the ridge readouts
  choose their penalty by macro F1, and `P2-esn-image` reads the ESN from one noisy image.
- [engineering] `P1-heterogeneity-64` reruns kinetic heterogeneity CV 0.3–0.5 at 64 substeps.
- [engineering] `P2-<design>-top` reruns the five best trials of each Phase 2 search at
  three more reservoir seeds.

### Added

- [engineering] The `v0.2` campaign configs in `experiments/v0.2/`: the tracking set, the
  Phase 1 grids (`P1-*`) and the Phase 2 searches (`P2-*`), checked by
  `tests/unit/test_experiment_configs.py`.
- [engineering] `python -m training.run study-table CONFIG [CONFIG ...] [--group]
  [--csv FILE]`: one row per trial of a study (searched values, reservoir seed, the
  metric's mean and SD over folds, wall time, simulation time, commit); `--group` averages
  each point over its reservoir seeds, and over several studies that differ only in their
  seeds (`training.study.trials_table`, `seed_table`).
- [engineering] Raw-beat runs (`reservoir: {type: none}`) and a `HistGB` readout.
- [engineering] `training.run compare --margin M --ceiling RUN_C`: non-inferiority and the
  gap closed.
- [engineering] Configs can name a `base:` config and list only what differs, and studies
  can search the run `seed`; the `v0.2` studies name designs in `experiments/v0.2/designs/`.
- [engineering] Realism options, off by default: heterogeneity of cell density and
  dilution, discrete input levels and an imperfect reset (`bioreservoir.realism`).
- [engineering] `bioreservoir.diagnostics.oscillation_amplitudes`, reported by
  `scripts/analysis/drive_bounds.py --oscillation`.
- [engineering] Controls: random features and a leaky-integrator bank as reservoir types,
  and a linear surrogate of a saved run (`reservoir.surrogate: linear`).
- [engineering] `utils.preprocessing.class_indices(Y)`: the class of each beat from labels,
  one-hot labels or scores.

### Removed

- [engineering] Plotting code that nothing used or that repeated another figure:
  `plot_cv_results.py`, `analyse_fold_diagnostics.py`, the t-SNE and bar-chart plots and
  unused `utils.visualisation` functions.

### Fixed

- [engineering] `scripts/analysis/drive_bounds.py` works on diffusive and uncoupled arrays.
- [science] The RK4 step check counts the whole membrane exchange, for every coupling, and a
  run stops with `UnstableIntegrationError` when colony states blow up.
- [engineering] A study records a trial that blows up as failed and goes on, and every
  trial records the machine it ran on.
- [engineering] `optimization/deploy.py` checks for a running session before it copies files.
- [engineering] Simulation workers print no progress bar per beat, which filled the logs.
- [engineering] Optuna's plots no longer warn that they are experimental, and
  `analyse_fold_results.py` reports MCC and kappa, also for `v0.1` fold files.

## [0.2.1] - 2026-10-25

### Added

- [engineering] Diffusive coupling, a chemically realisable alternative to the abstract
  coupling of `v0.1` (which stays the default): with `BioReservoir(coupling="diffusion")`
  colonies exchange extracellular AHL through a graph Laplacian, `dHe_i/dt` gaining
  `sum_j G_ij (He_j - He_i)` at every RK4 stage (`solve_dde_substep(conductance=...)`).
  The conductances come from a non-negative kernel (`distance_matrix(signed=False)`, or
  the new `lattice_matrix()` for a trap array on a near-square grid), scaled so that a
  colony's conductances sum to `diffusion_rate` on average (`scale_conductances`);
  symmetric conductances conserve AHL, and `relative_asymmetry` measures how far a kernel
  is from symmetric.
  `coupling="none"` gives independent oscillators. The input can be a non-negative
  inducer level (`input_encoding="nonnegative"`: a fixed offset on the clipped sample,
  noise cannot make it negative) delivered with non-negative weights
  (`input_sign="positive"`). Run configs and studies take the new options (`coupling`,
  `diffusion_rate`, `input_encoding`, `input_range`, `input_sign`, `topology_type:
  lattice`; trials record them), the `qs_diffusion` preset sets the structure of the
  realisable model, and only the abstract coupling needs `rc_scaling` and `noise_rc` (it
  refuses the lattice, whose weights are not normalised to unit spectral radius).
  The solver is checked against an independent integrator (fourth-order convergence
  without delay, second order with it), for AHL conservation, and for non-negative
  states with non-negative input. It runs on one thread, and refuses an exchange too fast
  for its step (the explicit solver would blow up); `drive_shares` applies to the
  abstract coupling only, and the ablation and heterogeneity scripts accept
  `--preset v0.1` only.

## [0.2] - 2026-10-24

### Changed

- [engineering] Worker processes read the run's reservoir from a temporary file when they
  start. It used to travel with each worker's start-up, so the workers started one after
  another (about 0.5 s each): simulating the 4,015 beats of the balanced pool with a
  1,000-node reservoir and 120 workers took 74 s, 63 s of it starting the workers, and
  now takes 16 s. The features are unchanged.
- [engineering] `optimization/deploy.py` installs each host's environment from `uv.lock`:
  it copies the local `uv` to the host's workspace, and uv installs Python 3.12 and the
  locked packages there, so the host's own Python version no longer matters. The code
  copy includes `experiments/`, `uv.lock` and `README.md`; remote commands clear
  `PYTHONPATH` and `PYTHONHOME`; a host can have its own workspace
  (`BIORESERVOIR_DEPLOY_WORKSPACES`); SSH waits up to 60 s to connect and never prompts.
  Hosts with glibc older than 2.28 get the environment without the plotting packages,
  which have no wheels for them, so runs there cannot draw figures; `--inventory` checks
  the glibc version instead of the system Python.
- [engineering] Python 3.12, and current releases of the tools: ReservoirPy 0.3.16, Optuna
  5.0.0, joblib 1.6.0, matplotlib 3.11.2, pillow 13.0.0 and psutil 7.2.2 (wfdb, which
  nothing imports, is no longer installed). The numerical libraries stay at the versions
  `v0.1` used (numpy 1.26.4, numba 0.60.0, scipy 1.13.0, scikit-learn 1.5.0, pandas
  2.2.3), so every `v0.1` result reproduces exactly: the balanced and ESN headline reruns
  match the archived predictions beat for beat. Optuna 5 reads the archived study
  databases unchanged, and runs with several worker processes no longer print a
  `ResourceTracker` error at exit. Scientific dependencies are upgraded only between sets
  of reported results.
- [science] A run simulates each beat once and every fold fits its readout on these
  features (`cv.simulate: once`, the new default; `training.simulate`). The folds and the
  readouts are the same as before, the simulation is done once instead of once per fold,
  and `--processes` runs beats in worker processes with one thread each, which keeps every
  core busy; the reservoir warms up once per run and each worker gets a copy of it. Each
  beat draws its noise from its own seed (the reservoir seed and the beat's
  position), so a beat's features no longer depend on its fold or on the beats run before
  it; a reservoir with noise therefore gives slightly different numbers than before.
  `cv.simulate: per_fold` keeps the previous path, and the `v0.1` preset and configs that
  take their reservoir from a study trial use it, so the `v0.1` benchmarks are unchanged.
  Augmentation and scalings fitted on a training fold need `per_fold`. A held-out
  evaluation simulates the pool and the split once in the same way.
- [engineering] Study names follow Supplementary Table S5 (`P1-scaling`, `P2-distance`,
  `P4-readout-balanced`, `ESN-balanced`, `B1-balanced`, ...). Runs are named
  `<study>-best[-<variant>]`; the ablation condition `baseline` is now `full`.
- [engineering] Working outputs are written under `results/` (`optuna/`, `runs/`,
  `metrics/`, `figures/`) and log files under `logs/`. Paths and settings come from
  `utils/config.py` (environment variable, then `config.env`, then default).
  `BIORESERVOIR_RESULTS_DIR` now names the results root rather than the runs folder.
- [engineering] `optimize rerun-best` and `optimize apply-best` go through the run
  framework: they write a run folder with config, metrics, manifest and a registry line,
  and refuse to overwrite an existing run folder. `apply-best` writes a new run
  (`results/runs/<study>-best/`) instead of rewriting the source run's fold files.
  PyYAML (already installed with Optuna) is a declared dependency.
- [engineering] The code passes ruff (pycodestyle errors, pyflakes, import order,
  pyupgrade, bugbear) and the CI lint job is blocking, with ruff pinned to 0.15.0. Most of
  the 695 findings were fixed automatically (annotations, import order); the rest by hand
  (strict `zip`, exception chaining, unused variables, a mutable default argument).
  Line length (E501) waits for a formatting-only change; `I` (LuxI) is allowed in the
  solver.
- [engineering] Comments, docstrings, tests, configs and documentation describe what the
  code does and why, and call the design of the first version of the paper `v0.1`;
  `AGENTS.md` describes how to write code and documentation.
  The held-out draws are described as evaluated during model development (draws 1 and 2)
  or as the final test set, not yet evaluated (draw 3); their beat IDs are unchanged.

### Added

- [engineering] A `RidgeCV` readout: ridge whose strength is chosen by leave-one-out on
  each training fold (scikit-learn's `RidgeClassifierCV`, from `params.alphas`, by default
  10^-4 to 10^6, two per decade). One decomposition of the training features gives the
  leave-one-out errors for every alpha, so the choice costs about one fit, where
  `readout.search` fits once per alpha and inner fold. Each fold's alpha is recorded in
  `metrics.json`.
- [engineering] Optuna studies defined in a run config (`training.study`): a `study`
  section names the values to search by their dotted paths in the config
  (`reservoir.params.input_scaling`, `readout.params.alpha`, ...), the metric, the sampler
  (seeded TPE, random or grid) and points to try first. Every trial runs the same pipeline
  as `training.run run` and records its resolved config, code version and fold scores;
  `python -m training.run study` runs a study one trial at a time and resumes it, and
  `study-best` writes the best trial as a plain run config. A seeded study runs the same
  trials again. Search paths must name settings the run reads, and a study keeps its
  definition: rerunning it with another config (other than `study.trials`) is refused.
  With `reservoir.states_from` a study refits only the readout on the saved states of a
  run, to tune readout settings without simulating again.
- [engineering] Run configs set what a measurement of the reservoir sees
  (`training.observation`): `features.variables` lists the observed variables (`[I]` for
  the LuxI reporter alone), `features.aggregation: images:K` takes K snapshots evenly
  spaced over the beat, `features.measurement_noise` multiplies each measured value by
  1 + σ·N(0, 1) with draws of the beat's own (after aggregation: one draw per image, or one
  per average over the beat), and `features.calibration_error` gives the
  channels of each test fold a gain error the readout does not see in training. They need
  `cv.simulate: once`; the defaults observe every output variable without error, as
  before.
- [engineering] `--archive VERSION` reads frozen studies from `archive/<version>/`.
- [engineering] `optimization/deploy.py` and `optimization/collect.py` are now part of the
  repository. Host addresses, user, remote workspace, pip proxy and service come from
  `config.env`; hosts are referred to by label, and `--inventory` records each host's
  CPU, memory, OS and Python in `results/metrics/hosts.json`.
- [engineering] `uv.lock` locks every dependency (`uv sync`). Python 3.10-3.12 is
  supported (`requires-python = ">=3.10,<3.13"`); Python 3.9 (end of life) is no longer.
  The scientific dependencies keep their `v0.1` versions; pillow 12.3.0, kagglehub 1.0.2,
  pytest 9.1.1 and pytest-cov 7.1.0 are upgraded for security fixes that need Python
  3.10, and the unused pyarrow pin is removed.
- [engineering] CI installs from the lock, runs the integration tests on the real dataset
  (Kaggle secrets), and requires 80% coverage of changed lines, measured on the unit and
  integration tests together.
- [engineering] Regression tests against the `v0.1` code: the model regression test
  (N = 20, synthetic beats; reservoir states and features) and the pipeline regression
  test (N = 200, 600 real beats, 2 folds; macro-F1 within 0.02). Both references were
  generated with the `v0.1` code (`tests/regression_reference.py`).
- [engineering] `python -m utils.preprocessing` prepares the dataset (Kaggle download if
  needed, five-class and binary caches) and prints content checksums;
  `tests/integration/test_dataset.py` checks the datasets and the `v0.1` CV pools against
  the `v0.1` checksums.
- [engineering] Held-out splits: `python -m utils.heldout --write` draws balanced sets of
  N, S, V and Q beats that no `v0.1` optimisation study used and records their beat IDs
  and checksums in `data/splits/heldout.json` (`data/splits/` is tracked, the rest of
  `data/` is not); `--check` rebuilds and verifies them. The splits `validation-1` and
  `validation-2` (draws 1 and 2) have been evaluated during model development; `test`
  (draw 3) is the final test set and has not been evaluated.
- [engineering] Held-out evaluation: a run config with `test: {split: <name>}` fits the
  readout on the whole pool and evaluates it on the split (`utils.heldout.load_split`
  loads its beats, scaled like the pool): macro F1 over the split's classes (it has no F
  beats) with a bootstrap 95% CI, the split's beat IDs in the manifest and the split in
  the registry. The run refuses split beats that are in its pool, and the `test` split
  needs `--final-test`. `evaluate_performance` and `run_fold` take the classes to average
  over, and `compare` averages over the classes of the test labels (unchanged for
  cross-validated runs).
- [engineering] Run manifests (code version, data and lock-file checksums, environment)
  and an append-only run registry (`registry.jsonl`, `python -m utils.provenance`).
- [engineering] Unit tests of the current behaviour of the model, training and
  preprocessing code: the solver against the written-out model equations and a reference
  ODE solver (fourth-order convergence), the distance kernel against its interaction-range
  calibration, and cross-validation folds (stratified, reproducible, and identical
  predictions in worker processes).
- [engineering] Run framework: `python -m training.run run <config>` runs a
  classification task from a YAML config (task, reservoir, features, readout, folds;
  explicit values or Optuna study trials, `resolve` prints the explicit config) into one
  folder per run (config, metrics with sample SDs, folds with test-beat positions,
  figures, log, manifest) and the registry; `compare` reports per-fold metrics,
  prediction agreement and a paired bootstrap CI between two runs, including archived
  ones. The registry records the paper items a run backs (`mark --uses`,
  `list --used`). Configs of the `v0.1` headline runs (`experiments/benchmarks/v0.1/`)
  are checked against the archived models and pools, and rerun in full as benchmarks
  (`pytest -m benchmark`).
- [engineering] Tests of the optimisation package: the study runner, evaluation report
  and plots, the readout, baseline and reservoir objectives (checked fold by fold against
  direct scikit-learn computations), their commands, the command line and the
  deployment and collection helpers; integration tests for a reservoir trial through the
  study runner, pruned and failed trials, cross-validation, and the archived B1-B3 best
  trials reproduced fold by fold. Line coverage of `optimization/` (unit and integration
  tests) is 90-100% per module, up from 0-90%; the unused `classification.py` is not
  tested.

### Removed

- [engineering] `optimization/classification.py`, an unused older copy of the reservoir
  objective in `optimization/training.py`. Nothing imported it; the optimisation
  commands use `optimization/training.py`.

### Fixed

Effect on the balanced headline (P2-distance trial 416 with the P4 random forest, the
`v0.1` hyperparameters, no re-tuning): with every fix, macro F1 is 0.8660 instead of
0.8915 (paired difference -0.026, 95% CI -0.035 to -0.017). One fix at a time: one
position per oscillator -0.015, spectral radius on the final matrix -0.016, substep
delay -0.003 (not significant), scaled readout features -0.003. The first two change the
coupling the hyperparameters were tuned for, so the studies need re-tuning.

- [science] Standard deviations over folds or seeds are sample SDs (ddof = 1)
  everywhere: Optuna trial attributes (`metric_f1_std`, fold timings), per-class SDs,
  run metrics, the headline-metrics and fold-analysis scripts and the heterogeneity
  figure. SDs over five folds were population SDs and are now 1.118 times larger.
- [science] Readouts standardise the state features inside each training fold
  (`readout.scale`, on by default: a StandardScaler fitted on the training fold only).
  The `v0.1` readouts were fitted on unscaled features; the `v0.1` run preset
  keeps that for reproduction.
- [engineering] Readout hyperparameters can be chosen by nested cross-validation inside
  each training fold (`readout.search`, `readout.inner_folds`); the choice per fold is
  recorded in `metrics.json`. The Phase 4 studies chose the readout on the reported folds.
- [science] A per-class cap (`max_per_class`) caps each class's whole pool, test beats
  included, as the Supporting Information states. The `v0.1` loader capped only the
  training part, so capped classes kept the cap plus their test beats (4,175 instead of
  4,015 per class at 5x). `cap_includes_test=False` (and the `v0.1` preset)
  keeps the `v0.1` rule; the loader logs the beats per class.
- [science] With two observed variables (`output_variables="IHe"`) the reservoir has one
  position per oscillator: oscillator j receives `sum_k K_jk (c_I I_k + c_He He_k)` from
  the N x N signed distance kernel K, with weights `couple_vars` (default 1, 1). The `v0.1`
  code built the kernel over 2N random points, so an oscillator's I and He coordinates
  sat at different positions and only the first N of the 2N kernel rows drove an
  oscillator. `geometry="v0.1"` (and the `v0.1` preset) keeps the `v0.1`
  construction.
- [engineering] The sequential solver (`parallel=False`) works when two variables are
  observed; it failed with a broadcast error.
- [science] The delayed Hi in the promoter term follows the solution within an input
  step. Hi is kept at every RK4 substep and each stage reads Hi(t + c h - delay), linear
  between substeps when the delay is not a whole number of substeps, so the solver
  converges at second order to the delay equation, and a delay of 0 is the ordinary
  differential equation. The `v0.1` solver read the delayed Hi once per input step and
  held it for all the substeps: more substeps did not reduce that error, and the
  "no delay" ablation was not delay-free. `delay_scheme="frozen"` (and the
  `v0.1` preset) keeps the `v0.1` solver. The unit tests run the new kernel compiled and
  as plain Python, and check it against closed-form solutions.
- [science] The distance kernel is normalised after the random signs and sparsity, so
  its spectral radius is 1 as stated. The `v0.1` code normalised first, which left the
  headline topology (P2-distance trial 416) at 0.43; the new order is the old matrix
  times 1/0.43, the same as a larger `rc_scaling`. Row normalisation uses absolute
  weights. `normalize_after_masks=False` (and the `v0.1` preset) keeps the
  `v0.1` order.
- [engineering] The distance kernel option is `decay_mode="gaussian"`
  (`exp(-(d/l)^2)`); `"exp"`, its `v0.1` name, still works.
- [engineering] The `signal_mean` readout aggregation is an alias of `mean`, the mean
  over all 187 steps. It was meant to skip each beat's zero-padded tail, detected as
  inputs equal to 0, but the beats reach the reservoir z-scored, where the padding is
  not 0: none of the 4,015 balanced or 109,446 unbalanced beats had a tail detected, so
  every `v0.1` result used the full mean and none changes.
- [engineering] Removed the RK4 substep aggregation option
  (`rk4_substep_aggregation_window`, `rk4_substep_aggregation` in study parameters and
  `classify.py --rk4_substep_aggregation_window`): the node only ever received the state
  at the end of each input step, so the option had no effect. Run configs log a
  warning for reservoir parameters the builder does not read.
- [science] Per-oscillator kinetic heterogeneity draws mean-preserving log-normal
  factors (mean 1, coefficient of variation `param_heterogeneity_cv`). The `v0.1` factors
  had mean sqrt(1 + cv^2), so every varied rate was biased upwards (4.4% at cv 0.3, 11.8%
  at 0.5). `param_heterogeneity_mean_preserving=False` (and the `v0.1` preset)
  keeps the `v0.1` draws.
- [science] The heterogeneity sweep (`scripts/run_heterogeneity.py`) no longer augments
  the training folds by default (`v0.1` used rates 0.2999 and 0.2862 there, unlike the
  headline runs), and each grid seed now sets the reservoir array as well as the
  parameter draws, so the spread at CV = 0 is the spread between reservoirs; in `v0.1`
  the five CV = 0 runs were the same reservoir and gave identical predictions.
  `--fixed_array`, `--noise_rate`, `--noise_ratio` and `--preset v0.1`
  reproduce the `v0.1` sweep.
- [engineering] The ablation script defaults to no augmentation, as the v0.1
  ablations were run, logs the augmentation, and accepts `--preset v0.1` to
  build the reservoir as the `v0.1` code did.
- [engineering] Every cross-validation fold runs a copy of the run's reservoir (same
  weights, warm-up state and noise generator state) and fits its own copy of the
  readout: one reservoir per run, folds differing only in their data. The code meant to
  rebuild each fold's nodes with a fold seed, but BioReservoir and the ESN reservoir
  cannot be rebuilt from their hyperparameters, so it silently copied them; the results
  do not change.
- [engineering] `scripts/analysis/readout_tables.py` writes Tables S13-S15 (the best
  trial of each classifier family in the Phase 4 readout studies) from the archived
  databases; a snapshot test keeps them in step with the archive. The `v0.1` tables
  had the right F1 values, but most hyperparameters outside the random forest rows came
  from earlier databases (for example KNN k = 4 instead of 1 for the balanced study, and
  a gradient boosting loss that five-class studies cannot use), and the MLP and
  perceptron families were left out.
- [engineering] `bioreservoir.diagnostics.drive_and_state_ranges` and
  `scripts/analysis/drive_bounds.py` report how often the drive x_t exceeds the bound in
  the model description (5e-3) and how low the states go. For the `v0.1` balanced
  headline reservoir (10 beats per class), |x_t| reaches 8.5e-3 (above 5e-3 in 0.1% of
  oscillator-steps) and the AHL levels Hi and He go negative (down to -0.35) in 1.2% of
  oscillator-steps.
- [engineering] `bioreservoir.diagnostics.drive_shares` (also in
  `scripts/analysis/drive_bounds.py`) splits the drive into the input term and the
  coupling through each observed variable. For the `v0.1` balanced headline reservoir the
  coupling through LuxI (I) makes up 98.8% of the drive, the input 1.2% and He 0.009%.
- [engineering] Binary runs can reproduce `v0.1`. The `v0.1` binary runs read the binary
  dataset file the hosts had then: a pre-balanced pool of 37,714 beats (all 18,857
  arrhythmia beats and 18,857 normal beats) written by an earlier loader. The preparation
  command writes all 109,446 beats instead, from which the loader draws other normal
  beats, so binary folds and scores differed from the archive (binary headline 0.9617
  instead of 0.9613, no-reservoir B3-binary 0.9308 instead of 0.9355). A task can name
  its dataset file (`task.data_file`, checked against `task.data_sha256`); the binary
  headline benchmark reads the `v0.1` file and is compared fold by fold like the other
  benchmarks, and `tests/integration/test_dataset.py` checks the `v0.1` binary pool
  against that file.
- [engineering] A study trial resolves to the values it recorded (reservoir, topology,
  augmentation, readout aggregation, pool size and labels) rather than to today's
  BASELINE, in run configs (`reservoir.study`), `optimize rerun-best` and the
  confusion-matrix plot. BASELINE changed after Phase 1, so the Phase 1 trials (all but
  P1-units, which ran later) resolved to other reservoirs (the P1-scaling best trial
  with warm-up 50 instead of 40, coupling 8.862 instead of 16.28 and augmentation
  0.2999/0.2862 instead of 0.0825/0.2348, for example); the Phase 2-4 and ESN studies
  resolved correctly.
- [engineering] The no-reservoir baseline studies B1-B3 run without augmentation, as
  every archived B1-B3 trial ran (and records). Their registry entries named the
  BASELINE augmentation (0.2999/0.2862), so `optimize research --type baseline`
  augmented the training folds unless told otherwise; the reported baselines are
  unchanged.
- [engineering] `optimize research --processes N` runs the requested number of trials:
  each process ran `trials // N`, so 500 trials on 96 processes ran 480 (the B1-B3
  studies ran on a 96-thread host and have 480 trials each), and fewer trials than
  processes ran none. Without `processes` the study runs in one process instead of
  failing.
- [engineering] Readout studies optimise macro F1 unless `--primary_average` says
  otherwise; they defaulted to weighted F1, unlike the reservoir and baseline studies.
  Every archived readout study was run with macro F1.
- [engineering] Trials store the macro and weighted averages of precision, recall, F1
  and support (`metric_macro_*`, `metric_weighted_*`), which the evaluation report's
  per-class table shows. The objectives averaged the folds with a helper that keeps the
  class rows only, so no trial (archived or new) had them; the reported metrics are
  stored separately and are unaffected.
- [engineering] A reservoir trial on the original train/test split (`--preserve_split`)
  is recorded as one fold (`metric_n_folds`, `fold_0_*`, per-class means); it recorded
  0 folds and no fold metrics. No archived trial records using the original split.
- [engineering] The environment metadata timestamp uses a timezone-aware UTC clock
  (`datetime.utcnow()` is deprecated from Python 3.12); the recorded format is unchanged.
  So do the trial start and end times of the optimisation objectives.
- [engineering] `load_ecg_data` reads the preprocessed cache without requiring (or
  downloading) the source CSV files first; the CSVs are only needed to build the cache.
- [engineering] Trial metadata no longer records the host name, IP address, working
  directory, or script, output and interpreter paths, and the platform string keeps only
  the OS, numeric kernel release and architecture. Trials now record the full git SHA, a
  dirty flag, the OS name and version, and an optional machine label from `config.env`.
- [engineering] `optimization.deploy --cleanup` stops the remote session with every
  process it started (the session runs in its own process group). It used to stop the
  shell running the cleanup itself, whose command line matched its `pkill` pattern, so
  the workspace was not deleted.

## [0.1] - 2026-09-24

- The code behind the first version of the paper, archived on Zenodo:
  https://doi.org/10.5281/zenodo.23436888
