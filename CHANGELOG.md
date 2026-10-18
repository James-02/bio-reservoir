# Changelog

Notable changes to this project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow SemVer.
Each entry is marked **[science]** if it changes a reported number, or
**[engineering]** if it does not.

## [Unreleased] (v0.2)

### Changed

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

### Added

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
- [engineering] Held-out beat draws: `python -m utils.heldout --write` draws balanced sets
  of beats that no `v0.1` optimisation phase used and records their beat IDs and checksums
  in `experiments/heldout/draws.json`; `--check` rebuilds and verifies them. Draws 1-2
  were consulted in the revision pilots; draw 3 is sealed for the confirmatory evaluation.
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
  ones. The registry records the manuscript items a run backs (`mark --uses`,
  `list --used`). Configs of the `v0.1` headline runs (`experiments/benchmarks/v0.1/`)
  are checked against the archived models and pools, and rerun in full as benchmarks
  (`pytest -m benchmark`).

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
  `v0.1` preset) keeps the `v0.1` solver.
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
- [engineering] The environment metadata timestamp uses a timezone-aware UTC clock
  (`datetime.utcnow()` is deprecated from Python 3.12); the recorded format is unchanged.
- [engineering] `load_ecg_data` reads the preprocessed cache without requiring (or
  downloading) the source CSV files first; the CSVs are only needed to build the cache.
- [engineering] Trial metadata no longer records the host name, IP address, working
  directory, or script, output and interpreter paths, and the platform string keeps only
  the OS, numeric kernel release and architecture. Trials now record the full git SHA, a
  dirty flag, the OS name and version, and an optional machine label from `config.env`.

## [0.1] (manuscript submission)

- Code archived on Zenodo: https://doi.org/10.5281/zenodo.23436888
