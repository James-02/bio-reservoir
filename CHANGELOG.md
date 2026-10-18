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
