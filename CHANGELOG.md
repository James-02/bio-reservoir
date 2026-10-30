# Changelog

Notable changes to this project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow SemVer.
Each entry is marked **[science]** if it changes a reported number, or
**[engineering]** if it does not.

## [Unreleased]

### Added

- [engineering] The `v0.2` campaign configs in `experiments/v0.2/` (designs, tracking set,
  Phase 1 grids `P1-*`, Phase 2 searches `P2-*`), checked by `test_experiment_configs.py`.
- [engineering] Configs can name a `base:` config and list only what differs; studies can
  search the run `seed`.
- [engineering] `training.run study-table`: one row per study trial (searched values, seed,
  mean and SD over folds, times, commit); `--group` averages over reservoir seeds.
- [engineering] Raw-beat runs (`reservoir: {type: none}`) and a `HistGB` readout.
- [engineering] `training.run compare --margin M --ceiling RUN`: a non-inferiority test and the
  share of the gap closed.
- [engineering] Realism options, off by default: cell-density and dilution heterogeneity,
  discrete input levels and an imperfect reset (`bioreservoir.realism`).
- [engineering] `reservoir.params.washout_rate` and `cell_density` set the AHL washout rate
  (the flow, `mu`) and the cell density (`d`) of every colony; unset, the defaults apply.
- [engineering] `warmup_input: rest` warms the colonies up at the resting inducer level, so a
  beat starts without a step in the inducer.
- [engineering] Controls: random features, a leaky-integrator bank, and a linear surrogate of
  a saved run (`reservoir.surrogate: linear`).
- [engineering] Ablations: `reservoir.params.delay: 0` removes the feedback delay and
  `task.scramble: true` puts each beat's time steps in a random order.
- [engineering] `features.aggregation: images_at:t1,t2,...` takes images at chosen minutes
  of the beat.
- [engineering] `features.reference: start` divides each image by the trap's image of the
  start state, which cancels calibration error.
- [engineering] `bioreservoir.diagnostics.oscillation_amplitudes` (`drive_bounds.py
  --oscillation`) and `utils.analysis.class_indices`, which every label conversion uses.
- [engineering] A run warns, and a study trial records (`grid_edges`), when a fold picks a
  readout setting at the end of its grid.

### Changed

- [engineering] The figure scripts draw the `v0.1` figure designs (Figs. 3b, 4, 5, S10) from a
  run or study config, in one style, and warn when a figure breaks the ACS rules.
- [engineering] Run, comparison and fold-analysis figures name the classes and show each
  fold's score with the mean and SD.
- [engineering] The `v0.2` bioreservoir studies use 32 RK4 substeps and choose ridge penalties
  by macro F1; `P2-esn-image` reads the ESN from one noisy image.
- [engineering] New studies: `P1-heterogeneity-64` (kinetic heterogeneity at 64 substeps) and
  `P2-<design>-top` (the five best Phase 2 trials at three more seeds).
- [engineering] `optimization/deploy.py` also copies the tests, so the suite runs on a host.
- [engineering] `pyproject.toml` gives the version as 0.2.1 and pins ruff to CI's 0.15.0; the
  Pillow pin is removed (matplotlib installs it).
- [engineering] The README and this changelog are shorter.
- [engineering] The P designs no longer list settings they never read (`rc_scaling`,
  `noise_rc`, the distance-kernel settings); their simulations are unchanged.
- [engineering] The default ridge penalties are written out, so every CPU records the same.
- [engineering] A readout grid search uses the run's `--processes` when folds run in turn.
- [engineering] A relaunch with the same `--label` adds to the job log instead of replacing it.

### Removed

- [engineering] Unused plotting code: `plot_cv_results.py`, `analyse_fold_diagnostics.py`,
  the t-SNE and bar-chart plots.
- [engineering] `deploy` no longer stops or starts a service on the hosts: the
  `BIORESERVOIR_DEPLOY_SERVICE` setting and `--start-service` are removed.

### Fixed

- [science] The RK4 step check counts the whole membrane exchange for every coupling, and a
  run stops with `UnstableIntegrationError` when colony states blow up.
- [engineering] A study records a trial that blows up as failed and goes on; every trial
  records its machine.
- [engineering] `deploy` checks for a running session before copying files.
- [engineering] Workers no longer print a progress bar per beat.
- [engineering] `drive_bounds.py` works on diffusive and uncoupled arrays;
  `analyse_fold_results.py` reports MCC and kappa, also for `v0.1` files.
- [engineering] The oscillation diagnostic and the colony figure continue a `rest` warm-up
  under the resting input, not without input.
- [engineering] A refit on saved states refuses task and fold settings, which it ignored
  (`task.scramble` included).
- [engineering] A `deploy` launch fails unless its session starts, and records the commit whose
  files it copied.
- [engineering] A job's log holds the errors of every part of its command (`a && b`); those of
  `a` went to the session log only.
- [engineering] A run without the plotting packages skips its figures instead of stopping.
- [engineering] `study-table` lists the run seed of raw-beat and refit trials (it showed None).
- [engineering] `evaluate_performance` reads class-index labels (it read all as class 0).
- [engineering] `reservoir.params.delay` must be a number (`true` read as 1).
- [engineering] The archive tests skip a host without a full copy of the archive.

## [0.2.1] - 2026-10-25

### Added

- [engineering] Diffusive coupling (`coupling: diffusion`): colonies exchange extracellular
  AHL through a graph Laplacian, on a trap lattice (`topology_type: lattice`) or a distance
  kernel.
- [engineering] Uncoupled colonies (`coupling: none`), a non-negative inducer input
  (`input_encoding: nonnegative`, `input_sign: positive`) and the `qs_diffusion` preset.

## [0.2] - 2026-10-24

### Changed

- [science] Each beat is simulated once and every fold reads the same features
  (`cv.simulate: once`); noise is drawn per beat, so noisy reservoirs give slightly
  different numbers. The `v0.1` preset keeps `per_fold`.
- [engineering] Workers start in parallel (balanced pool, 120 workers: 74 s → 16 s).
- [engineering] Hosts install Python 3.12 and the locked packages from `uv.lock`.
- [engineering] Python 3.12 and current tools; numpy, numba, scipy, scikit-learn and pandas
  keep their `v0.1` versions, so every `v0.1` result reproduces exactly.
- [engineering] Study names follow Supplementary Table S5; outputs go to `results/` and
  `logs/`, set in `utils/config.py`.
- [engineering] `optimize rerun-best` and `apply-best` write run folders through the run
  framework.
- [engineering] The code passes ruff, and CI's lint job blocks a merge.
- [engineering] Code and docs describe what the code does and call the first paper's design
  `v0.1`.

### Added

- [engineering] Run framework: `python -m training.run run CONFIG` writes one folder per run
  (config, metrics, folds, figures, manifest) and a registry line; `compare` pairs two runs.
- [engineering] Studies in run configs (`training.study`): seeded Optuna searches that run
  the same pipeline and record each trial's config.
- [engineering] Observation options (`training.observation`): observed variables, images,
  measurement noise and calibration error.
- [engineering] A `RidgeCV` readout: the ridge penalty is chosen by leave-one-out on each
  training fold.
- [engineering] Held-out splits (`validation-1`, `validation-2`, `test`) and evaluation on
  them; the `test` split needs `--final-test`.
- [engineering] Run manifests and an append-only run registry (`utils.provenance`).
- [engineering] `python -m utils.preprocessing` prepares the dataset and prints its checksums.
- [engineering] `uv.lock`; CI installs from it and needs 80% coverage of changed lines.
- [engineering] Regression tests against `v0.1`, and tests of the model, training,
  preprocessing and optimisation code.
- [engineering] `deploy.py` and `collect.py` run studies on remote hosts and fetch results.
- [engineering] `--archive VERSION` reads frozen results from `archive/<version>/`.

### Removed

- [engineering] `optimization/classification.py`, an unused copy of the reservoir objective.

### Fixed

Together these fixes lower the balanced headline at the `v0.1` hyperparameters from 0.8915
to 0.8660 macro F1 (paired difference −0.026, 95% CI −0.035 to −0.017). The first two change
the coupling the hyperparameters were tuned for, so the studies need re-tuning. The `v0.1`
preset keeps every `v0.1` behaviour.

- [science] One position per oscillator: `v0.1` built the kernel over 2N points and only
  half its rows drove the oscillators (−0.015; `geometry="v0.1"` keeps it).
- [science] The kernel is normalised after signs and sparsity, so its spectral radius is 1
  (`v0.1`: 0.43; −0.016; `normalize_after_masks=False` keeps it).
- [science] The delayed Hi is updated at every RK4 substep, not held over the input step
  (−0.003, not significant); delay 0 gives the ODE (`delay_scheme="frozen"` keeps `v0.1`).
- [science] Readouts standardise the features inside each training fold (−0.003).
- [science] Kinetic heterogeneity is mean-preserving; `v0.1` rates were biased up by 4.4% at
  CV 0.3.
- [science] Standard deviations are sample SDs (×1.118 over five folds).
- [science] A per-class cap includes the test beats, as the SI states (4,015 per class at
  5×, not 4,175).
- [science] The heterogeneity sweep runs without augmentation and draws a new array per seed
  (`--preset v0.1` reproduces the old sweep).
- [engineering] Readout hyperparameters can be chosen inside each training fold
  (`readout.search`).
- [engineering] Study trials resolve to the values they recorded, not to today's defaults
  (the Phase 1 trials resolved to other reservoirs).
- [engineering] Binary runs can read the `v0.1` binary file, so the binary headline
  reproduces (0.9613).
- [engineering] Baseline and ablation runs use no augmentation, as recorded; `--processes N`
  runs every requested trial.
- [engineering] Readout studies optimise macro F1, and trials store macro and weighted
  metrics.
- [engineering] Tables S13–S15 come from the archived databases (some `v0.1` values came
  from older databases).
- [engineering] Diagnostics report the drive range and shares (`v0.1` headline: drive up to
  8.5e-3; 98.8% of it through LuxI).
- [engineering] `signal_mean` is the full mean (it never found a padded tail), and the
  unused RK4 aggregation option is removed.
- [engineering] Every fold runs a copy of the run's reservoir; results do not change.
- [engineering] Smaller fixes: the sequential solver with two variables, the cache read
  without the CSVs, a timezone-aware clock, no host details in trial metadata, `deploy
  --cleanup`, `decay_mode="gaussian"`, `--preserve_split` recorded as one fold.

## [0.1] - 2026-09-24

- The code behind the first version of the paper, archived on Zenodo:
  https://doi.org/10.5281/zenodo.23436888
