# Reservoir Computing with Genetic Oscillators

[![Tests](https://github.com/James-02/bio-reservoir/actions/workflows/tests.yml/badge.svg)](https://github.com/James-02/bio-reservoir/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/James-02/bio-reservoir/branch/main/graph/badge.svg)](https://codecov.io/gh/James-02/bio-reservoir)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23436888.svg)](https://doi.org/10.5281/zenodo.23436888)

A reservoir computing framework built on coupled genetic oscillators for ECG arrhythmia classification. Each reservoir node is a quorum-sensing oscillator modelled by delay differential equations (DDEs), implemented as a [ReservoirPy](https://github.com/reservoirpy/reservoirpy) `Node`.

> For a comprehensive overview, see our associated research paper.

## Project Overview

![Graphical TOC](.github/static/graphical-toc.png)

High-level overview of the biological reservoir computing pipeline: ECG sequences perturb quorum-sensing genetic oscillators, the coupled reservoir transforms those inputs into a high-dimensional dynamical state, and a lightweight readout performs the final classification.

## Environment Setup

Dependencies are declared in `pyproject.toml` and locked in `uv.lock` (Python 3.12), so [uv](https://docs.astral.sh/uv/) recreates the exact environment:

```bash
export UV_PROJECT_ENVIRONMENT=venv
uv sync --extra dev
source venv/bin/activate
cp config.env.example config.env   # Kaggle credentials and local paths
```

Without uv, `pip install -e .[dev]` installs the pinned runtime dependencies.

## Example Usage

```python
from bioreservoir import BioReservoir, distance_matrix
from utils.preprocessing import load_ecg_data
from training import cross_validate
from utils.analysis import compute_mean_metrics

# Load arrhythmia ECG data (balanced 5-class)
X_train, Y_train, X_test, Y_test = load_ecg_data(rows=1000)

# Initialise reservoir and readout
import reservoirpy as rpy
from reservoirpy.nodes import Ridge

reservoir = BioReservoir(units=100)
readout = Ridge(ridge=1e-6)

# Combine train+test for cross-validation
import numpy as np
X = np.concatenate([X_train, X_test])
Y = np.concatenate([Y_train, Y_test])

# 5-fold cross-validation
results = cross_validate(reservoir, readout, X, Y, folds=5)
```

## Running Experiments

A run is defined by a YAML config: the task (labels, pool size, balancing, scaling), the reservoir (explicit parameters or an Optuna study trial), the state features, the readout and the cross-validation. The configs of the `v0.1` headline runs are in [experiments/benchmarks/v0.1](experiments/benchmarks/v0.1).

```bash
python -m training.run run experiments/benchmarks/v0.1/balanced.yaml --processes 5
python -m training.run resolve my-config.yaml          # print the explicit config
python -m training.run compare archive:v0.1/P4-readout-balanced-best results/runs/<run>
python -m utils.provenance list                        # the run registry
python -m utils.provenance mark <run id> ok --uses "Table 2, balanced"
```

Each run writes `results/runs/<name>-<UTC time>/`: the resolved `config.yaml`, `metrics.json` (per fold, mean and sample SD, pooled), `fold-<k>.npz`, `figures/`, `run.log` and `manifest.json` (code version, data checksums, environment), and adds a line to `registry.jsonl`. Runs from uncommitted code need `--allow-dirty` and are marked non-citable. `compare` prints per-fold metrics, the prediction agreement and a paired bootstrap 95% CI of the macro-F1 difference, and saves figures with `--out`; `--margin -0.02` tests B for non-inferiority to A, and `--ceiling RUN_C` gives the share of the gap from A to C that B closes. `optimize rerun-best` and `optimize apply-best` use the same runner.

A reservoir starts every beat from the same state, so a beat's features do not depend on the fold. A run therefore simulates each beat once and every fold fits its readout on these features (`cv.simulate: once`, the default); `--processes` sets the number of worker processes that simulate beats in parallel, each with one thread. Each beat draws its noise from its own seed, so the features do not depend on how the beats are shared between processes. `cv.simulate: per_fold` simulates the beats of every fold, as `v0.1` did, with `--processes` folds in parallel; the `v0.1` preset uses it, and so does augmentation (`cv.noise_rate`, `cv.noise_ratio`), whose noisy copies of a fold's training beats belong to that fold only.

Readouts are fitted on standardised features (`readout.scale`, on by default). A readout can choose its hyperparameters on each training fold: `classifier: RidgeCV` picks the ridge strength by leave-one-out (from `params.alphas`, by default 10^-4 to 10^6, two per decade), and `readout.search` runs a grid search over inner folds (`readout.inner_folds`). `metrics.json` records each fold's choice.

`reservoir: {type: none}` fits the readout on the scaled beat itself (187 values), on the same pool and folds as a reservoir run, so `compare` pairs the two beat by beat; `classifier: HistGB` is histogram gradient boosting.

The `features` section also sets what a measurement sees ([training/observation.py](training/observation.py)). In a device a microscope images the colonies' reporter, so a readout can be restricted to some variables, a few images, measurement noise and calibration error:

```yaml
features:
  variables: [I]            # LuxI only (default: every variable the reservoir outputs)
  aggregation: images:4     # four snapshots evenly spaced over the beat; final is one
  measurement_noise: 0.01   # each value times 1 + 0.01 N(0, 1), drawn per beat
  calibration_error: 0.01   # each test-fold channel times its own 1 + 0.01 N(0, 1)
```

The draws depend only on the run seed and the beat (measurement) or the fold (calibration), so runs with the same seed and the same feature layout are compared under the same errors. The noise multiplies the feature row after aggregation: each image gets its own draws, and an average over the beat (`mean`, `last_N:K`) is one noisy measurement of the average.

### Studies

A run config with a `study` section is an Optuna search over its own values. Each trial puts the sampled values into the config at their dotted paths and runs it through the same pipeline as a single run; the trials record their resolved configs, and `study-best` writes the best one as a plain config to run for the reported numbers. A search may also set the run's `seed` (with `task.seed` and `cv.seed` fixed). A config can name another with `base: <file>` and list only what differs: its sections are merged key by key over the base's.

```yaml
study:
  trials: 300
  metric: f1                  # mean cross-validation F1 (cv.primary_average)
  sampler: {type: tpe, seed: 6337, multivariate: true, n_startup_trials: 20}
  enqueue:
    - {reservoir.params.input_scaling: 0.083}
  search:
    reservoir.params.input_scaling: {type: float, low: 0.005, high: 0.5, log: true}
    reservoir.params.units: {type: categorical, choices: [256, 1024]}
```

```bash
python -m training.run study my-study.yaml --processes 128   # resumes an existing study
python -m training.run study-best my-study.yaml --out best.yaml
python -m training.run study-table my-study.yaml --group --csv my-study.csv
```

Trials run one at a time with the beats in parallel, so a study with a seeded sampler runs the same trials again; run one process per study. `sampler: {type: grid}` runs every combination of the `choices`. The study is stored in `results/optuna/optuna-<name>.db` (or `--storage`). Running the command again resumes the study; only `study.trials` may change between runs, and every search path must name a setting the run reads. A study whose config takes its states from another run (`reservoir: {states_from: <run folder>}`, a run made with `outputs.save_states: true`) refits only the readout on the saved states of each fold, so readout settings can be tuned without simulating the reservoir again.

`study-table` prints one row per trial: the searched values, the reservoir seed, the metric's mean and SD over the folds, the trial's wall time, the time spent simulating the beats and the commit. With `--group` it averages the trials that differ only in `reservoir.seed`, or only in the study (several configs may be given), so a grid run at three seeds reads as one row per point, with the SD over seeds and the mean SD over folds. `--csv` writes the table. The configs of the `v0.2` campaign (tracking set, Phase 1 grids, Phase 2 searches) are in [experiments/v0.2](experiments/v0.2/README.md).

### Held-out evaluation

Besides the cross-validation pool, three held-out splits of the five-class dataset hold beats (N, S, V and Q) that no `v0.1` optimisation study used: `validation-1` and `validation-2` (803 beats per class each) and `test` (370 beats per class). Their beat IDs and checksums are tracked in [data/splits/heldout.json](data/splits/heldout.json); `python -m utils.heldout --check` rebuilds them from the data and verifies them. Select models by cross-validation on the pool; use the validation splits to check how a choice generalises, and the `test` split once, for the final evaluation.

A run evaluates on a split when its config has a `test` section; the readout is then fitted on the whole pool and scored on the split (macro F1 over the split's classes, with a bootstrap 95% CI):

```yaml
test:
  split: validation-1   # or validation-2, or test (needs --final-test)
```

```bash
python -m training.run run my-config.yaml                # validation split
python -m training.run run my-final-config.yaml --final-test
```

The run refuses beats of the split that are also in its pool, records the split's beat IDs in the manifest and the split in the registry, and runs on the same split pair beat by beat in `compare`.

`pytest -m benchmark` reruns the `v0.1` headline configs and compares them fold by fold with the archived runs; it needs a capable host (`BIORESERVOIR_BENCHMARK_PROCESSES` sets the parallel folds). The balanced, unbalanced, binary-full and ESN reruns reproduce the archived predictions exactly. The binary-full config reads the binary file the `v0.1` runs used (`data/ecg_archive/binary_ecg_data.npz`, a pre-balanced pool of 37,714 beats, checked by its SHA-256); it is not built by `python -m utils.preprocessing`, and without it the binary tests are skipped.

The code defaults include the fixes listed in the [CHANGELOG](CHANGELOG.md) (one position per oscillator, the delay resolved at every RK4 substep, unit spectral radius, scaled readout features, and others). `preset: v0.1` in a run config, which the benchmark configs use, restores the v0.1 code path; the ablation and heterogeneity scripts take `--preset v0.1`. `python scripts/analysis/drive_bounds.py <config>` reports a reservoir's drive range, negative states and drive shares.

## Project Structure

- **bioreservoir/** — Core library: DDE solver, `BioReservoir` node, topology generation (distance-based spatial kernels).
- **training/** — Cross-validation pipeline, classification helpers, profiling utilities.
- **optimization/** — Optuna hyperparameter studies, study registry, and CLI commands for running/evaluating studies.
- **utils/** — Preprocessing (ECG loading, augmentation, scaling), analysis metrics, visualisation, and result management.
- **scripts/** — Standalone experiment scripts and analysis helpers.
- **experiments/** — Run configs (`benchmarks/v0.1/`: the `v0.1` headline runs; `v0.2/`: the `v0.2` campaign studies).
- **data/ecg/** — MIT-BIH arrhythmia dataset files (not tracked; `python -m utils.preprocessing`).
- **data/splits/** — Held-out validation and test splits (beat IDs and checksums, tracked).
- **results/** — Working outputs (gitignored): Optuna studies, fold runs, metrics and figures; see [Results Layout](#results-layout).
- **logs/** — Log files from studies and experiment scripts (gitignored).

## How It Works

1. **Genetic oscillator nodes** — Each node simulates two coupled genes (luxI and aiiA) that produce an intracellular AHL signal (Hi) which diffuses externally (He). The system is driven by a system of DDEs with delayed negative feedback.

2. **External input** — ECG time-series samples are fed through He, representing the influence of the extracellular environment on the oscillator dynamics.

3. **Reservoir network** — A spatial topology of oscillator nodes is constructed with distance-dependent coupling weights (a Gaussian kernel of distance, with random signs and sparsity, normalised to unit spectral radius). This is the abstract coupling of `v0.1`; diffusive AHL exchange is the alternative (see [Coupling Models](#coupling-models)).

4. **State extraction** — At each timestep the concentration of each node's luxI gene is read out, producing a high-dimensional state representation of the input.

5. **Classification** — A readout layer (Ridge, KNN, or Random Forest) is trained on the final reservoir states to classify the input into arrhythmia categories.

## Coupling Models

`BioReservoir(coupling=...)` (in a run config, `reservoir.params.coupling`) sets how the colonies interact:

- `abstract` (default, the `v0.1` model): once per input step each oscillator receives `rc_scaling * W r`, a signed kernel applied to the observed states of the previous step, added to its input drive. It is a computational abstraction: the coupling can be negative and carries LuxI between colonies.
- `diffusion`: colonies exchange extracellular AHL (He) through the medium. At every RK4 stage, `dHe_i/dt` gains `sum_j G_ij (He_j - He_i)`, a graph Laplacian over non-negative conductances `G`. The kernel comes from `W` (`distance_matrix(signed=False)` or `lattice_matrix()`, a trap array with nearest-neighbour exchange; `topology_type: distance` or `lattice` in a config) and is scaled so that a colony's conductances sum to `diffusion_rate` per model time unit on average. Symmetric conductances conserve AHL: the exchange only moves it between colonies.
- `none`: independent oscillators.

The input can be delivered as a non-negative inducer level: `input_encoding: nonnegative` clips each (z-scored) sample to `input_range` (default `[-3, 8]`) and shifts it by the lower end, a fixed and causal offset, and input noise cannot take it below zero; `input_sign: positive` makes the input weights non-negative. `preset: qs_diffusion` sets these choices together with symmetric, distance-based conductances:

```yaml
preset: qs_diffusion
reservoir:
  params:
    units: 1000
    topology_type: lattice      # or distance, with interaction_diameters and fade_alpha
    diffusion_rate: 0.5         # mean AHL exchange rate of a colony
    # ... and the other parameters: warmup, rk4_substeps, input_scaling, dde_scaling, ...
```

Each step from the `v0.1` model towards a chemically realisable one is one config change, so the steps can be run and compared one at a time:

| Model | Config |
|---|---|
| `v0.1` code path | `preset: v0.1` |
| Corrected abstract coupling | `coupling: abstract` (the defaults) |
| AHL exchange, non-negative conductances | `coupling: diffusion`, `signed: false` (with `sparsity < 1` the conductances are not symmetric, so AHL is not conserved) |
| + symmetric conductances, non-negative input | `preset: qs_diffusion` |
| + reporter readout: LuxI only, one end-point image | `features: {variables: [I], aggregation: final}` |
| + measurement noise | `features: {measurement_noise: 0.01}` |
| + colony-to-colony variability | `param_heterogeneity_cv` |

The diffusive solver advances all colonies together and runs on one thread (a parallel region per RK4 stage costs more than the stage itself); run folds or beats in parallel processes instead (`--processes`). The exchange is integrated explicitly, so an RK4 step must be short compared with the fastest exchange; the node refuses a step that is too long (16 substeps allow lattice rates up to about 16). A lattice (`lattice_matrix`, a near-square grid) is for the diffusive coupling: its weights are not normalised, so the abstract coupling refuses it. `bioreservoir.topology.relative_asymmetry` measures how far a coupling matrix is from symmetric.

## Visualisation

### Biopixel Reservoir Model
![Biopixel Reservoir Model](.github/static/biopixel-reservoir-model-3-subfigures.png)

**(a)** Schematic of a quorum-sensing genetic oscillator (biopixel) comprising luxI, aiiA, intracellular AHL (Hi), and extracellular AHL (He). **(b)** Effective input scales showing how external ECG drive and recurrent coupling modulate gene expression dynamics. **(c)** Reservoir computing architecture: ECG time-series inputs are fed through a network of coupled oscillator nodes, and the final luxI concentrations are read out for classification.

### Reservoir Dynamics and Topology
![Reservoir Drive and Topology](.github/static/reservoir_drive_topology_heatmap.png)

Characterisation of reservoir dynamics and spatial topology. Panels show the relative magnitude of input and recurrent drive contributions, reservoir state heatmaps over time, distance-based spatial weight kernels, and topology connectivity structure.

### ECG Dataset and Preprocessing
![ECG Dataset Characterisation](.github/static/samples_noise_scalers-subfigures.png)

**(a)** Class-averaged ECG waveforms for each heartbeat category with 10–90% percentile bands showing intra-class variability. **(b)** Additive Gaussian noise augmentation applied to a single ECG instance. **(c)** Effect of different preprocessing scalers on the input waveform shape.

### Raw MIT-BIH ECG Segment
![Raw ECG Segment](.github/static/ecg-10s-segment.png)

Example 10-second segment of a raw MIT-BIH record, showing the morphology and annotation context before conversion into single-beat classification instances.

### DDE Solver Validation
![Solver Validation](.github/static/solver_validation.png)

Convergence analysis of the RK4 DDE solver, validating numerical accuracy across step sizes.

### Classification Results (Balanced 5-Class)
![Balanced Categorical Confusion Matrix](.github/static/balanced-categorical-confusion-matrix.png)

Aggregated confusion matrix across five folds for the balanced five-class arrhythmia task (N=1000, KNN k=4, distance-weighted). Off-diagonal values indicate common misclassification patterns.

### Classification Results (Binary)
![Binary Confusion Matrix](.github/static/binary-confusion-matrix.png)

Aggregated confusion matrix across five folds for the binary Normal vs Arrhythmia task (N=1000, Random Forest).

### Hyperparameter Optimisation
![Optimisation Slice Plot](.github/static/P2-distance-slice.png)

Updated Optuna slice plot generated from the training study evaluation command for the Phase 2 joint optimisation (P2-distance, distance-based topology). Each panel shows per-trial macro F1-score as a function of one hyperparameter.

### Readout Optimisation
![Readout Slice Plot](.github/static/P4-readout-balanced-slice.png)

Optuna slice plot generated from the balanced readout optimisation study (P4-readout-balanced), showing how classifier-specific hyperparameters affect macro F1 across the frozen-state readout search.

## Optimisation CLI

Use the unified optimisation CLI to run studies, inspect the best trials, and regenerate diagnostic plots. See `python -m optimization.optimize --help` for the full option set.

Study names follow Supplementary Table S5 (`P1-scaling`, `P2-distance`, `P4-readout-balanced`, `ESN-balanced`, `B1-balanced`, ...), and each study is stored in `results/optuna/optuna-<study>.db`.

### Run Studies

```bash
python -m optimization.optimize research --type training --study P2-distance --processes 32
python -m optimization.optimize research --type readout --study_name P4-readout-balanced --trial_name P2-distance-best --processes 32
```

### Evaluate Studies

```bash
python -m optimization.optimize evaluate --type readout --study_name P4-readout-balanced --plots slice --show metrics
python -m optimization.optimize evaluate --type training --study P2-distance --plots slice --show metrics
```

Both commands generate plots into `results/figures/optimization/`.

### Results Layout

Paths and settings come from [utils/config.py](utils/config.py). Each setting is read from the environment first, then from an optional `config.env` file (copy [config.env.example](config.env.example)), then the default. Relative paths resolve against the repository root.

| Default path | Setting | Contents |
|---|---|---|
| `results/optuna/` | `BIORESERVOIR_RESULTS_DIR` (root) | Optuna studies, `optuna-<study>.db` |
| `results/runs/<run>/` | | Run folders: `config.yaml`, `metrics.json`, `fold-<k>.npz`, `figures/`, `run.log`, `manifest.json` |
| `results/metrics/` | | Derived metric tables and statistics |
| `results/figures/<topic>/` | | Figures |
| `logs/` | `BIORESERVOIR_LOGS_DIR` | Log files, `<command>-<study>-<UTC time>.log` |
| `archive/<version>/` | `BIORESERVOIR_ARCHIVE_DIR` (root) | Frozen results of a released version |

An archive has the same `optuna/`, `runs/` and `metrics/` folders, plus `MANIFEST.csv` (what each file supports in the paper) and `SHA256SUMS`. Archives are read-only and are not part of the repository; new outputs always go to `results/` and `logs/`. To inspect an archived study or run:

```bash
python -m optimization.optimize evaluate --type training --study P2-distance --archive v0.1
python scripts/analysis/analyse_fold_results.py --glob 'archive/v0.1/runs/P4-readout-balanced-best/fold-*.npz' --plots confusion
```

### Host-Agnostic Reproducibility Scripts

Some studies are intentionally script-driven rather than Optuna-driven. These remain reproducible because they are self-contained Python entrypoints with no host-specific orchestration:

- [scripts/run_ablation.py](scripts/run_ablation.py) — Ablation conditions with fixed reservoir and RF readout.
- [scripts/run_heterogeneity.py](scripts/run_heterogeneity.py) — Parameter heterogeneity robustness grid.

Run these locally in the same Python environment as the optimisation CLI.

### Remote Deployment

[optimization/deploy.py](optimization/deploy.py) runs studies or commands on remote hosts in a detached session, and [optimization/collect.py](optimization/collect.py) copies the databases, run folders and logs back. Each host needs Linux with glibc 2.17 or newer (2.28 or newer to draw figures), `ssh` and `rsync`, and key-based SSH login; its own Python is not used. Setting up a host (`--setup`, and before every launch) copies the code, the tests, the configs and `uv.lock` to its workspace with your local `uv`, which installs Python 3.12 and the locked packages there (with the test tools of the `dev` extra). Data files that `data/` links to (for example from a worktree) are copied as files. Host addresses, the SSH user and the remote workspace (shared, or one per host) are set in `config.env` (see [config.env.example](config.env.example)); hosts are otherwise referred to by label (`host1`, ...), and every remote run records its label, CPU model, core counts and RAM with its results.

```bash
python -m optimization.deploy --inventory          # hardware per host -> results/metrics/hosts.json
python -m optimization.deploy --host host1 --study P2-distance
python -m optimization.deploy --host host5 --label tests --run "python -m pytest tests/unit tests/integration"
python -m optimization.deploy --status
python -m optimization.collect --host host1 --study P2-distance
```

> [!TIP]
> Use `--help` to see what else is available

Example output from the readout evaluation:

```text
========================================================================
  ALL classifiers — P4-readout-balanced
========================================================================
    f1  trial classifier rf_criterion  rf_max_depth rf_max_features  rf_min_samples_leaf  rf_min_samples_split  rf_n_estimators rf_oob_score  accuracy     f1  f1_macro   f1_std  f1_weighted  kappa    mcc  n_folds  precision  precision_macro  precision_weighted primary_average  recall  recall_macro  recall_weighted  runtime
0.8953    833         RF     log_loss            23            sqrt                    1                     2              326         True    0.8949 0.8953    0.8953  0.00737       0.8953 0.8686 0.8691        5     0.8973           0.8973              0.8973           macro  0.8949        0.8949           0.8949    243.7

  Summary — top   10: mean=0.8950  std=0.0002  min=0.8948  max=0.8953
  Summary — all   946: mean=0.8108  std=0.2076  min=0.0665  max=0.8953
```

Example output from the training evaluation:

```text
========================================================================
	study — P2-distance
========================================================================
		f1  trial  cell_coupling  dde_scaling  fade_alpha  input_connectivity  input_scaling  p_excite  rc_scaling  sparsity  units  warmup  accuracy     f1  precision  recall  runtime
0.9047    416          12.75    0.0006439   2.803e-05              0.5282         0.0277    0.8354    0.007036     0.576   1000      50    0.9041 0.9047     0.9096  0.9041     2401

	Summary — top   10: mean=0.9014  std=0.0019  min=0.8988  max=0.9047
	Summary — all   768: mean=0.8481  std=0.0748  min=0.1924  max=0.9047
```

## Acknowledgements

**ReservoirPy** — Our `BioReservoir` class extends the ReservoirPy `Node` class.
- [reservoirpy](https://github.com/reservoirpy/reservoirpy)

**Optuna** — Hyperparameter optimisation framework used for study-based tuning.
- [Optuna](https://github.com/optuna/optuna)

**Genetic Oscillators** — Our approach builds upon research into coupled genetic oscillators within bacterial colonies:
- [A sensing array of radically coupled genetic 'biopixels'](https://doi.org/10.1038/nature10722) — Prindle et al. (2011)
- [A synchronized quorum of genetic clocks](https://doi.org/10.1038/nature08753) — Danino et al. (2010)

**Arrhythmia Dataset** — MIT-BIH dataset from Kaggle:
- [Kaggle Heartbeat Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) — Shayan Fazeli
- [ECG Heartbeat Classification: A Deep Transferable Representation](http://dx.doi.org/10.1109/ICHI.2018.00092) — Kachuee et al. (2018)
