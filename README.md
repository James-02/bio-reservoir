# Reservoir Computing with Genetic Oscillators

[![Tests](https://github.com/James-02/bio-reservoir/actions/workflows/tests.yml/badge.svg)](https://github.com/James-02/bio-reservoir/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/James-02/bio-reservoir/branch/main/graph/badge.svg)](https://codecov.io/gh/James-02/bio-reservoir)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23436888.svg)](https://doi.org/10.5281/zenodo.23436888)

This project classifies heartbeats from ECG recordings with a reservoir of simulated
bacterial colonies. Each colony carries a genetic clock (a quorum-sensing oscillator,
modelled by delay differential equations). The heartbeat drives the colonies, their
response is measured, and a simple classifier reads the arrhythmia class from that
response. The reservoir is a [ReservoirPy](https://github.com/reservoirpy/reservoirpy)
`Node`. The paper describes the method and the results.

![Graphical TOC](.github/static/graphical-toc.png)

## How It Works

1. **Colonies.** Each colony is an oscillator of two genes, luxI and aiiA, and the signal
   molecule AHL inside (Hi) and outside (He) the cells, with a delayed negative feedback.
2. **Input.** Each ECG sample is added to the extracellular AHL of the colonies it reaches.
3. **Coupling.** The colonies interact, either through an abstract signed kernel (the
   `v0.1` model) or by exchanging AHL through the medium (see [Model Options](#model-options)).
4. **Measurement.** The colonies' states are measured over the beat: by default every
   variable, or only the LuxI reporter in a few images, with noise.
5. **Readout.** A classifier (ridge, k-nearest neighbours, random forest and others) is
   trained on the measurements to name the beat's class.

## Install

Dependencies are locked in `uv.lock` (Python 3.12), so [uv](https://docs.astral.sh/uv/)
recreates the exact environment:

```bash
export UV_PROJECT_ENVIRONMENT=venv
uv sync --extra dev
source venv/bin/activate
cp config.env.example config.env   # Kaggle credentials and local paths
python -m utils.preprocessing      # download and prepare the MIT-BIH dataset
```

Without uv, `pip install -e .[dev]` installs the pinned dependencies.

## Quick Start

A run is described by a YAML config: the task, the reservoir, what is measured, the readout
and the cross-validation folds.

```bash
python -m training.run run experiments/benchmarks/v0.1/balanced.yaml --processes 5
python -m training.run resolve my-config.yaml      # print the full config
python -m training.run compare RUN_A RUN_B         # pair two runs fold by fold
python -m utils.provenance list                    # every run so far
```

Each run writes a folder `results/runs/<name>-<time>/` with its config, metrics (per fold,
mean and sample SD), predictions, figures, log and a manifest (code version, data checksums,
environment), and adds a line to the run registry. Runs from uncommitted code need
`--allow-dirty` and are marked as not citable.

The same pieces can be used from Python:

```python
import numpy as np
from reservoirpy.nodes import Ridge

from bioreservoir import BioReservoir
from training import cross_validate
from utils.preprocessing import load_ecg_data

X_train, Y_train, X_test, Y_test = load_ecg_data(rows=1000)
X, Y = np.concatenate([X_train, X_test]), np.concatenate([Y_train, Y_test])
results = cross_validate(BioReservoir(units=100), Ridge(ridge=1e-6), X, Y, folds=5)
```

## Configs

The main sections of a run config:

| Section | Sets | Example |
|---|---|---|
| `task` | Labels, pool size, balancing, scaling | `labels: five-class`, `rows: 4015` |
| `reservoir` | Type and parameters, or a study trial | `params: {units: 1024, coupling: diffusion}` |
| `features` | What is measured ([training/observation.py](training/observation.py)) | `variables: [I]`, `aggregation: images:4`, `measurement_noise: 0.01` |
| `readout` | Classifier and its parameters | `classifier: RidgeCV` |
| `cv` | Folds and seed | `folds: 5` |
| `test` | A held-out split to score on | `split: validation-1` |

- A config can start from another with `base: <file>` and list only what differs.
- `preset: v0.1` restores the code of the first paper; the `v0.1` headline configs in
  [experiments/benchmarks/v0.1](experiments/benchmarks/v0.1) use it.
- Each beat is simulated once and every fold reads the same features (`cv.simulate: once`).
- Readouts standardise the features on each training fold. `RidgeCV` picks its penalty by
  leave-one-out, and `readout.search` picks hyperparameters on inner folds.
- `reservoir: {type: none}` classifies the raw beat, the baseline without a reservoir.

`aggregation: images_at:155,187` takes an image at each listed minute of the beat (the state after that many steps; `images_at:187` is `final` for a 187-step beat).

### Studies

A config with a `study` section is a seeded Optuna search over its own values. Each trial
runs the same pipeline as a single run and records its full config.

```yaml
study:
  trials: 300
  metric: f1
  sampler: {type: tpe, seed: 6337, multivariate: true, n_startup_trials: 20}
  search:
    reservoir.params.input_scaling: {type: float, low: 0.005, high: 0.5, log: true}
    reservoir.params.units: {type: categorical, choices: [256, 1024]}
```

```bash
python -m training.run study my-study.yaml --processes 128      # resumes if it exists
python -m training.run study-table my-study.yaml --group        # one row per point
python -m training.run study-best my-study.yaml --out best.yaml # best trial as a run config
```

A study is stored in `results/optuna/optuna-<name>.db` (or `--storage`). `sampler: {type:
grid}` runs every combination. With `reservoir: {states_from: <run>}` a study tunes only the
readout on a run's saved states. The `v0.2` campaign configs are in
[experiments/v0.2](experiments/v0.2/README.md).

### Held-out Evaluation

Three held-out splits contain beats that no `v0.1` study used: `validation-1` and
`validation-2` (803 beats per class each) and `test` (370 per class). Their beat IDs and
checksums are in [data/splits/heldout.json](data/splits/heldout.json), and
`python -m utils.heldout --check` verifies them. Choose models by cross-validation, check
them on the validation splits, and use `test` once, for the final result:

```yaml
test:
  split: validation-1   # `test` also needs --final-test on the command line
```

## Model Options

All options are off by default, so the defaults change no result. Each table row is one
config change, so the effect of each change can be measured on its own.

**Coupling** (`reservoir.params`):

| Option | Effect |
|---|---|
| `coupling: abstract` | Default (`v0.1` model): a signed kernel of the colonies' states adds to their input |
| `coupling: diffusion` | Colonies exchange extracellular AHL through the medium; `diffusion_rate` sets the rate |
| `coupling: none` | Independent colonies |
| `topology_type: lattice` | Colonies in a square trap array (diffusion only) |
| `input_encoding: nonnegative`, `input_sign: positive` | The input is a non-negative inducer level |
| `preset: qs_diffusion` | The chemically realisable model: these choices together |

**Realism** (`reservoir.params`, [bioreservoir/realism.py](bioreservoir/realism.py)):

| Option | Effect |
|---|---|
| `param_heterogeneity_cv` | Colony-to-colony variation of the kinetic rates |
| `heterogeneity_density_cv`, `heterogeneity_dilution_cv` | Variation of cell density and washout |
| `input_levels: K` | The input reaches the traps at K discrete levels |
| `reset_noise: cv` | Each beat starts from a slightly different state |
| `warmup_input: rest` | The colonies warm up at the resting inducer level |

**Controls** ask what the colonies add over simpler feature maps of the beat:

| Control | Config |
|---|---|
| Random nonlinear features | `reservoir: {type: random_features, params: {units: 8000}}` |
| A bank of leaky integrators | `reservoir: {type: leaky_bank, params: {units: 24}}` |
| A linear filter fitted to a run's features | `reservoir: {states_from: <run>, surrogate: linear}` |

**Ablations** remove one part of the model:

| Ablation | Config |
|---|---|
| No coupling | `reservoir.params.coupling: none` |
| No feedback delay | `reservoir.params.delay: 0` (default 10 input steps) |
| Scrambled beats | `task.scramble: true` |

Controls and ablations use the same pool, folds and readout as the reservoir, so
`training.run compare` pairs them beat by beat.

The colonies are integrated with an explicit RK4 solver. A run refuses a step too long for
the coupling, and stops with `UnstableIntegrationError` if the states blow up; large
heterogeneity needs more substeps (`rk4_substeps: 32` or 64).
`python scripts/analysis/drive_bounds.py <config>` reports a reservoir's drive range.

## Figures

The figure scripts draw the `v0.1` paper figures from this repository's configs and runs.
Each saves a PNG under `results/figures/<topic>/` (`--formats pdf,svg` adds more) and warns
if the figure would break the ACS rules at its printed size.

| Script | Draws (`v0.1` figure) | Input |
|---|---|---|
| [visualise_dde.py](scripts/figures/visualise_dde.py) | One colony at several input scalings (3b) | Run config |
| [visualise_reservoir_states.py](scripts/figures/visualise_reservoir_states.py) | The array's states during a beat (4c); `--drive` adds its drive (4a) | Run config |
| [visualise_topology.py](scripts/figures/visualise_topology.py) | The kernel (4b) and the strongest links (4d) | Run config |
| [visualise_dataset.py](scripts/figures/visualise_dataset.py) | Class means, noise and scalings (5) | Run config |
| [visualise_heterogeneity.py](scripts/figures/visualise_heterogeneity.py) | Macro F1 against heterogeneity (S10) | Study config or `--archive v0.1` |
| [analyse_fold_results.py](scripts/analysis/analyse_fold_results.py) | Metric tables; `--plots` adds confusion, scores and PCA | Fold files |

### Figures from the `v0.1` Paper

**The model.** (a) One colony: luxI, aiiA and AHL inside (Hi) and outside (He) the cells.
(b) How the ECG input and the coupling drive gene expression. (c) The reservoir: the ECG
drives the coupled colonies and a readout classifies their final LuxI levels.

![Biopixel Reservoir Model](.github/static/biopixel-reservoir-model-3-subfigures.png)

**Reservoir dynamics.** The input and coupling drive, the colonies' states over a beat, the
distance kernel and the strongest links.

![Reservoir Drive and Topology](.github/static/reservoir_drive_topology_heatmap.png)

**The dataset.** (a) The mean beat of each class with its 10–90% band. (b) A beat with
added noise. (c) A beat under three scalings.

![ECG Dataset Characterisation](.github/static/samples_noise_scalers-subfigures.png)

**A raw recording.** Ten seconds of a MIT-BIH record before it is cut into beats.

![Raw ECG Segment](.github/static/ecg-10s-segment.png)

**Solver check.** The error of the RK4 solver against the step size.

![Solver Validation](.github/static/solver_validation.png)

**Balanced five-class result.** Confusion matrix over five folds (`P2-distance-best`: 2,000
colonies, k-nearest-neighbour readout with k = 1).

![Balanced Categorical Confusion Matrix](.github/static/balanced-categorical-confusion-matrix.png)

**Binary result.** Normal against arrhythmia over five folds (`P3-binary-best`: 2,000
colonies, distance-weighted k-nearest neighbours with k = 4).

![Binary Confusion Matrix](.github/static/binary-confusion-matrix.png)

**Tuning.** Macro F1 of each trial against each parameter, for the reservoir search
(`P2-distance`) and the readout search (`P4-readout-balanced`).

![Optimisation Slice Plot](.github/static/P2-distance-slice.png)
![Readout Slice Plot](.github/static/P4-readout-balanced-slice.png)

## Reproducing `v0.1`

`pytest -m benchmark` reruns the `v0.1` headline configs and compares them fold by fold
with the archived runs (balanced, unbalanced, binary and ESN reproduce exactly). The binary
run reads the `v0.1` binary file `data/ecg_archive/binary_ecg_data.npz` (checked by its
SHA-256); without it, the binary tests are skipped.

The `v0.1` studies use the older optimisation command (study names follow Supplementary
Table S5):

```bash
python -m optimization.optimize research --type training --study P2-distance --processes 32
python -m optimization.optimize evaluate --type training --study P2-distance --archive v0.1 --plots slice
```

[scripts/run_ablation.py](scripts/run_ablation.py) and
[scripts/run_heterogeneity.py](scripts/run_heterogeneity.py) rerun the `v0.1` ablations and
heterogeneity grid (`--preset v0.1`).

## Outputs

Paths come from [utils/config.py](utils/config.py): environment variable, then `config.env`,
then the default.

| Path | Setting | Contents |
|---|---|---|
| `results/` | `BIORESERVOIR_RESULTS_DIR` | `optuna/` studies, `runs/` run folders, `metrics/` tables, `figures/` |
| `logs/` | `BIORESERVOIR_LOGS_DIR` | Log files |
| `archive/<version>/` | `BIORESERVOIR_ARCHIVE_DIR` | Frozen, read-only results of a release (`--archive VERSION`) |

## Remote Hosts

[optimization/deploy.py](optimization/deploy.py) copies the code to a Linux host over SSH,
installs the locked environment there with `uv`, and runs a study or command in a detached
session; [optimization/collect.py](optimization/collect.py) fetches the results. Host
addresses and the remote workspace are set in `config.env`; hosts are named by label.

```bash
python -m optimization.deploy --host host1 --run 'python -m training.run study my-study.yaml'
python -m optimization.deploy --status
python -m optimization.collect --host host1 --study my-study
```

## Project Structure

| Folder | Contents |
|---|---|
| `bioreservoir/` | The model: DDE solver, `BioReservoir` node, topologies, realism, diagnostics |
| `training/` | Runs, studies, cross-validation, measurement, comparison |
| `optimization/` | The `v0.1` optimisation command, controls, deployment |
| `utils/` | Data preparation, held-out splits, provenance, settings, plotting |
| `scripts/` | Figure and analysis scripts, `v0.1` experiment scripts |
| `experiments/` | Run and study configs: `benchmarks/v0.1/`, `v0.2/` |
| `data/splits/` | Held-out splits (tracked); the rest of `data/` is generated |
| `tests/` | Unit and integration tests |

See the [CHANGELOG](CHANGELOG.md) for what changed in each version.

## Acknowledgements

- [ReservoirPy](https://github.com/reservoirpy/reservoirpy): `BioReservoir` is a ReservoirPy
  `Node`.
- [Optuna](https://github.com/optuna/optuna): hyperparameter searches.
- Coupled genetic oscillators: [Prindle et al. (2011)](https://doi.org/10.1038/nature10722),
  *A sensing array of radically coupled genetic 'biopixels'*;
  [Danino et al. (2010)](https://doi.org/10.1038/nature08753), *A synchronized quorum of
  genetic clocks*.
- MIT-BIH heartbeats from [Kaggle](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)
  (Shayan Fazeli); [Kachuee et al. (2018)](http://dx.doi.org/10.1109/ICHI.2018.00092),
  *ECG Heartbeat Classification: A Deep Transferable Representation*.
