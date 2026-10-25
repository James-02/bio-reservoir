# v0.2 campaign configs

The configs of the `v0.2` tuning campaign. Each file is a run config with a `study`
section ([Studies](../../README.md#studies)): every number of the campaign comes from one of
these studies, run from a tagged commit through `optimization.deploy`, so each trial records
the commit, the host and its resolved config.

| Folder | What | Studies | Trials |
|---|---|---|---|
| `tracking/` | The tracking set, rerun at every tagged code version: the P centre, P lattice #208, A′ #109, U #68 and the `v0.1` ESN, each at three reservoir seeds | 5 | 15 |
| `phase-1/` | Phase 1 grids: one or two design parameters at a time around the P centre, at three reservoir seeds | 14 | 540 |
| `phase-2/` | Phase 2 seeded TPE searches over every design parameter: P lattice, A′, U and the ESN | 4 | 1,000 |

Study names give the phase and what the study varies: `P1-coupling` is the Phase 1 coupling
grid, `P2-lattice` the Phase 2 search of the P lattice, and `tracking-<design>` a design of the
tracking set. Later phases follow the same pattern: `phase-3/` (`P3-`, robustness at the
frozen designs), `phase-4/` (`P4-`, readouts and raw-beat baselines), `phase-5/` (`P5-`, final
runs) and `phase-6/` (`P6-`, ablations and controls). `tests/unit/test_experiment_configs.py`
checks every config here.

## Models

- **P**: colonies in a square trap lattice exchanging AHL by diffusion
  (`coupling: diffusion`), a non-negative inducer input to some of the traps, one image of
  the LuxI reporter at the end of the beat with 1% measurement noise
  (`features: {variables: [I], aggregation: final, measurement_noise: 0.01}`), and a
  leave-one-out ridge readout (`RidgeCV`).
- **P centre**, the reference design: N = 1,024 traps, input scaling 0.155113 (5.6 times the
  `v0.1` input scaling of 0.0277), coupling rate 0.6, input connectivity 0.66,
  mean-preserving kinetic heterogeneity of CV 0.065, warm-up 800 steps, 16 RK4 substeps.
- **A′**: the abstract coupling of `v0.1` with its corrections (a signed Gaussian kernel of
  random positions applied to the observed states); not chemically realisable.
- **U**: the colonies of P without coupling (`coupling: none`).
- **ESN**: the `v0.1` echo state network ([../benchmarks/v0.1/esn.yaml](../benchmarks/v0.1/esn.yaml)),
  read from the clean mean of its states over the beat.

Every config uses the balanced pool (4,015 beats), five folds and seed 6337 for the pool and
the folds. `param_heterogeneity_seed` is `null`, so the heterogeneity draw follows
`reservoir.seed`, and the three seeds 6337, 6338 and 6339 draw different reservoirs: input
weights, heterogeneity, start state and noise. Fold SDs are 0.01–0.02 and seed SDs about
0.005, so differences below about 0.005 are noise.

## Phase 1 grids (`phase-1/`)

| Study | Values | Trials |
|---|---|---|
| `P1-gain` | Input scaling 0.0277–0.554 (×1–20) | 33 |
| `P1-coupling` | Coupling rate 0–10 | 42 |
| `P1-connectivity` | Input connectivity 0.1–1 | 30 |
| `P1-heterogeneity` | Kinetic heterogeneity CV 0–0.5 | 27 |
| `P1-warmup` | Warm-up 0–2,000 steps | 33 |
| `P1-units` | Square lattices of 100–2,025 traps | 27 |
| `P1-gain-coupling` | Input ×2–12 × coupling rate 0.1–4 | 108 |
| `P1-connectivity-heterogeneity` | Input connectivity 0.4–1 × CV 0–0.3 | 60 |
| `P1-images-noise` | 1–4 images × measurement noise 0.5–3% | 48 |
| `P1-images-calibration-s6337`, `-s6338`, `-s6339` | 1–4 images × calibration error 0–5% | 3 × 24 |
| `P1-substeps` | 8–128 RK4 substeps | 15 |
| `P1-gaussian` | Random positions with a Gaussian kernel: range 2–8.2 cell diameters × coupling rate 0.3–1 (layout comparison only) | 45 |

The calibration error of a test fold is drawn from the run's top-level `seed`
([training/observation.py](../../training/observation.py)), which a study cannot search. So
the three calibration draws are three studies, each with its own seed for the calibration,
the measurement noise and the reservoir; `study-table --group` reads them as one grid.

## Phase 2 searches (`phase-2/`)

Seeded multivariate TPE (`seed: 6337`, 24 start-up trials), N = 1,024 traps and 16
substeps, reservoir seed 6337. The ranges are those of the exploration searches, whose input
and recurrent gains were multiples of the `v0.1` scalings; here they are absolute scalings.
Each search first runs the exploration's best design (P lattice #208 and the P centre, A′
#109, U #68, the `v0.1` ESN).

| Study | Searched | Trials |
|---|---|---|
| `P2-lattice` | Input scaling 0.0138–0.554 (log), coupling rate 0.01–10 (log), input connectivity 0.1–1, CV 0–0.5, warm-up 50–2,000 (log) | 300 |
| `P2-abstract` | Input scaling 0.00692–0.554 (log), recurrent scaling 7.04e-5–0.0211 (log), input connectivity 0.1–1, CV 0–0.5, warm-up 50–2,000 (log) | 300 |
| `P2-uncoupled` | Input scaling 0.0138–0.554 (log), input connectivity 0.1–1, CV 0–0.5, warm-up 50–2,000 (log) | 200 |
| `P2-esn` | Leak rate 0.05–1, spectral radius 0.1–3 (log), input scaling 0.0327–3.27 (log), recurrent connectivity 0.05–0.5, bias scaling 0–2 | 200 |

## Running and reading

Run a study on a host through deploy, from a clean checkout of a tagged commit, one study
per host at a time. A study resumes when its command runs again; a grid runs each point
once.

```bash
python -m optimization.deploy --host host3 --label P2-lattice \
    --run 'python -m training.run study experiments/v0.2/phase-2/lattice.yaml --processes $NPROC'
python -m optimization.collect --host host3 --study P2-lattice --logs
```

Several `--run` options run one after another in one session. The tracking set runs at every
tag; run it in a fresh workspace for each tag (or give each tag its own `--storage`), so that
a study does not resume the trials of another code version.

`study-table` prints the trials of one or more collected studies; `--group` averages each
grid point over its reservoir seeds (and over the studies given), and `--csv` writes the
table:

```bash
python -m training.run study-table experiments/v0.2/phase-1/gain.yaml --group
python -m training.run study-table experiments/v0.2/phase-1/images-calibration-s*.yaml --group --csv calibration.csv
python -m training.run study-table experiments/v0.2/phase-2/lattice.yaml --csv lattice.csv
python -m training.run study-best experiments/v0.2/phase-2/lattice.yaml --out lattice-best.yaml
```

A trial row holds the searched values, the reservoir seed, the mean macro F1 over the five
folds and its SD over the folds, the trial's wall time and the time spent simulating the
beats, and the commit. A grouped row holds the mean over seeds, the SD over seeds, the mean
SD over folds and the seeds.
