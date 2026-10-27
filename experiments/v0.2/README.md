# v0.2 campaign configs

The configs of the `v0.2` tuning campaign. Each study names one of the designs in `designs/`
(`base: ../designs/centre.yaml`) and lists only what differs and its `study` section
([Studies](../../README.md#studies)): every number of the campaign comes from one of
these studies, run from a tagged commit through `optimization.deploy`, so each trial records
the commit, the host and its resolved config.

| Folder | What | Studies | Trials |
|---|---|---|---|
| `designs/` | The five designs: the P centre, the best P lattice, A′ and U designs of the earlier searches, and the `v0.1` ESN | — | — |
| `tracking/` | The tracking set, rerun at every tagged code version: each design at three reservoir seeds | 5 | 15 |
| `phase-1/` | Phase 1 grids: one or two design parameters at a time around the P centre, at three reservoir seeds | 13 | 549 |
| `phase-2/` | Phase 2 seeded TPE searches over every design parameter: P lattice, A′, U and the ESN (clean mean and one noisy image); and the five best trials of each at reservoir seeds 6338–6340 (`P2-<design>-top`) | 5 + 5 | 1,200 + 75 |
| `slow-sensing/` | The designs on the eight inducer-pattern tasks, and the colony budget, coupling and memory of P (below) | 17 | 988 |

Study names follow the convention of the `v0.1` studies (SI Table S5): a phase prefix and
a descriptor, with each phase keeping its `v0.1` meaning. Phase 1 (`P1-`, folder `phase-1/`)
varies one or two parameters at a time, Phase 2 (`P2-`) searches all of a design's parameters
jointly, Phase 3 (`P3-`) evaluates the chosen designs on other tasks, and Phase 4 (`P4-`)
optimises the readout on saved states. Classifiers on the raw beat, without a reservoir, are
the baselines (`B1-`, `B2-`, ...), and `tracking-<design>` is a design of the tracking set.
So `P1-coupling` is the Phase 1 coupling grid and `P2-lattice` the Phase 2 search of the P
lattice. A `v0.2` study and a `v0.1` study of the same name are told apart by version: the
`v0.1` databases are in `archive/v0.1/`. `tests/unit/test_experiment_configs.py` checks every
config here.

## Models

- **P**: colonies in a square trap lattice exchanging AHL by diffusion
  (`coupling: diffusion`), a non-negative inducer input to some of the traps, one image of
  the LuxI reporter at the end of the beat with 1% measurement noise
  (`features: {variables: [I], aggregation: final, measurement_noise: 0.01}`), and a
  leave-one-out ridge readout whose penalty is chosen by macro F1 (`RidgeCV` with
  `scoring: f1_macro`; the default squared error scored 0.003–0.010 lower at the P centre).
- **P centre**, the reference design: N = 1,024 traps, input scaling 0.155113 (5.6 times the
  `v0.1` input scaling of 0.0277), coupling rate 0.6, input connectivity 0.66,
  mean-preserving kinetic heterogeneity of CV 0.065, warm-up 800 steps, 32 RK4 substeps
  (at 16, colonies whose degradation rates the heterogeneity raises blow up at CV 0.3 and
  above).
- **A′**: the abstract coupling of `v0.1` with its corrections (a signed Gaussian kernel of
  random positions applied to the observed states); not chemically realisable.
- **U**: the colonies of P without coupling (`coupling: none`).
- **ESN**: the `v0.1` echo state network ([../benchmarks/v0.1/esn.yaml](../benchmarks/v0.1/esn.yaml)),
  read from the clean mean of its states over the beat (as in `v0.1`), or, like the
  bioreservoirs, from one image of its states at the end of the beat with 1% measurement
  noise (`P2-esn-image`).

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
| `P1-heterogeneity-64` | CV 0.3–0.5 at 64 substeps (32 blow up colonies from CV ≈ 0.4) | 9 |
| `P1-warmup` | Warm-up 0–2,000 steps | 33 |
| `P1-units` | Square lattices of 100–2,025 traps | 27 |
| `P1-gain-coupling` | Input ×2–12 × coupling rate 0.1–4 | 108 |
| `P1-connectivity-heterogeneity` | Input connectivity 0.4–1 × CV 0–0.3 | 60 |
| `P1-images-noise` | 1–4 images × measurement noise 0.5–3% | 48 |
| `P1-images-calibration` | 1–4 images × calibration error 0–5%, at three run seeds | 72 |
| `P1-substeps` | 8–128 RK4 substeps | 15 |
| `P1-gaussian` | Random positions with a Gaussian kernel: range 2–8.2 cell diameters × coupling rate 0.3–1 (layout comparison only) | 45 |

The calibration error of a test fold is drawn from the run's top-level `seed`
([training/observation.py](../../training/observation.py)), so `P1-images-calibration`
searches `seed`, and the reservoir seed follows it.

## Phase 2 searches (`phase-2/`)

Seeded multivariate TPE (`seed: 6337`, 24 start-up trials), N = 1,024 traps and 32
substeps, reservoir seed 6337. The ranges are those of the exploration searches, whose input
and recurrent gains were multiples of the `v0.1` scalings; here they are absolute scalings.
Each search first runs the best design of the earlier search (and the P centre for the P
lattice; the `v0.1` ESN for the ESN).

| Study | Searched | Trials |
|---|---|---|
| `P2-lattice` | Input scaling 0.0138–0.554 (log), coupling rate 0.01–10 (log), input connectivity 0.1–1, CV 0–0.5, warm-up 50–2,000 (log) | 300 |
| `P2-abstract` | Input scaling 0.00692–0.554 (log), recurrent scaling 7.04e-5–0.0211 (log), input connectivity 0.1–1, CV 0–0.5, warm-up 50–2,000 (log) | 300 |
| `P2-uncoupled` | Input scaling 0.0138–0.554 (log), input connectivity 0.1–1, CV 0–0.5, warm-up 50–2,000 (log) | 200 |
| `P2-esn` | Leak rate 0.05–1, spectral radius 0.1–3 (log), input scaling 0.0327–3.27 (log), recurrent connectivity 0.05–0.5, bias scaling 0–2 | 200 |
| `P2-esn-image` | As `P2-esn`, read from one noisy image of the states | 200 |

For the selection table, `P2-<design>-top` reruns the five best trials of each search at
reservoir seeds 6338–6340: written from the finished search, the trials' values are
enqueued and run exactly (random sampler, 15 trials).

## Slow sensing (`slow-sensing/`)

The designs on the eight inducer-pattern tasks ([training/inducer.py](../../training/inducer.py);
`python scripts/figures/visualise_inducer_tasks.py` draws them): 400 traces per class, one
noisy image at the end of the trace, ridge readout, three reservoir seeds. The bioreservoirs
read the inducer from 0 (`input_range: [0.0, 20.0]`). Pump delivery is `condition: device`.
Draw a study's scores with `scripts/figures/visualise_slow_sensing.py`:

```bash
python scripts/figures/visualise_slow_sensing.py experiments/v0.2/slow-sensing/centre-delay.yaml
python scripts/figures/visualise_slow_sensing.py --compare \
    experiments/v0.2/slow-sensing/{centre,uncoupled-centre,esn,raw}.yaml \
    --fix task.condition=device --fix reservoir.params.input_scaling=0.155113
```

| Study | What | Trials |
|---|---|---|
| `slow-sensing-centre` | P centre: 8 tasks × ideal or pump × input gain ×0.5–2 | 144 |
| `slow-sensing-uncoupled-centre` | As above without coupling | 144 |
| `slow-sensing-uncoupled-best` | U at P2-uncoupled trial 182: 8 tasks × ideal or pump | 48 |
| `slow-sensing-esn` | The `v0.1` ESN read like the bioreservoirs: 8 tasks × ideal or pump | 48 |
| `slow-sensing-esn-image-best` | The ESN of P2-esn-image trial 178 (slow leak): 8 tasks × ideal or pump | 48 |
| `slow-sensing-leaky-bank` | Leaky-integrator bank, clean or noisy image: 8 tasks × ideal or pump | 32 |
| `slow-sensing-raw` | Ridge on the whole trace: 8 tasks × ideal or pump | 16 |
| `slow-sensing-raw-rf`, `-raw-histgb`, `-raw-svm`, `-raw-knn` | A nonlinear classifier on the whole trace, settings by inner CV; the best of the four per task is the ceiling for one image | 16 each |
| `slow-sensing-centre-units-noise` | P centre on 100–1,024 traps × 1% or 3% noise, 4 tasks | 96 |
| `slow-sensing-centre-units-traces` | P centre on 256–1,024 traps × 400 or 800 traces per class, 2 tasks | 60 |
| `slow-sensing-centre-coupling` | P centre at coupling rate 0–3, 4 tasks | 72 |
| `slow-sensing-centre-delay`, `-uncoupled-centre-delay`, `-esn-delay` | Image 0–240 min after the pattern, 4 tasks | 72 each |

## Running and reading

Run a study on a host through deploy, from a clean checkout of a tagged commit, one study
per host at a time. A study resumes when its command runs again; a grid runs each point
once. A trial whose reservoir the solver cannot integrate (the RK4 step check refuses it, or
its states blow up) is recorded as failed and the study goes on; `study-table` lists it
without a score. Each trial records its machine in the `environment` attribute.

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
python -m training.run study-table experiments/v0.2/phase-1/images-calibration.yaml --group --csv calibration.csv
python -m training.run study-table experiments/v0.2/phase-2/lattice.yaml --csv lattice.csv
python -m training.run study-best experiments/v0.2/phase-2/lattice.yaml --out lattice-best.yaml
```

A trial row holds the searched values, the reservoir seed, the mean macro F1 over the five
folds and its SD over the folds, the trial's wall time and the time spent simulating the
beats, and the commit. A grouped row holds the mean over seeds, the SD over seeds, the mean
SD over folds and the seeds.
