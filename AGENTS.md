# AGENTS.md

Guidance for AI coding agents (and new human contributors) working in this
repository. Keep this file short and factual — it is read every session.

## What this project is

A reservoir computing framework built on coupled quorum-sensing genetic
oscillators, applied to ECG arrhythmia classification. Each reservoir node is a
delay-differential-equation (DDE) oscillator implemented as a
[ReservoirPy](https://github.com/reservoirpy/reservoirpy) `Node`.
Hyperparameter tuning uses [Optuna](https://github.com/optuna/optuna).

The two anchor dependencies are **reservoirpy** and **optuna**. Prefer their
conventions and integrate with them rather than reinventing.

## Repository layout

- `bioreservoir/` — core library: DDE solver (`dde.py`), `BioReservoir` node
  (`node.py`), topology generators (`topology.py`).
- `training/` — cross-validation pipeline, classification helpers, profiling.
- `utils/` — preprocessing (ECG loading/augmentation), analysis metrics,
  logging, results, visualisation.
- `optimization/` — Optuna study registry (`studies.py`), reservoir builders
  (`reservoirs.py`), objectives, and the CLI (`optimize.py`).
- `scripts/` — standalone experiment/analysis scripts.
- `tests/unit/` and `tests/integration/` — pytest suite.

## Workspace data layout (all gitignored)

All paths and settings come from `utils/config.py` (environment variable, then
`config.env`, then default); do not hard-code `results/` or `logs/` paths.

- `results/` — working outputs (`BIORESERVOIR_RESULTS_DIR`): `optuna/optuna-<study>.db`,
  `runs/<run>/fold-<k>.npz`, `metrics/`, `figures/<topic>/`.
- `logs/` — log files only (`BIORESERVOIR_LOGS_DIR`), named
  `<command>-<study>-<UTC time>.log`.
- `archive/<version>/` — **frozen** results behind a manuscript version, read-only:
  `optuna/`, `runs/`, `metrics/`, `MANIFEST.csv` (what each file is and which
  manuscript item it supports) and `SHA256SUMS`
  (`cd archive/v0.1 && sha256sum -c SHA256SUMS`). Tools only read it
  (`--archive v0.1`, or an explicit path such as `--glob` / `--runs_dir`). Promote
  results from `results/` to a new `archive/<version>/` only when they are frozen
  for a manuscript version.
- `data/` — input datasets.
- Snapshots of earlier workspaces live outside the repository (`../backups/`);
  planning notes live in the private repository.

## Naming conventions

- **Studies** (SI Table S5): phase prefix `P1`–`P4`, `ESN` or `B1`–`B3`, then a
  lowercase descriptor, e.g. `P1-scaling`, `P2-distance`, `P4-readout-balanced`,
  `B1-balanced`. Database file: `optuna-<study>.db`, with the same study name inside.
- **Runs**: `<study>-best[-<variant>]` for re-runs of a study's best trial (the
  `rerun-best` default), e.g. `P4-readout-binary-best-full`; script-driven runs use a
  descriptive prefix (`ablation-<condition>`, `heterogeneity-cv<cv>-seed<seed>`).

## Build, test, and lint commands

```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"        # editable install with dev tools

pytest tests/unit/ -v          # fast unit tests
pytest tests/integration/ -v   # slower integration tests
pytest --cov --cov-report=term-missing   # coverage

ruff check .                   # lint
ruff format .                  # format
```

## Conventions

- **Docstrings:** NumPy style (matches numpy/reservoirpy). Napoleon-compatible.
- **Formatting/linting:** `ruff` (line length 100, target py39).
- **Commits:** Conventional Commits (`feat:`, `fix:`, `chore:`, `ci:`,
  `docs:`, `test:`). Versioning follows SemVer.
- **Python support:** 3.9–3.12 (runtime deps are pinned; see `pyproject.toml`).
- Library code must not call `logging.basicConfig`; use
  `utils.logger.setup_logging` from CLIs/scripts only.

## Hard rules (do not violate)

- **Never modify `archive/`.** It holds the frozen artefacts of a manuscript
  version (the `v0.1` code is archived on Zenodo, DOI 10.5281/zenodo.23436888).
  Write new results to `results/` and logs to `logs/`. Do not change pinned runtime
  dependency versions before `v1.0`.
- **No infrastructure secrets in committed files.** No hostnames, IPs,
  credentials, cluster names, or host-specific ops. Deployment wrappers
  (`scripts/deploy*.sh`, `optimization/deploy.py`, `optimization/collect.py`)
  are gitignored on purpose.
- Secrets come from the environment (see `config.env.example`); the only one
  currently required is a Kaggle API token for dataset download.

## Domain quick-reference (science → code)

- Each node models two genes: **luxI** and **aiiA**, producing intracellular
  AHL (**Hi**) that diffuses externally (**He**).
- ECG input drives the reservoir through **He**.
- The extracted reservoir **state is the luxI concentration** per node.
- Coupling weights are distance-dependent (Gaussian/exponential kernels) built
  in `bioreservoir/topology.py`.
- `BioReservoir` subclasses the ReservoirPy `Node`; respect that contract when
  extending (e.g. `run(X) -> states`).
