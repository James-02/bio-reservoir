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
- `results/`, `data/` — experiment artefacts and datasets (see rules below).
- `future/` — **gitignored** local planning notes (not published).

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

- **Do not edit committed results.** Everything under `results/` (and the ECG
  data manifests) are frozen paper artefacts. The manuscript is under review;
  code, experiment setup, and results must stay immutable until the Zenodo DOI
  is minted. Add new files/tooling; never alter research outputs or pinned
  runtime dependency versions.
- **No infrastructure secrets in committed files.** No hostnames, IPs,
  credentials, cluster names, or host-specific ops. Deployment wrappers
  (`scripts/deploy*.sh`, `optimization/deploy.py`, `optimization/collect.py`)
  are gitignored on purpose.
- **`future/` is gitignored planning** — put rough/strategic notes there, not
  in tracked files.
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
