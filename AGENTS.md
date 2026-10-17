# AGENTS.md

Guidance for AI coding agents (and new human contributors) working in this
repository. Keep this file short and factual — it is read every session.

## Working together

- Keep changes simple and standard: well-supported libraries, standard patterns,
  NumPy-style docstrings. Prefer the conventions of numpy, reservoirpy, optuna and
  scikit-learn over custom machinery.
- Explain every non-trivial design or change to the maintainer in plain language and
  agree it before implementing it.
- Work on task branches with small Conventional Commits (`feat(component): ...`,
  `fix(...)`, `test(...)`, `docs(...)`, `build(...)`, `ci(...)`, `chore(...)`).
  Branches are reviewed and merged by PR; `v0.2` is tagged for the resubmission.
- Record every change between `v0.1` and `v0.2` in `CHANGELOG.md`, marked
  `[science]` (changes a reported number) or `[engineering]`.

## What this project is

A reservoir computing framework built on coupled quorum-sensing genetic
oscillators, applied to ECG arrhythmia classification. Each reservoir node is a
delay-differential-equation (DDE) oscillator implemented as a
[ReservoirPy](https://github.com/reservoirpy/reservoirpy) `Node`.
Hyperparameter tuning uses [Optuna](https://github.com/optuna/optuna).

The two anchor dependencies are **reservoirpy** and **optuna**. Prefer their
conventions and integrate with them rather than reinventing.

## Repository layout

- `bioreservoir/` — the model: DDE solver (`dde.py`), `BioReservoir` node
  (`node.py`), topology generators (`topology.py`). Must be well tested, documented
  and easy to read.
- `training/` — cross-validation, training and evaluation, metrics, profiling.
- `optimization/` — Optuna study registry (`studies.py`), reservoir builders
  (`reservoirs.py`), objectives, the CLI (`optimize.py`), and the deployment helpers
  (`deploy.py`, `collect.py`).
- `utils/` — settings (`config.py`), preprocessing, analysis metrics, results,
  logging, visualisation.
- `scripts/` — independent argparse CLIs for experiments, analysis, figures and
  housekeeping.
- `tests/unit/` and `tests/integration/` — pytest suite.

## Workspace data layout (all gitignored)

All paths and settings come from `utils/config.py` (environment variable, then
`config.env`, then default); do not hard-code `results/` or `logs/` paths.

- `results/` — working outputs (`BIORESERVOIR_RESULTS_DIR`): `optuna/optuna-<study>.db`,
  `runs/<run>/fold-<k>.npz`, `metrics/`, `figures/<topic>/`.
- `logs/` — log files only (`BIORESERVOIR_LOGS_DIR`), one timestamped file per model
  run (`<command>-<study>-<UTC time>.log`); all modules of that run log to it.
  Analysis CLIs log to stdout unless given a log file.
- `archive/<version>/` — **frozen** results behind a manuscript version, read-only:
  `optuna/`, `runs/`, `metrics/`, `MANIFEST.csv` (what each file is and which
  manuscript item it supports) and `SHA256SUMS` (`sha256sum -c SHA256SUMS` inside the
  version folder). Its location is `BIORESERVOIR_ARCHIVE_DIR` (default `archive/`); the
  maintainers keep it in the private repository. Tools only read it
  (`--archive v0.1`, or an explicit path such as `--glob` / `--runs_dir`). Promote
  results from `results/` to a new `archive/<version>/` only when they are frozen
  for a manuscript version.
- `data/` — raw datasets and the preprocessed caches (`BIORESERVOIR_DATA_DIR`); build
  them with `python -m utils.preprocessing`.
- `config.env` — local settings: paths, Kaggle credentials, deployment hosts.
- Planning notes and the original `v0.1` files live in the private repository.

## Naming conventions

- **Studies** (SI Table S5): phase prefix `P1`–`P4`, `ESN` or `B1`–`B3`, then a
  lowercase descriptor, e.g. `P1-scaling`, `P2-distance`, `P4-readout-balanced`,
  `B1-balanced`. Database file: `optuna-<study>.db`, with the same study name inside.
- **Runs**: `<study>-best[-<variant>]` for re-runs of a study's best trial (the
  `rerun-best` default), e.g. `P4-readout-binary-best-full`; script-driven runs use a
  descriptive prefix (`ablation-<condition>`, `heterogeneity-cv<cv>-seed<seed>`).

## Tests

- Unit-test as much as possible: fast, no dataset, no network. Integration tests use
  the real dataset (`data/`) and are skipped without it.
- The archive tests (`tests/integration/test_archive.py`) check the frozen `v0.1`
  archive (`BIORESERVOIR_ARCHIVE_DIR`); they are skipped where it is absent, including CI.
- CI combines the coverage of the unit and integration tests in one report; at least 80%
  of the lines a PR changes must be covered.
- Every bug fix comes with a test that fails before the fix.

## Build, test, and lint commands

```bash
export UV_PROJECT_ENVIRONMENT=venv   # the project venv is venv/, not .venv/
uv sync --extra dev                  # exact environment from uv.lock (editable install)
uv lock                              # after changing dependencies in pyproject.toml

pytest tests/unit/                   # fast unit tests (CI on every PR)
pytest tests/integration/            # slower; need data/ (and the archive for its tests)
pytest --cov --cov-report=term-missing

ruff check .                         # lint
ruff format .                        # format
```

## Conventions

- **Docstrings:** NumPy style (matches numpy/reservoirpy). Napoleon-compatible.
- **Formatting/linting:** `ruff` (line length 100, target py310).
- **Commits:** Conventional Commits with a component scope, e.g.
  `feat(provenance): ...`. Versioning follows SemVer.
- **Python support:** 3.10–3.12. Runtime dependencies are pinned in `pyproject.toml`;
  every dependency is locked in `uv.lock`.
- **Settings:** paths, secrets and host details only through `utils.config`
  (environment, then `config.env`, then default).
- Library code must not call `logging.basicConfig`; use
  `utils.logger.setup_logging` from CLIs/scripts only.
- **CI** (GitHub Actions, every PR): ruff, unit tests on Python 3.10–3.12,
  integration tests, coverage.

## Hard rules (do not violate)

- **Never modify `archive/`.** It holds the frozen artefacts of a manuscript
  version (the `v0.1` code is archived on Zenodo, DOI 10.5281/zenodo.23436888).
  Write new results to `results/` and logs to `logs/`. Keep the scientific
  dependencies (numpy, numba, scipy, scikit-learn, reservoirpy, optuna, pandas,
  matplotlib) at their pinned versions before `v1.0`; other dependencies may be
  upgraded for security fixes, with a CHANGELOG entry.
- **No infrastructure details in committed files.** No hostnames, IPs, user names,
  paths on remote hosts, proxies, systemd units or credentials. The deployment helpers
  (`optimization/deploy.py`, `optimization/collect.py`) read all of these from the
  gitignored `config.env` and refer to hosts by label (`host1`, ...) only;
  `scripts/deploy*.sh` stay gitignored.
- Secrets and host details come from the environment or `config.env` (see
  `config.env.example`), read through `utils.config`.

## Domain quick-reference (science → code)

- Each node models two genes: **luxI** and **aiiA**, producing intracellular
  AHL (**Hi**) that diffuses externally (**He**).
- ECG input drives the reservoir through **He**.
- The extracted reservoir **state is the luxI concentration** per node.
- Coupling weights are distance-dependent (Gaussian/exponential kernels) built
  in `bioreservoir/topology.py`.
- `BioReservoir` subclasses the ReservoirPy `Node`; respect that contract when
  extending (e.g. `run(X) -> states`).
