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

### Added

- [engineering] `--archive VERSION` reads frozen studies from `archive/<version>/`.
- [engineering] `optimization/deploy.py` and `optimization/collect.py` are now part of the
  repository. Host addresses, user, remote workspace, pip proxy and service come from
  `config.env`; hosts are referred to by label, and `--inventory` records each host's
  CPU, memory, OS and Python in `results/metrics/hosts.json`.
- [engineering] `uv.lock` locks every dependency for Python 3.9-3.12 (`uv sync`);
  `requires-python` is now `>=3.9,<3.13`, matching the supported versions.
- [engineering] Golden-master test (N = 20, synthetic beats) and a statistical sentinel
  (N = 200, 600 real beats, 2 folds) pin the `v0.1` model; both references were generated
  with the `v0.1` code (`tests/golden_master.py`).

### Fixed

- [engineering] Trial metadata no longer records the host name, IP address, working
  directory, or script, output and interpreter paths, and the platform string keeps only
  the OS, numeric kernel release and architecture. Trials now record the full git SHA, a
  dirty flag, the OS name and version, and an optional machine label from `config.env`.

## [0.1] (manuscript submission)

- Code archived on Zenodo: https://doi.org/10.5281/zenodo.23436888
