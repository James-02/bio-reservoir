# Copilot instructions

This repository's agent guidance lives in [`AGENTS.md`](../AGENTS.md). Read it
first — it covers the project overview, layout, commands, conventions, and hard
rules.

Key reminders (see `AGENTS.md` for detail):

- Anchor dependencies are **reservoirpy** and **optuna**; follow their patterns.
- **Never modify `archive/`** (frozen manuscript artefacts) or change pinned
  runtime dependency versions; new results go to `results/`, logs to `logs/`.
- **No infrastructure secrets** (hostnames, IPs, credentials, cluster names) in
  tracked files.
- NumPy-style docstrings, `ruff` for lint/format, Conventional Commits, SemVer.
- Run `pytest tests/unit/` for fast feedback; `ruff check .` before committing.
