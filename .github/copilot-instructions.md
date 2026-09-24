# Copilot instructions

This repository's agent guidance lives in [`AGENTS.md`](../AGENTS.md). Read it
first — it covers the project overview, layout, commands, conventions, and hard
rules.

Key reminders (see `AGENTS.md` for detail):

- Anchor dependencies are **reservoirpy** and **optuna**; follow their patterns.
- **Never edit committed results** under `results/` or change pinned runtime
  dependency versions — the paper snapshot must stay immutable until the Zenodo
  DOI is minted.
- **No infrastructure secrets** (hostnames, IPs, credentials, cluster names) in
  tracked files.
- NumPy-style docstrings, `ruff` for lint/format, Conventional Commits, SemVer.
- Run `pytest tests/unit/` for fast feedback; `ruff check .` before committing.
