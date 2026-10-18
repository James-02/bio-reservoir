# Copilot instructions

This repository's agent guidance lives in [`AGENTS.md`](../AGENTS.md). Read it
first — it covers the project overview, layout, commands, conventions, and hard
rules.

Key reminders (see `AGENTS.md` for detail):

- Keep changes simple and standard; explain non-trivial designs to the maintainer and
  agree them before implementing.
- Anchor dependencies are **reservoirpy** and **optuna**; follow their patterns.
- **Never modify `archive/`** (frozen results of released versions), never tune on the
  held-out draws, or change the pinned scientific dependency versions; new results go to
  `results/`, logs to `logs/`.
- Write for the people who use and extend the framework: describe what the code does
  and why, and cite the paper, releases, public issues or PRs, or the literature.
- Name options after what they do; behaviour kept only to reproduce an earlier release
  is named after that release (`v0.1`) and is removed in the next major version.
- **No infrastructure details** (hostnames, IPs, user names, remote paths, proxies,
  systemd units, credentials) in tracked files; they belong in `config.env`.
- Every bug fix comes with a test that fails before the fix; record changes in
  `CHANGELOG.md` (`[science]` or `[engineering]`).
- NumPy-style docstrings, `ruff` for lint/format, Conventional Commits, SemVer;
  dependencies via `uv` (`uv.lock`).
- Run `pytest tests/unit/` for fast feedback; `ruff check .` before committing.
