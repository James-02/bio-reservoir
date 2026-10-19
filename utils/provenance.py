"""Run manifests and the append-only run registry.

Every citable run writes ``manifest.json`` into its run folder and appends one line to
``registry.jsonl`` at the repository root.  Runs from a working tree with uncommitted
changes are refused unless explicitly allowed; they are then marked non-citable.  Status
changes (for example ``superseded``) are appended as new lines; nothing is deleted::

    python -m utils.provenance list [--exp EXP_ID] [--status ok] [--used]
    python -m utils.provenance show RUN_ID
    python -m utils.provenance diff RUN_A RUN_B
    python -m utils.provenance mark RUN_ID superseded [--by RUN_ID] [--note TEXT]
    python -m utils.provenance mark RUN_ID ok --uses "Table 2, balanced"

Manifests follow the environment metadata policy: no host names, IP addresses, user
names or absolute paths; files are recorded relative to the repository root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from utils import config

REGISTRY_FILE = config.PROJECT_ROOT / "registry.jsonl"
LOCK_FILE = config.PROJECT_ROOT / "uv.lock"
STATUSES = ("ok", "failed", "superseded", "exploratory")
ENVIRONMENT_KEYS = (
    "python_version", "platform", "os_name", "os_version", "machine_label", "cpu_model",
    "cpu_count_physical", "cpu_count_logical", "ram_total_gb", "numpy_version",
    "reservoirpy_version", "optuna_version", "numba_version", "sklearn_version", "scipy_version",
)


def _utc(moment: datetime | None = None) -> str:
    return (moment or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def spec_hash(spec: dict) -> str:
    """SHA-256 of the spec as canonical JSON (sorted keys, no white space)."""
    text = json.dumps(spec, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def git_info() -> dict[str, Any]:
    """Return the commit, ``git describe`` and dirty flag of the working tree.

    Falls back to ``BIORESERVOIR_GIT_COMMIT`` / ``BIORESERVOIR_GIT_DIRTY`` (set by the
    deployment helper) when the code runs outside a git checkout.
    """
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=config.PROJECT_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    try:
        return {"commit": git("rev-parse", "HEAD"),
                "describe": git("describe", "--tags", "--always", "--dirty"),
                "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.CalledProcessError):
        dirty = (config.get("BIORESERVOIR_GIT_DIRTY") or "").lower()
        return {"commit": config.get("BIORESERVOIR_GIT_COMMIT", "N/A"), "describe": "N/A",
                "dirty": {"true": True, "false": False}.get(dirty, True)}


def require_clean(git: dict[str, Any], allow_dirty: bool = False) -> bool:
    """Refuse a dirty (or unknown) tree unless *allow_dirty*; return whether citable."""
    clean = git["dirty"] is False and git["commit"] != "N/A"
    if not clean and not allow_dirty:
        raise SystemExit("Uncommitted changes (or unknown commit): commit first, or pass "
                         "--allow-dirty to mark the run as non-citable.")
    return clean


def make_run_id(exp_id: str, spec: dict, commit: str, moment: datetime | None = None) -> str:
    """``<UTC>-<exp_id>-<short SHA>-<spec hash8>``."""
    stamp = (moment or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{exp_id}-{commit[:7]}-{spec_hash(spec)[:8]}"


def _relative(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(config.PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def file_record(path: Path) -> dict[str, Any]:
    size = Path(path).stat().st_size
    return {"path": _relative(path), "sha256": sha256_file(path), "bytes": size}


def build_manifest(
    exp_id: str,
    spec: dict,
    *,
    seeds: Sequence[int],
    started: datetime,
    ended: datetime,
    data_files: Iterable[Path] = (),
    beat_ids: np.ndarray | None = None,
    outputs: Iterable[Path] = (),
    peak_ram_gb: float | None = None,
    allow_dirty: bool = False,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Collect the provenance of one run (see the module docstring)."""
    from optimization.environment import get_environment

    git = git_info()
    citable = require_clean(git, allow_dirty)
    env = get_environment()
    return {
        "run_id": run_id or make_run_id(exp_id, spec, git["commit"], started),
        "exp_id": exp_id,
        "citable": citable,
        "git": git,
        "spec": spec,
        "spec_hash": spec_hash(spec),
        "lock_sha256": sha256_file(LOCK_FILE) if LOCK_FILE.is_file() else None,
        "environment": {key: env.get(key) for key in ENVIRONMENT_KEYS},
        "data": [file_record(p) for p in data_files],
        "beat_ids_sha256": (hashlib.sha256(np.asarray(beat_ids, dtype=np.int64).tobytes())
                            .hexdigest() if beat_ids is not None else None),
        "seeds": [int(s) for s in seeds],
        "timing": {"start_utc": _utc(started), "end_utc": _utc(ended),
                   "wall_s": round((ended - started).total_seconds(), 3)},
        "peak_ram_gb": peak_ram_gb,
        "outputs": [file_record(p) for p in outputs],
    }


def write_manifest(run_dir: Path, manifest: dict[str, Any]) -> Path:
    path = Path(run_dir) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True, default=str) + "\n")
    return path


def _append(entry: dict[str, Any], registry: Path) -> None:
    with open(registry, "a") as f:
        f.write(json.dumps(entry, sort_keys=True, default=str) + "\n")


def register(manifest: dict[str, Any], run_dir: Path, *, status: str = "ok",
             metrics: dict[str, float] | None = None, evaluation: str | None = None,
             registry: Path = REGISTRY_FILE) -> dict[str, Any]:
    """Append a run to the registry.

    *evaluation* names the held-out split a run was evaluated on (``None`` for a
    cross-validated run).
    """
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    entry = {
        "run_id": manifest["run_id"], "exp_id": manifest["exp_id"],
        "git_commit": manifest["git"]["commit"], "spec_hash": manifest["spec_hash"],
        "citable": manifest["citable"], "status": status, "metrics": metrics or {},
        "path": _relative(run_dir), "registered_utc": _utc(),
    }
    if evaluation:
        entry["evaluation"] = evaluation
    _append(entry, registry)
    return entry


def mark(run_id: str, status: str, *, by: str | None = None, note: str = "",
         uses: str | None = None, registry: Path = REGISTRY_FILE) -> None:
    """Append a status change (runs are flagged, never deleted).

    *uses* names a paper item the run backs, e.g. "Table 2, balanced"; items
    accumulate over successive marks.
    """
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    runs = load_registry(registry)
    if run_id not in runs:
        raise SystemExit(f"unknown run {run_id!r}")
    entry = {"run_id": run_id, "status": status, "superseded_by": by, "note": note,
             "registered_utc": _utc()}
    if uses:
        entry["uses"] = [*runs[run_id].get("uses", []), uses]
    _append(entry, registry)


def load_registry(registry: Path = REGISTRY_FILE) -> dict[str, dict[str, Any]]:
    """Return one entry per run: the registration with later status changes applied."""
    runs: dict[str, dict[str, Any]] = {}
    if not Path(registry).is_file():
        return runs
    for line in Path(registry).read_text().splitlines():
        if line.strip():
            entry = json.loads(line)
            runs.setdefault(entry["run_id"], {}).update(
                {k: v for k, v in entry.items() if v is not None})
    return runs


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            out.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return {prefix: value}


def diff_runs(a: dict[str, Any], b: dict[str, Any],
              manifests: tuple[dict, dict] | None = None) -> dict[str, Any]:
    """Metric deltas (b - a) and the manifest fields that differ."""
    metrics = {key: b["metrics"][key] - a["metrics"][key]
               for key in sorted(set(a.get("metrics", {})) & set(b.get("metrics", {})))}
    changed = {}
    if manifests:
        fa, fb = (_flatten({k: v for k, v in m.items() if k not in ("run_id", "timing")})
                  for m in manifests)
        changed = {key: [fa.get(key), fb.get(key)] for key in sorted(set(fa) | set(fb))
                   if fa.get(key) != fb.get(key)}
    return {"metrics": metrics, "manifest": changed}


def _manifest_for(entry: dict[str, Any]) -> dict | None:
    path = config.PROJECT_ROOT / entry.get("path", "") / "manifest.json"
    return json.loads(path.read_text()) if path.is_file() else None


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m utils.provenance", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_list = sub.add_parser("list", help="List runs.")
    p_list.add_argument("--exp")
    p_list.add_argument("--status", choices=STATUSES)
    p_list.add_argument("--used", action="store_true", help="Only runs that back paper items.")
    sub.add_parser("show", help="Show one run.").add_argument("run_id")
    p_diff = sub.add_parser("diff", help="Compare two runs.")
    p_diff.add_argument("run_a")
    p_diff.add_argument("run_b")
    p_mark = sub.add_parser("mark", help="Change the status of a run.")
    p_mark.add_argument("run_id")
    p_mark.add_argument("status", choices=STATUSES)
    p_mark.add_argument("--by")
    p_mark.add_argument("--note", default="")
    p_mark.add_argument("--uses", help="Paper item (table, figure) this run backs.")
    args = parser.parse_args(argv)

    runs = load_registry()
    if args.command == "list":
        for run_id, entry in runs.items():
            if (args.exp and entry.get("exp_id") != args.exp) or (
                    args.status and entry.get("status") != args.status) or (
                    args.used and not entry.get("uses")):
                continue
            flag = "" if entry.get("citable") else "  (not citable)"
            split = f"  split: {entry['evaluation']}" if entry.get("evaluation") else ""
            uses = f"  uses: {'; '.join(entry['uses'])}" if entry.get("uses") else ""
            print(f"{run_id}  {entry.get('status')}  {json.dumps(entry.get('metrics', {}))}"
                  f"{split}{flag}{uses}")
    elif args.command == "mark":
        mark(args.run_id, args.status, by=args.by, note=args.note, uses=args.uses)
    else:
        wanted = [args.run_id] if args.command == "show" else [args.run_a, args.run_b]
        missing = [r for r in wanted if r not in runs]
        if missing:
            raise SystemExit(f"unknown run(s): {', '.join(missing)}")
        if args.command == "show":
            entry = runs[args.run_id]
            print(json.dumps({"registry": entry, "manifest": _manifest_for(entry)}, indent=1))
        else:
            a, b = runs[args.run_a], runs[args.run_b]
            ma, mb = _manifest_for(a), _manifest_for(b)
            print(json.dumps(diff_runs(a, b, (ma, mb) if ma and mb else None), indent=1))


if __name__ == "__main__":
    main()
