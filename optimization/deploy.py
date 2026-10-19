#!/usr/bin/env python3
"""Deploy optimisation studies and experiment commands to remote Linux hosts.

Host requirements: Linux with glibc 2.17 or newer (2.28 or newer to draw figures), the
standard ``ssh`` and ``rsync`` tools and key-based SSH login.  The host's own Python is
not used: setting up a host (``--setup``, and before every launch) copies the local
``uv`` to the workspace, and uv installs Python 3.12 and the packages locked in
``uv.lock`` there.

Nothing host-specific is stored in the repository.  Host addresses, the SSH user, the
remote workspace (shared, or one per host), an optional proxy and an optional systemd
unit to pause during runs are read from the gitignored ``config.env`` (see
``config.env.example``)::

    BIORESERVOIR_DEPLOY_HOSTS=host1=<address>,host2=<address>
    BIORESERVOIR_DEPLOY_USER=<ssh user>
    BIORESERVOIR_DEPLOY_WORKSPACE=<absolute remote path>
    BIORESERVOIR_DEPLOY_WORKSPACES=host2=<absolute remote path>

Hosts are referred to by label only (``host1``, ...).  Every remote run records its
label as the machine label, with the CPU model, core counts, RAM, OS and the local git
SHA (see ``optimization/environment.py``).  ``--inventory`` records the hardware of
each host in ``results/metrics/hosts.json`` and checks the requirements.

Usage
-----
  python -m optimization.deploy --inventory
  python -m optimization.deploy --setup [--host host1]
  python -m optimization.deploy --host host1 --study P2-distance [--processes 32] [-- --trials 50]
  python -m optimization.deploy --host host2 --type baseline --study B1-balanced
  python -m optimization.deploy --host host2 --push results/optuna/optuna-P2-distance.db \\
      --run "python -m optimization.optimize rerun-best --study P2-distance --folds 5"
  python -m optimization.deploy --status | --progress | --logs [N] | --watch [SEC]
  python -m optimization.deploy --wait                 # wait for sessions, then collect
  python -m optimization.deploy --reset-running --host host1 --study P2-distance
  python -m optimization.deploy --cleanup [--host host1]
  python -m optimization.deploy --start-service [--host host1]

Add ``--dry-run`` to print the commands without running them.  Collect results with
``python -m optimization.collect``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import textwrap
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import PurePosixPath

from utils import config
from utils.logger import setup_logging

log = logging.getLogger("deploy")
q = shlex.quote

SYNC_SOURCES = ("bioreservoir/", "optimization/", "training/", "utils/", "scripts/",
                "experiments/", "pyproject.toml", "uv.lock", "README.md")
SSH_OPTIONS = ("-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
               "-o", "ConnectTimeout=60", "-o", "ServerAliveInterval=30")
RSYNC = ("rsync", "-az", "--exclude=__pycache__", "--exclude=*.pyc", "--exclude=*.egg-info")
# pillow and contourpy have no wheels for glibc older than 2.28; matplotlib and seaborn need them.
PLOTTING_PACKAGES = ("matplotlib", "seaborn", "pillow", "contourpy")
GLIBC_MIN, GLIBC_PLOTTING = (2, 17), (2, 28)
NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    hosts: dict[str, str]
    user: str | None
    workspace: str
    python: str
    pip_proxy: str | None
    service: str | None
    workspaces: dict[str, str] = field(default_factory=dict)

    def workspace_of(self, host: str) -> str:
        """The host's own workspace if it has one, else the shared one."""
        return self.workspaces.get(host, self.workspace)

    def venv_of(self, host: str) -> str:
        return f"{self.workspace_of(host)}/venv"


def parse_hosts(value: str | None, setting: str = "BIORESERVOIR_DEPLOY_HOSTS") -> dict[str, str]:
    """Parse ``label=value`` pairs separated by commas or white space."""
    hosts: dict[str, str] = {}
    for item in (value or "").replace(",", " ").split():
        label, sep, address = item.partition("=")
        if not (sep and NAME_PATTERN.match(label) and address):
            raise SystemExit(f"{setting}: expected label=value pairs, got {item!r}")
        hosts[label] = address
    return hosts


def _workspace(value: str, setting: str) -> str:
    workspace = value.rstrip("/")
    path = PurePosixPath(workspace)
    if not path.is_absolute() or len(path.parts) < 3 or ".." in path.parts:
        raise SystemExit(f"{setting} must hold absolute paths at least two levels deep "
                         "(they are deleted by --cleanup).")
    return workspace


def load_settings() -> Settings:
    """Read the deployment settings (environment, then ``config.env``)."""
    hosts = parse_hosts(config.get("BIORESERVOIR_DEPLOY_HOSTS"))
    workspace = config.get("BIORESERVOIR_DEPLOY_WORKSPACE")
    if not hosts or not workspace:
        raise SystemExit("Set BIORESERVOIR_DEPLOY_HOSTS and BIORESERVOIR_DEPLOY_WORKSPACE in "
                         "config.env (see config.env.example).")
    workspaces = parse_hosts(config.get("BIORESERVOIR_DEPLOY_WORKSPACES"),
                             "BIORESERVOIR_DEPLOY_WORKSPACES")
    for label in set(workspaces) - set(hosts):
        raise SystemExit(f"BIORESERVOIR_DEPLOY_WORKSPACES: {label!r} is not a configured host")
    return Settings(
        hosts=hosts,
        user=config.get("BIORESERVOIR_DEPLOY_USER"),
        workspace=_workspace(workspace, "BIORESERVOIR_DEPLOY_WORKSPACE"),
        python=config.get("BIORESERVOIR_DEPLOY_PYTHON", "python3"),
        pip_proxy=config.get("BIORESERVOIR_DEPLOY_PIP_PROXY"),
        service=config.get("BIORESERVOIR_DEPLOY_SERVICE"),
        workspaces={label: _workspace(path, "BIORESERVOIR_DEPLOY_WORKSPACES")
                    for label, path in workspaces.items()},
    )


def git_state() -> tuple[str, str]:
    """Return the local git SHA and dirty flag (``"N/A"`` when git is unavailable)."""
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=config.PROJECT_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    try:
        dirty = git("status", "--porcelain", "--untracked-files=no")
        return git("rev-parse", "HEAD"), "true" if dirty else "false"
    except (OSError, subprocess.CalledProcessError):
        return "N/A", "N/A"


# ---------------------------------------------------------------------------
# Remote operations
# ---------------------------------------------------------------------------

class Deployer:
    """Run SSH and rsync commands on labelled hosts.  Dry runs only record commands."""

    def __init__(self, settings: Settings, *, dry_run: bool = False) -> None:
        self.settings = settings
        self.dry_run = dry_run
        self.commands: list[list[str]] = []

    def target(self, host: str) -> str:
        if host not in self.settings.hosts:
            raise SystemExit(f"Unknown host {host!r}; configured: {', '.join(self.settings.hosts)}")
        address = self.settings.hosts[host]
        return f"{self.settings.user}@{address}" if self.settings.user else address

    def remote(self, host: str, path: str) -> str:
        return f"{self.target(host)}:{path}"

    def ssh(self, host: str, script: str, *, capture: bool = False, check: bool = True) -> str:
        # Hosts may set these for their system Python; they break the workspace's Python.
        script = f"unset PYTHONPATH PYTHONHOME\n{script}"
        cmd = ["ssh", *SSH_OPTIONS, self.target(host), script]
        self.commands.append(cmd)
        if self.dry_run:
            log.info("[%s] dry run, remote script:\n%s", host,
                     textwrap.indent(textwrap.dedent(script).strip(), "    "))
            return ""
        result = subprocess.run(cmd, capture_output=capture, text=True, check=check)
        return (result.stdout or "").rstrip() if capture else ""

    def rsync(self, host: str, *args: str, check: bool = True) -> bool:
        cmd = [*RSYNC, "-e", " ".join(("ssh",) + SSH_OPTIONS), *args]
        self.commands.append(cmd)
        if self.dry_run:
            log.info("[%s] dry run: %s", host, " ".join(cmd))
            return True
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return True
        message = result.stderr.strip() or f"exit {result.returncode}"
        if check:
            raise SystemExit(f"[{host}] rsync failed: {message}")
        log.warning("[%s] rsync: %s", host, message)
        return False

    # -- deployment steps ----------------------------------------------------

    def sync(self, host: str) -> None:
        """Copy the code (replacing older copies) and any missing data files."""
        ws, root = self.settings.workspace_of(host), config.PROJECT_ROOT
        self.ssh(host, f"mkdir -p {q(ws)}/logs {q(ws)}/results")
        for source in SYNC_SOURCES:
            self.rsync(host, "--delete", f"{root}/{source}", self.remote(host, f"{ws}/{source}"))
        self.rsync(host, "--ignore-existing", f"{root}/data/", self.remote(host, f"{ws}/data/"))
        log.info("[%s] code and data synced", host)

    def push(self, host: str, paths: Sequence[str]) -> None:
        """Copy local ``results/`` or ``archive/`` paths to the same place on *host*."""
        ws = self.settings.workspace_of(host)
        for path in paths:
            rel = PurePosixPath(path)
            if rel.is_absolute() or ".." in rel.parts or rel.parts[0] not in ("results", "archive"):
                raise SystemExit(f"--push takes paths under results/ or archive/, got {path!r}")
            local = config.PROJECT_ROOT / rel
            if not local.exists():
                raise SystemExit(f"--push: {path} does not exist")
            slash = "/" if local.is_dir() else ""
            self.ssh(host, f"mkdir -p {q(f'{ws}/{rel.parent}')}")
            self.rsync(host, f"{local}{slash}", self.remote(host, f"{ws}/{rel}{slash}"))

    def setup_venv(self, host: str) -> None:
        """Copy the local uv to *host*; it installs Python and the locked environment there."""
        s, ws, venv = self.settings, self.settings.workspace_of(host), self.settings.venv_of(host)
        uv = shutil.which("uv")
        if not uv:
            raise SystemExit("uv is not installed here; it is copied to the hosts to install "
                             "the environment (https://docs.astral.sh/uv/).")
        self.ssh(host, f"mkdir -p {q(ws)}/bin")
        self.rsync(host, os.path.realpath(uv), self.remote(host, f"{ws}/bin/uv"))
        proxy = (f"export HTTPS_PROXY={q(s.pip_proxy)} HTTP_PROXY={q(s.pip_proxy)}\n"
                 if s.pip_proxy else "")
        low = ".".join(map(str, GLIBC_PLOTTING))
        skip = " ".join(f"--no-install-package {name}" for name in PLOTTING_PACKAGES)
        self.ssh(host, f"""
set -e
{proxy}export UV_CACHE_DIR={q(ws)}/uv-cache UV_PYTHON_INSTALL_DIR={q(ws)}/python \\
    UV_PYTHON_INSTALL_BIN=0 UV_PROJECT_ENVIRONMENT={q(venv)}
cd {q(ws)}
GLIBC=$(getconf GNU_LIBC_VERSION | cut -d ' ' -f 2)
SKIP=
if [ "$(printf '%s\\n' {low} "$GLIBC" | sort -V | head -n 1)" != {low} ]; then
    echo "glibc $GLIBC: no plotting packages, so runs on this host cannot draw figures" >&2
    SKIP="{skip}"
fi
bin/uv sync --frozen --quiet $SKIP
{q(venv)}/bin/python -c "import numba, optuna, reservoirpy, sklearn"
""")
        log.info("[%s] environment ready", host)

    def stop_service(self, host: str) -> None:
        if self.settings.service:
            self.ssh(host, f"systemctl stop {q(self.settings.service)} 2>/dev/null || true")
            log.info("[%s] configured service paused", host)

    def start_service(self, host: str) -> None:
        if not self.settings.service:
            log.info("[%s] no BIORESERVOIR_DEPLOY_SERVICE configured", host)
            return
        unit = q(self.settings.service)
        self.ssh(host, f"systemctl start {unit} && echo 'service started' "
                       f"|| echo 'could not start the service'", check=False)

    def launch(self, host: str, jobs: Sequence[tuple[str, str]]) -> None:
        """Run *jobs* (log name, shell command) one after another in a detached session.

        Each job logs to ``<workspace>/logs/run-<name>.log``.  The session exports the
        host label and the local git SHA so that every trial records them, and runs in its
        own process group, which ``--cleanup`` stops.
        """
        ws = self.settings.workspace_of(host)
        sha, dirty = git_state()
        if dirty == "true":
            log.warning("Uncommitted local changes: runs will record git_dirty=true.")
        lines = [
            "set -o pipefail",
            f"cd {q(ws)}",
            f"source {q(self.settings.venv_of(host))}/bin/activate",
            f"export BIORESERVOIR_MACHINE_LABEL={q(host)} BIORESERVOIR_GIT_COMMIT={q(sha)} "
            f"BIORESERVOIR_GIT_DIRTY={q(dirty)}",
            "NPROC=$(nproc)",
            "FAILED=0",
        ]
        for name, command in jobs:
            run_log = q(f"{ws}/logs/run-{name}.log")
            lines += [
                f'echo "=== {name} started $(date -u +%FT%TZ) ===" | tee {run_log}',
                f"{command} 2>&1 | tee -a {run_log}",
                "RC=$?",
                f'echo "=== {name} finished (exit $RC) $(date -u +%FT%TZ) ===" | tee -a {run_log}',
                '[ "$RC" -eq 0 ] || FAILED=$((FAILED + 1))',
            ]
        lines.append('exit "$FAILED"')
        pid = q(f"{ws}/logs/session.pid")
        session = q(f"{ws}/logs/session.sh")
        script = "\n".join([
            "set -e",
            f"mkdir -p {q(ws)}/logs",
            f'if [ -f {pid} ] && kill -0 "$(cat {pid})" 2>/dev/null; then',
            '    echo "a session is already running; wait for it or use --cleanup" >&2',
            "    exit 3",
            "fi",
            f"cat > {session} << 'BIORESERVOIR_SESSION'",
            *lines,
            "BIORESERVOIR_SESSION",
            f"setsid nohup bash {session} > {q(f'{ws}/logs/session.log')} 2>&1 < /dev/null &",
            f"echo $! > {pid}",
            f'echo "session started (PID $(cat {pid}))"',
        ])
        log.info("[%s] launching: %s", host, ", ".join(name for name, _ in jobs))
        self.ssh(host, script)

    def session_alive(self, host: str) -> bool:
        pid = q(f"{self.settings.workspace_of(host)}/logs/session.pid")
        out = self.ssh(host, f'[ -f {pid} ] && kill -0 "$(cat {pid})" 2>/dev/null '
                             f"&& echo alive || echo idle", capture=True, check=False)
        return out.strip() == "alive"

    def cleanup(self, host: str) -> None:
        """Stop the session and every process it started, then delete the remote workspace."""
        ws = self.settings.workspace_of(host)
        pid = q(f"{ws}/logs/session.pid")
        self.ssh(host, f"""
if [ -f {pid} ]; then
    kill -- -"$(cat {pid})" 2>/dev/null || kill "$(cat {pid})" 2>/dev/null || true
    sleep 1
fi
rm -rf -- {q(ws)}
""")
        log.info("[%s] workspace removed", host)


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

def study_jobs(studies: Sequence[str], processes: int | None, extra: Sequence[str],
               study_type: str = "training") -> list[tuple[str, str]]:
    """Build ``optimize research`` commands; without *processes* all remote CPUs are used."""
    from optimization.studies import STUDY_REGISTRY

    jobs = []
    for study in studies:
        if study_type == "training" and study not in STUDY_REGISTRY:
            raise SystemExit(f"Unknown study {study!r}; see: python -m optimization.optimize "
                             "list-studies")
        if not NAME_PATTERN.match(study):
            raise SystemExit(f"Invalid study name {study!r}")
        flag = "--study" if study_type == "training" else "--study_name"
        n_proc = str(processes) if processes else "$NPROC"
        command = (f"python -m optimization.optimize research --type {study_type} "
                   f'{flag} {study} --processes "{n_proc}"')
        command += "".join(f" {q(arg)}" for arg in extra)
        jobs.append((study, command))
    return jobs


def command_jobs(commands: Sequence[str], label: str | None) -> list[tuple[str, str]]:
    """Name free-form ``--run`` commands ``<label>`` or ``<label>-<k>``."""
    label = label or "run"
    if not NAME_PATTERN.match(label):
        raise SystemExit(f"Invalid --label {label!r}")
    if len(commands) == 1:
        return [(label, commands[0])]
    return [(f"{label}-{k}", command) for k, command in enumerate(commands, 1)]


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------

_PROGRESS_SCRIPT = """
import glob, os, sys
import optuna
from optuna.trial import TrialState
optuna.logging.set_verbosity(optuna.logging.ERROR)
wanted = set(sys.argv[2:])
for db in sorted(glob.glob(os.path.join(sys.argv[1], "optuna-*.db"))):
    name = os.path.basename(db)[len("optuna-"):-len(".db")]
    if wanted and name not in wanted:
        continue
    try:
        study = optuna.load_study(study_name=name, storage="sqlite:///" + db + "?timeout=5")
        states = [t.state for t in study.trials]
        try:
            best = "%.4f" % study.best_value
        except Exception:
            best = "n/a"
        counts = [states.count(s) for s in (TrialState.COMPLETE, TrialState.RUNNING,
                                            TrialState.FAIL, TrialState.PRUNED)]
        print("|".join([name] + [str(c) for c in counts] + [best]))
    except Exception as exc:
        print(name + "|?|?|?|?|error: " + str(exc).replace("|", "/"))
"""

_RESET_SCRIPT = """
import sqlite3, sys
for name in sys.argv[2:]:
    path = "%s/optuna-%s.db" % (sys.argv[1], name)
    try:
        with sqlite3.connect("file:%s?mode=rw" % path, uri=True, timeout=5) as con:
            n = con.execute("UPDATE trials SET state = 'FAIL' WHERE state = 'RUNNING'").rowcount
        print("%s: %d RUNNING trial(s) marked FAIL" % (name, n))
    except sqlite3.Error as exc:
        print("%s: %s" % (name, exc))
"""

_INVENTORY_SCRIPT = """
import json, os, platform, re, shutil, subprocess
info, release = {}, {}
try:
    with open("/etc/os-release") as f:
        for line in f:
            key, sep, value = line.strip().partition("=")
            if sep:
                release[key] = value.strip('"')
except OSError:
    pass
info["os_name"] = release.get("NAME", platform.system())
info["os_version"] = release.get("VERSION_ID", "")
info["os_family"] = " ".join([release.get("ID", "")] + release.get("ID_LIKE", "").split())
match = re.match(r"[0-9]+(?:[.][0-9]+){0,2}", platform.release())
info["kernel"] = match.group(0) if match else ""
info["arch"] = platform.machine()
try:
    info["glibc"] = os.confstr("CS_GNU_LIBC_VERSION").split()[-1]
except (AttributeError, OSError, ValueError):
    info["glibc"] = ""
model, sockets, cores, logical, socket_id = "", set(), set(), 0, None
with open("/proc/cpuinfo") as f:
    for line in f:
        key, _, value = (part.strip() for part in line.partition(":"))
        if key == "processor":
            logical += 1
        elif key == "model name" and not model:
            model = value
        elif key == "physical id":
            socket_id = value
            sockets.add(value)
        elif key == "core id":
            cores.add((socket_id, value))
info["cpu_model"] = model
info["sockets"] = len(sockets) or 1
info["physical_cores"] = len(cores) or logical
info["logical_cpus"] = logical
with open("/proc/meminfo") as f:
    mem = dict(line.split(":", 1) for line in f if ":" in line)
info["ram_gb"] = round(int(mem["MemTotal"].split()[0]) / 1024 ** 2, 1)
info["configured_python"] = platform.python_version()
info["pythons"] = {}
for minor in range(9, 13):
    exe = shutil.which("python3.%d" % minor)
    if exe:
        out = subprocess.run([exe, "-c", "import platform; print(platform.python_version())"],
                             stdout=subprocess.PIPE, universal_newlines=True)
        info["pythons"]["3.%d" % minor] = out.stdout.strip()
print(json.dumps(info))
"""


def _python_heredoc(python: str, script: str, *args: str) -> str:
    argv = " ".join(q(a) for a in args)
    return f"{q(python)} - {argv} << 'BIORESERVOIR_PY'\n{script.strip()}\nBIORESERVOIR_PY"


def check_requirements(info: dict) -> list[str]:
    """Return the ways a host inventory falls short of the glibc the environment needs."""
    found = info.get("glibc") or "unknown"
    try:
        glibc = tuple(int(x) for x in found.split(".")[:2])
    except ValueError:
        glibc = (0, 0)
    if glibc < GLIBC_MIN:
        return [f"glibc {found}: uv's Python 3.12 needs glibc 2.17 or newer"]
    if glibc < GLIBC_PLOTTING:
        return [f"glibc {found}: the plotting packages need glibc 2.28 or newer, so runs "
                "on this host cannot draw figures"]
    return []


def for_hosts(hosts: Sequence[str], action: Callable[[str], object]) -> dict[str, object]:
    """Run *action* on every host in parallel; errors are logged per host."""
    def run(host: str) -> object:
        try:
            return action(host)
        except (subprocess.CalledProcessError, SystemExit, OSError) as exc:
            log.error("[%s] failed: %s", host, exc)
            return None
    with ThreadPoolExecutor(max_workers=max(1, len(hosts))) as pool:
        return dict(zip(hosts, pool.map(run, hosts), strict=True))


def inventory(deployer: Deployer, hosts: Sequence[str]) -> None:
    """Record CPU, memory, OS, glibc and Python versions per host label in hosts.json."""
    s = deployer.settings
    outputs = for_hosts(hosts, lambda h: deployer.ssh(
        h, _python_heredoc(s.python, _INVENTORY_SCRIPT), capture=True))
    if deployer.dry_run:
        return
    path = config.metrics_dir() / "hosts.json"
    record = json.loads(path.read_text()) if path.is_file() else {"hosts": {}}
    for host, out in outputs.items():
        if not out:
            continue
        info = json.loads(out.splitlines()[-1])
        info["problems"] = check_requirements(info)
        info["checked_utc"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        record["hosts"][host] = info
        print(f"  {host:<8} {info['cpu_model']}, {info['sockets']} socket(s), "
              f"{info['physical_cores']} cores / {info['logical_cpus']} threads, "
              f"{info['ram_gb']} GB RAM, {info['os_name']} {info['os_version']}, "
              f"glibc {info.get('glibc')}, Python {info['configured_python']}")
        for problem in info["problems"]:
            print(f"           requirement not met: {problem}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(f"  written to {path}")


def show_status(deployer: Deployer, hosts: Sequence[str]) -> None:
    def status(host: str) -> str:
        ws = deployer.settings.workspace_of(host)
        pid = q(f"{ws}/logs/session.pid")
        return deployer.ssh(host, f"""
if [ -f {pid} ] && kill -0 "$(cat {pid})" 2>/dev/null; then
    SECS=$(ps -o etimes= -p "$(cat {pid})" | tr -d ' ')
    printf 'running|%dh%02dm|' $((SECS / 3600)) $((SECS % 3600 / 60))
else
    printf 'idle|-|'
fi
tail -q -n 30 {q(ws)}/logs/run-*.log 2>/dev/null | grep -v -e '^$' -e '^==>' | tail -n 1
""", capture=True, check=False)

    outputs = for_hosts(hosts, status)
    print(f"\n  {'Host':<8} {'Session':<8} {'Up':>7}  Last log line")
    for host in hosts:
        parts = (outputs.get(host) or "unreachable|-|").split("|", 2)
        parts += [""] * (3 - len(parts))
        print(f"  {host:<8} {parts[0]:<8} {parts[1]:>7}  {parts[2][:80]}")


def show_progress(deployer: Deployer, hosts: Sequence[str], studies: Sequence[str]) -> None:
    s = deployer.settings
    outputs = for_hosts(hosts, lambda h: deployer.ssh(h, _python_heredoc(
        f"{s.venv_of(h)}/bin/python", _PROGRESS_SCRIPT, f"{s.workspace_of(h)}/results/optuna",
        *studies), capture=True, check=False))
    for host in hosts:
        print(f"\n  [{host}]\n  {'Study':<32} {'Done':>5} {'Run':>5} {'Fail':>5} {'Prnd':>5}  Best")
        for line in (outputs.get(host) or "").splitlines():
            fields = line.split("|")
            if len(fields) == 6:
                print(f"  {fields[0]:<32} {fields[1]:>5} {fields[2]:>5} {fields[3]:>5} "
                      f"{fields[4]:>5}  {fields[5]}")


def show_logs(deployer: Deployer, hosts: Sequence[str], n_lines: int) -> None:
    def logs(host: str) -> str:
        return deployer.ssh(host, (
            f"for f in {q(deployer.settings.workspace_of(host))}/logs/run-*.log; do "
            f'[ -f "$f" ] || continue; echo "--- $(basename "$f")"; tail -n {int(n_lines)} "$f"; '
            "done"), capture=True, check=False)

    outputs = for_hosts(hosts, logs)
    for host in hosts:
        print(f"\n=== {host} ===\n{outputs.get(host) or '(no logs)'}")


def reset_running(deployer: Deployer, hosts: Sequence[str], studies: Sequence[str]) -> None:
    """Mark orphaned RUNNING trials as FAIL (e.g. after workers were killed)."""
    s = deployer.settings
    outputs = for_hosts(hosts, lambda h: deployer.ssh(h, _python_heredoc(
        f"{s.venv_of(h)}/bin/python", _RESET_SCRIPT, f"{s.workspace_of(h)}/results/optuna",
        *studies), capture=True))
    for host, out in outputs.items():
        print(f"  [{host}] {out or ''}")


def wait_and_collect(deployer: Deployer, hosts: Sequence[str], poll_s: int = 120) -> None:
    from optimization.collect import collect_host

    try:
        while True:
            alive = [h for h, a in for_hosts(hosts, deployer.session_alive).items() if a]
            if not alive:
                break
            log.info("still running on: %s", ", ".join(alive))
            time.sleep(poll_s)
    except KeyboardInterrupt:
        log.warning("interrupted; collecting now")
    for_hosts(hosts, lambda h: collect_host(deployer, h))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m optimization.deploy", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--inventory", action="store_true",
                      help="Record each host's CPU, memory, OS, glibc and Python in "
                           "results/metrics/hosts.json and check the requirements.")
    mode.add_argument("--setup", action="store_true",
                      help="Sync code and data; install Python and the locked environment.")
    mode.add_argument("--status", action="store_true", help="Session state and last log line.")
    mode.add_argument("--progress", action="store_true", help="Trial counts per remote study.")
    mode.add_argument("--logs", nargs="?", const=30, type=int, metavar="N",
                      help="Last N lines (default 30) of each remote run log.")
    mode.add_argument("--watch", nargs="?", const=60, type=int, metavar="SEC",
                      help="Repeat --status every SEC seconds (default 60).")
    mode.add_argument("--wait", action="store_true",
                      help="Wait until the sessions finish, then collect the databases.")
    mode.add_argument("--reset-running", action="store_true",
                      help="Mark RUNNING trials of --study as FAIL in the remote database.")
    mode.add_argument("--cleanup", action="store_true",
                      help="Stop the session and delete the remote workspace.")
    mode.add_argument("--start-service", action="store_true",
                      help="Start the configured BIORESERVOIR_DEPLOY_SERVICE again.")
    p.add_argument("--host", action="append", metavar="LABEL",
                   help="Host label (repeatable; default: all configured hosts).")
    p.add_argument("--study", "--studies", dest="studies", nargs="+", metavar="STUDY",
                   help="Studies to run one after another with 'optimize research'.")
    p.add_argument("--type", dest="study_type", default="training",
                   choices=["training", "readout", "baseline"], help="Study type (research).")
    p.add_argument("--run", action="append", metavar="CMD",
                   help="Shell command to run remotely instead of a study (repeatable).")
    p.add_argument("--label", help="Log name for --run commands (default: run).")
    p.add_argument("--push", nargs="+", default=[], metavar="PATH",
                   help="Copy results/ or archive/ paths to the host before running.")
    p.add_argument("--processes", type=int, help="Worker processes (default: all remote CPUs).")
    p.add_argument("--dry-run", action="store_true", help="Print the commands only.")
    p.add_argument("--yes", "-y", action="store_true", help="Do not ask for confirmation.")
    p.add_argument("--verbose", "-v", action="store_true", help="Debug logging.")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    if extra[:1] == ["--"]:
        extra = extra[1:]
    if extra and not args.studies:
        parser.error(f"unrecognised arguments: {' '.join(extra)}")
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    deployer = Deployer(load_settings(), dry_run=args.dry_run)
    hosts = args.host or sorted(deployer.settings.hosts)
    for host in hosts:
        deployer.target(host)

    try:
        if args.inventory:
            inventory(deployer, hosts)
        elif args.setup:
            for_hosts(hosts, lambda h: (deployer.sync(h), deployer.setup_venv(h)))
        elif args.status:
            show_status(deployer, hosts)
        elif args.progress:
            show_progress(deployer, hosts, args.studies or [])
        elif args.logs is not None:
            show_logs(deployer, hosts, args.logs)
        elif args.watch is not None:
            while True:
                print(f"\n{datetime.now().strftime('%H:%M:%S')}  (Ctrl-C to stop)")
                show_status(deployer, hosts)
                time.sleep(args.watch)
        elif args.wait:
            wait_and_collect(deployer, hosts)
        elif args.reset_running:
            if not args.studies:
                parser.error("--reset-running needs --study")
            reset_running(deployer, hosts, args.studies)
        elif args.cleanup:
            if not (args.yes or args.dry_run) and input(
                    f"Delete the remote workspace on {', '.join(hosts)}? [y/N] ").lower() != "y":
                raise SystemExit("aborted")
            for_hosts(hosts, deployer.cleanup)
        elif args.start_service:
            for_hosts(hosts, deployer.start_service)
        else:
            if not args.host or len(args.host) != 1:
                parser.error("launching needs exactly one --host")
            if bool(args.studies) == bool(args.run):
                parser.error("give either --study or --run")
            jobs = (study_jobs(args.studies, args.processes, extra, args.study_type)
                    if args.studies else command_jobs(args.run, args.label))
            host = args.host[0]
            deployer.sync(host)
            deployer.push(host, args.push)
            deployer.setup_venv(host)
            deployer.stop_service(host)
            deployer.launch(host, jobs)
            log.info("[%s] follow with: python -m optimization.deploy --logs --host %s",
                     host, host)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"remote command failed (exit {exc.returncode})") from None
    except KeyboardInterrupt:
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
