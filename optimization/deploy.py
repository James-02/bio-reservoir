#!/usr/bin/env python3
"""Deploy optimisation studies and experiment commands to remote Linux hosts.

Host requirements: RHEL 9 (or a compatible distribution) with Python 3.10-3.12, plus
the standard ``ssh`` and ``rsync`` tools and key-based SSH login.

Nothing host-specific is stored in the repository.  Host addresses, the SSH user, the
remote workspace, an optional pip proxy and an optional systemd unit to pause during
runs are read from the gitignored ``config.env`` (see ``config.env.example``)::

    BIORESERVOIR_DEPLOY_HOSTS=host1=<address>,host2=<address>
    BIORESERVOIR_DEPLOY_USER=<ssh user>
    BIORESERVOIR_DEPLOY_WORKSPACE=<absolute remote path>

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
import re
import shlex
import subprocess
import textwrap
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Callable

from utils import config
from utils.logger import setup_logging

log = logging.getLogger("deploy")
q = shlex.quote

SYNC_SOURCES = ("bioreservoir/", "optimization/", "training/", "utils/", "scripts/",
                "pyproject.toml")
SSH_OPTIONS = ("-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15")
RSYNC = ("rsync", "-az", "--exclude=__pycache__", "--exclude=*.pyc", "--exclude=*.egg-info")
PYTHON_RANGE = ((3, 10), (3, 12))
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

    @property
    def venv(self) -> str:
        return f"{self.workspace}/venv"


def parse_hosts(value: str | None) -> dict[str, str]:
    """Parse ``label=address`` pairs separated by commas or white space."""
    hosts: dict[str, str] = {}
    for item in (value or "").replace(",", " ").split():
        label, sep, address = item.partition("=")
        if not (sep and NAME_PATTERN.match(label) and address):
            raise SystemExit(f"BIORESERVOIR_DEPLOY_HOSTS: expected label=address, got {item!r}")
        hosts[label] = address
    return hosts


def load_settings() -> Settings:
    """Read the deployment settings (environment, then ``config.env``)."""
    hosts = parse_hosts(config.get("BIORESERVOIR_DEPLOY_HOSTS"))
    workspace = (config.get("BIORESERVOIR_DEPLOY_WORKSPACE") or "").rstrip("/")
    if not hosts or not workspace:
        raise SystemExit("Set BIORESERVOIR_DEPLOY_HOSTS and BIORESERVOIR_DEPLOY_WORKSPACE in "
                         "config.env (see config.env.example).")
    path = PurePosixPath(workspace)
    if not path.is_absolute() or len(path.parts) < 3 or ".." in path.parts:
        raise SystemExit("BIORESERVOIR_DEPLOY_WORKSPACE must be an absolute path at least two "
                         "levels deep (it is deleted by --cleanup).")
    return Settings(
        hosts=hosts,
        user=config.get("BIORESERVOIR_DEPLOY_USER"),
        workspace=workspace,
        python=config.get("BIORESERVOIR_DEPLOY_PYTHON", "python3"),
        pip_proxy=config.get("BIORESERVOIR_DEPLOY_PIP_PROXY"),
        service=config.get("BIORESERVOIR_DEPLOY_SERVICE"),
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
        ws, root = self.settings.workspace, config.PROJECT_ROOT
        self.ssh(host, f"mkdir -p {q(ws)}/logs {q(ws)}/results")
        for source in SYNC_SOURCES:
            self.rsync(host, "--delete", f"{root}/{source}", self.remote(host, f"{ws}/{source}"))
        self.rsync(host, "--ignore-existing", f"{root}/data/", self.remote(host, f"{ws}/data/"))
        log.info("[%s] code and data synced", host)

    def push(self, host: str, paths: Sequence[str]) -> None:
        """Copy local ``results/`` or ``archive/`` paths to the same place on *host*."""
        ws = self.settings.workspace
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
        """Check the remote Python version, then create or refresh the venv."""
        s = self.settings
        proxy = f" --proxy {q(s.pip_proxy)}" if s.pip_proxy else ""
        (low, high) = PYTHON_RANGE
        check = (f"import sys; v = sys.version_info[:2]; sys.exit(0 if {low} <= v <= {high} "
                 f"else 'Python 3.10-3.12 required, found %d.%d' % v)")
        self.ssh(host, f"""
set -e
{q(s.python)} -c {q(check)}
if [ ! -x {q(s.venv)}/bin/python ]; then
    {q(s.python)} -m venv {q(s.venv)}
    {q(s.venv)}/bin/pip install --quiet --upgrade pip{proxy}
fi
{q(s.venv)}/bin/pip install --quiet{proxy} --editable {q(s.workspace)}
""")
        log.info("[%s] venv ready", host)

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
        host label and the local git SHA so that every trial records them.
        """
        s, ws = self.settings, self.settings.workspace
        sha, dirty = git_state()
        if dirty == "true":
            log.warning("Uncommitted local changes: runs will record git_dirty=true.")
        lines = [
            "set -o pipefail",
            f"cd {q(ws)}",
            f"source {q(s.venv)}/bin/activate",
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
            f"nohup bash {session} > {q(f'{ws}/logs/session.log')} 2>&1 &",
            f"echo $! > {pid}",
            f'echo "session started (PID $(cat {pid}))"',
        ])
        log.info("[%s] launching: %s", host, ", ".join(name for name, _ in jobs))
        self.ssh(host, script)

    def session_alive(self, host: str) -> bool:
        pid = q(f"{self.settings.workspace}/logs/session.pid")
        out = self.ssh(host, f'[ -f {pid} ] && kill -0 "$(cat {pid})" 2>/dev/null '
                             f"&& echo alive || echo idle", capture=True, check=False)
        return out.strip() == "alive"

    def cleanup(self, host: str) -> None:
        """Stop the session and its Python processes, then delete the remote workspace."""
        ws = self.settings.workspace
        pid = q(f"{ws}/logs/session.pid")
        self.ssh(host, f"""
if [ -f {pid} ]; then kill "$(cat {pid})" 2>/dev/null || true; sleep 1; fi
pkill -f {q(f'{self.settings.venv}/bin/python')} 2>/dev/null || true
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
import json, platform, re, shutil, subprocess
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
    """Return the ways a host inventory falls short of RHEL 9 + Python 3.10-3.12."""
    problems = []
    if "rhel" not in info.get("os_family", "").split() or not str(
            info.get("os_version", "")).startswith("9"):
        problems.append(f"expected RHEL 9 or compatible, found {info.get('os_name')} "
                        f"{info.get('os_version')}")
    version = tuple(int(x) for x in info.get("configured_python", "0.0").split(".")[:2])
    if not PYTHON_RANGE[0] <= version <= PYTHON_RANGE[1]:
        problems.append(f"Python {info.get('configured_python')} is outside 3.10-3.12")
    return problems


def for_hosts(hosts: Sequence[str], action: Callable[[str], object]) -> dict[str, object]:
    """Run *action* on every host in parallel; errors are logged per host."""
    def run(host: str) -> object:
        try:
            return action(host)
        except (subprocess.CalledProcessError, SystemExit, OSError) as exc:
            log.error("[%s] failed: %s", host, exc)
            return None
    with ThreadPoolExecutor(max_workers=max(1, len(hosts))) as pool:
        return dict(zip(hosts, pool.map(run, hosts)))


def inventory(deployer: Deployer, hosts: Sequence[str]) -> None:
    """Record CPU, memory, OS and Python versions per host label in hosts.json."""
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
        info["checked_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        record["hosts"][host] = info
        print(f"  {host:<8} {info['cpu_model']}, {info['sockets']} socket(s), "
              f"{info['physical_cores']} cores / {info['logical_cpus']} threads, "
              f"{info['ram_gb']} GB RAM, {info['os_name']} {info['os_version']}, "
              f"Python {info['configured_python']}")
        for problem in info["problems"]:
            print(f"           requirement not met: {problem}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(f"  written to {path}")


def show_status(deployer: Deployer, hosts: Sequence[str]) -> None:
    ws = deployer.settings.workspace
    pid = q(f"{ws}/logs/session.pid")
    script = f"""
if [ -f {pid} ] && kill -0 "$(cat {pid})" 2>/dev/null; then
    SECS=$(ps -o etimes= -p "$(cat {pid})" | tr -d ' ')
    printf 'running|%dh%02dm|' $((SECS / 3600)) $((SECS % 3600 / 60))
else
    printf 'idle|-|'
fi
tail -q -n 30 {q(ws)}/logs/run-*.log 2>/dev/null | grep -v -e '^$' -e '^==>' | tail -n 1
"""
    outputs = for_hosts(hosts, lambda h: deployer.ssh(h, script, capture=True, check=False))
    print(f"\n  {'Host':<8} {'Session':<8} {'Up':>7}  Last log line")
    for host in hosts:
        parts = (outputs.get(host) or "unreachable|-|").split("|", 2)
        parts += [""] * (3 - len(parts))
        print(f"  {host:<8} {parts[0]:<8} {parts[1]:>7}  {parts[2][:80]}")


def show_progress(deployer: Deployer, hosts: Sequence[str], studies: Sequence[str]) -> None:
    s = deployer.settings
    script = _python_heredoc(f"{s.venv}/bin/python", _PROGRESS_SCRIPT,
                             f"{s.workspace}/results/optuna", *studies)
    outputs = for_hosts(hosts, lambda h: deployer.ssh(h, script, capture=True, check=False))
    for host in hosts:
        print(f"\n  [{host}]\n  {'Study':<32} {'Done':>5} {'Run':>5} {'Fail':>5} {'Prnd':>5}  Best")
        for line in (outputs.get(host) or "").splitlines():
            fields = line.split("|")
            if len(fields) == 6:
                print(f"  {fields[0]:<32} {fields[1]:>5} {fields[2]:>5} {fields[3]:>5} "
                      f"{fields[4]:>5}  {fields[5]}")


def show_logs(deployer: Deployer, hosts: Sequence[str], n_lines: int) -> None:
    script = (f"for f in {q(deployer.settings.workspace)}/logs/run-*.log; do "
              f'[ -f "$f" ] || continue; echo "--- $(basename "$f")"; tail -n {int(n_lines)} "$f"; '
              "done")
    outputs = for_hosts(hosts, lambda h: deployer.ssh(h, script, capture=True, check=False))
    for host in hosts:
        print(f"\n=== {host} ===\n{outputs.get(host) or '(no logs)'}")


def reset_running(deployer: Deployer, hosts: Sequence[str], studies: Sequence[str]) -> None:
    """Mark orphaned RUNNING trials as FAIL (e.g. after workers were killed)."""
    s = deployer.settings
    script = _python_heredoc(f"{s.venv}/bin/python", _RESET_SCRIPT,
                             f"{s.workspace}/results/optuna", *studies)
    for host, out in for_hosts(hosts, lambda h: deployer.ssh(h, script, capture=True)).items():
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
                      help="Record each host's CPU, memory, OS and Python in "
                           "results/metrics/hosts.json and check the requirements.")
    mode.add_argument("--setup", action="store_true", help="Sync code and data, build the venv.")
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
