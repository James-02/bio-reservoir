"""Runtime environment metadata collection.

Collected once per worker process and cached for the lifetime of that
process.  Used by both the training (classification.py) and readout
(readout.py) objectives to stamp every Optuna trial with full
provenance information.

Privacy policy: record the code version (git SHA and dirty flag), OS name and
version, CPU model, core counts, RAM and package versions.  Never record host
names, IP addresses, user names, working directories or absolute paths.  An
optional machine label (``BIORESERVOIR_MACHINE_LABEL``, e.g. ``Host 1``) comes
from ``config.env``.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Optional

from utils import config

logger = logging.getLogger(__name__)

_ENV_CACHE: Optional[Dict[str, Any]] = None


def _platform_summary() -> str:
    """Return OS, numeric kernel release and architecture, e.g. ``Linux-5.14.0-x86_64``."""
    import platform

    match = re.match(r"\d+(?:\.\d+){0,2}", platform.release())
    release = match.group(0) if match else "unknown"
    return f"{platform.system()}-{release}-{platform.machine()}"


def _os_release() -> Dict[str, str]:
    """Parse ``/etc/os-release`` (empty if unavailable)."""
    values: Dict[str, str] = {}
    try:
        with open("/etc/os-release") as f:
            for line in f:
                key, sep, value = line.strip().partition("=")
                if sep:
                    values[key] = value.strip('"')
    except OSError:
        pass
    return values


def _git(*args: str) -> str:
    import subprocess

    return subprocess.check_output(
        ["git", *args], cwd=config.PROJECT_ROOT, stderr=subprocess.DEVNULL
    ).decode().strip()


def _collect_environment() -> Dict[str, Any]:
    import platform, sys as _sys, importlib, datetime

    def _safe(fn, fallback="N/A"):
        try:
            return fn()
        except Exception:
            return fallback

    os_release = _os_release()
    env: Dict[str, Any] = {}
    env["timestamp_utc"]     = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    env["python_version"]    = _sys.version.split()[0]
    env["platform"]          = _safe(_platform_summary)
    env["os_name"]           = os_release.get("NAME", platform.system())
    env["os_version"]        = os_release.get("VERSION_ID", "N/A")
    env["machine_label"]     = config.get("BIORESERVOIR_MACHINE_LABEL", "N/A")
    env["cpu_count_logical"] = _safe(os.cpu_count, "N/A")
    env["cpu_count_physical"] = "N/A"
    env["cpu_model"]          = "N/A"
    env["cpu_freq_mhz"]       = "N/A"

    try:
        with open("/proc/cpuinfo") as f:
            cpuinfo = f.read()
        for line in cpuinfo.splitlines():
            if "model name" in line:
                env["cpu_model"] = line.split(":", 1)[1].strip()
                break
        cores = set()
        cur_p = cur_c = None
        for line in cpuinfo.splitlines():
            if "physical id" in line:
                cur_p = line.split(":", 1)[1].strip()
            if "core id" in line:
                cur_c = line.split(":", 1)[1].strip()
                if cur_p is not None:
                    cores.add((cur_p, cur_c))
        if cores:
            env["cpu_count_physical"] = len(cores)
    except Exception as exc:
        logger.warning("Could not read /proc/cpuinfo: %s", exc)

    try:
        import psutil
        freq = psutil.cpu_freq()
        if freq:
            env["cpu_freq_mhz"] = round(freq.max, 1)
    except Exception:
        pass

    env["ram_total_gb"] = env["ram_available_gb"] = "N/A"
    try:
        with open("/proc/meminfo") as f:
            meminfo = {
                line.split(":")[0].strip(): line.split(":")[1].strip()
                for line in f if ":" in line
            }
        env["ram_total_gb"]     = round(int(meminfo.get("MemTotal",     "0 kB").split()[0]) / 1024**2, 2)
        env["ram_available_gb"] = round(int(meminfo.get("MemAvailable", "0 kB").split()[0]) / 1024**2, 2)
    except Exception as exc:
        logger.warning("Could not read /proc/meminfo: %s", exc)

    for pkg in ("numpy", "reservoirpy", "optuna", "numba", "sklearn", "scipy", "psutil"):
        try:
            m = importlib.import_module(pkg)
            env[pkg + "_version"] = getattr(m, "__version__", "unknown")
        except ImportError:
            env[pkg + "_version"] = "not_installed"

    dirty_setting = config.get("BIORESERVOIR_GIT_DIRTY", "N/A")
    dirty_fallback = {"true": True, "false": False}.get(dirty_setting.lower(), dirty_setting)
    env["git_commit"] = _safe(lambda: _git("rev-parse", "HEAD"),
                              config.get("BIORESERVOIR_GIT_COMMIT", "N/A"))
    env["git_dirty"] = _safe(lambda: bool(_git("status", "--porcelain", "--untracked-files=no")),
                             dirty_fallback)
    env["git_branch"] = _safe(lambda: _git("rev-parse", "--abbrev-ref", "HEAD"))
    return env


def get_environment() -> Dict[str, Any]:
    """Return the cached environment snapshot, collecting it on first call."""
    global _ENV_CACHE
    if _ENV_CACHE is None:
        _ENV_CACHE = _collect_environment()
    return _ENV_CACHE
