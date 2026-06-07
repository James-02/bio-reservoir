"""Runtime environment metadata collection.

Collected once per worker process and cached for the lifetime of that
process.  Used by both the training (classification.py) and readout
(readout.py) objectives to stamp every Optuna trial with full
provenance information.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_ENV_CACHE: Optional[Dict[str, Any]] = None


def _collect_environment() -> Dict[str, Any]:
    import platform, sys as _sys, importlib, subprocess, socket, datetime

    def _safe(fn, fallback="N/A"):
        try:
            return fn()
        except Exception:
            return fallback

    env: Dict[str, Any] = {}
    env["timestamp_utc"]     = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    env["python_version"]    = _sys.version.split()[0]
    env["python_executable"] = _sys.executable
    env["platform"]          = platform.platform()
    env["hostname"]          = _safe(socket.gethostname)
    env["ip_address"]        = _safe(lambda: socket.gethostbyname(socket.gethostname()))
    env["cwd"]               = _safe(os.getcwd)
    env["script_path"]       = _safe(lambda: os.path.abspath(__file__))
    env["output_dir"]        = _safe(lambda: os.path.abspath("results"))
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

    env["git_commit"] = _safe(lambda: subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip())
    env["git_branch"] = _safe(lambda: subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL).decode().strip())
    return env


def get_environment() -> Dict[str, Any]:
    """Return the cached environment snapshot, collecting it on first call."""
    global _ENV_CACHE
    if _ENV_CACHE is None:
        _ENV_CACHE = _collect_environment()
    return _ENV_CACHE
