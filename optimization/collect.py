#!/usr/bin/env python3
"""Collect Optuna databases, run folders and logs from the hosts used by ``deploy.py``.

Databases go to ``results/optuna/``, run folders to ``results/runs/<run>/`` and logs to
``logs/remote/<host>/``.  Collect a database after its session has finished (see
``python -m optimization.deploy --status``): copying a database while it is being
written can give an inconsistent file.

Usage
-----
  python -m optimization.collect                                   # all databases, all hosts
  python -m optimization.collect --host host1 --study P2-distance
  python -m optimization.collect --host host2 --runs P2-distance-best
  python -m optimization.collect --logs                            # also the remote logs
  python -m optimization.collect --list                            # list remote databases

Add ``--dry-run`` to print the commands without running them.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from optimization.deploy import NAME_PATTERN, Deployer, for_hosts, load_settings
from utils import config
from utils.logger import setup_logging

log = logging.getLogger("collect")


def collect_host(deployer: Deployer, host: str, *, studies: Sequence[str] = (),
                 runs: Sequence[str] = (), logs: bool = False) -> None:
    """Copy databases (all, or *studies*), *runs* and optionally logs from *host*."""
    for name in (*studies, *runs):
        if not NAME_PATTERN.match(name):
            raise SystemExit(f"Invalid study or run name {name!r}")
    ws = deployer.settings.workspace
    optuna_dir = config.results_root() / "optuna"
    targets = [optuna_dir] + [config.runs_dir() / run for run in runs]
    if logs:
        targets.append(config.logs_dir() / "remote" / host)
    if not deployer.dry_run:
        for path in targets:
            path.mkdir(parents=True, exist_ok=True)

    if studies:
        for study in studies:
            deployer.rsync(host, deployer.remote(host, f"{ws}/results/optuna/optuna-{study}.db"),
                           f"{optuna_dir}/", check=False)
    else:
        deployer.rsync(host, "--include=optuna-*.db", "--exclude=*",
                       deployer.remote(host, f"{ws}/results/optuna/"), f"{optuna_dir}/",
                       check=False)
    for run in runs:
        deployer.rsync(host, deployer.remote(host, f"{ws}/results/runs/{run}/"),
                       f"{config.runs_dir() / run}/")
    if logs:
        deployer.rsync(host, "--include=*.log", "--exclude=*",
                       deployer.remote(host, f"{ws}/logs/"),
                       f"{config.logs_dir() / 'remote' / host}/", check=False)
    log.info("[%s] collected into %s", host, config.results_root())


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m optimization.collect", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", action="append", metavar="LABEL",
                   help="Host label (repeatable; default: all configured hosts).")
    p.add_argument("--study", "--studies", dest="studies", nargs="+", default=[],
                   metavar="STUDY", help="Collect only these study databases.")
    p.add_argument("--runs", nargs="+", default=[], metavar="RUN",
                   help="Also collect these results/runs/<run>/ folders.")
    p.add_argument("--logs", action="store_true", help="Also collect the remote log files.")
    p.add_argument("--list", action="store_true", help="List remote databases only.")
    p.add_argument("--dry-run", action="store_true", help="Print the commands only.")
    args = p.parse_args(argv)

    setup_logging(level=logging.INFO)
    deployer = Deployer(load_settings(), dry_run=args.dry_run)
    hosts = args.host or sorted(deployer.settings.hosts)
    for host in hosts:
        deployer.target(host)

    if args.list:
        script = f"ls {deployer.settings.workspace}/results/optuna/ 2>/dev/null || true"
        for host, out in for_hosts(hosts, lambda h: deployer.ssh(
                h, script, capture=True, check=False)).items():
            print(f"  [{host}]\n" + "\n".join(f"    {line}" for line in (out or "").splitlines()
                                             if line.endswith(".db")))
        return
    for_hosts(hosts, lambda h: collect_host(deployer, h, studies=args.studies, runs=args.runs,
                                            logs=args.logs))


if __name__ == "__main__":
    main()
