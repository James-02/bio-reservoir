import logging
import os
import sys


def setup_logging(
    *,
    level: int | str = logging.INFO,
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    log_file: str | None = None,
    quiet_loggers: tuple[str, ...] = ("numba",),
) -> None:
    """Configure application logging for scripts/CLIs.

    Library code should never call ``basicConfig``. Instead, CLIs/examples should
    call this once at startup.

    Parameters
    ----------
    level:
        Root logging level.
    log_file:
        Optional file to write logs to.
    quiet_loggers:
        Logger names to silence (set to ERROR).
    """
    # Respect an existing configuration (avoid duplicate handlers).
    root = logging.getLogger()
    if not root.handlers:
        fmt = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
        handlers = [logging.StreamHandler(sys.stdout)]
        if log_file:
            os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
            handlers.append(logging.FileHandler(log_file))
        logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)
    else:
        # Root already configured; just update the level.
        root.setLevel(level)

    for name in quiet_loggers:
        logging.getLogger(name).setLevel(logging.ERROR)
        logging.getLogger(name).propagate = False
