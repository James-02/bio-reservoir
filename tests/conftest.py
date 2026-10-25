# Shared fixtures and configuration for the test suite.
#
# The project is installed in editable mode (`pip install -e .`) so that all
# packages (bioreservoir, training, utils, optimization) are importable without
# any sys.path manipulation.  Run:
#
#   pip install -e ".[dev]"
#
# once before executing the test suite.

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture
def pyplot():
    """``matplotlib.pyplot``; every figure is closed and every setting restored after the test."""
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.close("all")
    with plt.rc_context():
        yield plt
    plt.close("all")


@pytest.fixture
def figure_dir(tmp_path, monkeypatch, pyplot):
    """Figures are written under a temporary results folder, at a low resolution for speed."""
    from utils import visualisation

    monkeypatch.setenv("BIORESERVOIR_RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(visualisation, "DPI", 20)
    return tmp_path / "figures"


@pytest.fixture
def load_script(monkeypatch):
    """Import ``scripts/<name>.py`` as a module, e.g. ``load_script("figures/visualise_dde")``."""

    def load(name):
        spec = importlib.util.spec_from_file_location(f"script_{name.replace('/', '_')}",
                                                      SCRIPTS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        # Registered first, as a module is on import: dataclasses look their module up there.
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        return module

    return load


@pytest.fixture
def run_script(monkeypatch, load_script):
    """Run ``scripts/<name>.py`` in this process, as ``python scripts/<name>.py ARGS`` would."""

    def run(name, *args):
        module = load_script(name)
        monkeypatch.setattr(sys, "argv", [f"{name}.py", *map(str, args)])
        return module.main()

    return run
