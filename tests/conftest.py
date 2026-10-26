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
def ecg_data(tmp_path, monkeypatch):
    """A small five-class dataset (40 beats per class) in place of the prepared ECG cache."""
    import numpy as np

    rng = np.random.default_rng(1)
    labels = np.repeat(np.arange(5), 40)
    t = np.linspace(0.0, 1.0, 187)
    peak = np.exp(-(((t[None, :] - 0.1 - 0.1 * labels[:, None]) / 0.03) ** 2))
    beats = np.clip(0.2 + 0.7 * peak + 0.05 * rng.normal(size=peak.shape), 0.0, 1.0)
    cache = tmp_path / "data" / "ecg" / "ecg_data.npz"
    cache.parent.mkdir(parents=True)
    np.savez(cache, X=beats, Y=labels.astype(float))
    monkeypatch.setenv("BIORESERVOIR_DATA_DIR", str(tmp_path / "data"))


@pytest.fixture
def small_design(tmp_path):
    """A small copy of a ``v0.2`` design (16 colonies, a warm-up of 5 steps, 100 beats):
    ``small_design("centre")`` is the path of its config."""
    designs = Path(__file__).resolve().parents[1] / "experiments" / "v0.2" / "designs"

    def write(design, **params):
        path = tmp_path / f"{design}.yaml"
        params = {"units": 16, "warmup": 5, **params}
        values = ", ".join(f"{key}: {value}" for key, value in params.items())
        path.write_text(f"base: {designs / design}.yaml\nname: toy-{design}\n"
                        f"task: {{rows: 100}}\nreservoir: {{params: {{{values}}}}}\n")
        return path

    return write


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
