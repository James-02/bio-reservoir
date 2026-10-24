"""Sanity checks of the realism options: the oscillation of each colony
(bioreservoir.diagnostics.oscillation_amplitudes) and scripts/analysis/drive_bounds.py on a
diffusive array with --set and --oscillation."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import yaml

from bioreservoir import BioReservoir
from bioreservoir.diagnostics import oscillation_amplitudes
from optimization.reservoirs import build_reservoir
from training import run as run_module

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "analysis" / "drive_bounds.py"
P_LIKE = dict(units=16, warmup=300, rk4_substeps=16, input_scaling=0.155113,
              input_connectivity=0.66, input_bias=False, bias_scaling=0.1, dde_scaling=0.01,
              cell_coupling=12.75, noise_in=0.01442, output_variables="IHe",
              topology_type="lattice", coupling="diffusion", diffusion_rate=0.6,
              input_encoding="nonnegative", input_sign="positive", param_heterogeneity_cv=0.065)


def test_oscillating_colonies_have_amplitudes_of_order_one():
    node, _ = build_reservoir(dict(P_LIKE), seed=6337)
    states = node.states.copy()
    amplitude = oscillation_amplitudes(node, 200)
    assert amplitude.shape == (16,) and np.all(amplitude > 0.5)
    np.testing.assert_array_equal(node.states, states)


def test_a_colony_without_promoter_activity_settles():
    node = BioReservoir(units=4, warmup=300, rk4_substeps=16, seed=5, dde_kwargs={"alpha": 0.0})
    np.testing.assert_allclose(oscillation_amplitudes(node, 100), 0.0, atol=1e-6)


def _load_script():
    spec = importlib.util.spec_from_file_location("script_drive_bounds", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_script(tmp_path, monkeypatch, *options):
    config = tmp_path / "p.yaml"
    config.write_text(yaml.safe_dump({
        "name": "p", "reservoir": {"seed": 6337, "params": P_LIKE},
        "readout": {"classifier": "RidgeCV"}}))
    beats = np.random.default_rng(0).normal(size=(10, 20, 1))
    labels = np.eye(5)[np.arange(10) % 5]
    monkeypatch.setattr(run_module, "load_task", lambda task: (beats, labels, []))
    out = tmp_path / "summary.json"
    monkeypatch.setattr("sys.argv", [str(SCRIPT), str(config), "--beats", "1", "--json",
                                     str(out), *options])
    _load_script().main()
    return json.loads(out.read_text())


def test_drive_bounds_reports_a_diffusive_array_without_drive_shares(tmp_path, monkeypatch):
    """The drive shares belong to the abstract coupling; a diffusive array reports its drive
    and state ranges instead of failing."""
    summary = _run_script(tmp_path, monkeypatch)
    assert "drive_shares" not in summary and summary["beats"] == 5
    assert min(summary["state_min"].values()) > 0


def test_drive_bounds_applies_settings_and_reports_the_oscillation(tmp_path, monkeypatch,
                                                                    capsys):
    summary = _run_script(tmp_path, monkeypatch, "--oscillation", "50",
                          "--set", "reservoir.params.reset_noise=0.1",
                          "--set", "reservoir.seed=6338")
    assert summary["settings"] == {"reservoir.params.reset_noise": 0.1, "reservoir.seed": 6338}
    assert summary["oscillation"]["colonies"] == 16
    assert summary["oscillation"]["fraction_oscillating"] == 1.0
    assert "oscillation over 50 steps" in capsys.readouterr().out
