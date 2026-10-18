"""Unit tests for the scripts (their settings; nothing large is run)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import optuna
import pytest

from training.run import PRESETS

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
V0_1 = PRESETS["v0.1"]["reservoir"]


def _load(name):
    spec = importlib.util.spec_from_file_location(f"script_{name.replace('/', '_')}",
                                                  SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def capture(monkeypatch):
    """Record the topology and reservoir arguments of a script instead of building them."""

    def patch(module):
        calls = {}

        def fake_matrix(**kwargs):
            calls["W"] = kwargs
            return "W"

        def fake_node(**kwargs):
            calls["node"] = kwargs
            return "node"

        monkeypatch.setattr(module, "distance_matrix", fake_matrix)
        monkeypatch.setattr(module, "BioReservoir", fake_node)
        return calls

    return patch


class TestHeterogeneityScript:
    """The v0.1 sweep augmented the training folds, and its five CV = 0 runs were the
    same reservoir (only the parameter draws were seeded per point)."""

    def test_training_folds_are_not_augmented_by_default(self):
        het = _load("run_heterogeneity")
        assert het.NOISE_RATE == 0.0 and het.NOISE_RATIO == 0.0

    def test_each_grid_seed_has_its_own_array(self):
        het = _load("run_heterogeneity")
        points = het.grid_points(seed=6337)
        assert len(points) == len(het.CV_LEVELS) * len(het.HET_SEEDS)
        at_zero = [(het_seed, array_seed) for cv, het_seed, array_seed in points if cv == 0.0]
        assert sorted(array for _, array in at_zero) == sorted(het.HET_SEEDS)
        fixed = het.grid_points(seed=6337, fixed_array=True)
        assert {array for _, _, array in fixed} == {6337}

    def test_seeds_and_preset_reach_the_reservoir(self, capture):
        het = _load("run_heterogeneity")
        calls = capture(het)
        het._build_reservoir(0.2, het_seed=100, seed=300)
        assert calls["W"]["seed"] == calls["node"]["seed"] == 300
        assert calls["node"]["param_heterogeneity_seed"] == 100
        assert calls["node"]["param_heterogeneity_cv"] == 0.2
        assert calls["node"]["param_heterogeneity_mean_preserving"] is True
        assert calls["node"]["geometry"] == "per_node"
        assert calls["W"]["normalize_after_masks"] is True

        het._build_reservoir(0.2, het_seed=100, seed=300, overrides=V0_1)
        assert calls["node"]["param_heterogeneity_mean_preserving"] is False
        assert calls["node"]["geometry"] == "v0.1"
        assert calls["node"]["delay_scheme"] == "frozen"
        assert calls["W"]["normalize_after_masks"] is False


class TestAblationScript:
    """The v0.1 ablation script augmented the training folds unless told not to; the
    v0.1 ablations were run without augmentation."""

    def test_training_folds_are_not_augmented_by_default(self):
        ablation = _load("run_ablation")
        assert ablation.NOISE_RATE == 0.0 and ablation.NOISE_RATIO == 0.0

    @pytest.mark.parametrize("condition, rc_scaling, dde_kwargs", [
        ("full", None, {}), ("no-coupling", 0.0, {}), ("no-delay", None, {"delay": 0}),
    ])
    def test_conditions_and_preset_reach_the_reservoir(self, capture, condition, rc_scaling,
                                                       dde_kwargs):
        ablation = _load("run_ablation")
        calls = capture(ablation)
        ablation._build_reservoir(condition, seed=7)
        expected = ablation.RESERVOIR_PARAMS["rc_scaling"] if rc_scaling is None else rc_scaling
        assert calls["node"]["rc_scaling"] == expected
        assert calls["node"]["dde_kwargs"] == dde_kwargs
        assert calls["node"]["geometry"] == "per_node"
        assert calls["node"]["delay_scheme"] == "substep"
        assert calls["W"]["normalize_after_masks"] is True

        ablation._build_reservoir(condition, seed=7, overrides=V0_1)
        assert calls["node"]["geometry"] == "v0.1"
        assert calls["node"]["delay_scheme"] == "frozen"
        assert calls["W"]["normalize_after_masks"] is False


class _InProcess:
    """Stands in for ProcessPoolExecutor: runs the folds in this process."""

    def __init__(self, max_workers=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, fn, items):
        return map(fn, items)


class TestBinaryFullBaselineScript:
    """The v0.1 no-reservoir binary-full result read the v0.1 binary file."""

    @pytest.fixture
    def script(self, tmp_path, monkeypatch):
        module = _load("rerun_baseline_binary_full")
        monkeypatch.setenv("BIORESERVOIR_DATA_DIR", str(tmp_path))
        rng = np.random.default_rng(0)
        t = np.linspace(0.0, 1.0, 16)
        X = np.concatenate([t + 0.1 * rng.normal(size=(20, 16)),
                            1.0 - t + 0.1 * rng.normal(size=(20, 16))])[..., None]
        Y = np.eye(2)[np.repeat([0, 1], 20)][:, None, :]
        module.calls = []
        monkeypatch.setattr(module, "load_ecg_data", lambda **kwargs: module.calls.append(
            kwargs) or (X[::2], Y[::2], X[1::2], Y[1::2]))
        monkeypatch.setattr(module, "ProcessPoolExecutor", _InProcess)
        return module

    def test_the_named_data_file_is_read(self, script, tmp_path, monkeypatch, capsys):
        data_file = tmp_path / "ecg_archive" / "binary_ecg_data.npz"
        data_file.parent.mkdir()
        data_file.write_bytes(b"v0.1 pool")
        monkeypatch.setattr(sys, "argv", ["script", "--folds", "2", "--workers", "1",
                                          "--data-file", "ecg_archive/binary_ecg_data.npz"])
        script.main()
        (call,) = script.calls
        assert call["data_dir"] == str(data_file.parent)
        assert call["binary_save_file"] == "binary_ecg_data.npz"
        assert "F1:       1.0000" in capsys.readouterr().out

    def test_the_prepared_dataset_is_the_default(self, script, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["script", "--folds", "2", "--workers", "1"])
        script.main()
        (call,) = script.calls
        assert "data_dir" not in call and "binary_save_file" not in call

    def test_a_missing_data_file_is_an_error(self, script, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["script", "--data-file", "missing.npz"])
        with pytest.raises(SystemExit, match="--data-file not found"):
            script.main()


class TestReadoutTables:
    """Tables S13-S15 are generated from the Phase 4 study databases."""

    @staticmethod
    def _study():
        from optuna.distributions import CategoricalDistribution, IntDistribution

        distributions = {"classifier": CategoricalDistribution(["RF", "KNN"]),
                         "rf_n_estimators": IntDistribution(150, 350),
                         "rf_criterion": CategoricalDistribution(["entropy", "log_loss"]),
                         "knn_n_neighbors": IntDistribution(1, 25)}
        study = optuna.create_study(direction="maximize")
        for value, params in [
            (0.80, {"classifier": "RF", "rf_n_estimators": 200, "rf_criterion": "entropy"}),
            (0.90, {"classifier": "RF", "rf_n_estimators": 300, "rf_criterion": "log_loss"}),
            (0.85, {"classifier": "KNN", "knn_n_neighbors": 3}),
            (0.90, {"classifier": "RF", "rf_n_estimators": 250, "rf_criterion": "entropy"}),
        ]:
            study.add_trial(optuna.trial.create_trial(
                params=params, value=value,
                distributions={k: distributions[k] for k in params}))
        return study

    def test_rows_are_the_best_trial_of_each_family(self):
        tables = _load("analysis/readout_tables")
        rows = tables.best_per_family(self._study())
        assert [(r["classifier"], r["f1"], r["trial"]) for r in rows] == [
            ("RF", 0.90, 1), ("KNN", 0.85, 2)]
        assert rows[0]["params"] == {"rf_n_estimators": 300, "rf_criterion": "log_loss"}

    def test_rows_are_written_as_latex(self):
        tables = _load("analysis/readout_tables")
        lines = tables.latex_rows(tables.best_per_family(self._study()))
        assert lines[0].strip() == (r"Random Forest & \textbf{0.900} & "
                                    r"$n_{\mathrm{est}} = 300$, criterion = log\_loss \\")
        assert lines[1].strip() == r"KNN & 0.850 & $k = 3$ \\"

    @pytest.mark.parametrize("value, text", [
        (326, "326"), (8150.3, "8150"), (62000.0, r"6.2 \times 10^{4}"), (0.0571, "0.0571"),
        (1e-5, "10^{-5}"), (2.66, "2.66"), ("log_loss", r"log\_loss"), (True, "True"),
    ])
    def test_numbers_are_formatted_for_the_tables(self, value, text):
        assert _load("analysis/readout_tables")._number(value) == text


class TestFoldAnalysisScript:
    """The per-fold table and the class plots asked for RMSE and R2, which fold files no
    longer store: the table showed columns of zeros and the class plots failed."""

    @staticmethod
    def _folds():
        from utils.analysis import evaluate_performance
        from utils.results import FoldArtifact

        rng = np.random.default_rng(0)
        folds = []
        for k in range(2):
            y_test, y_pred = (np.eye(5)[rng.integers(0, 5, 40)] for _ in range(2))
            folds.append(FoldArtifact(trial_name="run", fold_index=k, Y_train=y_test,
                                      Y_test=y_test, Y_pred=y_pred,
                                      metrics=evaluate_performance(y_test, y_pred, 1.0)))
        return folds

    def test_fold_table_shows_the_stored_metrics(self, capsys):
        _load("analysis/analyse_fold_results")._print_fold_table(self._folds())
        header = capsys.readouterr().out.splitlines()[1].split()
        assert header == ["fold", "F1", "Accuracy", "Precision", "Recall", "MCC", "Kappa",
                          "Runtime(s)"]

    def test_class_plots_are_saved(self, tmp_path, monkeypatch):
        from utils import visualisation

        monkeypatch.setenv("BIORESERVOIR_RESULTS_DIR", str(tmp_path))
        monkeypatch.setattr(visualisation, "DPI", 20)
        _load("analysis/analyse_fold_results")._run_plots(self._folds(), "run", {"class"},
                                                         show=False)
        saved = sorted(p.name for p in (tmp_path / "figures" / "metrics" / "run").iterdir())
        assert saved == ["class-metrics.png", "metrics-across-folds.png"]
