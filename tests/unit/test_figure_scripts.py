"""The figure scripts in scripts/ run end to end on small inputs (Agg backend, no dataset).

Each test runs a script as ``python scripts/<name>.py ARGS`` would and checks the figures
it writes, so the scripts that draw the ``v0.1`` figures keep working.
"""

import logging

import numpy as np
import pytest


def _saved(folder):
    """The figure files under ``folder``, as sorted relative paths."""
    return sorted(str(p.relative_to(folder)) for p in folder.rglob("*.png"))


@pytest.fixture
def ecg_data(tmp_path, monkeypatch):
    """A small five-class dataset (40 beats per class) in place of the prepared ECG cache."""
    rng = np.random.default_rng(1)
    labels = np.repeat(np.arange(5), 40)
    t = np.linspace(0.0, 1.0, 187)
    peak = np.exp(-(((t[None, :] - 0.1 - 0.1 * labels[:, None]) / 0.03) ** 2))
    beats = np.clip(0.2 + 0.7 * peak + 0.05 * rng.normal(size=peak.shape), 0.0, 1.0)
    cache = tmp_path / "data" / "ecg" / "ecg_data.npz"
    cache.parent.mkdir(parents=True)
    np.savez(cache, X=beats, Y=labels.astype(float))
    monkeypatch.setenv("BIORESERVOIR_DATA_DIR", str(tmp_path / "data"))


def _fold_files(folder, *, folds=2, beats=40, classes=5, states=True):
    """Fold files of a small five-class run named ``run``; return their glob pattern."""
    from utils.analysis import evaluate_performance
    from utils.results import FoldArtifact

    rng = np.random.default_rng(0)
    for k in range(folds):
        y_test = np.eye(classes)[rng.integers(0, classes, beats)][:, None, :]
        y_pred = rng.random((beats, 1, classes))
        y_pred /= y_pred.sum(axis=-1, keepdims=True)
        FoldArtifact(trial_name="run", fold_index=k, Y_train=y_test, Y_test=y_test,
                     Y_pred=y_pred, metrics=evaluate_performance(y_test, y_pred, 1.0),
                     test_states=rng.normal(size=(beats, 6)) if states else None,
                     ).save(results_dir=str(folder), save_states=states)
    return str(folder / "run" / "fold-*.npz")


class TestFoldDiagnostics:
    """scripts/analysis/analyse_fold_diagnostics.py. Matplotlib 3.11 removed the box plot's
    ``labels`` argument (now ``tick_labels``), so ``--plots`` stopped at its second figure."""

    def test_every_plot_is_saved(self, tmp_path, figure_dir, run_script):
        folds = _fold_files(tmp_path / "runs")
        assert run_script("analysis/analyse_fold_diagnostics", "--glob", folds, "--plots") == 0
        assert _saved(figure_dir / "diagnostics") == [
            "confidence_correct_vs_incorrect.png", "entropy_by_class.png",
            "per_class_recall.png", "states_pca.png", "states_pca_explained_variance.png"]


class TestFoldAnalysisFigures:
    """The fold-analysis scripts draw the ``v0.1`` confusion matrices from fold files."""

    def test_analyse_fold_results_saves_every_plot_type(self, tmp_path, figure_dir, run_script):
        folds = _fold_files(tmp_path / "runs")
        assert run_script("analysis/analyse_fold_results", "--glob", folds, "--plots") == 0
        assert _saved(figure_dir / "metrics" / "run") == [
            "class-metrics.png", "confusion-matrix.png", "metrics-across-folds.png",
            "tsne-predictions.png", "tsne-states.png"]

    def test_plot_cv_results_saves_its_plots(self, tmp_path, figure_dir, run_script):
        folds = _fold_files(tmp_path / "runs", states=False)
        assert run_script("analysis/plot_cv_results", folds, "--tsne") == 0
        assert _saved(figure_dir / "metrics") == [
            "cv_class_metrics.png", "cv_confusion_matrix_sum.png",
            "cv_metrics_across_folds.png", "cv_tsne.png"]


class TestModelFigures:
    """The scripts behind the ``v0.1`` model, drive and topology figures."""

    def test_dde_figure(self, figure_dir, ecg_data, run_script, monkeypatch):
        monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: None)
        run_script("figures/visualise_dde", "--timesteps", 40, "--warmup", 10, "--rows", 100)
        assert _saved(figure_dir) == ["states/dde_ecg_multiscale.png"]

    def test_reservoir_state_drive_and_topology_figures(self, figure_dir, ecg_data, run_script):
        run_script("figures/visualise_reservoir_states", "--nodes", 20, "--timesteps", 30,
                   "--plot-topology", "--plot-drive")
        assert _saved(figure_dir) == ["states/reservoir_drive_diag_nodes20_inst0.png",
                                      "states/reservoir_heatmap_luxI_nodes20_inst0.png",
                                      "topology/reservoir_topology_nodes20.png"]

    def test_topology_figures(self, figure_dir, run_script):
        run_script("figures/visualise_topology", "--units", 30)
        assert _saved(figure_dir) == ["topology/distance_histograms.png",
                                      "topology/spatial_topology.png",
                                      "topology/weight_vs_distance_matrix.png"]


class TestDatasetFigures:
    def test_average_noise_scaler_and_sample_figures(self, figure_dir, ecg_data, run_script):
        run_script("figures/visualise_dataset", "--rows", 100, "--save-samples")
        saved = _saved(figure_dir)
        assert [p for p in saved if p.startswith("preprocessing/")] == [
            "preprocessing/ecg_average_instances.png", "preprocessing/ecg_noise_example.png",
            "preprocessing/ecg_scalers_comparison.png"]
        assert len([p for p in saved if p.startswith("dataset/samples/")]) == 5
