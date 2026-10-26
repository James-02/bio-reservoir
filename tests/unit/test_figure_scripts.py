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


class TestFoldAnalysisFigures:
    """The fold-analysis scripts draw the ``v0.1`` confusion matrices from fold files."""

    def test_analyse_fold_results_saves_every_plot_type(self, tmp_path, figure_dir, run_script):
        folds = _fold_files(tmp_path / "runs")
        assert run_script("analysis/analyse_fold_results", "--glob", folds, "--plots") == 0
        assert _saved(figure_dir / "metrics" / "run") == ["confusion-matrix.png",
                                                          "states-pca.png"]

    def test_an_unknown_plot_type_is_an_error(self, tmp_path, figure_dir, run_script):
        folds = _fold_files(tmp_path / "runs")
        with pytest.raises(SystemExit, match="Unknown plot types"):
            run_script("analysis/analyse_fold_results", "--glob", folds, "--plots", "tsne")

    def test_the_states_of_each_class_have_its_name_and_colour(self, load_script, pyplot,
                                                               monkeypatch):
        from matplotlib.colors import to_hex

        from utils.visualisation import CLASS_COLOURS, CLASS_NAMES

        script = load_script("analysis/analyse_fold_results")
        drawn = []
        monkeypatch.setattr(script, "save_figure", lambda fig, name, **kw: drawn.append(fig))
        classes = np.repeat(np.arange(5), 4)
        states = np.random.default_rng(0).normal(size=(20, 6)) + classes[:, None]
        script.plot_state_pca(states, np.eye(5)[classes][:, None, :], "run/states-pca")
        ax = drawn[0].axes[0]
        assert [t.get_text() for t in ax.get_legend().get_texts()] == list(CLASS_NAMES)
        colours = [to_hex(c.get_facecolor()[0]) for c in ax.collections]
        assert colours == [CLASS_COLOURS[name].lower() for name in CLASS_NAMES]
        assert ax.get_xlabel().startswith("PC 1 (")


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
                                      "states/reservoir_heatmap_LuxI_nodes20_inst0.png",
                                      "topology/reservoir_topology_nodes20.png"]

    def test_topology_figures(self, figure_dir, run_script):
        run_script("figures/visualise_topology", "--units", 30)
        assert _saved(figure_dir) == ["topology/distance_histograms.png",
                                      "topology/spatial_topology.png",
                                      "topology/weight_vs_distance_matrix.png"]


class TestDatasetFigure:
    """scripts/figures/visualise_dataset.py draws the beats of a run config's task."""

    def test_each_class_has_its_mean_band_and_count(self, load_script):
        script = load_script("figures/visualise_dataset")
        beats = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [9.0, 9.0]])
        bands = script.class_bands(beats[:, :, None], np.eye(2)[[0, 0, 0, 1]][:, None, :])
        mean, low, high, count = bands[0]
        np.testing.assert_allclose(mean, [2.0, 3.0])
        np.testing.assert_allclose([low, high], [[0.4, 1.4], [3.6, 4.6]])
        assert (count, bands[1][-1]) == (3, 1)

    def test_one_row_per_class_within_the_journal_rules(self, load_script, pyplot):
        from utils.visualisation import CLASS_NAMES, STYLE, figure_problems

        script = load_script("figures/visualise_dataset")
        beats = np.random.default_rng(0).normal(size=(50, 187))
        with pyplot.style.context(STYLE):
            fig = script.draw(script.class_bands(beats, np.repeat(np.arange(5), 10)),
                              CLASS_NAMES, "ECG")
        assert [ax.texts[0].get_text() for ax in fig.axes] == [f"{name} (n = 10)"
                                                               for name in CLASS_NAMES]
        assert figure_problems(fig) == []

    def test_the_script_reads_the_task_of_a_run_config(self, tmp_path, figure_dir, ecg_data,
                                                       run_script, capsys):
        config = tmp_path / "toy.yaml"
        config.write_text("name: toy\ntask: {rows: 100}\nreservoir: {type: none}\n"
                          "readout: {classifier: RidgeCV}\n")
        run_script("figures/visualise_dataset", config)
        assert _saved(figure_dir) == ["dataset/toy-beats.png"]
        counts = [int(line.split()[1]) for line in capsys.readouterr().out.splitlines()[:5]]
        assert sum(counts) == 100


class TestHeterogeneityFigure:
    """scripts/figures/visualise_heterogeneity.py draws the ``v0.1`` heterogeneity figure from
    the run folders of scripts/run_heterogeneity.py."""

    RUNS = {"heterogeneity-cv0.000-seed100": 0.88, "heterogeneity-cv0.000-seed200": 0.90,
            "heterogeneity-cv0.100-seed100": 0.86, "heterogeneity-cv0.100-seed200-array7": 0.87,
            "heterogeneity-cv0.300-seed100": 0.84, "heterogeneity-cv0.300-seed200": 0.83,
            "another-run": 0.5}

    @classmethod
    def _runs(cls, folder):
        """Two folds per run, with macro F1 0.01 below and above the run's value."""
        from utils.analysis import Metrics
        from utils.results import FoldArtifact

        labels = np.eye(5)[:, None, :]
        for name, value in cls.RUNS.items():
            for k, f1 in enumerate((value - 0.01, value + 0.01)):
                metrics = Metrics(f1=f1, accuracy=f1, precision=f1, recall=f1, mcc=0.0, kappa=0.0,
                                  runtime=1.0, confusion_matrix=np.eye(5), class_metrics={},
                                  f1_macro=f1, f1_weighted=f1)
                FoldArtifact(trial_name=name, fold_index=k, Y_train=labels, Y_test=labels,
                             Y_pred=labels, metrics=metrics).save(results_dir=str(folder))
        return folder

    def test_each_run_is_its_mean_over_folds_grouped_by_cv(self, tmp_path, load_script):
        scores = load_script("figures/visualise_heterogeneity").run_scores(
            self._runs(tmp_path / "runs"))
        assert list(scores) == [0.0, 0.1, 0.3]
        np.testing.assert_allclose(scores[0.0], [0.88, 0.90])
        np.testing.assert_allclose(scores[0.1], [0.86, 0.87])
        np.testing.assert_allclose(scores[0.3], [0.84, 0.83])

    def test_the_figure_shows_the_runs_their_means_and_the_baseline(self, tmp_path, pyplot,
                                                                    load_script):
        script = load_script("figures/visualise_heterogeneity")
        ax = script.draw(script.run_scores(self._runs(tmp_path / "runs"))).axes[0]
        baseline, means = ax.lines
        assert list(baseline.get_ydata()) == pytest.approx([0.89, 0.89])
        np.testing.assert_allclose(means.get_ydata(), [0.89, 0.865, 0.835])
        assert ax.get_legend_handles_labels()[1] == [
            "Baseline F$_1$ = 0.890", "Individual seeds", "Mean F$_1$ (\u00b11\u03c3)"]

    def test_the_script_saves_the_figure_and_prints_the_means(self, tmp_path, figure_dir,
                                                              run_script, capsys):
        run_script("figures/visualise_heterogeneity", "--runs_dir", self._runs(tmp_path / "runs"))
        assert sorted(p.name for p in (figure_dir / "diagnostics").iterdir()) == [
            "heterogeneity_robustness.pdf", "heterogeneity_robustness.png"]
        assert "0.1     0.8650" in capsys.readouterr().out

    def test_a_folder_without_heterogeneity_runs_is_an_error(self, tmp_path, pyplot, run_script):
        (tmp_path / "runs").mkdir()
        with pytest.raises(SystemExit, match="No heterogeneity"):
            run_script("figures/visualise_heterogeneity", "--runs_dir", tmp_path / "runs")
