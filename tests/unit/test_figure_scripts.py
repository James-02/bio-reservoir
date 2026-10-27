"""The figure scripts in scripts/ run end to end on small inputs (Agg backend, a small
dataset in place of the prepared one): each test runs a script as
``python scripts/<name>.py ARGS`` would, or calls its functions, and checks what it draws.
"""


import numpy as np
import pytest


def _saved(folder):
    """The figure files under ``folder``, as sorted relative paths."""
    return sorted(str(p.relative_to(folder)) for p in folder.rglob("*.png"))


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
    """scripts/analysis/analyse_fold_results.py: the tables and figures of a run's fold files."""

    def test_analyse_fold_results_saves_every_plot_type(self, tmp_path, figure_dir, run_script,
                                                        capsys):
        folds = _fold_files(tmp_path / "runs")
        assert run_script("analysis/analyse_fold_results", "--glob", folds, "--plots") == 0
        assert _saved(figure_dir / "metrics" / "run") == ["confusion-matrix.png",
                                                          "fold-metrics.png", "states-pca.png"]
        out = capsys.readouterr().out
        assert ("Per-fold metrics (F1, precision and recall: the weighted average over the "
                "classes):") in out
        printed = [line.removeprefix("Saved: ") for line in out.splitlines()
                   if line.startswith("Saved: ")]
        assert sorted(printed) == sorted(str(p) for p in figure_dir.rglob("*.png"))

    def test_each_plot_type_alone(self, tmp_path, figure_dir, run_script):
        folds = _fold_files(tmp_path / "runs")
        for plot, figure in (("metrics", "fold-metrics.png"), ("pca", "states-pca.png")):
            run_script("analysis/analyse_fold_results", "--glob", folds, "--plots", plot)
            assert figure in _saved(figure_dir / "metrics" / "run")
        assert len(_saved(figure_dir / "metrics" / "run")) == 2

    def test_an_unknown_plot_type_is_an_error(self, tmp_path, figure_dir, run_script):
        folds = _fold_files(tmp_path / "runs")
        with pytest.raises(SystemExit, match="Unknown plot types"):
            run_script("analysis/analyse_fold_results", "--glob", folds, "--plots", "states")

    def test_older_fold_files_get_mcc_and_kappa_from_their_labels(self, tmp_path, load_script):
        from sklearn.metrics import cohen_kappa_score, matthews_corrcoef

        from utils.preprocessing import class_indices
        from utils.results import FoldArtifact

        _fold_files(tmp_path / "runs")
        artifacts = [FoldArtifact.load(p) for p in sorted((tmp_path / "runs" / "run")
                                                          .glob("fold-*.npz"))]
        stored = artifacts[1].metrics.mcc
        artifacts[0].metrics.mcc = artifacts[0].metrics.kappa = float("nan")  # as in v0.1 files
        load_script("analysis/analyse_fold_results")._fill_mcc_and_kappa(artifacts)
        y_true, y_pred = class_indices(artifacts[0].Y_test), class_indices(artifacts[0].Y_pred)
        assert artifacts[0].metrics.mcc == pytest.approx(matthews_corrcoef(y_true, y_pred))
        assert artifacts[0].metrics.kappa == pytest.approx(cohen_kappa_score(y_true, y_pred))
        assert artifacts[1].metrics.mcc == stored  # a file that has them keeps its own

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


class TestDDEFigure:
    """scripts/figures/visualise_dde.py: one colony of a design at several input scalings."""

    WARM, REST = np.ones((3, 4)), np.full((5, 4), 2.0)

    def test_a_row_per_scaling_under_the_colony_without_input(self, load_script, pyplot):
        script = load_script("figures/visualise_dde")
        fig = script.draw(self.WARM, self.REST, {0.05: self.REST, 0.5: self.REST},
                          np.arange(5.0), "Normal", 0.5)
        *rows, beat = fig.axes
        assert [ax.texts[0].get_text() for ax in rows] == ["Baseline (0 input)", "Scale=0.05",
                                                           "Scale=0.5 (config)"]
        assert [t.get_text() for t in rows[0].get_legend().get_texts()] == [
            "LuxI", "AiiA", "Warmup (0 input)"]
        assert [t.get_text() for t in beat.get_legend().get_texts()] == ["ECG Class: Normal",
                                                                         "Warmup (0 input)"]
        assert (beat.get_ylabel(), fig.get_supylabel()) == ("ECG Input",
                                                            "Genetic Biopixel States")

    def test_the_warm_up_is_dashed_and_the_beat_starts_at_0(self, load_script, pyplot):
        script = load_script("figures/visualise_dde")
        fig = script.draw(self.WARM, self.REST, {}, np.arange(5.0), "Normal")
        warm_up, driven = fig.axes[0].lines[:2]
        assert list(warm_up.get_xdata()) == [-3, -2, -1] and warm_up.get_linestyle() == "--"
        assert list(driven.get_xdata()) == [-1, 0, 1, 2, 3, 4]
        assert list(fig.axes[-1].lines[-1].get_ydata()) == [0.0, 0.0, 1.0, 2.0, 3.0, 4.0]

    def test_without_a_warm_up_nothing_is_dashed(self, load_script, pyplot):
        script = load_script("figures/visualise_dde")
        fig = script.draw(np.empty((0, 4)), self.REST, {}, np.arange(5.0), "Normal")
        assert [line.get_linestyle() for line in fig.axes[0].lines] == ["-", "--"]
        assert list(fig.axes[0].lines[0].get_xdata()) == [0, 1, 2, 3, 4]

    def test_the_script_draws_a_design(self, small_design, figure_dir, ecg_data, run_script,
                                       capsys):
        run_script("figures/visualise_dde", small_design("centre"), "--scales", "0.1,1",
                   "--warmup-steps", "3")
        assert _saved(figure_dir) == ["states/toy-centre-dde-beat0.png"]
        assert capsys.readouterr().out.startswith("Beat 0: ")


class TestStatesFigure:
    """scripts/figures/visualise_reservoir_states.py: the array's states and drive terms."""

    def test_a_row_per_colony_and_a_column_per_step(self, load_script, pyplot):
        script = load_script("figures/visualise_reservoir_states")
        states = np.arange(5 * 3 * 4, dtype=float).reshape(5, 3, 4)  # steps, colonies, vars
        fig = script.draw_states(states, "I")
        ax, colorbar = fig.axes
        np.testing.assert_array_equal(ax.images[0].get_array(), states[:, :, 1].T)
        assert (ax.get_xlabel(), ax.get_ylabel()) == ("Timestep", "Colony index")
        assert colorbar.get_ylabel() == "LuxI concentration"
        norm = script.draw_states(states, "I", log=True).axes[0].images[0].norm
        assert type(norm).__name__ == "LogNorm" and norm.vmin == np.percentile(states[:, :, 1], 1)

    def test_the_drive_terms_above_the_beat(self, load_script, pyplot):
        script = load_script("figures/visualise_reservoir_states")
        terms = np.array([[0.1, 0.2, 0.3], [0.2, 0.4, 0.6]])
        ax, ax_ecg = script.draw_drive(terms, np.array([1.0, 2.0]), "Fusion", 0.209,
                                       0.00186).axes
        assert [line.get_label().split(":")[0] for line in ax.lines] == ["Input term",
                                                                         "Recurrent term",
                                                                         "Drive"]
        assert ax.texts[0].get_text() == "$\\alpha_u = 0.21$,\u2002$\\alpha_r = 0.0019$"
        assert ax_ecg.get_legend().get_texts()[0].get_text() == "ECG: Fusion"
        assert ax.get_yscale() == "log"

    def test_without_a_recurrent_term_only_the_input_term_is_drawn(self, load_script, pyplot):
        script = load_script("figures/visualise_reservoir_states")
        terms = np.array([[0.1, 0.0, 0.1], [0.2, 0.0, 0.2]])
        ax = script.draw_drive(terms, np.array([1.0, 2.0]), "Normal", 0.155, 0.007).axes[0]
        assert len(ax.lines) == 1 and ax.texts[0].get_text() == r"$\alpha_u = 0.15$"

    def test_the_script_draws_a_design(self, small_design, figure_dir, ecg_data, run_script):
        run_script("figures/visualise_reservoir_states", small_design("abstract"), "--drive",
                   "--var", "He", "--beat", "2")
        assert _saved(figure_dir) == ["states/toy-abstract-drive-beat2.png",
                                      "states/toy-abstract-states-beat2-He.png"]

    def test_an_inducer_trace_is_labelled_as_one(self, small_design, load_script, pyplot,
                                                 monkeypatch, capsys):
        design = small_design("centre")
        config = design.with_name("inducer.yaml")
        config.write_text(f"base: {design.name}\n"
                          "task: {type: inducer, pattern: shape, n_per_class: 5}\n")
        script = load_script("figures/visualise_reservoir_states")
        drawn = []
        monkeypatch.setattr(script, "save_figure",
                            lambda fig, name, **kw: drawn.append(fig) or [])
        monkeypatch.setattr("sys.argv", ["visualise_reservoir_states", str(config), "--drive"])
        script.main()
        beat_axes = drawn[1].axes[1]
        assert beat_axes.get_ylabel() == "Inducer input"
        assert beat_axes.get_legend().get_texts()[0].get_text() == "Inducer: Square"
        assert capsys.readouterr().out.startswith("Beat 0: Square")


class TestInducerTasksFigure:
    """scripts/figures/visualise_inducer_tasks.py: example traces of every class of every task."""

    def test_a_panel_per_pattern_and_a_labelled_row_per_class(self, load_script, pyplot):
        from training import inducer

        script = load_script("figures/visualise_inducer_tasks")
        fig = script.draw("device", per_class=2, image_delay=30)
        assert len(fig.axes) == len(inducer.PATTERNS)
        for ax, pattern in zip(fig.axes, inducer.PATTERNS, strict=True):
            names = inducer.CLASS_NAMES[pattern]
            assert [t.get_text() for t in ax.texts] == list(names)
            assert len(ax.lines) == 2 * len(names)
            assert len(ax.lines[0].get_xdata()) == 240 + 30

    def test_the_script_saves_the_figure(self, figure_dir, run_script):
        run_script("figures/visualise_inducer_tasks", "--per-class", "1")
        assert _saved(figure_dir) == ["inducer/inducer-tasks-ideal.png"]


class TestSlowSensingFigure:
    """scripts/figures/visualise_slow_sensing.py: slow-sensing scores from study trials."""

    @staticmethod
    def _trials():
        import pandas as pd

        rows = [{"task.pattern": pattern, "task.image_delay": delay, "reservoir.seed": seed,
                 "f1": 0.9 - delay / 1000 + seed / 1e5, "state": "COMPLETE"}
                for pattern in ("order", "shape") for delay in (0, 60)
                for seed in (6337, 6338)]
        return pd.DataFrame(rows)

    def test_a_sweep_has_a_line_per_task_at_the_mean_over_seeds(self, load_script, pyplot):
        script = load_script("figures/visualise_slow_sensing")
        ax = script.sweep(self._trials(), "task.image_delay", "toy").axes[0]
        assert [t.get_text() for t in ax.get_legend().get_texts()] == ["order", "shape"]
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [0.9 + 0.063375, 0.84 + 0.063375])
        assert [t.get_text() for t in ax.get_xticklabels()] == ["0", "60"]
        assert ax.get_xlabel() == "Image delay after the pattern (min)"

    def test_a_comparison_has_a_marker_per_study_and_the_chance_of_each_task(self, load_script,
                                                                           pyplot):
        script = load_script("figures/visualise_slow_sensing")
        trials = self._trials()
        ax = script.compare({"a": trials, "b": trials}, "toy").axes[0]
        assert [t.get_text() for t in ax.get_legend().get_texts()] == ["Chance", "a", "b"]
        assert [t.get_text() for t in ax.get_xticklabels()] == ["order", "shape"]
        chance = ax.collections[0].get_segments()
        assert [segment[0][1] for segment in chance] == [0.5, 0.25]


class TestTopologyFigure:
    """scripts/figures/visualise_topology.py: the kernel and the colonies with their links."""

    WEIGHTS = np.array([[0.0, 2.0, -1.0], [2.0, 0.0, 0.5], [-1.0, 0.5, 0.0]])
    POSITIONS = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    def test_the_kernel_against_distance(self, load_script, pyplot):
        script = load_script("figures/visualise_topology")
        fig = script.draw_kernel(np.array([0.1, 0.5]), np.array([0.9, 0.0]), 0.03, 0.25, 8.2)
        ax = fig.axes[0]
        np.testing.assert_array_equal(ax.collections[0].get_offsets(), [[0.1, 0.9], [0.5, 0.0]])
        assert [line.get_xdata()[0] for line in ax.lines] == [0.03, 0.25]
        assert ax.get_legend_handles_labels()[1] == ["~1 diameter (mean NN)", "~8.2 diameters"]
        assert (ax.get_xlabel(), ax.get_ylabel()) == ("Distance", "Weight (raw)")

    def test_positive_links_are_solid_and_negative_ones_dashed(self, load_script, pyplot):
        from scipy import sparse

        script = load_script("figures/visualise_topology")
        ax = script.draw_topology(self.POSITIONS, sparse.coo_matrix(self.WEIGHTS), 1).axes[0]
        assert [t.get_text() for t in ax.get_legend().get_texts()] == ["colony",
                                                                       "excitation (+)",
                                                                       "inhibition (-)"]
        positive, negative = ax.collections[1:]
        assert len(positive.get_segments()) == 2 and len(negative.get_segments()) == 1

    def test_diffusive_links_exchange_ahl_and_no_coupling_has_none(self, load_script, pyplot):
        from scipy import sparse

        script = load_script("figures/visualise_topology")
        weights = sparse.coo_matrix(np.abs(self.WEIGHTS))
        ax = script.draw_topology(self.POSITIONS, weights, 4, diffusive=True).axes[0]
        assert [t.get_text() for t in ax.get_legend().get_texts()] == ["colony", "AHL exchange"]
        assert len(ax.collections[1].get_segments()) == 6
        ax = script.draw_topology(self.POSITIONS, None, 3).axes[0]
        assert len(ax.collections) == 1 and len(ax.get_legend().get_texts()) == 1

    def test_the_script_draws_a_kernel_only_for_a_distance_topology(self, small_design,
                                                                    figure_dir, run_script):
        run_script("figures/visualise_topology", small_design("abstract"), "--units", "25")
        run_script("figures/visualise_topology", small_design("centre"))
        assert _saved(figure_dir) == ["topology/toy-abstract-25-kernel.png",
                                      "topology/toy-abstract-25-topology.png",
                                      "topology/toy-centre-topology.png"]


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

    def test_a_beat_with_noise_added(self, load_script, pyplot):
        script = load_script("figures/visualise_dataset")
        ax = script.draw_noise(np.array([1.0, 2.0]), np.array([0.5, -0.5]), 0.3).axes[0]
        assert [line.get_label() for line in ax.lines] == ["Training Data",
                                                           "Noise ($\\sigma$=0.3)",
                                                           "Augmented Data"]
        np.testing.assert_array_equal(ax.lines[2].get_ydata(), [1.5, 1.5])

    def test_the_scaling_of_the_runs_is_marked(self, load_script, pyplot):
        script = load_script("figures/visualise_dataset")
        scaled = {name: np.arange(3.0) * k for k, (name, *_) in enumerate(script.SCALERS)}
        ax = script.draw_scalers(scaled, "standard").axes[0]
        assert [line.get_label() for line in ax.lines] == ["Min-max (reference)",
                                                           "Z-score (standard) [used]",
                                                           "Z-score (per-seq)"]

    def test_the_script_reads_the_task_of_a_run_config(self, tmp_path, figure_dir, ecg_data,
                                                       run_script, capsys):
        config = tmp_path / "toy.yaml"
        config.write_text("name: toy\ntask: {rows: 100}\nreservoir: {type: none}\n"
                          "readout: {classifier: RidgeCV}\n")
        run_script("figures/visualise_dataset", config, "--beat", "3", "--noise-std", "0.1")
        assert _saved(figure_dir) == ["dataset/toy-beats.png", "dataset/toy-noise-beat3.png",
                                      "dataset/toy-scalers-beat3.png"]
        counts = [int(line.split()[1]) for line in capsys.readouterr().out.splitlines()[:5]]
        assert sum(counts) == 100


class TestHeterogeneityFigure:
    """scripts/figures/visualise_heterogeneity.py: the design of Supplementary Fig. S10, from
    the heterogeneity runs or from a study that sweeps a heterogeneity setting."""

    RUNS = {"heterogeneity-cv0.000-seed100": 0.88, "heterogeneity-cv0.000-seed200": 0.90,
            "heterogeneity-cv0.100-seed100": 0.86, "heterogeneity-cv0.100-seed200-array7": 0.87,
            "heterogeneity-cv0.300-seed100": 0.84, "heterogeneity-cv0.300-seed200": 0.83,
            "another-run": 0.5}
    DENSITY, KINETIC = ("reservoir.params.heterogeneity_density_cv",
                        "reservoir.params.param_heterogeneity_cv")

    @classmethod
    def _runs(cls, folder):
        """Two folds per run, with macro F1 0.01 below and above the run's value."""
        from utils.analysis import Metrics
        from utils.results import FoldArtifact

        labels = np.eye(5)[:, None, :]
        for name, value in cls.RUNS.items():
            for k, f1 in enumerate((value - 0.01, value + 0.01)):
                metrics = Metrics(f1=f1, accuracy=f1, precision=f1, recall=f1, mcc=0.0,
                                  kappa=0.0, runtime=1.0, confusion_matrix=np.eye(5),
                                  class_metrics={}, f1_macro=f1, f1_weighted=f1)
                FoldArtifact(trial_name=name, fold_index=k, Y_train=labels, Y_test=labels,
                             Y_pred=labels, metrics=metrics).save(results_dir=str(folder))
        return folder

    @classmethod
    def _trials(cls):
        import pandas as pd

        rows = [{"state": "FAIL", cls.DENSITY: 0.0, cls.KINETIC: 0.0, "f1": None}]
        for kinetic in (0.0, 0.065):
            for density in (0.0, 0.1, 0.3):
                for seed in (1, 2, 3):
                    rows.append({"state": "COMPLETE", cls.DENSITY: density, cls.KINETIC: kinetic,
                                 "f1": 0.85 + kinetic - density / 10 + seed / 1000})
        return pd.DataFrame(rows)

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
        fig = script.draw(script.run_scores(self._runs(tmp_path / "runs")), band=(0.1, 0.3))
        ax = fig.axes[0]
        baseline, means = ax.lines
        assert list(baseline.get_ydata()) == pytest.approx([0.89, 0.89])
        np.testing.assert_allclose(means.get_ydata(), [0.89, 0.865, 0.835])
        assert ax.get_legend_handles_labels()[1] == [
            "Baseline F$_1$ = 0.890", "Individual seeds", "Mean F$_1$ (\u00b11\u03c3)"]
        assert ax.texts[0].get_text() == "Biological\nrange"
        assert tuple(fig.get_size_inches()) == (5.5, 3.8)

    def test_the_script_draws_the_runs_and_prints_the_means(self, tmp_path, figure_dir,
                                                            run_script, capsys):
        run_script("figures/visualise_heterogeneity", "--runs_dir", self._runs(tmp_path / "runs"))
        assert _saved(figure_dir) == ["diagnostics/heterogeneity_robustness.png"]
        assert "0.1     0.8650    0.0050  2" in capsys.readouterr().out

    def test_a_folder_without_heterogeneity_runs_is_an_error(self, tmp_path, pyplot, run_script):
        (tmp_path / "runs").mkdir()
        with pytest.raises(SystemExit, match="No heterogeneity"):
            run_script("figures/visualise_heterogeneity", "--runs_dir", tmp_path / "runs")

    def test_a_study_is_drawn_at_one_value_of_its_other_settings(self, load_script):
        script = load_script("figures/visualise_heterogeneity")
        scores = script.study_scores(self._trials(), self.DENSITY, {self.KINETIC: 0.065})
        assert list(scores) == [0.0, 0.1, 0.3]
        np.testing.assert_allclose(scores[0.1], [0.906, 0.907, 0.908])

    def test_the_script_draws_a_study_at_the_config_s_value(self, tmp_path, figure_dir,
                                                            load_script, monkeypatch, capsys):
        import sys

        config = tmp_path / "density.yaml"
        config.write_text("name: toy-density\nreadout: {classifier: RidgeCV}\n"
                          "reservoir:\n  type: bioreservoir\n  params:\n"
                          f"    units: 4\n    param_heterogeneity_cv: 0.065\nstudy:\n  search:\n"
                          f"    {self.DENSITY}: {{type: categorical, choices: [0.0, 0.1, 0.3]}}\n"
                          f"    {self.KINETIC}: {{type: categorical, choices: [0.0, 0.065]}}\n"
                          "    reservoir.seed: {type: categorical, choices: [1, 2, 3]}\n")
        script = load_script("figures/visualise_heterogeneity")
        monkeypatch.setattr(script, "trials_table", lambda cfg, storage: self._trials())
        for fix in ([], ["--fix", f"{self.KINETIC}=0"]):
            monkeypatch.setattr(sys, "argv", ["visualise_heterogeneity.py", str(config), *fix])
            script.main()
        assert _saved(figure_dir) == [
            "diagnostics/toy-density-param_heterogeneity_cv0.065.png",
            "diagnostics/toy-density-param_heterogeneity_cv0.png"]
        out = capsys.readouterr().out
        assert "0.1     0.9070    0.0010  3" in out and "0.1     0.8420    0.0010  3" in out


class TestFormats:
    """Every figure script saves PNGs, and PDFs or SVGs as well on request (``--formats``)."""

    SCRIPTS = ("figures/visualise_dataset", "figures/visualise_dde",
               "figures/visualise_reservoir_states", "figures/visualise_topology",
               "figures/visualise_heterogeneity", "analysis/analyse_fold_results")

    @pytest.mark.parametrize("script", SCRIPTS)
    def test_every_figure_script_has_the_option(self, script, run_script, capsys):
        with pytest.raises(SystemExit):
            run_script(script, "--help")
        assert "--formats FORMATS" in capsys.readouterr().out

    def test_a_png_unless_other_formats_are_asked_for(self, tmp_path, figure_dir, run_script):
        runs = TestHeterogeneityFigure._runs(tmp_path / "runs")

        def saved():
            return sorted(p.name for p in (figure_dir / "diagnostics").iterdir())

        run_script("figures/visualise_heterogeneity", "--runs_dir", runs)
        assert saved() == ["heterogeneity_robustness.png"]
        run_script("figures/visualise_heterogeneity", "--runs_dir", runs, "--formats", "pdf,svg")
        assert saved() == ["heterogeneity_robustness.pdf", "heterogeneity_robustness.png",
                           "heterogeneity_robustness.svg"]

    def test_an_unknown_format_is_an_error(self, run_script, capsys):
        with pytest.raises(SystemExit):
            run_script("figures/visualise_heterogeneity", "--formats", "jpg")
        assert "choose one or more of png, pdf, svg" in capsys.readouterr().err
