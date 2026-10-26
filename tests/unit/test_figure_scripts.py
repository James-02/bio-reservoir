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

    def test_the_script_reads_the_task_of_a_run_config(self, tmp_path, figure_dir, ecg_data,
                                                       run_script, capsys):
        config = tmp_path / "toy.yaml"
        config.write_text("name: toy\ntask: {rows: 100}\nreservoir: {type: none}\n"
                          "readout: {classifier: RidgeCV}\n")
        run_script("figures/visualise_dataset", config)
        assert _saved(figure_dir) == ["dataset/toy-beats.png"]
        counts = [int(line.split()[1]) for line in capsys.readouterr().out.splitlines()[:5]]
        assert sum(counts) == 100


class TestSweepFigure:
    """scripts/figures/visualise_sweep.py draws a study's score against what it searches."""

    CV, GAIN = "reservoir.params.param_heterogeneity_cv", "reservoir.params.input_scaling"

    @classmethod
    def _trials(cls):
        import pandas as pd

        rows = [{"trial": 0, "state": "FAIL", cls.GAIN: 1.0, cls.CV: 0.0, "reservoir.seed": 3,
                 "f1": None}]
        for gain in (1.0, 2.0):
            for cv in (0.0, 0.1, 0.3):
                for seed in (1, 2):
                    rows.append({"trial": len(rows), "state": "COMPLETE", cls.GAIN: gain,
                                 cls.CV: cv, "reservoir.seed": seed,
                                 "f1": 0.9 - cv / 10 - gain / 100 + seed / 1000})
        return pd.DataFrame(rows)

    def test_finished_trials_are_grouped_by_the_settings(self, load_script):
        scores = load_script("figures/visualise_sweep").sweep_scores(self._trials(), self.CV,
                                                                     self.GAIN)
        assert list(scores) == [1.0, 2.0] and list(scores[1.0]) == [0.0, 0.1, 0.3]
        np.testing.assert_allclose(scores[2.0][0.1], [0.871, 0.872])

    def test_one_line_for_each_value_of_the_other_setting(self, load_script, pyplot):
        from utils.visualisation import STYLE, figure_problems

        script = load_script("figures/visualise_sweep")
        with pyplot.style.context(STYLE):
            fig = script.draw(script.sweep_scores(self._trials(), self.CV, self.GAIN), self.CV,
                              self.GAIN, reference=0.0)
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert [t.get_text() for t in legend.get_texts()] == ["Mean at 0", "Seeds", "1.0",
                                                              "2.0"]
        assert legend.get_title().get_text() == "Input gain"
        assert ax.get_xlabel() == "Kinetic heterogeneity (CV)"
        assert list(ax.lines[0].get_ydata()) == pytest.approx([0.8915, 0.8915])
        assert figure_problems(fig) == []

    def _run(self, load_script, monkeypatch, tmp_path, search, *args):
        import sys

        config = tmp_path / "sweep.yaml"
        config.write_text("name: toy-sweep\nstudy:\n  search:\n" + "".join(
            f"    {path}: {{type: categorical, choices: [0, 1]}}\n" for path in search))
        script = load_script("figures/visualise_sweep")
        trials = self._trials()
        monkeypatch.setattr(script, "trials_table", lambda cfg, storage: trials)
        monkeypatch.setattr(sys, "argv", ["visualise_sweep.py", str(config), *args])
        script.main()

    def test_the_script_draws_the_study_of_a_config(self, tmp_path, figure_dir, load_script,
                                                    monkeypatch, capsys):
        self._run(load_script, monkeypatch, tmp_path, [self.GAIN, self.CV, "reservoir.seed"])
        assert _saved(figure_dir) == ["sweeps/toy-sweep.png"]
        rows = capsys.readouterr().out.splitlines()[1:7]
        assert [row.split()[:2] for row in rows[:2]] == [["1.0", "0.0"], ["1.0", "0.1"]]
        assert rows[0].split()[2:] == ["0.8915", "0.0007", "2"]

    def test_one_seed_per_point_is_a_score(self, load_script, pyplot):
        script = load_script("figures/visualise_sweep")
        trials = self._trials()
        one_seed = trials[(trials["reservoir.seed"] == 1) & (trials[self.GAIN] == 1.0)]
        fig = script.draw(script.sweep_scores(one_seed, self.CV), self.CV)
        assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == ["Score"]

    def test_a_third_setting_needs_by(self, tmp_path, figure_dir, load_script, monkeypatch):
        with pytest.raises(SystemExit, match="--by"):
            self._run(load_script, monkeypatch, tmp_path, ["a.b", self.GAIN, self.CV])
