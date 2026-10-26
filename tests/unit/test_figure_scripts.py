"""The figure scripts in scripts/ run end to end on small inputs (Agg backend, a small
dataset in place of the prepared one): each test runs a script as
``python scripts/<name>.py ARGS`` would, or calls its functions, and checks what it draws.
"""

import logging

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


class TestModelFigures:
    """scripts/figures/visualise_model.py draws a run config's model, built as a run builds it."""

    def test_the_script_draws_each_figure_within_the_journal_rules(self, small_design,
                                                                   figure_dir, ecg_data,
                                                                   run_script, caplog):
        with caplog.at_level(logging.WARNING, logger="utils.visualisation"):
            run_script("figures/visualise_model", small_design("centre"), "--gains", "1,2")
            run_script("figures/visualise_model", small_design("uncoupled"),
                       "--plots", "coupling")
        assert _saved(figure_dir) == ["model/toy-centre-array-beat0-I.png",
                                      "model/toy-centre-colony-beat0.png",
                                      "model/toy-centre-coupling.png",
                                      "model/toy-uncoupled-coupling.png"]
        assert caplog.messages == []

    def test_an_unknown_plot_is_an_error(self, small_design, run_script):
        with pytest.raises(SystemExit, match="Unknown plots"):
            run_script("figures/visualise_model", small_design("centre"), "--plots",
                       "topology")


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
