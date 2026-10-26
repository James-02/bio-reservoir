"""utils.visualisation: the house style, its check of the journal's rules, and the shared plots."""

import argparse
import logging
import subprocess
import sys

import numpy as np
import pytest

from utils import visualisation as vis


class TestStyle:
    def test_importing_the_module_changes_no_matplotlib_setting(self):
        code = ("import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
                "before = dict(plt.rcParams)\nimport utils.visualisation\n"
                "print(sorted(k for k, v in plt.rcParams.items() if before[k] != v))")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             check=True)
        assert out.stdout.strip() == "[]"

    def test_the_style_is_the_v0_1_look(self, pyplot):
        with pyplot.style.context(vis.STYLE):
            rc = pyplot.rcParams
            assert rc["font.size"] == 15 and not rc["axes.grid"]
            assert (rc["xtick.direction"], rc["xtick.bottom"], rc["ytick.left"]) == ("out", True,
                                                                                     True)

    def test_each_class_has_its_own_okabe_ito_colour(self):
        assert set(vis.CLASS_COLOURS) == {*vis.CLASS_NAMES, *vis.BINARY_CLASS_NAMES}
        assert set(vis.CLASS_COLOURS.values()) <= set(vis.OKABE_ITO)
        assert len({vis.CLASS_COLOURS[name] for name in vis.CLASS_NAMES}) == 5
        assert vis.CLASS_COLOURS["Normal"] != vis.CLASS_COLOURS["Arrhythmia"]

    def test_class_names_follow_the_number_of_classes(self):
        assert vis.class_names(2) == ("Normal", "Arrhythmia")
        assert vis.class_names(5) == vis.CLASS_NAMES
        assert vis.class_names(3) == ("Normal", "Supraventricular", "Ventricular")


class TestFigureProblems:
    """The check measures a figure at its printed size: at most two columns, 7 in."""

    def test_a_large_figure_in_the_house_style_has_none(self, pyplot):
        with pyplot.style.context(vis.STYLE):
            fig, ax = pyplot.subplots(figsize=(10, 6))
            ax.plot([0, 1], [1, 0], label="A line")
            ax.set(xlabel="Time (min)", ylabel="LuxI")
            ax.legend()
        assert vis.printed_width(fig) == vis.DOUBLE_COLUMN
        assert vis.figure_problems(fig) == []

    def test_text_lines_and_height_are_measured_after_the_reduction(self, pyplot):
        fig, ax = pyplot.subplots(figsize=(14, 20))
        ax.plot([0, 1], [1, 0], lw=1.5)
        ax.set_xlabel("small", fontsize=7)
        assert vis.printed_width(fig) == vis.DOUBLE_COLUMN
        box = fig.get_tightbbox()  # cropped to about 11 x 16 in
        scale = vis.DOUBLE_COLUMN / box.width
        assert 0.6 < scale < 0.66
        assert vis.figure_problems(fig) == [
            f"it is {box.height * scale:.2f} in high; a page is 9.5 in",
            f"1 texts are under 5 pt, such as 'small' at {7 * scale:.1f} pt",
            f"1 lines are under 1 pt, the thinnest {1.5 * scale:.2f} pt"]

    def test_a_narrower_printed_width_can_be_asked_for(self, pyplot):
        fig, ax = pyplot.subplots(figsize=(7, 4))
        ax.set_xlabel("label", fontsize=8)
        assert vis.figure_problems(fig) == []  # printed as cropped, about 5.7 in wide
        scale = vis.SINGLE_COLUMN / fig.get_tightbbox().width
        assert vis.figure_problems(fig, vis.SINGLE_COLUMN)[0] == (
            f"1 texts are under 5 pt, such as 'label' at {8 * scale:.1f} pt")
        assert vis.figure_problems(fig, 8.0)[0] == "8.00 in is wider than a page (7 in)"


class TestSaveFigure:
    def test_a_png_by_default_and_the_figure_closed(self, figure_dir, pyplot):
        fig, ax = pyplot.subplots()
        ax.plot([0, 1], [1, 0])
        (png,) = vis.save_figure(fig, "line", topic="checks")
        assert png.relative_to(figure_dir).as_posix() == "checks/line.png"
        assert png.stat().st_size > 0 and not pyplot.fignum_exists(fig.number)

    def test_pdf_and_svg_keep_their_text_as_text(self, figure_dir, pyplot):
        fig, ax = pyplot.subplots()
        ax.set_xlabel("Time (min)")
        png, pdf, svg = vis.save_figure(fig, "text", formats=("png", "pdf", "svg"))
        content = pdf.read_bytes()
        assert b"/FontFile2" in content and b"/Type3" not in content
        assert "Time (min)" in svg.read_text()

    def test_the_png_has_600_dots_per_inch_at_the_printed_size(self, tmp_path, pyplot):
        fig, ax = pyplot.subplots(figsize=(14, 4))
        ax.plot([0, 1], [1, 0])
        (png,) = vis.save_figure(fig, "wide", folder=tmp_path)
        assert pyplot.imread(png).shape[1] == pytest.approx(vis.DPI * vis.DOUBLE_COLUMN, rel=0.01)

    def test_a_figure_that_breaks_the_rules_is_saved_with_a_warning(self, figure_dir, pyplot,
                                                                     caplog):
        fig, ax = pyplot.subplots(figsize=(6, 4))
        ax.set_title("small", fontsize=3)
        with caplog.at_level(logging.WARNING, logger="utils.visualisation"):
            (png,) = vis.save_figure(fig, "small")
        assert png.is_file()
        (message,) = caplog.messages
        assert message.startswith("Figure small, printed ")
        assert message.endswith("in wide: 1 texts are under 5 pt, such as 'small' at 3.0 pt.")

    def test_a_folder_can_be_given(self, tmp_path, pyplot):
        fig, _ = pyplot.subplots()
        (png,) = vis.save_figure(fig, "fig", folder=tmp_path / "run" / "figures", dpi=20)
        assert png == tmp_path / "run" / "figures" / "fig.png" and png.is_file()

    def test_matplotlib_is_given_python_floats(self, tmp_path, pyplot, monkeypatch):
        fig, _ = pyplot.subplots()
        given = []
        monkeypatch.setattr(fig, "savefig", lambda path, dpi, **kwargs: given.append(dpi))
        vis.save_figure(fig, "fig", folder=tmp_path)
        assert type(given[0]) is float and type(vis.printed_width(fig)) is float


class TestFormatsOption:
    def test_formats_are_read_from_a_comma_separated_list(self):
        assert vis.figure_formats("png") == ("png",)
        assert vis.figure_formats("png, PDF,svg,pdf") == ("png", "pdf", "svg")
        with pytest.raises(argparse.ArgumentTypeError, match="choose one or more"):
            vis.figure_formats("jpg")

    def test_scripts_save_pngs_unless_asked(self):
        parser = argparse.ArgumentParser()
        vis.add_formats_option(parser)
        assert parser.parse_args([]).formats == ("png",)
        assert parser.parse_args(["--formats", "pdf,svg"]).formats == ("pdf", "svg")


class TestConfusionMatrix:
    COUNTS = np.array([[8, 2, 0, 0, 0], [1, 9, 0, 0, 0], [0, 0, 5, 5, 0], [0, 0, 0, 10, 0],
                       [0, 0, 0, 0, 4]])

    def test_rows_are_divided_by_their_sums_and_the_classes_named(self, pyplot):
        ax = vis.plot_confusion_matrix(self.COUNTS)
        shares = self.COUNTS / self.COUNTS.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(ax.images[0].get_array(), shares)
        assert [t.get_text() for t in ax.texts[:3]] == ["0.80", "0.20", "0.00"]
        assert [t.get_text() for t in ax.get_xticklabels()] == list(vis.CLASS_NAMES)
        assert (ax.get_xlabel(), ax.get_ylabel()) == ("Predicted label", "True label")

    def test_two_classes_are_normal_and_arrhythmia(self, pyplot):
        ax = vis.plot_confusion_matrix(np.array([[3, 1], [1, 3]]))
        assert [t.get_text() for t in ax.get_yticklabels()] == ["Normal", "Arrhythmia"]

    def test_a_class_without_beats_is_a_row_of_zeros(self, pyplot):
        ax = vis.plot_confusion_matrix(np.array([[2, 0], [0, 0]]))
        np.testing.assert_array_equal(ax.images[0].get_array(), [[1, 0], [0, 0]])

    def test_alone_it_is_the_v0_1_figure_in_the_house_style(self, pyplot):
        pyplot.rcParams["font.size"] = 12
        ax = vis.plot_confusion_matrix(self.COUNTS)
        assert tuple(ax.figure.get_size_inches()) == (10, 8)
        assert ax.texts[0].get_fontsize() == vis.STYLE["font.size"]
        assert pyplot.rcParams["font.size"] == 12
        assert vis.figure_problems(ax.figure) == []

    def test_it_draws_on_the_axes_it_is_given(self, pyplot):
        fig, axes = pyplot.subplots(1, 2)
        assert vis.plot_confusion_matrix(np.eye(2), axes[1], colorbar=False) is axes[1]
        assert len(fig.axes) == 2 and len(axes[1].images) == 1


class TestSweep:
    """plot_sweep: a score against the levels of a swept setting, as in S10 of v0.1."""

    SCORES = {0.0: [0.80, 0.82, 0.84], 0.5: [0.70, 0.74], 0.1: [0.90, 0.91, 0.92]}
    MEANS = [0.82, 0.91, 0.72]  # at the sorted levels 0, 0.1, 0.5

    def _draw(self, pyplot, scores=SCORES, **kwargs):
        _, ax = pyplot.subplots()
        return vis.plot_sweep(scores, ax, **kwargs)

    def test_levels_are_evenly_spaced_and_labelled_with_their_values(self, pyplot):
        ax = self._draw(pyplot)
        assert list(ax.get_xticks()) == [0, 1, 2]
        assert [t.get_text() for t in ax.get_xticklabels()] == ["0", "0.1", "0.5"]
        assert ax.get_xlim() == (-0.5, 2.5)

    def test_every_seed_is_a_blue_dot_near_its_level(self, pyplot):
        dots = self._draw(pyplot, jitter=0.15).collections[0]
        offsets = dots.get_offsets()
        np.testing.assert_allclose(offsets[:, 1], [0.80, 0.82, 0.84, 0.90, 0.91, 0.92, 0.70,
                                                   0.74])
        assert np.all(np.abs(offsets[:, 0] - [0, 0, 0, 1, 1, 1, 2, 2]) <= 0.15)
        assert pyplot.matplotlib.colors.to_hex(dots.get_facecolor()[0]) == "#1f77b4"

    @pytest.mark.parametrize("ddof", [1, 0])
    def test_a_red_line_joins_the_means_and_the_band_spans_one_sd(self, pyplot, ddof):
        ax = self._draw(pyplot, ddof=ddof)
        np.testing.assert_allclose(ax.lines[0].get_ydata(), self.MEANS)
        assert ax.lines[0].get_color() == "#d62728"
        corners = ax.collections[1].get_paths()[0].vertices
        for x, level in enumerate((0.0, 0.1, 0.5)):
            sd = np.std(self.SCORES[level], ddof=ddof)
            for y in (self.MEANS[x] - sd, self.MEANS[x] + sd):
                assert np.isclose(corners, [x, y]).all(axis=1).any()

    def test_a_figure_is_the_same_every_time(self, pyplot):
        first, second = (self._draw(pyplot).collections[0].get_offsets() for _ in range(2))
        np.testing.assert_array_equal(first, second)

    def test_the_reference_line_and_the_legend(self, pyplot):
        ax = self._draw(pyplot, reference=0.82, reference_label="Without the setting")
        assert list(ax.lines[0].get_ydata()) == [0.82, 0.82]
        assert ax.get_legend_handles_labels()[1] == ["Without the setting", "Seeds",
                                                     "Mean (\u00b11 SD)"]

    def test_a_level_with_one_seed_has_a_mean_but_no_band(self, pyplot, recwarn):
        ax = self._draw(pyplot, scores={0.0: [0.8, 0.9], 1.0: [0.7]})
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [0.85, 0.7])
        assert not [w for w in recwarn.list if issubclass(w.category, RuntimeWarning)]

    def test_one_seed_per_level_draws_the_scores_without_dots(self, pyplot):
        from matplotlib.collections import PathCollection

        ax = self._draw(pyplot, scores={0.055397: [0.8], 0.110795: [0.85]})
        assert not [c for c in ax.collections if isinstance(c, PathCollection)]
        assert [t.get_text() for t in ax.get_xticklabels()] == ["0.0554", "0.111"]

    def test_levels_can_be_names_and_shared(self, pyplot):
        ax = self._draw(pyplot, scores={"final": [0.8, 0.82], "images:4": [0.85, 0.86]},
                        levels=["final", "images:2", "images:4"])
        assert [t.get_text() for t in ax.get_xticklabels()] == ["final", "images:2",
                                                                "images:4"]
        assert list(ax.lines[0].get_xdata()) == [0, 2]

    def test_alone_it_is_the_size_of_s10(self, pyplot):
        ax = vis.plot_sweep(self.SCORES)
        assert tuple(ax.figure.get_size_inches()) == (5.5, 3.8)
        assert vis.figure_problems(ax.figure) == []

    def test_categories_have_their_own_error_bars_and_nothing_joins_them(self, pyplot):
        from matplotlib.collections import PolyCollection

        scores = {"F1": [0.80, 0.84], "MCC": [0.70, 0.74, 0.72]}
        ax = self._draw(pyplot, scores=scores, levels=list(scores), connect=False)
        (bars,) = ax.containers
        means, _, (vertical,) = bars.lines
        np.testing.assert_allclose(means.get_ydata(), [0.82, 0.72])
        half = [np.ptp(segment[:, 1]) / 2 for segment in vertical.get_segments()]
        np.testing.assert_allclose(half, [np.std([0.80, 0.84], ddof=1),
                                          np.std([0.70, 0.74, 0.72], ddof=1)])
        assert means.get_linestyle() == "None"
        assert not [c for c in ax.collections if isinstance(c, PolyCollection)]  # no band
        assert [t.get_text() for t in ax.get_xticklabels()] == ["F1", "MCC"]


class TestFoldMetrics:
    """plot_fold_metrics: each fold's scores and the F1 of each class, as dots with the mean."""

    @staticmethod
    def _fold(y_true, y_pred, **scores):
        from sklearn.metrics import confusion_matrix

        return {"f1": 0.8, "accuracy": 0.9, "precision": 0.7, "recall": 0.6, "mcc": 0.5,
                "kappa": 0.4, **scores, "confusion_matrix": confusion_matrix(y_true, y_pred)}

    def test_every_score_and_the_f1_of_each_class_from_the_confusion_matrix(self, pyplot):
        from sklearn.metrics import f1_score

        rng = np.random.default_rng(0)
        labels = [(rng.integers(0, 5, 60), rng.integers(0, 5, 60)) for _ in range(3)]
        folds = [self._fold(t, p, f1=0.8 + k / 100) for k, (t, p) in enumerate(labels)]
        fig = vis.plot_fold_metrics(folds, "weighted")
        ax, ax_class = fig.axes
        assert [t.get_text() for t in ax.get_xticklabels()] == [
            "F1 (weighted)", "Accuracy", "Precision (weighted)", "Recall (weighted)", "MCC",
            "Cohen's \u03ba"]
        np.testing.assert_allclose(ax.containers[0].lines[0].get_ydata(),
                                   [0.81, 0.9, 0.7, 0.6, 0.5, 0.4])
        expected = np.mean([f1_score(t, p, average=None) for t, p in labels], axis=0)
        np.testing.assert_allclose(ax_class.containers[0].lines[0].get_ydata(), expected)
        assert [t.get_text() for t in ax_class.get_xticklabels()] == list(vis.CLASS_NAMES)
        assert ax.get_legend_handles_labels()[1] == ["Folds", "Mean (\u00b11 SD)"]
        assert not [line for line in ax.lines if line.get_linestyle() not in ("None", "")]

    def test_a_class_that_a_fold_neither_has_nor_predicts_is_left_out(self, pyplot):
        folds = [self._fold([0, 1, 1], [0, 1, 1]),
                 {**self._fold([0, 0], [0, 0]), "confusion_matrix": np.array([[2, 0], [0, 0]])}]
        ax_class = vis.plot_fold_metrics(folds).axes[1]
        np.testing.assert_allclose(ax_class.containers[0].lines[0].get_ydata(), [1.0, 1.0])
        assert len(ax_class.collections[0].get_offsets()) == 3  # the second fold's class 1 is out

    def test_folds_with_different_classes_get_the_scores_only(self, pyplot):
        folds = [self._fold([0, 1, 2], [0, 1, 2]), self._fold([0, 1], [0, 1])]
        fig = vis.plot_fold_metrics(folds)
        assert len(fig.axes) == 1 and fig.axes[0].get_title() == "All classes"
