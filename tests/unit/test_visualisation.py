"""The shared figure helpers of utils.visualisation draw the data they are given."""

import subprocess
import sys

import numpy as np

from utils import visualisation as vis


class TestV0_1Style:
    """The ``v0.1`` style applies where the ``v0.1`` figures are drawn, not on import."""

    def test_importing_the_module_changes_no_matplotlib_setting(self):
        code = ("import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
                "before = dict(plt.rcParams)\nimport utils.visualisation\n"
                "print(sorted(k for k, v in plt.rcParams.items() if before[k] != v))")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             check=True)
        assert out.stdout.strip() == "[]"

    def test_helpers_draw_in_the_style_and_leave_the_settings(self, figure_dir, pyplot):
        pyplot.rcParams["font.size"] = 9
        vis.plot_confusion_matrix(np.array([[3, 1], [1, 3]]), show=False, filename="s.png")
        annotation = pyplot.gcf().axes[0].texts[0]
        assert annotation.get_fontsize() == vis.V0_1_STYLE["font.size"]
        assert pyplot.rcParams["font.size"] == 9

    def test_scripts_switch_their_process_to_the_style(self, pyplot):
        with pyplot.rc_context(vis.V0_1_STYLE):
            expected = dict(pyplot.rcParams)
        vis.use_v0_1_style()
        assert dict(pyplot.rcParams) == expected


class TestConfusionMatrix:
    def test_rows_are_normalised_and_the_classes_named(self, figure_dir, pyplot):
        counts = np.array([[8, 2, 0, 0, 0], [1, 9, 0, 0, 0], [0, 0, 5, 5, 0],
                           [0, 0, 0, 10, 0], [0, 0, 0, 0, 4]])
        vis.plot_confusion_matrix(counts, show=False, filename="cm.png")
        assert (figure_dir / "metrics" / "cm.png").is_file()
        ax = pyplot.gcf().axes[0]
        shown = np.asarray(ax.collections[0].get_array()).reshape(5, 5)
        np.testing.assert_allclose(shown, counts / counts.sum(axis=1, keepdims=True))
        assert [t.get_text() for t in ax.get_xticklabels()] == vis.classes

    def test_two_classes_are_normal_and_arrhythmia(self, figure_dir, pyplot):
        vis.plot_confusion_matrix(np.array([[3, 1], [1, 3]]), show=False, filename="b.png")
        ax = pyplot.gcf().axes[0]
        assert [t.get_text() for t in ax.get_yticklabels()] == vis.binary_classes


def test_genes_are_named_after_the_model_variables():
    assert (vis.gene_full_name("I"), vis.gene_full_name("A")) == ("luxI", "aiiA")
    assert vis.gene_full_name("He") == "He"


class TestSweep:
    """plot_sweep: a score against the levels of a swept setting, one dot per seed."""

    SCORES = {0.0: [0.80, 0.82, 0.84], 0.5: [0.70, 0.74], 0.1: [0.90, 0.91, 0.92]}
    MEANS = [0.82, 0.91, 0.72]  # at the sorted levels 0, 0.1, 0.5

    def _draw(self, pyplot, scores=SCORES, **kwargs):
        _, ax = pyplot.subplots()
        return vis.plot_sweep(ax, scores, **kwargs)

    def test_levels_are_evenly_spaced_and_labelled_with_their_values(self, pyplot):
        ax = self._draw(pyplot)
        assert list(ax.get_xticks()) == [0, 1, 2]
        assert [t.get_text() for t in ax.get_xticklabels()] == ["0", "0.1", "0.5"]
        assert ax.get_xlim() == (-0.5, 2.5)

    def test_every_seed_is_a_dot_near_its_level(self, pyplot):
        dots = self._draw(pyplot, jitter=0.15).collections[0].get_offsets()
        np.testing.assert_allclose(dots[:, 1], [0.80, 0.82, 0.84, 0.90, 0.91, 0.92, 0.70, 0.74])
        assert np.all(np.abs(dots[:, 0] - [0, 0, 0, 1, 1, 1, 2, 2]) <= 0.15)

    def test_the_line_joins_the_means_and_the_band_spans_one_sd(self, pyplot):
        for ddof in (1, 0):
            ax = self._draw(pyplot, ddof=ddof)
            np.testing.assert_allclose(ax.lines[0].get_ydata(), self.MEANS)
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


def test_save_figure_writes_each_format_and_closes_the_figure(figure_dir, pyplot):
    fig, ax = pyplot.subplots()
    ax.plot([0, 1], [1, 0])
    paths = vis.save_figure(fig, "line", topic="checks")
    assert [p.relative_to(figure_dir).as_posix() for p in paths] == ["checks/line.png",
                                                                     "checks/line.pdf"]
    assert all(p.stat().st_size > 0 for p in paths)
    assert not pyplot.fignum_exists(fig.number)
