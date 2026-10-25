"""The shared figure helpers of utils.visualisation draw the data they are given."""

import subprocess
import sys

import numpy as np
import pytest

from utils import visualisation as vis


def _one_hot(labels, n_classes):
    return np.eye(n_classes)[labels]


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


class TestFoldMetrics:
    def test_one_bar_per_fold_and_metric(self, figure_dir, pyplot):
        folds = [{"accuracy": 0.8, "f1": 0.7, "mcc": 0.6, "kappa": 0.5 - 0.1 * k}
                 for k in range(3)]
        vis.plot_metrics_across_folds(folds, show=False, filename="folds.png")
        assert (figure_dir / "metrics" / "folds.png").is_file()
        heights = [bar.get_height() for bar in pyplot.gcf().axes[0].patches]
        assert heights == pytest.approx([0.8] * 3 + [0.7] * 3 + [0.6] * 3 + [0.5, 0.4, 0.3])

    def test_precision_recall_and_f1_of_each_class(self, figure_dir, pyplot):
        report = {str(c): {"precision": 0.1 * c, "recall": 0.2, "f1-score": 0.3, "support": 8}
                  for c in range(5)}
        report["accuracy"] = 0.5
        vis.plot_class_metrics(report, show=False, filename="classes.png")
        assert (figure_dir / "metrics" / "classes.png").is_file()
        ax = pyplot.gcf().axes[0]
        heights = [bar.get_height() for bar in ax.patches]
        assert heights == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4] + [0.2] * 5 + [0.3] * 5)
        assert [t.get_text() for t in ax.get_xticklabels()] == vis.classes


class TestTsne:
    def test_predictions_are_marked_correct_or_incorrect(self, figure_dir, pyplot):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 5, 40)
        y_pred = y_true.copy()
        y_pred[:5] = (y_pred[:5] + 1) % 5
        vis.plot_tsne_clustering(_one_hot(y_pred, 5) + rng.normal(0, 0.05, (40, 5)),
                                 _one_hot(y_true, 5), show=False, filename="tsne.png")
        assert (figure_dir / "metrics" / "tsne.png").is_file()
        correct, incorrect = pyplot.gcf().axes[0].collections[2:4]
        assert (len(correct.get_offsets()), len(incorrect.get_offsets())) == (35, 5)


def test_genes_are_named_after_the_model_variables():
    assert (vis.gene_full_name("I"), vis.gene_full_name("A")) == ("luxI", "aiiA")
    assert vis.gene_full_name("He") == "He"
