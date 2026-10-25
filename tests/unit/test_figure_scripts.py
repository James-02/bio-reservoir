"""The figure scripts in scripts/ run end to end on small inputs (Agg backend, no dataset).

Each test runs a script as ``python scripts/<name>.py ARGS`` would and checks the figures
it writes, so the scripts that draw the ``v0.1`` figures keep working.
"""

import numpy as np


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


class TestFoldDiagnostics:
    """scripts/analysis/analyse_fold_diagnostics.py. Matplotlib 3.11 removed the box plot's
    ``labels`` argument (now ``tick_labels``), so ``--plots`` stopped at its second figure."""

    def test_every_plot_is_saved(self, tmp_path, figure_dir, run_script):
        folds = _fold_files(tmp_path / "runs")
        assert run_script("analysis/analyse_fold_diagnostics", "--glob", folds, "--plots") == 0
        assert _saved(figure_dir / "diagnostics") == [
            "confidence_correct_vs_incorrect.png", "entropy_by_class.png",
            "per_class_recall.png", "states_pca.png", "states_pca_explained_variance.png"]
