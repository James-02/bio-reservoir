"""Unit tests for optimization.base: running studies, the evaluation report and plots."""

import argparse
import logging

import matplotlib.pyplot as plt
import optuna
import pandas as pd
import pytest
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from optuna.trial import TrialState, create_trial

from optimization import base


@pytest.fixture(autouse=True)
def _keep_logging(monkeypatch):
    """_optimize_study configures logging for worker processes."""
    monkeypatch.setattr(base, "setup_logging", lambda **kwargs: None)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("BIORESERVOIR_CONFIG", str(tmp_path / "config.env"))
    monkeypatch.setenv("BIORESERVOIR_RESULTS_DIR", str(tmp_path / "results"))
    return tmp_path


@pytest.fixture
def saved(monkeypatch):
    """Record the file names plot_results saves instead of writing 800-dpi images."""
    names = []
    monkeypatch.setattr(base, "_save_and_close",
                        lambda fig, filename, also_pdf=False: names.append(filename))
    yield names
    plt.close("all")


def _objective(trial, **kwargs):
    return trial.suggest_float("x", 0.0, 1.0)


def _study():
    return optuna.create_study(direction="maximize",
                               sampler=optuna.samplers.RandomSampler(seed=0))


# Every kind of user attribute the objectives store, for trial values 0.9, 0.8, 0.85.
def _attrs(value):
    return {
        "metric_f1": value, "metric_accuracy": value - 0.01, "metric_runtime": 2.5,
        "metric_class_0_f1_mean": 0.9, "metric_class_0_f1_std": 0.01,
        "metric_class_1_f1_mean": 0.8, "metric_class_1_f1_std": 0.02,
        "metric_class_0_support_mean": 10.0, "metric_class_1_support_mean": 12.0,
        "metric_macro_f1": value, "metric_weighted_f1": value + 0.001,
        "metric_confusion_matrix": [[5, 1], [0, 6]],
        "fold_0_f1": value, "fold_1_f1": value - 0.02,
        "fold_0_runtime_s": 1.0, "fold_1_runtime_s": 1.5,
        "res_units": 10, "topo_type": "distance", "readout_classifier": "KNN",
        "dataset_instances": 40,
        "profile_train_reservoir_total_s": 2.0, "profile_train_reservoir_mean_s": 1.0,
        "profile_train_reservoir_std_s": 0.1, "profile_train_reservoir_calls_per_fold": 1,
        "profile_scaling_total_s": 0.2, "profile_scaling_mean_s": 0.1,
        "profile_scaling_std_s": 0.0, "profile_scaling_calls_per_fold": "n/a",
        "profile_instance_node_total_s": 0.5, "profile_instance_node_mean_s": 0.25,
        "profile_instance_node_std_s": 0.0, "profile_instance_node_calls_per_fold": 4,
        "fold_runtime_mean_s": 1.25, "fold_runtime_std_s": 0.35,
        "fold_runtime_min_s": 1.0, "fold_runtime_max_s": 1.5,
        "fold_runtime_fold_0_s": 1.0, "fold_runtime_fold_1_s": 1.5,
        "runtime_cpu_percent_mean": 90.0, "runtime_rss_mb_max": 100.0,
        "env_python_version": "3.12", "env_machine_label": "host1",
    }


def _trials(attrs=_attrs):
    study = optuna.create_study(direction="maximize")
    distributions = {"classifier": CategoricalDistribution(["KNN", "RF"]),
                     "knn_n_neighbors": IntDistribution(1, 25),
                     "rf_max_depth": IntDistribution(2, 50)}
    for value, params in [(0.9, {"classifier": "KNN", "knn_n_neighbors": 3}),
                          (0.8, {"classifier": "RF", "rf_max_depth": 10}),
                          (0.85, {"classifier": "KNN", "knn_n_neighbors": 5})]:
        study.add_trial(create_trial(
            params=params, distributions={k: distributions[k] for k in params},
            value=value, user_attrs=attrs(value)))
    return study


class TestResearch:
    def test_one_process_runs_every_trial(self):
        study = _study()
        assert base.research(study, 3, _objective, processes=1) == [study]
        assert len(study.trials) == 3

    def test_processes_default_to_one(self):
        study = _study()
        base.research(study, 2, _objective)
        assert len(study.trials) == 2

    @pytest.mark.parametrize("trials, processes, jobs", [(7, 3, 3), (500, 96, 96), (2, 4, 2)])
    def test_processes_share_out_every_requested_trial(self, monkeypatch, trials, processes,
                                                       jobs):
        """Each process used to run trials // processes, so 500 trials on 96 processes
        ran 480 and fewer trials than processes ran none."""
        created = []

        class InProcess:
            def __init__(self, n_jobs):
                created.append(n_jobs)

            def __call__(self, calls):
                return [fn(*args, **kwargs) for fn, args, kwargs in calls]

        monkeypatch.setattr(base.joblib, "Parallel", InProcess)
        study = _study()
        assert len(base.research(study, trials, _objective, processes=processes)) == jobs
        assert created == [processes]
        assert len(study.trials) == trials

    def test_failed_and_pruned_trials_do_not_stop_the_study(self):
        def objective(trial, **kwargs):
            if trial.number == 0:
                raise ValueError("a broken trial")
            if trial.number == 1:
                raise optuna.TrialPruned()
            return 0.5

        study = _study()
        base.research(study, 3, objective, processes=1)
        assert [t.state for t in study.trials] == [TrialState.FAIL, TrialState.PRUNED,
                                                   TrialState.COMPLETE]

    def test_options_reach_the_objective(self):
        seen = []
        base.research(_study(), 1, lambda trial, **kwargs: seen.append(kwargs) or 0.0,
                      processes=1, study_name="s", log_level=logging.INFO)
        assert seen == [{"study_name": "s", "log_level": logging.INFO}]


class TestEvaluateStudy:
    def test_top_trials_and_summaries_are_printed(self, capsys):
        frame = _trials().trials_dataframe()
        frame["user_attrs_seed"] = 7
        frame["user_attrs_runtime"] = 1.5
        base.evaluate_study(frame, objective_str="F1", top_n=2)
        out = capsys.readouterr().out
        assert out.count("=== Trial Rank") == 2
        assert "knn_n_neighbors : 3.0" in out and "Seed" in out and "Runtime" in out
        assert "Summary over top 2 trials:" in out and "F1 mean : 0.8750" in out
        assert "Summary over ALL 3 trials:" in out and "F1 max  : 0.9000" in out

    def test_the_report_can_go_to_the_log(self, capsys, caplog):
        with caplog.at_level(logging.INFO, logger="optimization.base"):
            base.evaluate_study(_trials(), top_n=1, use_print=False)
        assert capsys.readouterr().out == ""
        assert "=== Trial Rank 1 ===" in caplog.messages


class TestPrintTopnTrials:
    def test_every_section_is_reported(self, capsys):
        frame = _trials().trials_dataframe()
        base.print_topn_trials(frame, 2, "every section", group_col="params_classifier",
                               sections=set(base.SHOW_CHOICES), best_only_detail=True)
        out = capsys.readouterr().out
        for text in ("every section", "[per-fold metrics]",
                     "[per-class metrics (mean \u00b1 std across folds)]",
                     "[reservoir + topology]", "[readout]", "[dataset]",
                     "[pipeline timings]", "train_reservoir", "instance/node",
                     "Fold runtime:  mean=1.2500s", "[resource usage (CPU % / RSS MB)]",
                     "cross-trial summary (2 trials)", "[environment \u2014 trial 0]",
                     "Summary \u2014 top    2", "Summary \u2014 all     3"):
            assert text in out, text
        assert out.count("[per-fold metrics]") == 1  # detail for the best trial only
        assert "confusion_matrix" not in out

    def test_classifier_prefixes_are_stripped(self, capsys):
        frame = _trials().trials_dataframe()
        knn = frame[frame["params_classifier"] == "KNN"]
        base.print_topn_trials(knn, 5, "KNN only", sections={"params"}, strip_prefix="KNN")
        header = capsys.readouterr().out.splitlines()[4]
        assert "n_neighbors" in header and "knn_n_neighbors" not in header

    def test_missing_timings_and_resources_are_reported(self, capsys):
        frame = _trials(lambda value: {"metric_f1": value}).trials_dataframe()
        base.print_topn_trials(frame, 1, "bare", sections={"timings", "resources"})
        out = capsys.readouterr().out
        assert "(none \u2014 re-run research with --profile)" in out
        assert "(none \u2014 psutil may not be installed)" in out

    def test_class_tables_without_spreads_are_reported(self, capsys):
        frame = _trials(lambda value: {
            "metric_class_0_f1": 0.9, "metric_class_1_f1": 0.7,
            "metric_class_0_support": 10.0, "metric_class_1_support": 12.0,
            "metric_macro_f1": value}).trials_dataframe()
        base.print_topn_trials(frame, 1, "legacy", sections={"metrics"})
        out = capsys.readouterr().out
        assert "[per-class metrics]" in out and "macro" in out
        assert "0.8000" in out  # the mean over classes

    def test_frames_without_values_print_nothing(self, capsys):
        base.print_topn_trials(pd.DataFrame(), 3, "empty")
        base.print_topn_trials(pd.DataFrame({"number": [0]}), 3, "no values")
        assert capsys.readouterr().out == ""


class TestHelpers:
    def test_show_sections_resolve_aliases_and_drop_unknown_names(self, caplog):
        with caplog.at_level(logging.WARNING, logger="optimization.base"):
            sections = base.parse_show_sections("params, Environment, folds, bogus")
        assert sections == {"params", "env", "metrics"}
        assert any("bogus" in message for message in caplog.messages)

    @pytest.mark.parametrize("column, label", [
        ("user_attrs_metric_f1", "f1"), ("user_attrs_env_python", "python"),
        ("params_knn_p", "knn_p"), ("user_attrs_job_id", "job_id"), ("number", "number"),
    ])
    def test_labels_drop_attribute_prefixes(self, column, label):
        assert base._pretty_label(column) == label

    @pytest.mark.parametrize("value, expected", [
        (1.5, 1.5), ("2", 2.0), (float("nan"), -1.0), ("text", -1.0), (None, -1.0),
    ])
    def test_floats_fall_back_to_the_default(self, value, expected):
        assert base._get_float(pd.Series({"key": value}), "key", -1.0) == expected
        assert base._get_float(pd.Series(dtype=float), "missing", -1.0) == -1.0

    def test_common_arguments(self):
        parser = argparse.ArgumentParser()
        base.add_common_args(parser)
        args = parser.parse_args([])
        assert (args.study_name, args.storage, args.archive) == (None, None, None)
        assert (args.log_level, args.verbosity) == ("INFO", 0)


class TestBuildStudy:
    @pytest.mark.parametrize("sampler, expected", [
        ("tpe", optuna.samplers.TPESampler), ("random", optuna.samplers.RandomSampler),
    ])
    def test_samplers(self, sampler, expected):
        study = base.build_study(study_name="s", storage=None, sampler_override=sampler)
        assert isinstance(study.sampler, expected)
        assert study.direction == optuna.study.StudyDirection.MAXIMIZE

    def test_grid_studies_need_a_grid(self):
        study = base.build_study(study_name="g", storage=None,
                                 study_config={"sampler": "grid", "_grid": {"x": [1, 2]}})
        assert isinstance(study.sampler, optuna.samplers.GridSampler)
        with pytest.raises(ValueError, match="_grid"):
            base.build_study(study_name="g", storage=None, study_config={"sampler": "grid"})

    def test_unknown_samplers_are_rejected(self):
        with pytest.raises(ValueError, match="Unknown sampler"):
            base.build_study(study_name="s", storage=None, sampler_override="bayes")

    def test_existing_studies_are_loaded(self, tmp_path):
        storage = f"sqlite:///{tmp_path / 'study.db'}"
        first = base.build_study(study_name="s", storage=storage)
        first.optimize(_objective, n_trials=2)
        assert len(base.build_study(study_name="s", storage=storage).trials) == 2


class TestPlots:
    @staticmethod
    def _study(n_params=2):
        study = optuna.create_study(direction="maximize")
        names = [f"p{k}" for k in range(n_params)]
        for value in (0.1, 0.5, 0.9):
            study.add_trial(create_trial(
                params={name: value for name in names},
                distributions={name: FloatDistribution(0.0, 1.0) for name in names},
                value=value))
        return study

    def test_figures_are_saved_under_results(self, workspace):
        fig, ax = plt.subplots(figsize=(1, 1))
        base._save_and_close(fig, "tiny.png", also_pdf=True)
        folder = workspace / "results" / "figures" / "optimization"
        assert (folder / "tiny.png").is_file() and (folder / "tiny.pdf").is_file()
        plt.close(fig)

    def test_slice_plots_drop_fixed_parameters(self, saved):
        from optuna.visualization.matplotlib import plot_slice

        base.plot_results(self._study(), plot_slice, params=["p0", "fixed_window"],
                          filename="slice.png")
        assert saved == ["slice.png"]

    def test_wide_slice_plots_are_split_in_two(self, saved):
        from optuna.visualization.matplotlib import plot_slice

        study = self._study(n_params=10)
        base.plot_results(study, plot_slice, params=[f"p{k}" for k in range(10)],
                          filename="wide.png", max_params_per_plot=8)
        assert saved == ["wide-A.png", "wide-B.png"]

    def test_plots_without_params_are_drawn_without_them(self, saved):
        calls = []

        def study_only(study):
            calls.append(study)
            plt.figure()

        base.plot_results(self._study(), study_only, params=["p0"], filename="edf.png")
        assert len(calls) == 1 and saved == ["edf.png"]

    def test_plots_that_cannot_be_drawn_are_skipped(self, saved, caplog):
        def fails(**kwargs):
            raise ValueError("needs more trials")

        with caplog.at_level(logging.WARNING, logger="optimization.base"):
            base.plot_results(self._study(), fails, filename="none.png")
        assert saved == [] and any("Skipping plot" in m for m in caplog.messages)

    def test_optuna_titles_and_legends_are_tidied(self):
        fig, ax = plt.subplots()
        ax.set_title("Slice Plot")
        ax.text(0.5, 0.5, "Hyperparameter Importances")
        ax.plot([0, 1], [1, 2], label="Objective Value")
        ax.set_xticks([0, 1], ["a_long_parameter_name", "another_long_name"])
        ax.legend()
        fig.suptitle("Optuna")
        base._postprocess_optuna_figure(fig)
        assert ax.get_title() == ""
        assert [t.get_text() for t in ax.get_legend().get_texts()] == ["fANOVA"]
        assert "Hyperparameter Importances" not in [t.get_text() for t in ax.texts]
        assert ax.get_xticklabels()[0].get_rotation() == 45
        plt.close(fig)
