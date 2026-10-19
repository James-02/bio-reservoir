
import argparse
import logging
from typing import Any

import numpy as np
import optuna
import pandas as pd  # type: ignore
from optuna.visualization.matplotlib import (
    plot_edf,
    plot_param_importances,
    plot_slice,
)
from reservoirpy.nodes import ScikitLearnNode
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from optimization.base import (
    DEFAULT_SEED,
    SHOW_DEFAULT,
    build_study,
    fold_average_attrs,
    get_default_storage,
    parse_show_sections,
    plot_results,
    print_topn_trials,
    research,
)
from optimization.environment import get_environment
from training.pipeline import fit_readout
from training.pipeline import predict as predict_readout
from training.profiling import ResourceMonitor
from utils import config
from utils.analysis import compute_class_dicts, evaluate_performance, sample_sd
from utils.results import FoldArtifact

logger = logging.getLogger(__name__)


def _important_params(study: optuna.Study, max_n: int = 8) -> list[str]:
    """Return up to *max_n* non-fixed parameter names ranked by importance."""
    try:
        from optuna.importance import get_param_importances
        imps = get_param_importances(study)
        return [p for p in imps if not p.startswith("fixed_")][:max_n]
    except Exception:
        seen = set()
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                for pname in trial.params:
                    if not pname.startswith("fixed_"):
                        seen.add(pname)
        return sorted(seen)[:max_n]


def _compute_group_stats(df, top_n: int) -> dict[str, Any]:
    """Compute top-N and all-trials summary stats for a group of trials.

    Returns a dict with keys:
    - n_trials
    - top_mean, top_std, top_min, top_max
    - all_mean, all_std, all_min, all_max
    """
    # Drop NaNs and sort by value descending
    vals_all = df["value"].astype(float).dropna()
    n_trials = int(len(vals_all))
    if n_trials == 0:
        return {
            "n_trials": 0,
            "top_mean": np.nan,
            "top_std": np.nan,
            "top_min": np.nan,
            "top_max": np.nan,
            "all_mean": np.nan,
            "all_std": np.nan,
            "all_min": np.nan,
            "all_max": np.nan,
        }

    vals_sorted = vals_all.sort_values(ascending=False)
    vals_top = vals_sorted.head(int(top_n))

    def _s(x: "np.ndarray") -> float:
        return float(np.std(x, ddof=0))

    return {
        "n_trials": n_trials,
        "top_mean": float(vals_top.mean()),
        "top_std": _s(vals_top.values),
        "top_min": float(vals_top.min()),
        "top_max": float(vals_top.max()),
        "all_mean": float(vals_all.mean()),
        "all_std": _s(vals_all.values),
        "all_min": float(vals_all.min()),
        "all_max": float(vals_all.max()),
    }


def _print_classifier_summary_table(df, top_n: int) -> None:
    """Print a compact table of summary metrics per classifier and overall.

    The table includes, for each classifier and for ALL trials:
    - number of trials
    - mean/std/min/max over top-N trials
    - mean/std/min/max over all trials
    """
    if "value" not in df.columns:
        return

    rows: list[dict[str, Any]] = []

    # Global summary over all classifiers
    stats_all = _compute_group_stats(df, top_n)
    stats_all["classifier"] = "ALL"
    rows.append(stats_all)

    # Per-classifier summaries
    if "params_classifier" in df.columns:
        for clf in sorted(c for c in df["params_classifier"].dropna().unique()):
            df_clf = df[df["params_classifier"] == clf]
            stats = _compute_group_stats(df_clf, top_n)
            stats["classifier"] = str(clf)
            rows.append(stats)

    if not rows:
        return

    tab = pd.DataFrame(rows)
    # Order columns for readability
    col_order = [
        "classifier",
        "n_trials",
        "top_mean",
        "top_std",
        "top_min",
        "top_max",
        "all_mean",
        "all_std",
        "all_min",
        "all_max",
    ]
    tab = tab[col_order]

    # Pretty print as a plain-text table.
    print(f"\n=== Summary table (F1 over top {int(top_n)} and all trials) ===")
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def cmd_research(args: argparse.Namespace) -> None:
    study_name = args.study_name or getattr(args, "study", None)
    storage = args.storage or get_default_storage(study_name)
    study = build_study(study_name=study_name, storage=storage, seed=int(args.seed))

    params: dict[str, Any] = {
        "classifiers": list(args.classifiers),
        "study_name": study_name,
        "job_id": int(args.job_id),
        "trial_name": args.trial_name,
        "seed": int(args.seed),
    }

    # Pass objective_average if specified on the CLI (--primary_average).
    primary_avg = getattr(args, "primary_average", None)
    if primary_avg:
        params["objective_average"] = primary_avg

    # Uses optimization.base.research which handles parallelism.
    research(
        study,
        int(args.trials),
        objective,
        processes=int(args.processes),
        **params,
    )


def cmd_evaluate(args: argparse.Namespace) -> None:
    study_name = args.study_name or getattr(args, "study", None)
    if not study_name:
        raise SystemExit("--study_name (or --study) is required for evaluate.")
    storage = args.storage or get_default_storage(study_name, must_exist=True)
    study = optuna.load_study(study_name=study_name, storage=storage)
    df = study.trials_dataframe()
    sections = parse_show_sections(getattr(args, "show", None) or SHOW_DEFAULT)
    classifiers_filter = getattr(args, "classifiers", None)

    # ── Environment (once, at the top) ──────────────────────────────────────
    if "env" in sections:
        best_trial = study.best_trial
        env_attrs = {k: v for k, v in best_trial.user_attrs.items()
                     if k.startswith("env_")}
        if env_attrs:
            sep = "=" * 72
            print(f"\n{sep}")
            print(f"  Environment — trial {best_trial.number}")
            print(sep)
            maxw = max(len(k.replace('env_', '')) for k in env_attrs)
            for k in sorted(env_attrs):
                label = k.replace('env_', '')
                print(f"    {label:<{maxw}} : {env_attrs[k]}")
        # Don't repeat env inside print_topn_trials
        sections = sections - {"env"}

    # Global top-N table across all classifiers — classifier kept as a column
    # so you can see the mix in a single view.
    print_topn_trials(
        df, int(args.top_n),
        title=f"ALL classifiers — {study_name}",
        group_col="params_classifier",
        sections=sections,
        best_only_detail=True,
    )

    # Per-classifier top-N tables with narrower param sets.
    per_clf_sections = sections - {"env"}
    if "params_classifier" in df.columns:
        all_clfs = sorted({c for c in df["params_classifier"].dropna().unique()})
        if classifiers_filter:
            filter_set = {c.upper() for c in classifiers_filter}
            all_clfs = [c for c in all_clfs if c.upper() in filter_set]
        for clf in all_clfs:
            df_clf = df[df["params_classifier"] == clf]
            if df_clf.empty:
                continue
            print_topn_trials(
                df_clf, int(args.top_n),
                title=f"classifier={clf}",
                sections=per_clf_sections,
                strip_prefix=clf,
                best_only_detail=True,
            )

    # Summary table with aggregate stats per classifier and overall.
    _print_classifier_summary_table(df, int(args.top_n))

    # Plot diagnostics.
    overview_arg = args.plots if args.plots is not None else "slice"
    wanted = {p.strip().lower() for p in overview_arg.split(",") if p.strip()}
    wanted.discard("none")
    classifier_plots_arg = getattr(args, "classifier_plots", None) or ""
    classifier_wanted = {
        p.strip().lower() for p in classifier_plots_arg.split(",") if p.strip()
    }
    classifier_wanted.discard("none")
    if not wanted and not classifier_wanted:
        return

    max_params = int(getattr(args, "max_params", 8))
    top_params = _important_params(study, max_n=max_params)

    if "slice" in wanted:
        params = top_params if top_params else None
        plot_results(
            study,
            plot_slice,
            params=params,
            filename=f"{study_name}-slice.png",
        )
    if "importance" in wanted:
        params = top_params if top_params else None
        plot_results(
            study,
            plot_param_importances,
            params=params,
            filename=f"{study_name}-importance.png",
        )

    # --- Global EDF over all trials --------------------------------------
    # Gives an overall view of the score distribution across the study.
    if "edf" in wanted:
        plot_results(
            study,
            plot_edf,
            filename=f"{study_name}-edf.png",
        )

    # Discover classifier families actually present in the study.
    classifier_values = {
        t.params.get("classifier")
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and "classifier" in t.params
    }
    classifier_values.discard(None)

    for clf_name in sorted(classifier_values):
        # Filter study on this classifier only via the `target`/`target_name`
        # hooks in plot_*; simpler is to pass explicit param list scoped to
        # that classifier. We prefix output filenames with the classifier.
        label = str(clf_name)
        safe_label = label.replace(" ", "_")

        # Collect parameter names used for this classifier (excluding
        # `classifier` itself).
        clf_param_names = set()
        for t in study.trials:
            if t.state != optuna.trial.TrialState.COMPLETE:
                continue
            if t.params.get("classifier") != clf_name:
                continue
            for pname in t.params.keys():
                if pname != "classifier":
                    clf_param_names.add(pname)
        if not clf_param_names:
            continue

        params_list = sorted(clf_param_names)

        if "slice" in classifier_wanted:
            plot_results(
                study,
                plot_slice,
                params=params_list,
                filename=f"{study_name}-{safe_label}-slice.png",
            )
        if "importance" in classifier_wanted:
            plot_results(
                study,
                plot_param_importances,
                params=params_list,
                filename=f"{study_name}-{safe_label}-importance.png",
            )


def cmd_apply_best(args: argparse.Namespace) -> None:
    """Fit the best (or a chosen) readout trial on every fold of a run with saved states.

    Goes through the run framework (``training.run``): the readout is fitted on each
    fold's saved training states and scored on its test states, and the result is a new
    run, ``results/runs/<study>-best/``, with predictions, metrics and the states. The
    source run (``--trial_name``: a run under ``results/runs``, a registry run id or
    ``archive:<version>/<run>``) is not modified.
    """
    from training.run import execute

    study_name = args.study_name or getattr(args, "study", None)
    readout_trial = getattr(args, "readout_trial", None)
    run_config = {
        "name": f"{study_name}-best",
        "description": (f"Readout of {study_name} on the states of {args.trial_name} "
                        "(optimize apply-best)."),
        "seed": int(getattr(args, "seed", DEFAULT_SEED)),
        "reservoir": {"states_from": args.trial_name},
        "readout": {"study": study_name, "archive": getattr(args, "archive", None),
                    "trial": "best" if readout_trial is None else readout_trial,
                    "storage": getattr(args, "storage", None)},
        "outputs": {"save_states": True},
    }
    run_dir = execute(run_config, out=config.runs_dir() / f"{study_name}-best",
                      processes=int(getattr(args, "processes", 1)), allow_dirty=True)
    logger.info("Saved the readout run in %s", run_dir)


def choose_classifier(trial: optuna.trial.Trial, classifier_name: str, seed: int = DEFAULT_SEED, binary: bool = False):
    hypers = None
    clf = None

    if classifier_name == "Ridge":
        hypers = {
            "alpha": trial.suggest_float('ridge_alpha', 1e-5, 1e2, log=True),
            "fit_intercept": trial.suggest_categorical('ridge_fit_intercept', [True, False]),
            "tol": trial.suggest_float('ridge_tol', 1e-5, 1e-1, log=True),
            "solver": trial.suggest_categorical('ridge_solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']),
            "max_iter": 2000,
        }
        clf = RidgeClassifier

    elif classifier_name == "Bayes":
        hypers = {
            "var_smoothing": trial.suggest_float('bayes_var_smoothing', 1e-12, 1e-4),
        }
        clf = GaussianNB

    elif classifier_name == "LR":
        hypers = {
            "tol": trial.suggest_float('lr_tol', 1e-5, 1e-1, log=True),
            "C": trial.suggest_float('lr_C', 1e-5, 1e5, log=True),
            "fit_intercept": trial.suggest_categorical('lr_fit_intercept', [True, False]),
            "intercept_scaling": trial.suggest_float('lr_intercept_scaling', 0.1, 10),
            "solver": trial.suggest_categorical('lr_solver', ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']),
            "max_iter": 1000,
            "class_weight": "balanced",
            "random_state": seed,
        }
        clf = LogisticRegression

    elif classifier_name == "Perceptron":
        hypers = {
            "penalty": trial.suggest_categorical('perceptron_penalty', [None, 'l2', 'l1', 'elasticnet']),
            "alpha": trial.suggest_float('perceptron_alpha', 1e-5, 1e-2, log=True),
            "l1_ratio": trial.suggest_float('perceptron_l1_ratio', 0, 1),
            "fit_intercept": trial.suggest_categorical('perceptron_fit_intercept', [True, False]),
            "tol": trial.suggest_float('perceptron_tol', 1e-5, 1e-1, log=True),
            "eta0": trial.suggest_float('perceptron_eta0', 1e-4, 1, log=True),
            "random_state": seed,
        }
        clf = Perceptron

    elif classifier_name == "SVM":
        kernel = trial.suggest_categorical('svc_kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
        hypers = {
            "C": trial.suggest_float('svc_C', 1e-5, 1e4, log=True),
            "kernel": kernel,
            "gamma": trial.suggest_categorical('svc_gamma', ['scale', 'auto']) if kernel != 'linear' else 'scale',
            "tol": trial.suggest_float('svc_tol', 1e-5, 1e-2, log=True),
            "class_weight": "balanced",
            "random_state": seed,
        }
        if kernel == 'poly':
            hypers["degree"] = trial.suggest_int('svc_degree', 2, 5)
        clf = SVC

    elif classifier_name == "MLP":
        hypers = {
            "hidden_layer_sizes": trial.suggest_int('mlp_hidden_layer_sizes', 100, 300),
            "activation": trial.suggest_categorical('mlp_activation', ['logistic', 'tanh', 'relu']),
            "solver": trial.suggest_categorical('mlp_solver', ['lbfgs', 'adam']),
            "alpha": trial.suggest_float('mlp_alpha', 1e-5, 1e-3, log=True),
            "learning_rate": trial.suggest_categorical('mlp_learning_type', ['constant', 'invscaling']),
            "learning_rate_init": trial.suggest_float('mlp_learning_rate_init', 1e-5, 1e-3, log=True),
            "power_t": trial.suggest_float('mlp_power_t', 0.1, 1.0),
            "tol": trial.suggest_float('mlp_tol', 1e-5, 1e-1, log=True),
            "momentum": trial.suggest_float('mlp_momentum', 0.1, 0.9),
            "epsilon": trial.suggest_float('mlp_epsilon', 1e-8, 1e-6, log=True),
            "max_iter": 1000,
            "random_state": seed,
        }
        clf = MLPClassifier

    elif classifier_name == "KNN":
        hypers = {
            "n_neighbors": trial.suggest_int('knn_n_neighbors', 1, 25),
            "weights": trial.suggest_categorical('knn_weights', ['uniform', 'distance']),
            "p": trial.suggest_float('knn_p', 1.0, 4.0),
        }
        clf = KNeighborsClassifier

    elif classifier_name == "DT":
        hypers = {
            "criterion": trial.suggest_categorical('dt_criterion', ["entropy", "log_loss"]),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            "class_weight": "balanced",
            "random_state": seed,
        }
        clf = DecisionTreeClassifier

    elif classifier_name == "RF":
        hypers = {
                "n_estimators": trial.suggest_int('rf_n_estimators', 150, 350),
                "criterion": trial.suggest_categorical('rf_criterion', ["entropy", "log_loss"]),
                "oob_score": trial.suggest_categorical('rf_oob_score', [True, False]),
                "max_features": trial.suggest_categorical('rf_max_features', ['sqrt', 'log2']),
                "max_depth": trial.suggest_int('rf_max_depth', 10, 50),
                "min_samples_split": trial.suggest_int('rf_min_samples_split', 2, 20),
                "min_samples_leaf": trial.suggest_int('rf_min_samples_leaf', 1, 20),
                "class_weight": "balanced",
                "random_state": seed,
            }
        clf = RandomForestClassifier

    elif classifier_name == "GB":
        gb_losses = ['log_loss', 'exponential'] if binary else ['log_loss']
        hypers = {
            "loss": trial.suggest_categorical('gb_loss', gb_losses),
            "learning_rate": trial.suggest_float('gb_learning_rate', 0.001, 1.0),
            "n_estimators": trial.suggest_int('gb_n_estimators', 100, 500),
            "subsample": trial.suggest_float('gb_subsample', 0.1, 1.0),
            "criterion": trial.suggest_categorical('gb_criterion', ['friedman_mse', 'squared_error']),
            "random_state": seed,
        }
        clf = GradientBoostingClassifier

    else:
        raise ValueError(
            f"Unknown classifier '{classifier_name}'. "
            f"Supported: Ridge, SVM, MLP, KNN, DT, RF, GB."
        )

    return clf, hypers

def objective(trial, **kwargs):
    """Optuna objective for readout classifiers.

    In addition to returning the mean F1 across folds, this function can
    *update* the underlying fold NPZ files when a new best configuration is
    found. The comparison baseline is provided via ``kwargs['best_mean_f1']``
    (initialised from the existing metrics in those files).

    When a trial's mean F1 strictly exceeds ``best_mean_f1``, the function
    overwrites the ``Y_pred`` and ``metrics`` entries in each fold file with
    the predictions and metrics from this trial. Other saved content (states,
    labels, hyperparameters, etc.) is left untouched.
    """
    import datetime
    import time as _time

    trial.set_user_attr(
        "trial_start_utc",
        datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    monitor = ResourceMonitor(interval=0.5).start()
    _trial_t0 = _time.perf_counter()

    classifier_name = trial.suggest_categorical('classifier', kwargs['classifiers'])
    objective_average = str(kwargs.get("objective_average", "macro"))

    artifacts = FoldArtifact.load_trial(kwargs['trial_name'])
    f1s: list[float] = []
    accuracies: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    mccs: list[float] = []
    kappas: list[float] = []
    fold_runtimes: list[float] = []
    fold_fit_times: list[float] = []
    fold_predict_times: list[float] = []
    fold_class_metrics: list[dict] = []
    fold_confusion_matrices: list[np.ndarray] = []

    seed: int = int(kwargs.get("seed", DEFAULT_SEED))
    clf, hypers = choose_classifier(trial, classifier_name, seed=seed)

    if any(a.train_states is None or a.test_states is None for a in artifacts):
        raise SystemExit(
            f"Trial '{kwargs['trial_name']}' has folds with no saved states.\n"
            "Re-run classification with --save_states to enable readout optimisation."
        )

    # Store environment metadata
    for k, v in get_environment().items():
        trial.set_user_attr(f"env_{k}", v)

    # Store study/trial metadata
    trial.set_user_attr("study_name",    kwargs.get("study_name", ""))
    trial.set_user_attr("job_id",        kwargs.get("job_id", 0))
    trial.set_user_attr("trial_name",    kwargs.get("trial_name", ""))

    # Store readout configuration
    trial.set_user_attr("readout_classifier", classifier_name)
    trial.set_user_attr("readout_hypers",     str(hypers))

    for artifact in artifacts:
        trained_states = artifact.train_states.reshape(artifact.train_states.shape[0], 1, -1)
        Y_train = artifact.Y_train.reshape(artifact.Y_train.shape[0], 1, -1)
        tested_states = artifact.test_states.reshape(artifact.test_states.shape[0], 1, -1)
        Y_test = artifact.Y_test.reshape(artifact.Y_test.shape[0], 1, -1)

        node = ScikitLearnNode(clf, model_hypers=hypers)

        logger.debug("Fitting %s on fold %d", classifier_name, artifact.fold_index)
        _fold_t0 = _time.perf_counter()
        _fit_t0 = _time.perf_counter()
        fit_readout(node, list(trained_states), list(Y_train))
        _fit_s = _time.perf_counter() - _fit_t0

        logger.debug("Running %s on fold %d", classifier_name, artifact.fold_index)
        _predict_t0 = _time.perf_counter()
        Y_pred_list = predict_readout(node, list(tested_states))
        Y_pred = np.stack(Y_pred_list)  # restore (N, 1, C)
        _predict_s = _time.perf_counter() - _predict_t0
        _fold_runtime = round(_time.perf_counter() - _fold_t0, 4)
        metrics = evaluate_performance(
            Y_test, Y_pred, time=_fold_runtime, primary_average=objective_average
        )

        f1s.append(float(metrics.f1))
        accuracies.append(float(metrics.accuracy))
        precisions.append(float(metrics.precision))
        recalls.append(float(metrics.recall))
        mccs.append(float(metrics.mcc))
        kappas.append(float(metrics.kappa))
        fold_runtimes.append(_fold_runtime)
        fold_fit_times.append(_fit_s)
        fold_predict_times.append(_predict_s)
        fold_class_metrics.append(metrics.class_metrics)
        fold_confusion_matrices.append(metrics.confusion_matrix)

    mean_f1 = float(np.mean(f1s))
    trial_runtime = round(_time.perf_counter() - _trial_t0, 4)

    # Store per-fold metrics
    for i, (f1, acc, prec, rec, mcc, kap, rt, fit_t, pred_t) in enumerate(
        zip(f1s, accuracies, precisions, recalls, mccs, kappas,
            fold_runtimes, fold_fit_times, fold_predict_times, strict=True)
    ):
        trial.set_user_attr(f"fold_{i}_f1",          f1)
        trial.set_user_attr(f"fold_{i}_accuracy",    acc)
        trial.set_user_attr(f"fold_{i}_precision",   prec)
        trial.set_user_attr(f"fold_{i}_recall",      rec)
        trial.set_user_attr(f"fold_{i}_mcc",         mcc)
        trial.set_user_attr(f"fold_{i}_kappa",       kap)
        trial.set_user_attr(f"fold_{i}_runtime_s",   rt)
        trial.set_user_attr(f"fold_{i}_fit_s",       round(fit_t, 6))
        trial.set_user_attr(f"fold_{i}_predict_s",   round(pred_t, 6))

    # Store aggregate metrics
    trial.set_user_attr("metric_primary_average", objective_average)
    trial.set_user_attr("metric_f1",        mean_f1)
    trial.set_user_attr("metric_f1_std",    sample_sd(f1s))
    trial.set_user_attr("metric_f1_macro",  float(np.mean([m.get("macro avg", {}).get("f1-score", float("nan")) for m in fold_class_metrics])))
    trial.set_user_attr("metric_f1_weighted", float(np.mean([m.get("weighted avg", {}).get("f1-score", float("nan")) for m in fold_class_metrics])))
    trial.set_user_attr("metric_accuracy",  float(np.mean(accuracies)))
    trial.set_user_attr("metric_precision", float(np.mean(precisions)))
    trial.set_user_attr("metric_recall",    float(np.mean(recalls)))
    trial.set_user_attr("metric_precision_macro", float(np.mean([m.get("macro avg", {}).get("precision", float("nan")) for m in fold_class_metrics])))
    trial.set_user_attr("metric_precision_weighted", float(np.mean([m.get("weighted avg", {}).get("precision", float("nan")) for m in fold_class_metrics])))
    trial.set_user_attr("metric_recall_macro", float(np.mean([m.get("macro avg", {}).get("recall", float("nan")) for m in fold_class_metrics])))
    trial.set_user_attr("metric_recall_weighted", float(np.mean([m.get("weighted avg", {}).get("recall", float("nan")) for m in fold_class_metrics])))
    trial.set_user_attr("metric_mcc",       float(np.mean(mccs)))
    trial.set_user_attr("metric_kappa",     float(np.mean(kappas)))
    trial.set_user_attr("metric_runtime",   trial_runtime)
    trial.set_user_attr("metric_n_folds",   len(f1s))

    # Per-stage timing aggregates
    trial.set_user_attr("profile_fit_readout_mean_s",
                        round(float(np.mean(fold_fit_times)), 6))
    trial.set_user_attr("profile_fit_readout_std_s",
                        round(sample_sd(fold_fit_times), 6))
    trial.set_user_attr("profile_fit_readout_total_s",
                        round(float(np.sum(fold_fit_times)), 6))
    trial.set_user_attr("profile_predict_mean_s",
                        round(float(np.mean(fold_predict_times)), 6))
    trial.set_user_attr("profile_predict_std_s",
                        round(sample_sd(fold_predict_times), 6))
    trial.set_user_attr("profile_predict_total_s",
                        round(float(np.sum(fold_predict_times)), 6))

    # Per-class metrics: mean AND std across folds
    _CLASS_KEY_MAP = {"f1-score": "f1", "precision": "precision",
                      "recall": "recall", "support": "support"}
    _class_stats = compute_class_dicts(fold_class_metrics)
    for cls_key, cls_vals in _class_stats.items():
        for mk, mv in cls_vals.items():
            raw_name, stat = mk.rsplit("_", 1)  # e.g. "f1-score_mean"
            safe_mk = _CLASS_KEY_MAP.get(raw_name, raw_name.replace("-", "_"))
            trial.set_user_attr(f"metric_class_{cls_key}_{safe_mk}_{stat}", mv)

    # Macro/weighted averages over folds
    for key, value in fold_average_attrs(fold_class_metrics).items():
        trial.set_user_attr(key, value)

    # Aggregated confusion matrix (summed across folds, stored as nested list)
    agg_cm = sum(fold_confusion_matrices)
    trial.set_user_attr("metric_confusion_matrix", np.asarray(agg_cm).tolist())

    # Store resource monitor results
    for k, v in monitor.stop().items():
        trial.set_user_attr(k, v)

    trial.set_user_attr(
        "trial_end_utc",
        datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    return mean_f1
