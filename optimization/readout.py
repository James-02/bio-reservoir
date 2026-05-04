
import argparse
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import optuna
import pandas as pd  # type: ignore

from optuna.visualization.matplotlib import (
    plot_slice,
    plot_param_importances,
    plot_edf,
)

from reservoirpy.nodes import ScikitLearnNode
from sklearn.linear_model import RidgeClassifier, LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from joblib import Parallel, delayed

from utils.analysis import compute_class_dicts, compute_mean_dicts, evaluate_performance
from utils.results import FoldArtifact
from optimization.base import (
    DEFAULT_SEED,
    research, plot_results, print_topn_trials,
    parse_show_sections, SHOW_DEFAULT,
    get_default_storage, build_study,
)
from optimization.environment import get_environment
from training.pipeline import fit_readout, predict as predict_readout
from training.profiling import ResourceMonitor

logger = logging.getLogger(__name__)


def _important_params(study: optuna.Study, max_n: int = 8) -> List[str]:
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


def _compute_group_stats(df, top_n: int) -> Dict[str, Any]:
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

    rows: List[Dict[str, Any]] = []

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
    os.makedirs("logs", exist_ok=True)
    storage = args.storage or get_default_storage(study_name)
    study = build_study(study_name=study_name, storage=storage, seed=int(args.seed))

    params: Dict[str, Any] = {
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


def _apply_fold_worker(
    clf_cls,
    hypers: Dict[str, Any],
    artifact: "FoldArtifact",
    results_dir: str,
) -> int:
    """Fit clf on one fold's states, update its NPZ file, return fold index.

    Module-level so joblib can pickle it across processes.
    """
    trained_states = artifact.train_states.reshape(artifact.train_states.shape[0], 1, -1).copy()
    Y_train        = artifact.Y_train.reshape(artifact.Y_train.shape[0], 1, -1).copy()
    tested_states  = artifact.test_states.reshape(artifact.test_states.shape[0], 1, -1).copy()
    Y_test         = artifact.Y_test.reshape(artifact.Y_test.shape[0], 1, -1).copy()

    import time as _time
    node = ScikitLearnNode(clf_cls, model_hypers=hypers)
    _t0 = _time.perf_counter()
    fit_readout(node, list(trained_states), list(Y_train))
    Y_pred_list = predict_readout(node, list(tested_states))
    Y_pred = np.stack(Y_pred_list)  # restore (N, 1, C)
    _runtime = round(_time.perf_counter() - _t0, 4)
    metrics = evaluate_performance(Y_test, Y_pred, time=_runtime)

    updated_hypers = dict(artifact.model_hypers)
    updated_hypers["model"] = clf_cls
    updated_hypers["model_hypers"] = hypers

    updated = FoldArtifact(
        trial_name=artifact.trial_name,
        fold_index=artifact.fold_index,
        train_states=artifact.train_states,
        Y_train=artifact.Y_train,
        test_states=artifact.test_states,
        Y_test=artifact.Y_test,
        Y_pred=Y_pred,
        metrics=metrics,
        model_hypers=updated_hypers,
    )
    updated.save(results_dir=results_dir, save_states=True)
    return artifact.fold_index


def cmd_apply_best(args: argparse.Namespace) -> None:
    """Re-run the best Optuna trial across all folds and update NPZ files.

    Re-fits the best readout classifier on each fold's saved reservoir states
    and overwrites ``Y_pred`` and ``metrics`` in the NPZ files while
    preserving all other stored content (states, Y_train, Y_test, etc.).

    Pass ``--processes N`` to fit folds in parallel using joblib (one process
    per fold).  Default is sequential (``--processes 1``).
    """

    study_name = args.study_name or getattr(args, "study", None)
    storage = args.storage or get_default_storage(study_name)
    study = optuna.load_study(study_name=study_name, storage=storage)

    readout_trial_number = getattr(args, "readout_trial", None)
    if readout_trial_number is not None:
        best_trial = study.trials[readout_trial_number]
    else:
        best_trial = study.best_trial
    classifier_name = best_trial.params["classifier"]
    processes = int(getattr(args, "processes", 1))

    logger.info(
        "Applying best trial %d (classifier=%s, f1=%.6f) to fold NPZ files "
        "(processes=%d).",
        best_trial.number,
        classifier_name,
        float(best_trial.value),
        processes,
    )

    seed = int(getattr(args, "seed", DEFAULT_SEED))
    clf, hypers = choose_classifier(best_trial, classifier_name, seed=seed)
    artifacts = FoldArtifact.load_trial(args.trial_name)

    for artifact in artifacts:
        if artifact.train_states is None or artifact.test_states is None:
            raise SystemExit(
                f"Fold {artifact.fold_index} of trial '{args.trial_name}' has no saved states.\n"
                "Re-run classification with --save_states to enable readout optimisation."
            )

    # Derive absolute results_dir from the first artifact's path so that
    # worker processes (which may have a different CWD) save to the right place.
    abs_results_dir = os.path.abspath(
        os.path.dirname(os.path.dirname(artifacts[0].path))
    )

    if processes > 1:
        logger.info("Running %d folds in parallel across %d processes.", len(artifacts), processes)
        Parallel(n_jobs=processes, backend="loky")(
            delayed(_apply_fold_worker)(clf, hypers, a, abs_results_dir) for a in artifacts
        )
    else:
        for artifact in artifacts:
            logger.info("Fitting best %s on fold %d", classifier_name, artifact.fold_index)
            _apply_fold_worker(clf, hypers, artifact, abs_results_dir)
            logger.info("Updated Y_pred and metrics in fold %d", artifact.fold_index)



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
    import datetime, time as _time

    trial.set_user_attr(
        "trial_start_utc",
        datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )
    monitor = ResourceMonitor(interval=0.5).start()
    _trial_t0 = _time.perf_counter()

    classifier_name = trial.suggest_categorical('classifier', kwargs['classifiers'])
    objective_average = str(kwargs.get("objective_average", "weighted"))

    artifacts = FoldArtifact.load_trial(kwargs['trial_name'])
    f1s: List[float] = []
    accuracies: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []
    mccs: List[float] = []
    kappas: List[float] = []
    fold_runtimes: List[float] = []
    fold_fit_times: List[float] = []
    fold_predict_times: List[float] = []
    fold_class_metrics: List[dict] = []
    fold_confusion_matrices: List[np.ndarray] = []

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
            fold_runtimes, fold_fit_times, fold_predict_times)
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
    trial.set_user_attr("metric_f1_std",    float(np.std(f1s, ddof=0)))
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
                        round(float(np.std(fold_fit_times, ddof=0)), 6))
    trial.set_user_attr("profile_fit_readout_total_s",
                        round(float(np.sum(fold_fit_times)), 6))
    trial.set_user_attr("profile_predict_mean_s",
                        round(float(np.mean(fold_predict_times)), 6))
    trial.set_user_attr("profile_predict_std_s",
                        round(float(np.std(fold_predict_times, ddof=0)), 6))
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

    # Macro/weighted averages (from averaged class_metrics using legacy helper)
    _class_metrics_avg = compute_mean_dicts(fold_class_metrics)
    for cls_key, cls_vals in _class_metrics_avg.items():
        if not isinstance(cls_vals, dict):
            continue
        if cls_key in ("macro avg", "weighted avg"):
            tag = cls_key.split()[0]
            for mk, mv in cls_vals.items():
                safe_mk = _CLASS_KEY_MAP.get(mk, mk.replace("-", "_"))
                trial.set_user_attr(f"metric_{tag}_{safe_mk}", mv)

    # Aggregated confusion matrix (summed across folds, stored as nested list)
    agg_cm = sum(fold_confusion_matrices)
    trial.set_user_attr("metric_confusion_matrix", np.asarray(agg_cm).tolist())

    # Store resource monitor results
    for k, v in monitor.stop().items():
        trial.set_user_attr(k, v)

    trial.set_user_attr(
        "trial_end_utc",
        datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )

    return mean_f1
