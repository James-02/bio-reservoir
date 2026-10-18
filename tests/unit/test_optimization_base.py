"""Unit tests for optimization.base."""

import optuna
import pytest

from optimization import base


@pytest.fixture(autouse=True)
def _keep_logging(monkeypatch):
    """_optimize_study configures logging for worker processes."""
    monkeypatch.setattr(base, "setup_logging", lambda **kwargs: None)


def _objective(trial, **kwargs):
    return trial.suggest_float("x", 0.0, 1.0)


def _study():
    return optuna.create_study(direction="maximize",
                               sampler=optuna.samplers.RandomSampler(seed=0))


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
