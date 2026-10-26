"""The figure scripts on the frozen v0.1 runs (skipped where they are absent)."""

import numpy as np
import pytest

from utils import config

V0_1_RUNS = config.runs_dir("v0.1")


@pytest.mark.skipif(not (V0_1_RUNS / "heterogeneity-cv0.500-seed500").is_dir(),
                    reason="the v0.1 heterogeneity runs are not present")
def test_the_v0_1_heterogeneity_runs_give_the_published_values(load_script):
    scores = load_script("figures/visualise_heterogeneity").run_scores(V0_1_RUNS)
    assert list(scores) == [0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    assert [len(runs) for runs in scores.values()] == [5] * 8
    # Supplementary Fig. S10 and its text: F1 0.889 without heterogeneity and at CV 0.2,
    # and 0.885 at CV 0.5.
    means = [round(float(np.mean(scores[cv])), 3) for cv in (0.0, 0.2, 0.5)]
    assert means == [0.889, 0.889, 0.885]
