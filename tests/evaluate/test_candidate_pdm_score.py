"""Contract tests and optional real-cache single-call equivalence test."""
import lzma
import os
import pickle
from dataclasses import asdict
from unittest.mock import patch

import numpy as np
import pytest

from navsim.common.dataclasses import PDMResults, Trajectory
from navsim.evaluate.candidate_pdm_score import score_candidates, summarize_candidates
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


def result(score):
    return PDMResults(1.0, 1.0, score, 1.0, 1.0, 1.0, score)


def candidates(count):
    sampling = TrajectorySampling(num_poses=8, interval_length=0.5)
    return [Trajectory(np.zeros((8, 3)) + i, sampling) for i in range(count)]


@pytest.mark.parametrize("count", [1, 3, 64])
def test_independent_calls_and_order(count):
    trajectories = candidates(count)
    cache, sampling, simulator, scorer = [object() for _ in range(4)]
    expected = [result(i / count) for i in range(count)]
    with patch("navsim.evaluate.candidate_pdm_score.pdm_score", side_effect=expected) as call:
        actual = score_candidates(cache, trajectories, sampling, simulator, scorer)
    assert actual == expected
    assert call.call_count == count
    for i, args in enumerate(call.call_args_list):
        assert args.args[0] is cache
        assert args.args[1] is trajectories[i]
        assert args.args[2:] == (sampling, simulator, scorer)


def test_summary_is_uniform_expectation_and_selected_gap():
    assert summarize_candidates([result(0.1), result(0.6), result(0.8)], 1) == pytest.approx(
        dict(selected=0.6, mean=0.5, oracle=0.8, gap=0.2)
    )
    assert summarize_candidates([result(0.4)], 0) == pytest.approx(dict(selected=0.4, mean=0.4, oracle=0.4, gap=0.0))


@pytest.mark.parametrize("index", [-1, 2, 0.5, True])
def test_invalid_selection(index):
    with pytest.raises(ValueError):
        summarize_candidates([result(0.1), result(0.8)], index)


def test_empty_and_nonfinite_inputs_fail_before_scoring():
    with pytest.raises(ValueError):
        score_candidates(None, [], None, None, None)
    with pytest.raises(ValueError):
        summarize_candidates([], 0)
    bad = candidates(2)
    bad[1].poses[0, 0] = np.nan
    with patch("navsim.evaluate.candidate_pdm_score.pdm_score") as call:
        with pytest.raises(ValueError):
            score_candidates(None, bad, None, None, None)
        call.assert_not_called()


@pytest.mark.parametrize("score", [np.nan, np.inf, -0.1, 1.1])
def test_invalid_output(score):
    with patch("navsim.evaluate.candidate_pdm_score.pdm_score", return_value=result(score)):
        with pytest.raises(ValueError):
            score_candidates(None, candidates(1), None, None, None)


def test_real_cache_matches_single_calls_and_is_order_independent():
    path = os.environ.get("NAVSIM_TEST_METRIC_CACHE")
    if not path:
        pytest.skip("Set NAVSIM_TEST_METRIC_CACHE to a trusted v1 metric_cache.pkl")
    with lzma.open(path, "rb") as file:
        cache = pickle.load(file)
    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    trajectories = candidates(3)
    for index, trajectory in enumerate(trajectories):
        trajectory.poses[:] = 0
        trajectory.poses[:, 0] = np.arange(1, 9) * (index + 1) * 0.5
    def evaluate(items):
        return score_candidates(cache, items, sampling, PDMSimulator(sampling), PDMScorer(sampling))
    expected = [pdm_score(cache, t, sampling, PDMSimulator(sampling), PDMScorer(sampling)) for t in trajectories]
    actual = evaluate(trajectories)
    reversed_results = evaluate(trajectories[::-1])[::-1]
    singleton = evaluate(trajectories[:1])
    for want, got, reverse in zip(expected, actual, reversed_results):
        np.testing.assert_allclose(list(asdict(want).values()), list(asdict(got).values()), rtol=0, atol=1e-12)
        np.testing.assert_allclose(list(asdict(want).values()), list(asdict(reverse).values()), rtol=0, atol=1e-12)
    assert asdict(singleton[0]) == asdict(actual[0])
