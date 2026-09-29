"""Independent candidate diagnostics for NAVSIM v1 PDM scoring.

Each candidate is evaluated against the reference trajectory separately. Passing
all candidates to score_proposals at once changes progress normalization.
"""

from dataclasses import asdict
from typing import Dict, List, Sequence

import numpy as np

from navsim.common.dataclasses import PDMResults, Trajectory
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


def score_candidates(
    metric_cache: MetricCache,
    candidates: Sequence[Trajectory],
    future_sampling: TrajectorySampling,
    simulator: PDMSimulator,
    scorer: PDMScorer,
) -> List[PDMResults]:
    """Return official per-candidate results in input order, on the [0, 1] scale.

    Reuses the loaded cache and evaluator objects, but calls pdm_score separately
    for every candidate. Execution is sequential; callers may parallelize scenes
    with separate worker-local simulators/scorers. This is not a speedup over K
    single-trajectory evaluations and does not change the benchmark protocol.
    """
    if len(candidates) == 0:
        raise ValueError("At least one candidate is required")
    # Validate all candidates before starting potentially expensive evaluation.
    for candidate in candidates:
        poses = np.asarray(candidate.poses)
        if poses.shape != (candidate.trajectory_sampling.num_poses, 3) or not np.isfinite(poses).all():
            raise ValueError("Candidate poses must be finite [num_poses, 3] arrays")
    results = []
    for candidate in candidates:
        result = pdm_score(metric_cache, candidate, future_sampling, simulator, scorer)
        values = np.asarray(list(asdict(result).values()), dtype=np.float64)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("Official scorer returned non-finite or out-of-range metrics")
        results.append(result)
    return results


def summarize_candidates(results: Sequence[PDMResults], selected_index: int) -> Dict[str, float]:
    """Summarize a scene on [0, 1]; oracle/mean are diagnostics, not submissions.

    Mean is the expected score under uniform random candidate selection, not a
    lower bound. Oracle uses evaluation information unavailable to the agent.
    """
    if len(results) == 0:
        raise ValueError("At least one result is required")
    if isinstance(selected_index, (bool, np.bool_)) or not isinstance(selected_index, (int, np.integer)):
        raise ValueError("selected_index must be an integer")
    if not 0 <= selected_index < len(results):
        raise ValueError("selected_index is outside the candidate range")
    scores = np.asarray([result.score for result in results], dtype=np.float64)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Scores must be finite and in [0, 1]")
    selected = float(scores[selected_index])
    oracle = float(scores.max())
    return dict(selected=selected, mean=float(scores.mean()), oracle=oracle, gap=oracle - selected)
