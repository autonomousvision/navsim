"""Evaluate exported candidate trajectories without importing a model."""

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from navsim.common.dataclasses import Trajectory
from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.candidate_pdm_score import score_candidates, summarize_candidates
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="NPZ: tokens [N], poses [N,K,8,3], selected_index [N]")
    parser.add_argument("--metric-cache-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing paths are rejected")
    args = parser.parse_args()
    with np.load(args.predictions, allow_pickle=False) as data:
        tokens, poses, selected = data["tokens"], data["poses"], data["selected_index"]
    if tokens.ndim != 1 or tokens.dtype.kind != "U" or len(tokens) == 0 or len(set(tokens)) != len(tokens):
        raise ValueError("tokens must be nonempty, unique Unicode scene identifiers")
    if poses.ndim != 4 or poses.shape[0] != len(tokens) or poses.shape[1] == 0 or poses.shape[2:] != (8, 3):
        raise ValueError("poses must have shape [N, K, 8, 3], with K > 0")
    if not np.isfinite(poses).all():
        raise ValueError("poses contain non-finite values")
    if selected.shape != (len(tokens),) or selected.dtype.kind not in "iu":
        raise ValueError("selected_index must be an integer array of shape [N]")
    if ((selected < 0) | (selected >= poses.shape[1])).any():
        raise ValueError("selected_index is outside the candidate range")
    loader = MetricCacheLoader(args.metric_cache_path)
    missing = set(tokens) - set(loader.tokens)
    if missing:
        raise ValueError(f"Metric cache is missing {len(missing)} requested tokens")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    candidate_sampling = TrajectorySampling(num_poses=8, interval_length=0.5)
    simulator, scorer = PDMSimulator(sampling), PDMScorer(sampling)
    summaries = []
    with (args.output_dir / "candidates.csv").open("w", newline="") as candidate_file:
        writer = None
        for token, scene_poses, index in zip(tokens, poses, selected):
            results = score_candidates(
                loader.get_from_token(str(token)),
                [Trajectory(p.astype(np.float64), candidate_sampling) for p in scene_poses],
                sampling, simulator, scorer,
            )
            for candidate_index, result in enumerate(results):
                row = dict(token=str(token), candidate_index=candidate_index, **asdict(result))
                if writer is None:
                    writer = csv.DictWriter(candidate_file, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
            summaries.append(dict(token=str(token), selected_index=int(index), **summarize_candidates(results, int(index))))
    with (args.output_dir / "scenes.csv").open("w", newline="") as scene_file:
        writer = csv.DictWriter(scene_file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    # Written only after every requested scene succeeds; partial CSVs are not a completed run.
    summary = dict(
        num_scenes=len(tokens), num_candidates=poses.shape[1], score_scale="0..1",
        scope="NAVSIM v1 candidate diagnostics; oracle is not a leaderboard score",
        **{key: float(np.mean([row[key] for row in summaries])) for key in ("selected", "mean", "oracle", "gap")},
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
