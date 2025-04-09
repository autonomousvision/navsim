import logging
import os
import traceback
import uuid
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union

import hydra
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.geometry.convert import relative_to_absolute_poses
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.utils.multithreading.worker_utils import worker_map
from omegaconf import DictConfig

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import PDMResults, SensorConfig
from navsim.common.dataloader import MetricCacheLoader, SceneFilter, SceneLoader
from navsim.common.enums import SceneFrameType
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.scoring.scene_aggregator import SceneAggregator
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex
from navsim.traffic_agents_policies.abstract_traffic_agents_policy import AbstractTrafficAgentsPolicy

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[pd.DataFrame]:
    """
    Helper function to run PDMS evaluation in.
    :param args: input arguments
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert (
        simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    if cfg.traffic_agents == "non_reactive":
        traffic_agents_policy: AbstractTrafficAgentsPolicy = instantiate(
            cfg.traffic_agents_policy.non_reactive, simulator.proposal_sampling
        )
    elif cfg.traffic_agents == "reactive":
        traffic_agents_policy: AbstractTrafficAgentsPolicy = instantiate(
            cfg.traffic_agents_policy.reactive, simulator.proposal_sampling
        )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        original_sensor_path=Path(cfg.original_sensor_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    pdm_results: List[pd.DataFrame] = []
    for idx, (token) in enumerate(tokens_to_evaluate):
        logger.info(
            f"Processing scenario {idx + 1} / {len(tokens_to_evaluate)} in thread_id={thread_id}, node_id={node_id}"
        )
        try:
            metric_cache = metric_cache_loader.get_from_token(token)
            agent_input = scene_loader.get_agent_input_from_token(token)
            if agent.requires_scene:
                scene = scene_loader.get_scene_from_token(token)
                trajectory = agent.compute_trajectory(agent_input, scene)
            else:
                trajectory = agent.compute_trajectory(agent_input)

            score_row, ego_simulated_states = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
                traffic_agents_policy=traffic_agents_policy,
            )
            score_row["valid"] = True
            score_row["log_name"] = metric_cache.log_name
            score_row["frame_type"] = metric_cache.scene_type
            score_row["start_time"] = metric_cache.timepoint.time_s
            end_pose = StateSE2(
                x=trajectory.poses[-1, 0],
                y=trajectory.poses[-1, 1],
                heading=trajectory.poses[-1, 2],
            )
            absolute_endpoint = relative_to_absolute_poses(metric_cache.ego_state.rear_axle, [end_pose])[0]
            score_row["endpoint_x"] = absolute_endpoint.x
            score_row["endpoint_y"] = absolute_endpoint.y
            score_row["start_point_x"] = metric_cache.ego_state.rear_axle.x
            score_row["start_point_y"] = metric_cache.ego_state.rear_axle.y
            score_row["ego_simulated_states"] = [ego_simulated_states]  # used for two-frames extended comfort

        except Exception:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row = pd.DataFrame([PDMResults.get_empty_results()])
            score_row["valid"] = False
        score_row["token"] = token

        pdm_results.append(score_row)
    return pdm_results


def infer_start_adjacent_mapping(score_df: pd.DataFrame, time_gap_threshold: float = 0.55) -> Dict[str, str]:
    """
    Infers an adjacent mapping from the score_df DataFrame by start time.
    Each current-token is mapped to its previous-token if they are adjacent.
    Used to create the two-frame extended comfort score (reversed direction).

    :param score_df: DataFrame containing at least 'token', 'log_name', 'start_time'.
    :param time_gap_threshold: Maximum allowed gap (in seconds) between two frames to
                               consider them "adjacent".
    :return: Dictionary mapping each current-token to one previous-token.
    """
    adjacent_mapping: Dict[str, str] = {}

    for log_name, group_df in score_df[score_df["frame_type"] == SceneFrameType.ORIGINAL].groupby("log_name"):
        group_df = group_df.sort_values(by="start_time").reset_index(drop=True)

        for i in range(1, len(group_df)):
            prev_row = group_df.iloc[i - 1]
            current_row = group_df.iloc[i]

            prev_token = prev_row["token"]
            current_token = current_row["token"]
            time_diff = current_row["start_time"] - prev_row["start_time"]

            if abs(time_diff) <= time_gap_threshold:
                adjacent_mapping[current_token] = prev_token

    return adjacent_mapping


def compute_final_scores(pdm_score_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute final scores for each row in pdm_score_df after updating
    the weighted metrics with two-frame extended comfort.

    If 'two_frame_extended_comfort' is NaN for a row, the corresponding
    metric and its weight are set to zero, effectively ignoring it
    during normalization.

    :param pdm_score_df: DataFrame containing PDM scores and metrics.
    :return: A new DataFrame with the computed final scores.
    """
    df = pdm_score_df.copy()

    two_frame_scores = df["two_frame_extended_comfort"].to_numpy()  # shape: (N, )
    weighted_metrics = np.stack(df["weighted_metrics"].to_numpy())  # shape: (N, M)
    weighted_metrics_array = np.stack(df["weighted_metrics_array"].to_numpy())  # shape: (N, M)

    mask = np.isnan(two_frame_scores)
    two_frame_idx = WeightedMetricIndex.TWO_FRAME_EXTENDED_COMFORT

    weighted_metrics[mask, two_frame_idx] = 0.0
    weighted_metrics_array[mask, two_frame_idx] = 0.0

    non_mask = ~mask
    weighted_metrics[non_mask, two_frame_idx] = two_frame_scores[non_mask]

    weighted_sum = (weighted_metrics * weighted_metrics_array).sum(axis=1)
    total_weight = weighted_metrics_array.sum(axis=1)
    total_weight[total_weight == 0.0] = np.nan
    weighted_metric_scores = weighted_sum / total_weight

    df["score"] = df["multiplicative_metrics_prod"].to_numpy() * weighted_metric_scores
    df.drop(
        columns=["weighted_metrics", "weighted_metrics_array", "multiplicative_metrics_prod"],
        inplace=True,
    )

    return df


def create_scene_aggregators(
    all_mappings: Dict[str, str],
    full_score_df: pd.DataFrame,
    proposal_sampling: TrajectorySampling,
) -> pd.DataFrame:

    full_score_df["two_frame_extended_comfort"] = np.nan
    full_score_df = full_score_df.set_index("token")

    all_updates = []

    for now_frame, previous_frame in all_mappings.items():
        aggregator = SceneAggregator(
            now_frame=now_frame,
            previous_frame=previous_frame,
            score_df=full_score_df,
            proposal_sampling=proposal_sampling,
        )
        updated_rows = aggregator.aggregate_scores(one_stage_only=True)

        all_updates.append(updated_rows)

    all_updates_df = pd.concat(all_updates, ignore_index=True).set_index("token")
    full_score_df.update(all_updates_df)
    full_score_df.reset_index(inplace=True)
    full_score_df = full_score_df.drop(columns=["ego_simulated_states"])

    return full_score_df


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for running PDMS evaluation.
    :param cfg: omegaconf dictionary
    """

    build_logger(cfg)
    worker = build_worker(cfg)

    # Extract scenes based on scene-loader to know which tokens to distribute across workers
    # TODO: infer the tokens per log from metadata, to not have to load metric cache and scenes here
    scene_loader = SceneLoader(
        original_sensor_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    num_missing_metric_cache_tokens = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))
    num_unused_metric_cache_tokens = len(set(metric_cache_loader.tokens) - set(scene_loader.tokens))
    if num_missing_metric_cache_tokens > 0:
        logger.warning(f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")
    if num_unused_metric_cache_tokens > 0:
        logger.warning(f"Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens.")
    logger.info(f"Starting pdm scoring of {len(tokens_to_evaluate)} scenarios...")
    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]
    score_rows: List[pd.DataFrame] = worker_map(worker, run_pdm_score, data_points)

    pdm_score_df = pd.concat(score_rows)

    start_adjacent_mapping = infer_start_adjacent_mapping(pdm_score_df)
    pdm_score_df = create_scene_aggregators(
        start_adjacent_mapping, pdm_score_df, instantiate(cfg.simulator.proposal_sampling)
    )
    pdm_score_df = compute_final_scores(pdm_score_df)

    num_sucessful_scenarios = pdm_score_df["valid"].sum()
    num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios
    if num_failed_scenarios > 0:
        failed_tokens = pdm_score_df[~pdm_score_df["valid"]]["token"].to_list()
    else:
        failed_tokens = []

    score_cols = [
        c
        for c in pdm_score_df.columns
        if (
            (any(score.name in c for score in fields(PDMResults)) or c == "two_frame_extended_comfort" or c == "score")
            and c != "pdm_score"
        )
    ]

    # Calculate average score
    average_row = pdm_score_df[score_cols].mean(skipna=True)
    average_row["token"] = "average_all_frames"
    average_row["valid"] = pdm_score_df["valid"].all()

    # append average and pseudo closed loop scores
    pdm_score_df = pdm_score_df[["token", "valid"] + score_cols]
    pdm_score_df.loc[len(pdm_score_df)] = average_row

    save_path = Path(cfg.output_dir)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    pdm_score_df.to_csv(save_path / f"{timestamp}.csv")

    logger.info(
        f"""
        Finished running evaluation.
            Number of successful scenarios: {num_sucessful_scenarios}.
            Number of failed scenarios: {num_failed_scenarios}.
            Final average score of valid results: {pdm_score_df['score'].mean()}.
            Results are stored in: {save_path / f"{timestamp}.csv"}.
        """
    )

    if cfg.verbose:
        logger.info(
            f"""
            Detailed results:
            {pdm_score_df.iloc[-3:].T}
            """
        )
    if num_failed_scenarios > 0:
        logger.info(
            f"""
            List of failed tokens:
            {failed_tokens}
            """
        )


if __name__ == "__main__":
    main()
