import logging
import pickle
import traceback
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.geometry.convert import relative_to_absolute_poses
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from omegaconf import DictConfig
from tqdm import tqdm

from navsim.common.dataclasses import PDMResults, Trajectory
from navsim.common.dataloader import MetricCacheLoader
from navsim.common.enums import SceneFrameType
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.scoring.scene_aggregator import SceneAggregator
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex
from navsim.traffic_agents_policies.abstract_traffic_agents_policy import AbstractTrafficAgentsPolicy

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score_from_submission"


def run_pdm_score(
    cfg: DictConfig,
    first_stage_agent_output: Dict[str, Trajectory],
    second_stage_agent_output: Dict[str, Trajectory],
    simulator: PDMSimulator,
    scorer: PDMScorer,
    metric_cache_path: Path,
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """
    Evaluate submission file with PDM score.
    :param first stage agent output: first stage agent output
    :param second stage agent output: second stage agent output
    :param simulator: internal simulator object of PDM
    :param scorer: internal scoring objected in PDM
    :param metric_cache_path: path to metric cache
    :return: Tuple of two lists of pd.DataFrame, each containing the PDM results for the first and second stage agents
    """
    logger.info("Building SceneLoader")
    metric_cache_loader = MetricCacheLoader(metric_cache_path)

    pdm_results: List[pd.DataFrame] = []

    # first stage
    traffic_agents_policy_stage_one: AbstractTrafficAgentsPolicy = instantiate(
        cfg.traffic_agents_policy.reactive, simulator.proposal_sampling
    )

    for token in tqdm(first_stage_agent_output.keys(), desc="Compute PDM-Score for first stage reactive agents"):
        try:
            metric_cache = metric_cache_loader.get_from_token(token)
            trajectory = first_stage_agent_output[token]
            score_row_stage_one, ego_simulated_states = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
                traffic_agents_policy=traffic_agents_policy_stage_one,
            )
            score_row_stage_one["valid"] = True
            score_row_stage_one["log_name"] = metric_cache.log_name
            score_row_stage_one["frame_type"] = metric_cache.scene_type
            score_row_stage_one["start_time"] = metric_cache.timepoint.time_s
            end_pose = StateSE2(
                x=trajectory.poses[-1, 0],
                y=trajectory.poses[-1, 1],
                heading=trajectory.poses[-1, 2],
            )
            absolute_endpoint = relative_to_absolute_poses(metric_cache.ego_state.rear_axle, [end_pose])[0]
            score_row_stage_one["endpoint_x"] = absolute_endpoint.x
            score_row_stage_one["endpoint_y"] = absolute_endpoint.y
            score_row_stage_one["start_point_x"] = metric_cache.ego_state.rear_axle.x
            score_row_stage_one["start_point_y"] = metric_cache.ego_state.rear_axle.y
            score_row_stage_one["ego_simulated_states"] = [ego_simulated_states]  # used for two-frames extended comfort

        except Exception:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row_stage_one = pd.DataFrame([PDMResults.get_empty_results()])
            score_row_stage_one["valid"] = False
        score_row_stage_one["token"] = token

        pdm_results.append(score_row_stage_one)

    # second stage reactive scores

    traffic_agents_policy_stage_two: AbstractTrafficAgentsPolicy = instantiate(
        cfg.traffic_agents_policy.reactive, simulator.proposal_sampling
    )

    for token in tqdm(second_stage_agent_output.keys(), desc="Compute PDM-Score for second stage reactive agents"):
        try:
            metric_cache = metric_cache_loader.get_from_token(token)
            trajectory = second_stage_agent_output[token]
            score_row_stage_two, ego_simulated_states = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
                traffic_agents_policy=traffic_agents_policy_stage_two,
            )
            score_row_stage_two["valid"] = True
            score_row_stage_two["log_name"] = metric_cache.log_name
            score_row_stage_two["frame_type"] = metric_cache.scene_type
            score_row_stage_two["start_time"] = metric_cache.timepoint.time_s
            end_pose = StateSE2(
                x=trajectory.poses[-1, 0],
                y=trajectory.poses[-1, 1],
                heading=trajectory.poses[-1, 2],
            )
            absolute_endpoint = relative_to_absolute_poses(metric_cache.ego_state.rear_axle, [end_pose])[0]
            score_row_stage_two["endpoint_x"] = absolute_endpoint.x
            score_row_stage_two["endpoint_y"] = absolute_endpoint.y
            score_row_stage_two["start_point_x"] = metric_cache.ego_state.rear_axle.x
            score_row_stage_two["start_point_y"] = metric_cache.ego_state.rear_axle.y
            score_row_stage_two["ego_simulated_states"] = [ego_simulated_states]  # used for two-frames extended comfort

        except Exception:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row_stage_two = pd.DataFrame([PDMResults.get_empty_results()])
            score_row_stage_two["valid"] = False
        score_row_stage_two["token"] = token

        pdm_results.append(score_row_stage_two)

    return pdm_results


def compute_final_scores(pdm_score_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute final scores after injecting the two-frame extended comfort score
    into the weighted metrics.

    :param pdm_score_df: Original PDM score DataFrame with all tokens
    :return: DataFrame with final score computed
    """
    df = pdm_score_df.reset_index()

    assert (
        not df["two_frame_extended_comfort"].isna().any()
    ), "Found NaN in 'two_frame_extended_comfort'. Please check aggregator completeness."

    two_frame_scores = df["two_frame_extended_comfort"].to_numpy()
    weighted_metrics = np.stack(df["weighted_metrics"].to_numpy())  # shape: (N, M)
    weighted_metrics_array = np.stack(df["weighted_metrics_array"].to_numpy())  # shape: (N, M)

    two_frame_idx = WeightedMetricIndex.TWO_FRAME_EXTENDED_COMFORT
    weighted_metrics[:, two_frame_idx] = two_frame_scores

    weighted_sum = (weighted_metrics * weighted_metrics_array).sum(axis=1)
    total_weight = weighted_metrics_array.sum(axis=1)

    assert np.all(total_weight > 0), "Found total_weight == 0 during score computation."

    weighted_metric_scores = weighted_sum / total_weight
    df["score"] = df["multiplicative_metrics_prod"].to_numpy() * weighted_metric_scores

    df.drop(columns=["weighted_metrics", "weighted_metrics_array", "multiplicative_metrics_prod"], inplace=True)

    return df


def calculate_weighted_average_score(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the weighted average for all score columns in the dataframe.
    Automatically normalizes weights if they do not sum to 1.
    """
    score_cols = [c for c in df.columns if c not in {"weight", "token"}]

    if df.empty:
        return pd.Series([np.nan] * len(score_cols), index=score_cols)

    weights = df["weight"].to_numpy()
    scores = df[score_cols].to_numpy()
    total_weight = weights.sum()

    if total_weight == 0:
        return pd.Series([np.nan] * len(score_cols), index=score_cols)

    weighted_avg = (scores * weights[:, None]).sum(axis=0) / total_weight
    return pd.Series(weighted_avg, index=score_cols)


def calculate_individual_mapping_scores(
    df: pd.DataFrame, all_mappings: Dict[Tuple[str, str], List[Tuple[str, str]]]
) -> pd.DataFrame:
    """
    Compute the weighted average score for each mapping (now + prev),
    return a DataFrame where each row corresponds to one mapping.
    """
    mapping_level_averages = []

    for (first_now, first_prev), second_stage in all_mappings.items():

        tokens = [first_now, first_prev] + [tok for pair in second_stage for tok in pair]
        mapping_df = df[df["token"].isin(tokens)]

        mapping_avg = calculate_weighted_average_score(mapping_df)
        mapping_level_averages.append(mapping_avg)

    return pd.DataFrame(mapping_level_averages).mean(skipna=True)


def create_scene_aggregators(
    all_mappings: Dict[Tuple[str, str], List[Tuple[str, str]]],
    full_score_df: pd.DataFrame,
    proposal_sampling: TrajectorySampling,
) -> pd.DataFrame:

    full_score_df["two_frame_extended_comfort"] = np.nan
    full_score_df["weight"] = np.nan
    full_score_df = full_score_df.set_index("token")

    all_updates = []
    all_seen_tokens = set()

    for (now_frame, previous_frame), second_stage in all_mappings.items():
        aggregator = SceneAggregator(
            now_frame=now_frame,
            previous_frame=previous_frame,
            second_stage=second_stage,
            score_df=full_score_df,
            proposal_sampling=proposal_sampling,
        )
        updated_rows = aggregator.aggregate_scores()

        all_seen_tokens.update(updated_rows["token"])
        all_updates.append(updated_rows)

    all_updates_df = pd.concat(all_updates, ignore_index=True).set_index("token")
    full_score_df.update(all_updates_df)
    full_score_df.reset_index(inplace=True)
    full_score_df = full_score_df.drop(columns=["ego_simulated_states"])

    return full_score_df


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for running PDMS from submission pickle.
    :param cfg: omegaconf dictionary
    """
    submission_file_path = Path(cfg.submission_file_path)
    metric_cache_path = Path(cfg.metric_cache_path)
    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)

    build_logger(cfg)
    assert (
        simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"

    with open(submission_file_path, "rb") as f:
        submission_data = pickle.load(f)

    first_stage_output: Dict[str, Trajectory] = submission_data["first_stage_predictions"]
    second_stage_output: Dict[str, Trajectory] = submission_data["second_stage_predictions"]

    assert (
        len(first_stage_output) == 1 and len(second_stage_output) == 1
    ), "Multi-seed evaluation currently not supported in run_pdm_score!"
    first_stage_output = first_stage_output[0]
    second_stage_output = second_stage_output[0]

    score_rows = run_pdm_score(
        cfg=cfg,
        first_stage_agent_output=first_stage_output,
        second_stage_agent_output=second_stage_output,
        simulator=simulator,
        scorer=scorer,
        metric_cache_path=metric_cache_path,
    )

    pdm_score_df = pd.concat(score_rows)

    # score aggregation
    try:
        raw_mapping = cfg.train_test_split.reactive_all_mapping
        all_mappings: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}

        for orig_token, prev_token, two_stage_pairs in raw_mapping:
            if prev_token in set(first_stage_output.keys()) or orig_token in set(first_stage_output.keys()):
                all_mappings[(orig_token, prev_token)] = [tuple(pair) for pair in two_stage_pairs]

        # for stage one reactive
        pdm_score_df = create_scene_aggregators(
            all_mappings,
            pdm_score_df,
            instantiate(cfg.simulator.proposal_sampling),
        )
        pdm_score_df = compute_final_scores(pdm_score_df)
        pseudo_closed_loop_valid = True

    except Exception:
        logger.warning("----------- Failed to calculate pseudo closed-loop weights or comfort:")
        traceback.print_exc()
        pdm_score_df["weight"] = 1.0
        pseudo_closed_loop_valid = True

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
    average_all_frames_extended_pdm_score_row = pdm_score_df[score_cols].mean(skipna=True)
    average_all_frames_extended_pdm_score_row["token"] = "average_all_frames_extended_pdm_score"
    average_all_frames_extended_pdm_score_row["valid"] = pdm_score_df["valid"].all()

    # Calculate pseudo closed loop score with weighted average
    extended_pdm_score_row = calculate_individual_mapping_scores(
        pdm_score_df[score_cols + ["token", "weight"]], all_mappings
    )
    extended_pdm_score_row["token"] = "extended_pdm_score"
    extended_pdm_score_row["valid"] = pseudo_closed_loop_valid

    # Original frames average
    original_frames = pdm_score_df[pdm_score_df["frame_type"] == SceneFrameType.ORIGINAL]
    average_expert_frames_extended_pdm_score_row = original_frames[score_cols].mean(skipna=True)
    average_expert_frames_extended_pdm_score_row["token"] = "average_expert_frames_extended_pdm_score"
    average_expert_frames_extended_pdm_score_row["valid"] = original_frames["valid"].all()

    # append average and pseudo closed loop scores
    pdm_score_df = pdm_score_df[["token", "valid"] + score_cols]
    pdm_score_df.loc[len(pdm_score_df)] = average_all_frames_extended_pdm_score_row
    pdm_score_df.loc[len(pdm_score_df)] = average_expert_frames_extended_pdm_score_row
    pdm_score_df.loc[len(pdm_score_df)] = extended_pdm_score_row

    save_path = Path(cfg.output_dir)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    pdm_score_df.to_csv(save_path / f"{timestamp}.csv")

    logger.info(
        f"""
        Finished running evaluation.
            Number of successful scenarios: {num_sucessful_scenarios}.
            Number of failed scenarios: {num_failed_scenarios}.
            Final extended pdm score of valid results: {pdm_score_df[pdm_score_df["token"] == "extended_pdm_score"]["score"].iloc[0]}.
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
