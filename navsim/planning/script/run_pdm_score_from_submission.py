import logging
import pickle
import traceback
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Dict, List

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
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import ego_is_two_frame_extended_comfort
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex
from navsim.traffic_agents_policies.abstract_traffic_agents_policy import AbstractTrafficAgentsPolicy

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score_from_submission"


def run_pdm_score(
    submission_file_path: Path,
    simulator: PDMSimulator,
    scorer: PDMScorer,
    metric_cache_path: Path,
    traffic_agents_policy: AbstractTrafficAgentsPolicy,
) -> pd.DataFrame:
    """
    Evaluate submission file with PDM score.
    :param submission_file_path: path to submission pickle file
    :param simulator: internal simulator object of PDM
    :param scorer: internal scoring objected in PDM
    :param metric_cache_path: path to metric cache
    :param traffic_agents_policy: traffic agents policy
    :return: pandas data frame with PDMS results.
    """
    logger.info("Building SceneLoader")
    metric_cache_loader = MetricCacheLoader(metric_cache_path)
    with open(submission_file_path, "rb") as f:
        agent_output_list: List[Dict[str, Trajectory]] = pickle.load(f)["predictions"]

    assert len(agent_output_list) == 1, "Multi-seed evaluation currently not supported in run_pdm_score!"
    agent_output = agent_output_list[0]

    pdm_results: List[pd.DataFrame] = []
    for token in tqdm(metric_cache_loader.tokens, desc="Compute PDM-Score"):

        try:
            metric_cache = metric_cache_loader.get_from_token(token)
            trajectory = agent_output[token]
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


def infer_two_stage_mapping(score_df: pd.DataFrame, first_stage_duration: float) -> pd.DataFrame:
    initial_frames = score_df[(score_df["valid"]) & (score_df["frame_type"] == SceneFrameType.ORIGINAL)]

    two_stage_mapping = {}
    for _, row in initial_frames.iterrows():
        # Filter tokens in the same log starting at least T seconds later
        earliest_second_stage_start_time = row["start_time"] + first_stage_duration - 0.05
        latest_second_stage_start_time = row["start_time"] + first_stage_duration + 0.05
        second_stage_tokens: pd.DataFrame = score_df[
            (score_df["log_name"] == row["log_name"])
            & (score_df["start_time"] <= latest_second_stage_start_time)
            & (score_df["start_time"] >= earliest_second_stage_start_time)
            & (score_df["valid"])
            & (score_df["frame_type"] == SceneFrameType.SYNTHETIC)
        ]["token"].to_list()

        two_stage_mapping[row["token"]] = second_stage_tokens
    return two_stage_mapping


def validate_two_stage_mapping(
    score_df: pd.DataFrame,
    two_stage_mapping: Dict[str, List[str]],
    validate_start_times: bool = True,
) -> None:
    # make sure all tokens are unique
    all_tokens = [token for tokens in two_stage_mapping.values() for token in tokens] + list(two_stage_mapping.keys())
    assert len(all_tokens) == len(set(all_tokens)), "Tokens in the two stage mapping are not unique."

    # make sure all tokens are in the score dataframe
    assert set(all_tokens) == set(score_df["token"]), (
        f"Tokens in the two stage aggregation mapping and the results are not the same. "
        f"Missing tokens in the mapping: {set(all_tokens) - set(score_df['token'])}."
        f"Missing tokens in the results: {set(score_df['token']) - set(all_tokens)}."
    )

    # make sure subsequent tokens belong to the same log
    # make sure first stage and second stage tokens are 4s apart
    for first_stage_token, second_stage_tokens in two_stage_mapping.items():
        first_stage_log_name = score_df[score_df["token"] == first_stage_token].iloc[0]["log_name"]
        if validate_start_times:
            first_stage_start_time = score_df[score_df["token"] == first_stage_token].iloc[0]["start_time"]
        else:
            first_stage_start_time = 0.0
        for second_stage_token in second_stage_tokens:
            second_stage_log_name = score_df[score_df["token"] == second_stage_token].iloc[0]["log_name"]
            if validate_start_times:
                second_stage_start_time = score_df[score_df["token"] == second_stage_token].iloc[0]["start_time"]
            else:
                second_stage_start_time = 4.0
            assert first_stage_log_name == second_stage_log_name, (
                f"Tokens {first_stage_token} and {second_stage_token} belong to different logs."
                f"First stage log: {first_stage_log_name}, second stage log: {second_stage_log_name}."
            )
            assert np.abs(second_stage_start_time - first_stage_start_time - 4.0) < 0.05, (
                f"Tokens {first_stage_token} and {second_stage_token} are not 4s apart."
                f"First stage start time: {first_stage_start_time}, second stage start time: {second_stage_start_time}."
            )

    # make sure the frame_type of all first_stage tokens is ORIGINAL and all second_stage tokens is SYNTHETIC
    first_stage_tokens = list(two_stage_mapping.keys())
    second_stage_tokens = [token for tokens in two_stage_mapping.values() for token in tokens]
    first_stage_types = score_df.loc[score_df["token"].isin(first_stage_tokens), "frame_type"]
    second_stage_types = score_df.loc[score_df["token"].isin(second_stage_tokens), "frame_type"]
    assert (first_stage_types == SceneFrameType.ORIGINAL).all(), "Some first-stage tokens are not of type ORIGINAL."
    assert (second_stage_types == SceneFrameType.SYNTHETIC).all(), "Some second-stage tokens are not of type SYNTHETIC."


def calculate_pseudo_closed_loop_weights(
    score_df: pd.DataFrame, two_stage_mapping: Dict[str, List[str]]
) -> pd.DataFrame:
    """
    Calculate two stage scores for each scenario.
    :param score_rows: List of dataframes containing scores for each scenario.
    :param first_stage_duration: Duration of the first stage in seconds.
    """
    pd.options.mode.copy_on_write = True

    def _calc_distance(x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    first_stage_tokens = list(two_stage_mapping.keys())

    weights = []
    for first_stage_token in first_stage_tokens:
        first_stage_row = score_df[score_df["token"] == first_stage_token].iloc[0]
        second_stage_tokens = two_stage_mapping[first_stage_token]
        # set weight of first stage to one
        weights.append(pd.DataFrame([{"token": first_stage_token, "weight": 1.0}]))
        # compute weights for second stage
        second_stage_scores: pd.DataFrame = score_df[(score_df["token"].isin(second_stage_tokens))]
        second_stage_scores["distance"] = second_stage_scores.apply(
            lambda x: _calc_distance(
                first_stage_row["endpoint_x"],
                first_stage_row["endpoint_y"],
                x["start_point_x"],
                x["start_point_y"],
            ),
            axis=1,
        )
        second_stage_scores["weight"] = second_stage_scores["distance"].apply(lambda x: np.exp(-x))
        second_stage_scores["weight"] = second_stage_scores["weight"] / second_stage_scores["weight"].sum()

        weights.append(second_stage_scores[["token", "weight"]])

    weights = pd.concat(weights)
    return weights


def calculate_two_frame_extended_comfort(score_df: pd.DataFrame, proposal_sampling: TrajectorySampling) -> pd.DataFrame:
    """
    Calculates two-frame extended comfort by comparing only the overlapping parts of consecutive original frames.
    Handles varying observation intervals.

    :param score_df: DataFrame containing scores and states of frames.
    :param proposal_sampling: Sampling parameters for trajectory.
    :return: DataFrame containing two-frame extended comfort scores.
    """
    results = []
    interval_length = proposal_sampling.interval_length  # Default: 0.1s

    grouped_logs = score_df[score_df["frame_type"] == SceneFrameType.ORIGINAL].groupby("log_name")

    for log_name, group_df in grouped_logs:
        group_df = group_df.sort_values(by="start_time").reset_index(drop=True)

        for idx in range(len(group_df) - 1):  # Iterate over consecutive frames
            current_row = group_df.iloc[idx]
            next_row = group_df.iloc[idx + 1]

            observation_interval = next_row["start_time"] - current_row["start_time"]

            if abs(observation_interval) > 0.55:
                two_frame_comfort = np.nan
                next_token = np.nan
            else:
                overlap_start = int(observation_interval / interval_length)

                current_states = current_row["ego_simulated_states"]
                next_states = next_row["ego_simulated_states"]

                # Ensure they have the same shape
                assert current_states.shape == next_states.shape, "Trajectories must be of equal length"

                # Extract only the overlapping part
                current_states_overlap = current_states[overlap_start:]
                next_states_overlap = next_states[:-overlap_start]

                # Define corresponding time points for overlap
                n_overlap = current_states_overlap.shape[0]  # Compute the actual number of overlapping steps
                time_point_s = np.arange(n_overlap) * interval_length  # Generate aligned time steps

                # Compute two-frame extended comfort
                two_frame_comfort = ego_is_two_frame_extended_comfort(
                    current_states_overlap[None, :],
                    next_states_overlap[None, :],
                    time_point_s,
                )[0].astype(np.float64)

                next_token = next_row["token"]

            results.append(
                {
                    "current_token": current_row["token"],
                    "next_token": next_token,
                    "two_frame_extended_comfort": two_frame_comfort,
                }
            )

    return pd.DataFrame(results)


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


def calculate_weighted_average_score(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the weighted average score of a dataframe.
    :param df: Dataframe containing scores.
    """

    if df.empty:
        score_cols = [c for c in df.columns if c not in {"weight", "token"}]
        return pd.Series([np.nan] * len(score_cols), index=score_cols)

    weights = df["weight"]
    weighted_scores = df[[c for c in df.columns if c not in {"weight", "token"}]].mul(weights, axis=0)

    weighted_scores_row = weighted_scores.sum(skipna=False)
    return weighted_scores_row


def calculate_individual_mapping_scores(df: pd.DataFrame, two_stage_mapping: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Compute the weighted average score for each first_stage_token
    in the two_stage_mapping. The function returns a new DataFrame
    containing the weighted average for each mapping.

    :param df: A DataFrame that includes columns like 'token', 'weight', 'score', etc.
    :param two_stage_mapping: A dictionary where each key is a first-stage token (str),
        and each value is a list of second-stage tokens.
    :return: A DataFrame with one row per first-stage token, containing the
        weighted average scores for that token and its second-stage tokens.
    """
    # This list will hold the results (one row per mapping).
    rows_for_each_mapping = []

    for first_stage_token, second_stage_tokens in two_stage_mapping.items():

        stage1_df = df[df["token"] == first_stage_token]
        stage1_avg_series = calculate_weighted_average_score(stage1_df)
        stage2_df = df[df["token"].isin(second_stage_tokens)]
        stage2_avg_series = calculate_weighted_average_score(stage2_df)

        # Combine the two stages
        subset_average = pd.concat([stage1_avg_series, stage2_avg_series], axis=1).mean(axis=1, skipna=True)
        rows_for_each_mapping.append(subset_average)

    mapping_scores_df = pd.DataFrame(rows_for_each_mapping)
    mapping_scroes_row = mapping_scores_df.mean(skipna=True)

    return mapping_scroes_row


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
    traffic_agents_policy: AbstractTrafficAgentsPolicy = instantiate(
        cfg.traffic_agents_policy, simulator.proposal_sampling
    )
    build_logger(cfg)
    assert (
        simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"

    score_rows = run_pdm_score(
        submission_file_path=submission_file_path,
        simulator=simulator,
        scorer=scorer,
        metric_cache_path=metric_cache_path,
        traffic_agents_policy=traffic_agents_policy,
    )

    pdm_score_df = pd.concat(score_rows)

    # Calculate two-frame extended comfort
    two_frame_comfort_df = calculate_two_frame_extended_comfort(
        pdm_score_df, proposal_sampling=instantiate(cfg.simulator.proposal_sampling)
    )

    # Merge two-frame comfort scores and drop unnecessary columns in one step
    pdm_score_df = (
        pdm_score_df.drop(columns=["ego_simulated_states"])  # Remove the unwanted column first
        .merge(
            two_frame_comfort_df[["current_token", "two_frame_extended_comfort"]],
            left_on="token",
            right_on="current_token",
            how="left",
        )
        .drop(columns=["current_token"])  # Remove merged key after the merge
    )

    # Compute final scores
    pdm_score_df = compute_final_scores(pdm_score_df)

    try:
        if hasattr(cfg.train_test_split, "two_stage_mapping"):
            two_stage_mapping: Dict[str, List[str]] = dict(cfg.train_test_split.two_stage_mapping)
        else:
            # infer two stage mapping from results
            two_stage_mapping = infer_two_stage_mapping(pdm_score_df, first_stage_duration=4.0)
        validate_two_stage_mapping(pdm_score_df, two_stage_mapping)

        # calculate weights for pseudo closed loop using config
        weights = calculate_pseudo_closed_loop_weights(pdm_score_df, two_stage_mapping=two_stage_mapping)
        assert len(weights) == len(pdm_score_df), "Couldn't calculate weights for all tokens."
        pdm_score_df = pdm_score_df.merge(weights, on="token")
        pseudo_closed_loop_valid = True
    except Exception:
        logger.warning("----------- Failed to calculate pseudo closed-loop weights:")
        traceback.print_exc()
        pdm_score_df["weight"] = 1.0
        pseudo_closed_loop_valid = False

    num_sucessful_scenarios = pdm_score_df["valid"].sum()
    num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios
    if num_failed_scenarios > 0:
        pdm_score_df[not pdm_score_df["valid"]]["token"].to_list()
    else:
        pass

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

    # Calculate pseudo closed loop score with weighted average
    pseudo_closed_loop_row = calculate_individual_mapping_scores(
        pdm_score_df[score_cols + ["token", "weight"]], two_stage_mapping
    )
    pseudo_closed_loop_row["token"] = "pseudo_closed_loop"
    pseudo_closed_loop_row["valid"] = pseudo_closed_loop_valid

    # Original frames average
    original_frames = pdm_score_df[pdm_score_df["frame_type"] == SceneFrameType.ORIGINAL]
    average_original_row = original_frames[score_cols].mean(skipna=True)
    average_original_row["token"] = "average_expert_frames"
    average_original_row["valid"] = original_frames["valid"].all()

    # append average and pseudo closed loop scores
    pdm_score_df = pdm_score_df[["token", "valid"] + score_cols]
    pdm_score_df.loc[len(pdm_score_df)] = average_row
    pdm_score_df.loc[len(pdm_score_df)] = pseudo_closed_loop_row
    pdm_score_df.loc[len(pdm_score_df)] = average_original_row

    save_path = Path(cfg.output_dir)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    pdm_score_df.to_csv(save_path / f"{timestamp}.csv")


if __name__ == "__main__":
    main()
