from typing import List

import numpy as np
import numpy.typing as npt
import pandas as pd
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, TimePoint
from nuplan.common.geometry.convert import relative_to_absolute_poses
from nuplan.planning.simulation.planner.ml_planner.transform_utils import (
    _get_fixed_timesteps,
    _se2_vel_acc_to_ego_state,
)
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataclasses import Trajectory
from navsim.common.enums import SceneFrameType
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import ego_states_to_state_array
from navsim.traffic_agents_policies.abstract_traffic_agents_policy import AbstractTrafficAgentsPolicy


def transform_trajectory(pred_trajectory: Trajectory, initial_ego_state: EgoState) -> InterpolatedTrajectory:
    """
    Transform trajectory in global frame and return as InterpolatedTrajectory
    :param pred_trajectory: trajectory dataclass in ego frame
    :param initial_ego_state: nuPlan's ego state object
    :return: nuPlan's InterpolatedTrajectory
    """

    future_sampling = pred_trajectory.trajectory_sampling
    timesteps = _get_fixed_timesteps(initial_ego_state, future_sampling.time_horizon, future_sampling.interval_length)

    relative_poses = np.array(pred_trajectory.poses, dtype=np.float64)
    relative_states = [StateSE2.deserialize(pose) for pose in relative_poses]
    absolute_states = relative_to_absolute_poses(initial_ego_state.rear_axle, relative_states)

    # NOTE: velocity and acceleration ignored by LQR + bicycle model
    agent_states = [
        _se2_vel_acc_to_ego_state(
            state,
            [0.0, 0.0],
            [0.0, 0.0],
            timestep,
            initial_ego_state.car_footprint.vehicle_parameters,
        )
        for state, timestep in zip(absolute_states, timesteps)
    ]

    # NOTE: maybe make addition of initial_ego_state optional
    return InterpolatedTrajectory([initial_ego_state] + agent_states)


def get_trajectory_as_array(
    trajectory: InterpolatedTrajectory,
    future_sampling: TrajectorySampling,
    start_time: TimePoint,
) -> npt.NDArray[np.float64]:
    """
    Interpolated trajectory and return as numpy array
    :param trajectory: nuPlan's InterpolatedTrajectory object
    :param future_sampling: Sampling parameters for interpolation
    :param start_time: TimePoint object of start
    :return: Array of interpolated trajectory states.
    """

    times_s = np.arange(
        0.0,
        future_sampling.time_horizon + future_sampling.interval_length,
        future_sampling.interval_length,
    )
    times_s += start_time.time_s
    times_us = [int(time_s * 1e6) for time_s in times_s]
    times_us = np.clip(times_us, trajectory.start_time.time_us, trajectory.end_time.time_us)
    time_points = [TimePoint(time_us) for time_us in times_us]

    trajectory_ego_states: List[EgoState] = trajectory.get_state_at_times(time_points)

    return ego_states_to_state_array(trajectory_ego_states)


def pdm_score(
    metric_cache: MetricCache,
    model_trajectory: Trajectory,
    future_sampling: TrajectorySampling,
    simulator: PDMSimulator,
    scorer: PDMScorer,
    traffic_agents_policy: AbstractTrafficAgentsPolicy,
) -> pd.DataFrame:
    """
    FIXME: Output type hints and refactoring/debugging. Inconsistent with some evaluation scripts.
    Runs PDM-Score and saves results in the corresponding dataclass.
    :param metric_cache: Metric cache dataclass of the sample.
    :param model_trajectory: Predicted trajectory in ego frame.
    :param future_sampling: Sampling configuration of the model trajectory.
    :param simulator: Simulator applied on the model trajectory.
    :param scorer: Scoring object to retrieve the sub-scores
    :param traffic_agents_policy: background traffic used during simulation/scoring.
    :return: Dataclass of PDM sub-scores.
    """

    pred_trajectory = transform_trajectory(model_trajectory, metric_cache.ego_state)

    return pdm_score_from_interpolated_trajectory(
        metric_cache=metric_cache,
        pred_trajectory=pred_trajectory,
        future_sampling=future_sampling,
        simulator=simulator,
        scorer=scorer,
        traffic_agents_policy=traffic_agents_policy,
    )


def pdm_score_from_interpolated_trajectory(
    metric_cache: MetricCache,
    pred_trajectory: InterpolatedTrajectory,
    future_sampling: TrajectorySampling,
    simulator: PDMSimulator,
    scorer: PDMScorer,
    traffic_agents_policy: AbstractTrafficAgentsPolicy,
):
    """
    FIXME: Output type hints and refactoring/debugging. Inconsistent with some evaluation scripts.
    Computes PDM-Score from interpolated trajectory of an agent.
    :param metric_cache: Metric cache dataclass of the sample.
    :param pred_trajectory: Predicted (interpolated) trajectory in global frame.
    :param future_sampling: Sampling configuration of the trajectory.
    :param simulator: Simulator applied on the trajectory.
    :param scorer: Scoring object to retrieve the sub-scores.
    :param traffic_agents_policy: background traffic used during simulation/scoring.
    :return: Dataclass of PDM sub-scores.
    """

    initial_ego_state = metric_cache.ego_state
    pdm_trajectory = metric_cache.trajectory

    pdm_states, pred_states = (
        get_trajectory_as_array(pdm_trajectory, future_sampling, initial_ego_state.time_point),
        get_trajectory_as_array(pred_trajectory, future_sampling, initial_ego_state.time_point),
    )
    trajectory_states = np.concatenate([pdm_states[None, ...], pred_states[None, ...]], axis=0)

    simulated_states = simulator.simulate_proposals(trajectory_states, initial_ego_state)

    # infer traffic agents policy and update future observation
    simulated_agent_detections_tracks = traffic_agents_policy.simulate_environment(simulated_states[1], metric_cache)

    assert (
        len(simulated_agent_detections_tracks) == trajectory_states.shape[1]
    ), f"""
            Traffic agents policy returned trajectories of invalid length:
            Traffic agents trajectories must be of length ego_trajectory_length = {trajectory_states.shape[1]},
            but got {len(simulated_agent_detections_tracks)}
        """

    pred_idx = 1  # index of predicted trajectory in trajectory_states and simulated_states
    pdm_result = scorer.score_proposals(
        simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
        metric_cache.map_parameters,
        simulated_agent_detections_tracks,
        metric_cache.past_human_trajectory,
    )[pred_idx]

    if scorer._config.human_penalty_filter and metric_cache.scene_type == SceneFrameType.ORIGINAL:
        # human_penalty_filter

        human_trajectory = transform_trajectory(metric_cache.human_trajectory, initial_ego_state)

        human_states = get_trajectory_as_array(human_trajectory, future_sampling, initial_ego_state.time_point)

        human_simulated_states = simulator.simulate_proposals(human_states[None, ...], initial_ego_state)

        human_simulated_agent_detections_tracks = traffic_agents_policy.simulate_environment(
            human_simulated_states[0], metric_cache
        )

        human_pdm_result = scorer.score_proposals(
            human_simulated_states,
            metric_cache.observation,
            metric_cache.centerline,
            metric_cache.route_lane_ids,
            metric_cache.drivable_area_map,
            metric_cache.map_parameters,
            human_simulated_agent_detections_tracks,
        )[0]

        for column in human_pdm_result.columns:
            if column in [
                "multiplicative_metrics_prod",
                "weighted_metrics",
                "weighted_metrics_array",
            ]:
                continue
            if human_pdm_result[column].iloc[0] == 0:
                pdm_result.at[0, column] = 1

    return pdm_result, simulated_states[pred_idx]
