import numpy as np
import numpy.typing as npt

from typing import List

from nuplan.planning.simulation.planner.ml_planner.transform_utils import (
    _get_fixed_timesteps,
    _se2_vel_acc_to_ego_state,
)

from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    ego_states_to_state_array,
)
from navsim.planning.metric_caching.metric_cache import MetricCache

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.common.actor_state.state_representation import StateSE2, TimePoint
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.geometry.convert import relative_to_absolute_poses

from navsim.common.dataclasses import PDMResults, Trajectory

from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    MultiMetricIndex,
    WeightedMetricIndex,
)


def transform_trajectory(
    pred_trajectory: Trajectory, initial_ego_state: EgoState
) -> InterpolatedTrajectory:
    """
    Transform trajectory in global frame and return as InterpolatedTrajectory
    :param pred_trajectory: trajectory dataclass in ego frame
    :param initial_ego_state: nuPlan's ego state object
    :return: nuPlan's InterpolatedTrajectory
    """

    future_sampling = pred_trajectory.trajectory_sampling
    timesteps = _get_fixed_timesteps(
        initial_ego_state, future_sampling.time_horizon, future_sampling.interval_length
    )

    relative_states = [StateSE2.deserialize(pose) for pose in pred_trajectory.poses]
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
    scorer: PDMScorer
) -> PDMResults:
    """
    Runs PDM-Score and saves results in dataclass.
    :param metric_cache: Metric cache dataclass
    :param model_trajectory: Predicted trajectory in ego frame.
    :return: Dataclass of PDM-Subscores.
    """

    initial_ego_state = metric_cache.ego_state

    pdm_trajectory = metric_cache.trajectory
    pred_trajectory = transform_trajectory(model_trajectory, initial_ego_state)

    pdm_states, pred_states = (
        get_trajectory_as_array(pdm_trajectory, future_sampling, initial_ego_state.time_point),
        get_trajectory_as_array(pred_trajectory, future_sampling, initial_ego_state.time_point),
    )

    trajectory_states = np.concatenate([pdm_states[None, ...], pred_states[None, ...]], axis=0)

    simulated_states = simulator.simulate_proposals(trajectory_states, initial_ego_state)

    scores = scorer.score_proposals(
        simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
    )

    # TODO: Refactor & add / modify existing metrics.
    pred_idx = 1

    no_at_fault_collisions = scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, pred_idx]
    drivable_area_compliance = scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, pred_idx]
    driving_direction_compliance = scorer._multi_metrics[
        MultiMetricIndex.DRIVING_DIRECTION, pred_idx
    ]

    ego_progress = scorer._weighted_metrics[WeightedMetricIndex.PROGRESS, pred_idx]
    time_to_collision_within_bound = scorer._weighted_metrics[WeightedMetricIndex.TTC, pred_idx]
    comfort = scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE, pred_idx]

    score = scores[pred_idx]

    return PDMResults(
        no_at_fault_collisions,
        drivable_area_compliance,
        driving_direction_compliance,
        ego_progress,
        time_to_collision_within_bound,
        comfort,
        score,
    )
