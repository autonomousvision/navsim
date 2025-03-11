from typing import Optional

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.geometry.convert import relative_to_absolute_poses
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer


class PDMEmergencyBrake:
    """Class for emergency brake maneuver of PDM-Closed."""

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        time_to_infraction_threshold: float = 2.0,
        max_ego_speed: float = 5.0,
        max_long_accel: float = 2.40,
        min_long_accel: float = -4.05,
        infraction: str = "collision",
    ):
        """
        Constructor for PDMEmergencyBrake
        :param trajectory_sampling: Sampling parameters for final trajectory
        :param time_to_infraction_threshold: threshold for applying brake, defaults to 2.0
        :param max_ego_speed: maximum speed to apply brake, defaults to 5.0
        :param max_long_accel: maximum longitudinal acceleration for braking, defaults to 2.40
        :param min_long_accel: min longitudinal acceleration for braking, defaults to -4.05
        :param infraction: infraction to determine braking (collision or ttc), defaults to "collision"
        """

        # trajectory parameters
        self._trajectory_sampling = trajectory_sampling

        # braking parameters
        self._max_ego_speed: float = max_ego_speed  # [m/s]
        self._max_long_accel: float = max_long_accel  # [m/s^2]
        self._min_long_accel: float = min_long_accel  # [m/s^2]

        # braking condition parameters
        self._time_to_infraction_threshold: float = time_to_infraction_threshold
        self._infraction: str = infraction

        assert self._infraction in [
            "collision",
            "ttc",
        ], f"PDMEmergencyBraking: Infraction {self._infraction} not available as brake condition!"

    def brake_if_emergency(
        self, ego_state: EgoState, scores: npt.NDArray[np.float64], scorer: PDMScorer
    ) -> Optional[InterpolatedTrajectory]:
        """
        Applies emergency brake only if an infraction is expected within horizon.
        :param ego_state: state object of ego
        :param scores: array of proposal scores
        :param metric: scorer class of PDM
        :return: brake trajectory or None
        """

        trajectory = None
        ego_speed: float = ego_state.dynamic_car_state.speed

        proposal_idx = np.argmax(scores)

        # retrieve time to infraction depending on brake detection mode
        if self._infraction == "ttc":
            time_to_infraction = scorer.time_to_ttc_infraction(proposal_idx)

        elif self._infraction == "collision":
            time_to_infraction = scorer.time_to_at_fault_collision(proposal_idx)

        # check time to infraction below threshold
        if time_to_infraction <= self._time_to_infraction_threshold and ego_speed <= self._max_ego_speed:
            trajectory = self._generate_trajectory(ego_state)

        return trajectory

    def _generate_trajectory(self, ego_state: EgoState) -> InterpolatedTrajectory:
        """
        Generates trajectory for reach zero velocity.
        :param ego_state: state object of ego
        :return: InterpolatedTrajectory for braking
        """
        current_time_point = ego_state.time_point
        current_velocity = ego_state.dynamic_car_state.center_velocity_2d.x
        current_acceleration = ego_state.dynamic_car_state.center_acceleration_2d.x

        target_velocity = 0.0

        if current_velocity > 0.2:
            k_p = 10.0
            k_d = 0.0

            error = -current_velocity
            dt_error = -current_acceleration
            u_t = k_p * error + k_d * dt_error

            error = max(min(u_t, self._max_long_accel), self._min_long_accel)
            correcting_velocity = 11 / 10 * (current_velocity + error)

        else:
            k_p = 4
            k_d = 1

            error = target_velocity - current_velocity
            dt_error = -current_acceleration

            u_t = k_p * error + k_d * dt_error

            correcting_velocity = max(min(u_t, self._max_long_accel), self._min_long_accel)

        trajectory_states = []

        # Propagate planned trajectory for set number of samples
        for sample in range(self._trajectory_sampling.num_poses + 1):
            time_t = self._trajectory_sampling.interval_length * sample
            pose = relative_to_absolute_poses(ego_state.center, [StateSE2(correcting_velocity * time_t, 0, 0)])[0]

            ego_state_ = EgoState.build_from_center(
                center=pose,
                center_velocity_2d=StateVector2D(0, 0),
                center_acceleration_2d=StateVector2D(0, 0),
                tire_steering_angle=0.0,
                time_point=current_time_point,
                vehicle_parameters=ego_state.car_footprint.vehicle_parameters,
            )
            trajectory_states.append(ego_state_)

            current_time_point += TimePoint(int(self._trajectory_sampling.interval_length * 1e6))

        return InterpolatedTrajectory(trajectory_states)
