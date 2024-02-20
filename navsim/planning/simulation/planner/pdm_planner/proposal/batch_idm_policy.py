from typing import List, Union

import numpy as np
import numpy.typing as npt

from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    LeadingAgentIndex,
    StateIDMIndex,
)


class BatchIDMPolicy:
    """
    IDM policies operating on a batch of proposals.
    """

    def __init__(
        self,
        fallback_target_velocity: Union[List[float], float],
        speed_limit_fraction: Union[List[float], float],
        min_gap_to_lead_agent: Union[List[float], float],
        headway_time: Union[List[float], float],
        accel_max: Union[List[float], float],
        decel_max: Union[List[float], float],
    ):
        """
        Constructor for BatchIDMPolicy
        :param target_velocity: Desired fallback velocity in free traffic [m/s]
        :param speed_limit_fraction: Fraction of speed-limit desired in free traffic
        :param min_gap_to_lead_agent: Minimum relative distance to lead vehicle [m]
        :param headway_time: Desired time headway. Minimum time to the vehicle in front [s]
        :param accel_max: maximum acceleration [m/s^2]
        :param decel_max: maximum deceleration (positive value) [m/s^2]
        """
        parameter_list = [
            fallback_target_velocity,
            speed_limit_fraction,
            min_gap_to_lead_agent,
            headway_time,
            accel_max,
            decel_max,
        ]
        num_parameter_policies = [
            len(item) for item in parameter_list if isinstance(item, list)
        ]

        if len(num_parameter_policies) > 0:
            assert all(
                item == num_parameter_policies[0] for item in num_parameter_policies
            ), "BatchIDMPolicy initial parameters must be float, or lists of equal length"
            num_policies = max(num_parameter_policies)
        else:
            num_policies = 1

        self._num_policies: int = num_policies

        self._fallback_target_velocities: npt.NDArray[np.float64] = np.zeros(
            (self._num_policies), dtype=np.float64
        )
        self._speed_limit_fractions: npt.NDArray[np.float64] = np.zeros(
            (self._num_policies), dtype=np.float64
        )
        self._min_gap_to_lead_agent: npt.NDArray[np.float64] = np.zeros(
            (self._num_policies), dtype=np.float64
        )
        self._headway_time: npt.NDArray[np.float64] = np.zeros(
            (self._num_policies), dtype=np.float64
        )
        self._accel_max: npt.NDArray[np.float64] = np.zeros(
            (self._num_policies), dtype=np.float64
        )

        self._decel_max: npt.NDArray[np.float64] = np.zeros(
            (self._num_policies), dtype=np.float64
        )

        for i in range(self._num_policies):
            self._fallback_target_velocities[i] = (
                fallback_target_velocity
                if isinstance(fallback_target_velocity, float)
                else fallback_target_velocity[i]
            )
            self._speed_limit_fractions[i] = (
                speed_limit_fraction
                if isinstance(speed_limit_fraction, float)
                else speed_limit_fraction[i]
            )
            self._min_gap_to_lead_agent[i] = (
                min_gap_to_lead_agent
                if isinstance(min_gap_to_lead_agent, float)
                else min_gap_to_lead_agent[i]
            )
            self._headway_time[i] = (
                headway_time if isinstance(headway_time, float) else headway_time[i]
            )
            self._accel_max[i] = (
                accel_max if isinstance(accel_max, float) else accel_max[i]
            )
            self._decel_max[i] = (
                decel_max if isinstance(decel_max, float) else decel_max[i]
            )

        # lazy loaded
        self._target_velocities: npt.NDArray[np.float64] = np.zeros(
            (self._num_policies), dtype=np.float64
        )

    @property
    def num_policies(self) -> int:
        """
        Getter for number of policies
        :return: int
        """
        return self._num_policies

    @property
    def max_target_velocity(self):
        """
        Getter for highest target velocity of policies
        :return: target velocity [m/s]
        """
        return np.max(self._target_velocities)

    def update(self, speed_limit_mps: float):
        """
        Updates class with current speed limit
        :param speed_limit_mps: speed limit of current lane [m/s]
        """

        if speed_limit_mps is not None:
            self._target_velocities = self._speed_limit_fractions * speed_limit_mps
        else:
            self._target_velocities = (
                self._speed_limit_fractions * self._fallback_target_velocities
            )

    def propagate(
        self,
        previous_idm_states: npt.NDArray[np.float64],
        leading_agent_states: npt.NDArray[np.float64],
        longitudinal_idcs: List[int],
        sampling_time: float,
    ) -> npt.NDArray[np.float64]:
        """
        Propagates IDM policies for one time-step
        :param previous_idm_states: array containing previous state
        :param leading_agent_states: array contains leading vehicle information
        :param longitudinal_idcs: indices of policies to be applied over a batch-dim
        :param sampling_time: time to propagate forward [s]
        :return: array containing propagated state values
        """

        assert len(previous_idm_states) == len(longitudinal_idcs) and len(
            leading_agent_states
        ) == len(
            longitudinal_idcs
        ), "PDMIDMPolicy: propagate function requires equal length of input arguments!"

        # state variables
        x_agent, v_agent = (
            previous_idm_states[:, StateIDMIndex.PROGRESS],
            previous_idm_states[:, StateIDMIndex.VELOCITY],
        )

        x_lead, v_lead, l_r_lead = (
            leading_agent_states[:, LeadingAgentIndex.PROGRESS],
            leading_agent_states[:, LeadingAgentIndex.VELOCITY],
            leading_agent_states[:, LeadingAgentIndex.LENGTH_REAR],
        )

        # parameters
        target_velocity, min_gap_to_lead_agent, headway_time, accel_max, decel_max = (
            self._target_velocities[longitudinal_idcs],
            self._min_gap_to_lead_agent[longitudinal_idcs],
            self._headway_time[longitudinal_idcs],
            self._accel_max[longitudinal_idcs],
            self._decel_max[longitudinal_idcs],
        )

        # TODO: add as parameter
        acceleration_exponent = 10

        # convenience definitions
        s_star = (
            min_gap_to_lead_agent
            + v_agent * headway_time
            + (v_agent * (v_agent - v_lead)) / (2 * np.sqrt(accel_max * decel_max))
        )

        s_alpha = np.maximum(
            x_lead - x_agent - l_r_lead, min_gap_to_lead_agent
        )  # clamp to avoid zero division

        # differential equations
        x_agent_dot = v_agent
        v_agent_dot = accel_max * (
            1
            - (v_agent / target_velocity) ** acceleration_exponent
            - (s_star / s_alpha) ** 2
        )

        # clip values
        v_agent_dot = np.clip(v_agent_dot, -decel_max, accel_max)

        next_idm_states: npt.NDArray[np.float64] = np.zeros(
            (len(longitudinal_idcs), len(StateIDMIndex)), dtype=np.float64
        )
        next_idm_states[:, StateIDMIndex.PROGRESS] = (
            x_agent + sampling_time * x_agent_dot
        )
        next_idm_states[:, StateIDMIndex.VELOCITY] = (
            v_agent + sampling_time * v_agent_dot
        )

        return next_idm_states
