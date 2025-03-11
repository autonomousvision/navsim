import copy

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters, get_pacifica_parameters
from nuplan.common.geometry.compute import principal_value

from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import DynamicStateIndex, StateIndex


def forward_integrate(
    init: npt.NDArray[np.float64],
    delta: npt.NDArray[np.float64],
    sampling_time: TimePoint,
) -> npt.NDArray[np.float64]:
    """
    Performs a simple euler integration.
    :param init: Initial state
    :param delta: The rate of change of the state.
    :param sampling_time: The time duration to propagate for.
    :return: The result of integration
    """
    return init + delta * sampling_time.time_s


class BatchKinematicBicycleModel:
    """
    A batch-wise operating class describing the kinematic motion model where the rear axle is the point of reference.
    """

    def __init__(
        self,
        vehicle: VehicleParameters = get_pacifica_parameters(),
        max_steering_angle: float = np.pi / 3,
        accel_time_constant: float = 0.2,
        steering_angle_time_constant: float = 0.05,
    ):
        """
        Construct BatchKinematicBicycleModel.
        :param vehicle: Vehicle parameters.
        :param max_steering_angle: [rad] Maximum absolute value steering angle allowed by model.
        :param accel_time_constant: low pass filter time constant for acceleration in s
        :param steering_angle_time_constant: low pass filter time constant for steering angle in s
        """
        self._vehicle = vehicle
        self._max_steering_angle = max_steering_angle
        self._accel_time_constant = accel_time_constant
        self._steering_angle_time_constant = steering_angle_time_constant

    def get_state_dot(self, states: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Calculates the changing rate of state array representation.
        :param states: array describing the state of the ego-vehicle
        :return: change rate across several state values
        """
        state_dots = np.zeros(states.shape, dtype=np.float64)

        longitudinal_speeds = states[:, StateIndex.VELOCITY_X]

        state_dots[:, StateIndex.X] = longitudinal_speeds * np.cos(states[:, StateIndex.HEADING])
        state_dots[:, StateIndex.Y] = longitudinal_speeds * np.sin(states[:, StateIndex.HEADING])
        state_dots[:, StateIndex.HEADING] = (
            longitudinal_speeds * np.tan(states[:, StateIndex.STEERING_ANGLE]) / self._vehicle.wheel_base
        )

        state_dots[:, StateIndex.VELOCITY_2D] = states[:, StateIndex.ACCELERATION_2D]
        state_dots[:, StateIndex.ACCELERATION_2D] = 0.0

        state_dots[:, StateIndex.STEERING_ANGLE] = states[:, StateIndex.STEERING_RATE]

        return state_dots

    def _update_commands(
        self,
        states: npt.NDArray[np.float64],
        command_states: npt.NDArray[np.float64],
        sampling_time: TimePoint,
    ) -> EgoState:
        """
        This function applies some first order control delay/a low pass filter to acceleration/steering.

        :param state: Ego state
        :param ideal_dynamic_state: The desired dynamic state for propagation
        :param sampling_time: The time duration to propagate for
        :return: propagating_state including updated dynamic_state
        """

        propagating_state: npt.NDArray[np.float64] = copy.deepcopy(states)

        dt_control = sampling_time.time_s

        accel = states[:, StateIndex.ACCELERATION_X]
        steering_angle = states[:, StateIndex.STEERING_ANGLE]

        ideal_accel_x = command_states[:, DynamicStateIndex.ACCELERATION_X]
        ideal_steering_angle = dt_control * command_states[:, DynamicStateIndex.STEERING_RATE] + steering_angle

        updated_accel_x = dt_control / (dt_control + self._accel_time_constant) * (ideal_accel_x - accel) + accel
        updated_steering_angle = (
            dt_control / (dt_control + self._steering_angle_time_constant) * (ideal_steering_angle - steering_angle)
            + steering_angle
        )
        updated_steering_rate = (updated_steering_angle - steering_angle) / dt_control

        propagating_state[:, StateIndex.ACCELERATION_X] = updated_accel_x
        propagating_state[:, StateIndex.ACCELERATION_Y] = 0.0
        propagating_state[:, StateIndex.STEERING_RATE] = updated_steering_rate

        return propagating_state

    def propagate_state(
        self,
        states: npt.NDArray[np.float64],
        command_states: npt.NDArray[np.float64],
        sampling_time: TimePoint,
    ) -> npt.NDArray[np.float64]:
        """
        Propagates ego state array forward with motion model.
        :param states: state array representation of the ego-vehicle
        :param command_states: command array representation of controller
        :param sampling_time: time to propagate [s]
        :return: updated tate array representation of the ego-vehicle
        """

        assert len(states) == len(command_states), "Batch size of states and command_states does not match!"

        propagating_state = self._update_commands(states, command_states, sampling_time)
        output_state = copy.deepcopy(states)

        # Compute state derivatives
        state_dot = self.get_state_dot(propagating_state)

        output_state[:, StateIndex.X] = forward_integrate(
            states[:, StateIndex.X], state_dot[:, StateIndex.X], sampling_time
        )
        output_state[:, StateIndex.Y] = forward_integrate(
            states[:, StateIndex.Y], state_dot[:, StateIndex.Y], sampling_time
        )

        output_state[:, StateIndex.HEADING] = principal_value(
            forward_integrate(
                states[:, StateIndex.HEADING],
                state_dot[:, StateIndex.HEADING],
                sampling_time,
            )
        )

        output_state[:, StateIndex.VELOCITY_X] = forward_integrate(
            states[:, StateIndex.VELOCITY_X],
            state_dot[:, StateIndex.VELOCITY_X],
            sampling_time,
        )

        # Lateral velocity is always zero in kinematic bicycle model
        output_state[:, StateIndex.VELOCITY_Y] = 0.0

        # Integrate steering angle and clip to bounds
        output_state[:, StateIndex.STEERING_ANGLE] = np.clip(
            forward_integrate(
                propagating_state[:, StateIndex.STEERING_ANGLE],
                state_dot[:, StateIndex.STEERING_ANGLE],
                sampling_time,
            ),
            -self._max_steering_angle,
            self._max_steering_angle,
        )

        output_state[:, StateIndex.ANGULAR_VELOCITY] = (
            output_state[:, StateIndex.VELOCITY_X]
            * np.tan(output_state[:, StateIndex.STEERING_ANGLE])
            / self._vehicle.wheel_base
        )

        output_state[:, StateIndex.ACCELERATION_2D] = state_dot[:, StateIndex.VELOCITY_2D]

        output_state[:, StateIndex.ANGULAR_ACCELERATION] = (
            output_state[:, StateIndex.ANGULAR_VELOCITY] - states[:, StateIndex.ANGULAR_VELOCITY]
        ) / sampling_time.time_s

        output_state[:, StateIndex.STEERING_RATE] = state_dot[:, StateIndex.STEERING_ANGLE]

        return output_state
