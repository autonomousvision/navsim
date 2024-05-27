from typing import Optional, Tuple
from enum import IntEnum

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.vehicle_parameters import VehicleParameters, get_pacifica_parameters
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration

from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import DynamicStateIndex, StateIndex
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import normalize_angle
from navsim.planning.simulation.planner.pdm_planner.simulation.batch_lqr_utils import (
    _generate_profile_from_initial_condition_and_derivatives,
    get_velocity_curvature_profiles_with_derivatives_from_poses,
)


class LateralStateIndex(IntEnum):
    """
    Index mapping for the lateral dynamics state vector.
    """

    LATERAL_ERROR = 0  # [m] The lateral error with respect to the planner centerline at the vehicle's rear axle center.
    HEADING_ERROR = 1  # [rad] The heading error "".
    STEERING_ANGLE = 2  # [rad] The wheel angle relative to the longitudinal axis of the vehicle.


class BatchLQRTracker:
    """
    Implements an LQR tracker for a kinematic bicycle model.

    Tracker operates on a batch of proposals. Implementation directly based on the nuplan-devkit
    Link: https://github.com/motional/nuplan-devkit

    We decouple into two subsystems, longitudinal and lateral, with small angle approximations for linearization.
    We then solve two sequential LQR subproblems to find acceleration and steering rate inputs.

    Longitudinal Subsystem:
        States: [velocity]
        Inputs: [acceleration]
        Dynamics (continuous time):
            velocity_dot = acceleration

    Lateral Subsystem (After Linearization/Small Angle Approximation):
        States: [lateral_error, heading_error, steering_angle]
        Inputs: [steering_rate]
        Parameters: [velocity, curvature]
        Dynamics (continuous time):
            lateral_error_dot  = velocity * heading_error
            heading_error_dot  = velocity * (steering_angle / wheelbase_length - curvature)
            steering_angle_dot = steering_rate

    The continuous time dynamics are discretized using Euler integration and zero-order-hold on the input.
    In case of a stopping reference, we use a simplified stopping P controller instead of LQR.

    The final control inputs passed on to the motion model are:
        - acceleration
        - steering_rate
    """

    def __init__(
        self,
        q_longitudinal: npt.NDArray[np.float64] = [10.0],
        r_longitudinal: npt.NDArray[np.float64] = [1.0],
        q_lateral: npt.NDArray[np.float64] = [1.0, 10.0, 0.0],
        r_lateral: npt.NDArray[np.float64] = [1.0],
        discretization_time: float = 0.1,
        tracking_horizon: int = 10,
        jerk_penalty: float = 1e-4,
        curvature_rate_penalty: float = 1e-2,
        stopping_proportional_gain: float = 0.5,
        stopping_velocity: float = 0.2,
        vehicle: VehicleParameters = get_pacifica_parameters(),
    ):
        """
        Constructor for LQR controller
        :param q_longitudinal: The weights for the Q matrix for the longitudinal subystem.
        :param r_longitudinal: The weights for the R matrix for the longitudinal subystem.
        :param q_lateral: The weights for the Q matrix for the lateral subystem.
        :param r_lateral: The weights for the R matrix for the lateral subystem.
        :param discretization_time: [s] The time interval used for discretizing the continuous time dynamics.
        :param tracking_horizon: How many discrete time steps ahead to consider for the LQR objective.
        :param stopping_proportional_gain: The proportional_gain term for the P controller when coming to a stop.
        :param stopping_velocity: [m/s] The velocity below which we are deemed to be stopping and we don't use LQR.
        :param vehicle: Vehicle parameters
        """
        # Longitudinal LQR Parameters
        assert len(q_longitudinal) == 1, "q_longitudinal should have 1 element (velocity)."
        assert len(r_longitudinal) == 1, "r_longitudinal should have 1 element (acceleration)."
        self._q_longitudinal: float = q_longitudinal[0]
        self._r_longitudinal: float = r_longitudinal[0]

        # Lateral LQR Parameters
        assert len(q_lateral) == 3, "q_lateral should have 3 elements (lateral_error, heading_error, steering_angle)."
        assert len(r_lateral) == 1, "r_lateral should have 1 element (steering_rate)."
        self._q_lateral: npt.NDArray[np.float64] = np.diag(q_lateral)
        self._r_lateral: npt.NDArray[np.float64] = np.diag(r_lateral)

        # Common LQR Parameters
        # Note we want a horizon > 1 so that steering rate actually can impact lateral/heading error in discrete time.
        assert discretization_time > 0.0, "The discretization_time should be positive."
        assert (
            tracking_horizon > 1
        ), "We expect the horizon to be greater than 1 - else steering_rate has no impact with Euler integration."
        self._discretization_time = discretization_time
        self._tracking_horizon = tracking_horizon
        self._wheel_base = vehicle.wheel_base

        # Velocity/Curvature Estimation Parameters
        assert jerk_penalty > 0.0, "The jerk penalty must be positive."
        assert curvature_rate_penalty > 0.0, "The curvature rate penalty must be positive."
        self._jerk_penalty = jerk_penalty
        self._curvature_rate_penalty = curvature_rate_penalty

        # Stopping Controller Parameters
        assert stopping_proportional_gain > 0, "stopping_proportional_gain has to be greater than 0."
        assert stopping_velocity > 0, "stopping_velocity has to be greater than 0."
        self._stopping_proportional_gain = stopping_proportional_gain
        self._stopping_velocity = stopping_velocity

        # lazy loaded
        self._proposal_states: Optional[npt.NDArray[np.float64]] = None
        self._initialized: bool = False

    def update(self, proposal_states: npt.NDArray[np.float64]) -> None:
        """
        Loads proposal state array and resets velocity, and curvature profile.
        :param proposal_states: array representation of proposals.
        """
        self._proposal_states: npt.NDArray[np.float64] = proposal_states
        self._velocity_profile, self._curvature_profile = None, None
        self._initialized = True

    def track_trajectory(
        self,
        current_iteration: SimulationIteration,
        next_iteration: SimulationIteration,
        initial_states: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Calculates the command values given the proposals to track.
        :param current_iteration: current simulation iteration.
        :param next_iteration: desired next simulation iteration.
        :param initial_states: array representation of current ego states.
        :return: command values for motion model.
        """
        assert self._initialized, "BatchLQRTracker: Run update first to load proposal states!"

        batch_size = len(initial_states)
        (
            initial_velocity,
            initial_lateral_state_vector,
        ) = self._compute_initial_velocity_and_lateral_state(
            current_iteration, initial_states
        )  # (batch), (batch, 3)

        (
            reference_velocities,
            curvature_profiles,
        ) = self._compute_reference_velocity_and_curvature_profile(
            current_iteration
        )  # (batch), (batch, 10)

        # create output arrays
        accel_cmds = np.zeros(batch_size, dtype=np.float64)
        steering_rate_cmds = np.zeros(batch_size, dtype=np.float64)

        # 1. Stopping Controller
        should_stop_mask = np.logical_and(
            reference_velocities <= self._stopping_velocity,
            initial_velocity <= self._stopping_velocity,
        )
        stopping_accel_cmd, stopping_steering_rate_cmd = self._stopping_controller(
            initial_velocity[should_stop_mask], reference_velocities[should_stop_mask]
        )
        accel_cmds[should_stop_mask] = stopping_accel_cmd
        steering_rate_cmds[should_stop_mask] = stopping_steering_rate_cmd

        # 2. Regular Controller
        accel_cmds[~should_stop_mask] = self._longitudinal_lqr_controller(
            initial_velocity[~should_stop_mask], reference_velocities[~should_stop_mask]
        )

        velocity_profiles = _generate_profile_from_initial_condition_and_derivatives(
            initial_condition=initial_velocity[~should_stop_mask],
            derivatives=np.repeat(accel_cmds[~should_stop_mask, None], self._tracking_horizon, axis=-1),
            discretization_time=self._discretization_time,
        )[:, : self._tracking_horizon]

        steering_rate_cmds[~should_stop_mask] = self._lateral_lqr_controller(
            initial_lateral_state_vector[~should_stop_mask],
            velocity_profiles,
            curvature_profiles[~should_stop_mask],
        )

        command_states = np.zeros((batch_size, len(DynamicStateIndex)), dtype=np.float64)
        command_states[:, DynamicStateIndex.ACCELERATION_X] = accel_cmds
        command_states[:, DynamicStateIndex.STEERING_RATE] = steering_rate_cmds

        return command_states

    def _compute_initial_velocity_and_lateral_state(
        self,
        current_iteration: SimulationIteration,
        initial_values: npt.NDArray[np.float64],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        This method projects the initial tracking error into vehicle/Frenet frame.  It also extracts initial velocity.
        :param current_iteration: Used to get the current time.
        :param initial_state: The current state for ego.
        :param trajectory: The reference trajectory we are tracking.
        :return: Initial velocity [m/s] and initial lateral state.
        """
        # Get initial trajectory state.
        initial_trajectory_values = self._proposal_states[:, current_iteration.index]

        # Determine initial error state.
        x_errors = initial_values[:, StateIndex.X] - initial_trajectory_values[:, StateIndex.X]
        y_errors = initial_values[:, StateIndex.Y] - initial_trajectory_values[:, StateIndex.Y]
        heading_references = initial_trajectory_values[:, StateIndex.HEADING]

        lateral_errors = -x_errors * np.sin(heading_references) + y_errors * np.cos(heading_references)
        heading_errors = normalize_angle(initial_values[:, StateIndex.HEADING] - heading_references)

        # Return initial velocity and lateral state vector.
        initial_velocities = initial_values[:, StateIndex.VELOCITY_X]

        initial_lateral_state_vector = np.stack(
            [
                lateral_errors,
                heading_errors,
                initial_values[:, StateIndex.STEERING_ANGLE],
            ],
            axis=-1,
        )

        return initial_velocities, initial_lateral_state_vector

    def _compute_reference_velocity_and_curvature_profile(
        self,
        current_iteration: SimulationIteration,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        This method computes reference velocity and curvature profile based on the reference trajectory.
        We use a lookahead time equal to self._tracking_horizon * self._discretization_time.
        :param current_iteration: Used to get the current time.
        :param trajectory: The reference trajectory we are tracking.
        :return: The reference velocity [m/s] and curvature profile [rad] to track.
        """

        poses = self._proposal_states[..., StateIndex.STATE_SE2]

        if self._velocity_profile is None or self._curvature_profile is None:
            (
                self._velocity_profile,
                acceleration_profile,
                self._curvature_profile,
                curvature_rate_profile,
            ) = get_velocity_curvature_profiles_with_derivatives_from_poses(
                discretization_time=self._discretization_time,
                poses=poses,
                jerk_penalty=self._jerk_penalty,
                curvature_rate_penalty=self._curvature_rate_penalty,
            )

        batch_size, num_poses = self._velocity_profile.shape
        reference_idx = min(current_iteration.index + self._tracking_horizon, num_poses - 1)
        reference_velocities = self._velocity_profile[:, reference_idx]

        reference_curvature_profiles = np.zeros((batch_size, self._tracking_horizon), dtype=np.float64)

        reference_length = reference_idx - current_iteration.index
        reference_curvature_profiles[:, 0:reference_length] = self._curvature_profile[
            :, current_iteration.index : reference_idx
        ]

        if reference_length < self._tracking_horizon:
            reference_curvature_profiles[:, reference_length:] = self._curvature_profile[:, reference_idx, None]

        return reference_velocities, reference_curvature_profiles

    def _stopping_controller(
        self,
        initial_velocities: npt.NDArray[np.float64],
        reference_velocities: npt.NDArray[np.float64],
    ) -> Tuple[float, float]:
        """
        Apply proportional controller when at near-stop conditions.
        :param initial_velocity: [m/s] The current velocity of ego.
        :param reference_velocity: [m/s] The reference velocity to track.
        :return: Acceleration [m/s^2] and zero steering_rate [rad/s] command.
        """
        accel = -self._stopping_proportional_gain * (initial_velocities - reference_velocities)
        return accel, 0.0

    def _longitudinal_lqr_controller(
        self,
        initial_velocities: npt.NDArray[np.float64],
        reference_velocities: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        This longitudinal controller determines an acceleration input to minimize velocity error at a lookahead time.
        :param initial_velocity: [m/s] The current velocity of ego.
        :param reference_velocity: [m/s] The reference_velocity to track at a lookahead time.
        :return: Acceleration [m/s^2] command based on LQR.
        """
        # We assume that we hold the acceleration constant for the entire tracking horizon.
        # Given this, we can show the following where N = self._tracking_horizon and dt = self._discretization_time:
        # velocity_N = velocity_0 + (N * dt) * acceleration

        batch_size = len(initial_velocities)

        A: npt.NDArray[np.float64] = np.ones(batch_size, dtype=np.float64)

        B: npt.NDArray[np.float64] = np.zeros(batch_size, dtype=np.float64)
        B.fill(self._tracking_horizon * self._discretization_time)

        g: npt.NDArray[np.float64] = np.zeros(batch_size, dtype=np.float64)

        accel_cmds = self._solve_one_step_longitudinal_lqr(
            initial_state=initial_velocities,
            reference_state=reference_velocities,
            A=A,
            B=B,
            g=g,
        )

        return accel_cmds

    def _lateral_lqr_controller(
        self,
        initial_lateral_state_vector: npt.NDArray[np.float64],
        velocity_profile: npt.NDArray[np.float64],
        curvature_profile: npt.NDArray[np.float64],
    ) -> float:
        """
        This lateral controller determines a steering_rate input to minimize lateral errors at a lookahead time.
        It requires a velocity sequence as a parameter to ensure linear time-varying lateral dynamics.
        :param initial_lateral_state_vector: The current lateral state of ego.
        :param velocity_profile: [m/s] The velocity over the entire self._tracking_horizon-step lookahead.
        :param curvature_profile: [rad] The curvature over the entire self._tracking_horizon-step lookahead..
        :return: Steering rate [rad/s] command based on LQR.
        """
        assert velocity_profile.shape[-1] == self._tracking_horizon, (
            f"The linearization velocity sequence should have length {self._tracking_horizon} "
            f"but is {len(velocity_profile)}."
        )
        assert curvature_profile.shape[-1] == self._tracking_horizon, (
            f"The linearization curvature sequence should have length {self._tracking_horizon} "
            f"but is {len(curvature_profile)}."
        )

        batch_dim = velocity_profile.shape[0]

        # Set up the lateral LQR problem using the constituent linear time-varying (affine) system dynamics.
        # Ultimately, we'll end up with the following problem structure where N = self._tracking_horizon:
        # lateral_error_N = A @ lateral_error_0 + B @ steering_rate + g
        n_lateral_states = len(LateralStateIndex)

        I: npt.NDArray[np.float64] = np.eye(n_lateral_states, dtype=np.float64)

        in_matrix: npt.NDArray[np.float64] = np.zeros((n_lateral_states, 1), np.float64)  # no batch dim
        in_matrix[LateralStateIndex.STEERING_ANGLE] = self._discretization_time

        states_matrix_at_step: npt.NDArray[np.float64] = np.tile(
            I[None, None, ...], [self._tracking_horizon, batch_dim, 1, 1]
        )  # (horizon, batch, 3, 3)

        states_matrix_at_step[:, :, LateralStateIndex.LATERAL_ERROR, LateralStateIndex.HEADING_ERROR] = (
            velocity_profile.T * self._discretization_time
        )

        states_matrix_at_step[:, :, LateralStateIndex.HEADING_ERROR, LateralStateIndex.STEERING_ANGLE] = (
            velocity_profile.T * self._discretization_time / self._wheel_base
        )

        affine_terms: npt.NDArray[np.float64] = np.zeros(
            (self._tracking_horizon, batch_dim, n_lateral_states), dtype=np.float64
        )

        affine_terms[:, :, LateralStateIndex.HEADING_ERROR] = (
            -velocity_profile.T * curvature_profile.T * self._discretization_time
        )

        A: npt.NDArray[np.float64] = np.tile(I[None, ...], [batch_dim, 1, 1])  # (batch, 3, 3)
        B: npt.NDArray[np.float64] = np.zeros((batch_dim, n_lateral_states, 1), dtype=np.float64)  # (batch, 3, 1)
        g: npt.NDArray[np.float64] = np.zeros((batch_dim, n_lateral_states), dtype=np.float64)  # (batch, 3)

        for index_step, (state_matrix_at_step, affine_term) in enumerate(zip(states_matrix_at_step, affine_terms)):
            # state_matrix_at_step (batch, 3, 3)
            # affine_term (batch, 3)
            A = np.einsum("bij, bjk -> bik", state_matrix_at_step, A)  # (batch, 3, 3)
            B = np.einsum("bij, bjk -> bik", state_matrix_at_step, B) + in_matrix  # (batch, 3, 1)
            g = np.einsum("bij, bj  -> bi", state_matrix_at_step, g) + affine_term  # (batch, 3)

        steering_rate_cmd = self._solve_one_step_lateral_lqr(
            initial_state=initial_lateral_state_vector,
            A=A,
            B=B,
            g=g,
        )

        return np.squeeze(steering_rate_cmd, axis=-1)

    def _solve_one_step_longitudinal_lqr(
        self,
        initial_state: npt.NDArray[np.float64],
        reference_state: npt.NDArray[np.float64],
        A: npt.NDArray[np.float64],
        B: npt.NDArray[np.float64],
        g: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        This function uses LQR to find an optimal input to minimize tracking error in one step of dynamics.
        The dynamics are next_state = A @ initial_state + B @ input + g and our target is the reference_state.
        :param initial_state: The current state.
        :param reference_state: The desired state in 1 step (according to A,B,g dynamics).
        :param A: The state dynamics matrix.
        :param B: The input dynamics matrix.
        :param g: The offset/affine dynamics term.
        :return: LQR optimal input for the 1-step longitudinal problem.
        """
        state_error_zero_input = A * initial_state + g - reference_state
        inverse = -1 / (B * self._q_longitudinal * B + self._r_longitudinal)
        lqr_input = inverse * B * self._q_longitudinal * state_error_zero_input

        return lqr_input

    def _solve_one_step_lateral_lqr(
        self,
        initial_state: npt.NDArray[np.float64],
        A: npt.NDArray[np.float64],
        B: npt.NDArray[np.float64],
        g: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        This function uses LQR to find an optimal input to minimize tracking error in one step of dynamics.
        The dynamics are next_state = A @ initial_state + B @ input + g and our target is the reference_state.
        :param initial_state: The current state.
        :param A: The state dynamics matrix.
        :param B: The input dynamics matrix.
        :param g: The offset/affine dynamics term.
        :return: LQR optimal input for the 1-step lateral problem.
        """

        Q, R = self._q_lateral, self._r_lateral
        angle_diff_indices = [
            LateralStateIndex.HEADING_ERROR.value,
            LateralStateIndex.STEERING_ANGLE.value,
        ]
        BT = B.transpose(0, 2, 1)

        state_error_zero_input = np.einsum("bij, bj -> bi", A, initial_state) + g

        angle = state_error_zero_input[..., angle_diff_indices]
        state_error_zero_input[..., angle_diff_indices] = np.arctan2(np.sin(angle), np.cos(angle))

        BT_x_Q = np.einsum("bij, jk -> bik", BT, Q)
        Inv = -1 / (np.einsum("bij, bji -> bi", BT_x_Q, B) + R)
        Tail = np.einsum("bij, bj -> bi", BT_x_Q, state_error_zero_input)

        lqr_input = Inv * Tail

        return lqr_input
