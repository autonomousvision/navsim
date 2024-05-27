from typing import List

import numpy as np
import numpy.typing as npt
import shapely

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters

from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import BBCoordsIndex, SE2Index, StateIndex
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import translate_lon_and_lat


def array_to_state_se2(array: npt.NDArray[np.float64]) -> StateSE2:
    """
    Converts array representation to single StateSE2.
    :param array: array filled with (x,y,θ)
    :return: StateSE2 class
    """
    return StateSE2(array[0], array[1], array[2])


# use numpy vectorize function to apply on last dim
array_to_state_se2_vectorize = np.vectorize(array_to_state_se2, signature="(n)->()")


def array_to_states_se2(array: npt.NDArray[np.float64]) -> npt.NDArray[np.object_]:
    """
    Converts array representation to StateSE2 over last dim.
    :param array: array filled with (x,y,θ) on last dim
    :return: array of StateSE2 class
    """
    assert array.shape[-1] == len(SE2Index)
    return array_to_state_se2_vectorize(array)


def state_se2_to_array(state_se2: StateSE2) -> npt.NDArray[np.float64]:
    """
    Converts StateSE2 to array representation.
    :param state_se2: class containing (x,y,θ)
    :return: array containing (x,y,θ)
    """
    array = np.zeros(len(SE2Index), dtype=np.float64)
    array[SE2Index.X] = state_se2.x
    array[SE2Index.Y] = state_se2.y
    array[SE2Index.HEADING] = state_se2.heading
    return array


def states_se2_to_array(states_se2: List[StateSE2]) -> npt.NDArray[np.float64]:
    """
    Converts list of StateSE2 object to array representation
    :param states_se2: list of StateSE2 object's
    :return: array representation of states
    """
    state_se2_array = np.zeros((len(states_se2), len(SE2Index)), dtype=np.float64)
    for i, state_se2 in enumerate(states_se2):
        state_se2_array[i] = state_se2_to_array(state_se2)
    return state_se2_array


def ego_state_to_state_array(ego_state: EgoState) -> npt.NDArray[np.float64]:
    """
    Converts an ego state into an array representation (drops time-stamps and vehicle parameters)
    :param ego_state: ego state class
    :return: array containing ego state values
    """
    state_array = np.zeros(StateIndex.size(), dtype=np.float64)

    state_array[StateIndex.STATE_SE2] = ego_state.rear_axle.serialize()
    state_array[StateIndex.VELOCITY_2D] = ego_state.dynamic_car_state.rear_axle_velocity_2d.array
    state_array[StateIndex.ACCELERATION_2D] = ego_state.dynamic_car_state.rear_axle_acceleration_2d.array

    state_array[StateIndex.STEERING_ANGLE] = ego_state.tire_steering_angle
    state_array[StateIndex.STEERING_RATE] = ego_state.dynamic_car_state.tire_steering_rate

    state_array[StateIndex.ANGULAR_VELOCITY] = ego_state.dynamic_car_state.angular_velocity
    state_array[StateIndex.ANGULAR_ACCELERATION] = ego_state.dynamic_car_state.angular_acceleration

    return state_array


def ego_states_to_state_array(ego_states: List[EgoState]) -> npt.NDArray[np.float64]:
    """
    Converts a list of ego states into an array representation (drops time-stamps and vehicle parameters)
    :param ego_state: ego state class
    :return: array containing ego state values
    """
    state_array = np.array(
        [ego_state_to_state_array(ego_state) for ego_state in ego_states],
        dtype=np.float64,
    )
    return state_array


def state_array_to_ego_state(
    state_array: npt.NDArray[np.float64],
    time_point: TimePoint,
    vehicle_parameters: VehicleParameters,
) -> EgoState:
    """
    Converts array representation of ego state back to ego state class.
    :param state_array: array representation of ego states
    :param time_point: time point of state
    :param vehicle_parameters: vehicle parameter of ego
    :return: nuPlan's EgoState object
    """
    return EgoState.build_from_rear_axle(
        rear_axle_pose=StateSE2(*state_array[StateIndex.STATE_SE2]),
        rear_axle_velocity_2d=StateVector2D(*state_array[StateIndex.VELOCITY_2D]),
        rear_axle_acceleration_2d=StateVector2D(*state_array[StateIndex.ACCELERATION_2D]),
        tire_steering_angle=state_array[StateIndex.STEERING_ANGLE],
        time_point=time_point,
        vehicle_parameters=vehicle_parameters,
        is_in_auto_mode=True,
        angular_vel=state_array[StateIndex.ANGULAR_VELOCITY],
        angular_accel=state_array[StateIndex.ANGULAR_ACCELERATION],
        tire_steering_rate=state_array[StateIndex.STEERING_RATE],
    )


def state_array_to_ego_states(
    state_array: npt.NDArray[np.float64],
    time_points: List[TimePoint],
    vehicle_parameter: VehicleParameters,
) -> List[EgoState]:
    """
    Converts array representation of ego states back to list of ego state class.
    :param state_array: array representation of ego states
    :param time_point: list of time point of state array
    :param vehicle_parameters: vehicle parameter of ego
    :return: list nuPlan's EgoState object
    """
    ego_states_list: List[EgoState] = []
    for i, time_point in enumerate(time_points):
        state = state_array[i] if i < len(state_array) else state_array[-1]
        ego_states_list.append(state_array_to_ego_state(state, time_point, vehicle_parameter))
    return ego_states_list


def state_array_to_coords_array(
    states: npt.NDArray[np.float64],
    vehicle_parameters: VehicleParameters,
) -> npt.NDArray[np.float64]:
    """
    Converts multi-dim array representation of ego states to bounding box coordinates
    :param state_array: array representation of ego states
    :param vehicle_parameters: vehicle parameter of ego
    :return: multi-dim array bounding box coordinates
    """
    n_batch, n_time, n_states = states.shape

    half_length, half_width, rear_axle_to_center = (
        vehicle_parameters.half_length,
        vehicle_parameters.half_width,
        vehicle_parameters.rear_axle_to_center,
    )

    headings = states[..., StateIndex.HEADING]
    cos, sin = np.cos(headings), np.sin(headings)

    # calculate ego center from rear axle
    rear_axle_to_center_translate = np.stack([rear_axle_to_center * cos, rear_axle_to_center * sin], axis=-1)

    ego_centers: npt.NDArray[np.float64] = states[..., StateIndex.POINT] + rear_axle_to_center_translate

    coords_array: npt.NDArray[np.float64] = np.zeros((n_batch, n_time, len(BBCoordsIndex), 2), dtype=np.float64)

    coords_array[:, :, BBCoordsIndex.CENTER] = ego_centers

    coords_array[:, :, BBCoordsIndex.FRONT_LEFT] = translate_lon_and_lat(ego_centers, headings, half_length, half_width)
    coords_array[:, :, BBCoordsIndex.FRONT_RIGHT] = translate_lon_and_lat(
        ego_centers, headings, half_length, -half_width
    )
    coords_array[:, :, BBCoordsIndex.REAR_LEFT] = translate_lon_and_lat(ego_centers, headings, -half_length, half_width)
    coords_array[:, :, BBCoordsIndex.REAR_RIGHT] = translate_lon_and_lat(
        ego_centers, headings, -half_length, -half_width
    )

    return coords_array


def coords_array_to_polygon_array(
    coords: npt.NDArray[np.float64],
) -> npt.NDArray[np.object_]:
    """
    Converts multi-dim array of bounding box coords of to polygons
    :param coords: bounding box coords (including corners and center)
    :return: array of shapely's polygons
    """
    # create coords copy and use center point for closed exterior
    coords_exterior: npt.NDArray[np.float64] = coords.copy()
    coords_exterior[..., BBCoordsIndex.CENTER, :] = coords_exterior[..., BBCoordsIndex.FRONT_LEFT, :]

    # load new coordinates into polygon array
    polygons = shapely.creation.polygons(coords_exterior)

    return polygons
