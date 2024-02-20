from typing import List

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.state_representation import StateSE2

from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    SE2Index,
)


def normalize_angle(angle):
    """
    Map a angle in range [-π, π]
    :param angle: any angle as float
    :return: normalized angle
    """
    return np.arctan2(np.sin(angle), np.cos(angle))


def parallel_discrete_path(
    discrete_path: List[StateSE2], offset=float
) -> List[StateSE2]:
    """
    Creates a parallel discrete path for a given offset.
    :param discrete_path: baseline path (x,y,θ)
    :param offset: parall loffset
    :return: parallel discrete path
    """
    parallel_discrete_path = []
    for state in discrete_path:
        theta = state.heading + np.pi / 2
        x_new = state.x + np.cos(theta) * offset
        y_new = state.y + np.sin(theta) * offset
        parallel_discrete_path.append(StateSE2(x_new, y_new, state.heading))
    return parallel_discrete_path


def translate_lon_and_lat(
    centers: npt.NDArray[np.float64],
    headings: npt.NDArray[np.float64],
    lon: float,
    lat: float,
) -> npt.NDArray[np.float64]:
    """
    Translate the position component of an centers point array
    :param centers: array to be translated
    :param headings: array with heading angles
    :param lon: [m] distance by which a point should be translated in longitudinal direction
    :param lat: [m] distance by which a point should be translated in lateral direction
    :return array of translated coordinates
    """
    half_pi = np.pi / 2.0
    translation: npt.NDArray[np.float64] = np.stack(
        [
            (lat * np.cos(headings + half_pi)) + (lon * np.cos(headings)),
            (lat * np.sin(headings + half_pi)) + (lon * np.sin(headings)),
        ],
        axis=-1,
    )
    return centers + translation


def calculate_progress(path: List[StateSE2]) -> List[float]:
    """
    Calculate the cumulative progress of a given path.
    :param path: a path consisting of StateSE2 as waypoints
    :return: a cumulative list of progress
    """
    x_position = [point.x for point in path]
    y_position = [point.y for point in path]
    x_diff = np.diff(x_position)
    y_diff = np.diff(y_position)
    points_diff: npt.NDArray[np.float64] = np.concatenate(
        ([x_diff], [y_diff]), axis=0, dtype=np.float64
    )
    progress_diff = np.append(0.0, np.linalg.norm(points_diff, axis=0))
    return np.cumsum(progress_diff, dtype=np.float64)  # type: ignore


def convert_absolute_to_relative_se2_array(
    origin: StateSE2, state_se2_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Converts an StateSE2 array from global to relative coordinates.
    :param origin: origin pose of relative coords system
    :param state_se2_array: array of SE2 states with (x,y,θ) in last dim
    :return: SE2 coords array in relative coordinates
    """
    assert len(SE2Index) == state_se2_array.shape[-1]

    theta = -origin.heading
    origin_array = np.array([[origin.x, origin.y, origin.heading]], dtype=np.float64)

    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    points_rel = state_se2_array - origin_array
    points_rel[..., :2] = points_rel[..., :2] @ R.T
    points_rel[:, 2] = normalize_angle(points_rel[:, 2])

    return points_rel
