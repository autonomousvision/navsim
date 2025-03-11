# TODO: Move this file for common usage (not specific for PDM)

from __future__ import annotations

import warnings
from typing import Any, List, Tuple, Type, Union

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.state_representation import StateSE2
from scipy.interpolate import interp1d
from shapely.creation import linestrings
from shapely.geometry import LineString
from shapely.ops import substring

from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    array_to_states_se2,
    states_se2_to_array,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import SE2Index
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import calculate_progress, normalize_angle


class PDMPath:
    """Class representing a path to interpolate for PDM."""

    def __init__(self, discrete_path: List[StateSE2]):
        """
        Constructor for PDMPath
        :param discrete_path: list of (x,y,θ) values
        """

        # attribute
        self._discrete_path = discrete_path

        # loaded during initialization
        self._states_se2_array = states_se2_to_array(discrete_path)
        self._states_se2_array[:, SE2Index.HEADING] = np.unwrap(self._states_se2_array[:, SE2Index.HEADING], axis=0)
        self._progress = calculate_progress(discrete_path)
        self._linestring = linestrings(self._states_se2_array[:, : SE2Index.HEADING])
        self._interpolator = interp1d(self._progress, self._states_se2_array, axis=0)

    def __reduce__(self) -> Tuple[Type[PDMPath], Tuple[Any, ...]]:
        """Helper for pickling."""
        return self.__class__, (self._discrete_path,)

    @property
    def discrete_path(self):
        """Getter for discrete StateSE2 objects of path."""
        return self._discrete_path

    @property
    def length(self):
        """Getter for length of path."""
        return self._progress[-1]

    @property
    def linestring(self) -> LineString:
        """Getter for shapely's linestring of path."""
        return self._linestring

    def project(self, points: Any) -> Any:
        warnings.filterwarnings(
            "ignore", message="invalid value encountered in line_locate_point", category=RuntimeWarning
        )
        return self._linestring.project(points)

    def interpolate(
        self,
        distances: Union[List[float], npt.NDArray[np.float64]],
        as_array=False,
    ) -> Union[npt.NDArray[np.object_], npt.NDArray[np.float64]]:
        """
        Calculates (x,y,θ) for a given distance along the path.
        :param distances: list of array of distance values
        :param as_array: whether to return in array representation, defaults to False
        :return: array of StateSE2 class or (x,y,θ) values
        """
        clipped_distances = np.clip(distances, 1e-5, self.length)
        interpolated_se2_array = self._interpolator(clipped_distances)
        interpolated_se2_array[..., 2] = normalize_angle(interpolated_se2_array[..., 2])
        interpolated_se2_array[np.isnan(interpolated_se2_array)] = 0.0

        if as_array:
            return interpolated_se2_array

        return array_to_states_se2(interpolated_se2_array)

    def substring(self, start_distance: float, end_distance: float) -> LineString:
        """
        Creates a sub-linestring between start and ending distances.
        :param start_distance: distance along the path to start [m]
        :param end_distance:  distance along the path to end [m]
        :return: LineString
        """

        # try faster method fist
        start_distance = np.clip(start_distance, 0.0, self.length)
        end_distance = np.clip(end_distance, 0.0, self.length)
        in_interval = np.logical_and(start_distance <= self._progress, self._progress <= end_distance)
        coordinates = self._states_se2_array[in_interval, :2]
        if len(coordinates) > 1:
            return LineString(coordinates)

        # fallback to slower method of shapely
        return substring(self.linestring, start_distance, end_distance)
