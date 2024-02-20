from __future__ import annotations

from typing import Any, Tuple, Type, Union

import numpy as np
import numpy.typing as npt

from scipy.interpolate import interp1d
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    normalize_angle,
)


class StateInterpolator:

    def __init__(self, state_array: npt.NDArray[np.float64]):

        # attribute
        self._state_array = state_array

        # loaded during initialization
        self._time = state_array[:, 0]
        self._states = state_array[:, 1:]
        
        # unwrap heading angle
        self._states[:, 2] = np.unwrap(self._states[:, 2], axis=0)
        self._interpolator = interp1d(self._time, self._states, axis=0)

    def __reduce__(self) -> Tuple[Type[StateInterpolator], Tuple[Any, ...]]:
        """Helper for pickling."""
        return self.__class__, (self.state_array,)
    
    @property
    def start_time(self):
        return self._time[0]
    
    @property
    def end_time(self):
        return self._time[-1]

    def interpolate(
        self,
        time: float,
    ) -> Union[npt.NDArray[np.object_], npt.NDArray[np.float64]]:
        
        if self.start_time <= time <= self.end_time:

            interpolated_state = self._interpolator(time)
            interpolated_state[2] = normalize_angle(interpolated_state[2])
            return interpolated_state
            
        return None
