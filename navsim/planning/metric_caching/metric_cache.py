from __future__ import annotations

from typing import List
from pathlib import Path
from dataclasses import dataclass

import lzma
import pickle

from nuplan.common.utils.io_utils import save_buffer
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

from navsim.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_observation import PDMObservation
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap


@dataclass
class MetricCache:
    """Dataclass for storing metric computation information."""

    file_path: Path
    trajectory: InterpolatedTrajectory
    ego_state: EgoState

    observation: PDMObservation
    centerline: PDMPath
    route_lane_ids: List[str]
    drivable_area_map: PDMDrivableMap

    def dump(self) -> None:
        """Dump metric cache to pickle with lzma compression."""
        # TODO: check if file_path must really be pickled
        pickle_object = pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
        save_buffer(self.file_path, lzma.compress(pickle_object, preset=0))
