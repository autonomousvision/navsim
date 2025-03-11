from __future__ import annotations

import lzma
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.utils.io_utils import save_buffer
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

from navsim.common.dataclasses import Trajectory
from navsim.common.enums import SceneFrameType
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_observation import PDMObservation
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath


@dataclass
class MapParameters:
    map_root: str
    map_version: str
    map_name: str


@dataclass
class MetricCache:
    """Dataclass for storing metric computation information."""

    file_path: Path
    log_name: str
    timepoint: TimePoint
    scene_type: SceneFrameType
    trajectory: InterpolatedTrajectory
    human_trajectory: Optional[Trajectory]  # not available for synthetic scenes
    past_human_trajectory: InterpolatedTrajectory
    ego_state: EgoState

    observation: PDMObservation
    centerline: PDMPath
    route_lane_ids: List[str]
    drivable_area_map: PDMDrivableMap

    past_detections_tracks: List[DetectionsTracks]  # past objects at 2Hz
    current_tracked_objects: List[DetectionsTracks]  # List containing only current objects
    future_tracked_objects: List[DetectionsTracks]  # interpolated at 10Hz

    map_parameters: MapParameters

    def dump(self) -> None:
        """Dump metric cache to pickle with lzma compression."""
        # TODO: check if file_path must really be pickled
        pickle_object = pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
        save_buffer(self.file_path, lzma.compress(pickle_object, preset=0))
