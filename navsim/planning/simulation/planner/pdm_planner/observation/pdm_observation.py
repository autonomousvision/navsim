import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import shapely.creation
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.tracked_objects import TrackedObject
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import (
    TrafficLightStatusData,
    TrafficLightStatusType,
)
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import (
    DetectionsTracks,
    Observation,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from shapely.geometry import Polygon

from navsim.planning.simulation.planner.pdm_planner.observation.pdm_object_manager import (
    PDMObjectManager,
)
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import (
    PDMOccupancyMap,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import BBCoordsIndex


class PDMObservation:
    """PDM's observation class for forecasted occupancy maps."""

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        proposal_sampling: TrajectorySampling,
        map_radius: float,
        observation_sample_res: int = 2,
        extend_observation_for_ttc: bool = True,
    ):
        """
        Constructor of PDMObservation
        :param trajectory_sampling: Sampling parameters for final trajectory
        :param proposal_sampling: Sampling parameters for proposals
        :param map_radius: radius around ego to consider, defaults to 50
        :param observation_sample_res: sample resolution of forecast, defaults to 2
        :param extend_observation_for_ttc: extend observation for TTC metric, defaults to False
        """
        assert (
            trajectory_sampling.interval_length == proposal_sampling.interval_length
        ), "PDMObservation: Proposals and Trajectory must have equal interval length!"

        # observation needs length of trajectory horizon or proposal horizon +1s (for TTC metric)
        self._sample_interval: float = trajectory_sampling.interval_length  # [s]

        if extend_observation_for_ttc:
            self._observation_samples: int = (
                proposal_sampling.num_poses + int(1 / self._sample_interval)
                if proposal_sampling.num_poses + int(1 / self._sample_interval)
                > trajectory_sampling.num_poses
                else trajectory_sampling.num_poses
            )
        else:
            self._observation_samples: int = max(
                trajectory_sampling.num_poses, proposal_sampling.num_poses
            )

        self._map_radius: float = map_radius
        self._observation_sample_res: int = observation_sample_res

        # useful things
        self._global_to_local_idcs = [
            idx // observation_sample_res
            for idx in range(self._observation_samples + observation_sample_res)
        ]
        self._collided_track_ids: List[str] = []
        self._red_light_token = "red_light"

        # lazy loaded (during update)
        self._occupancy_maps: Optional[List[PDMOccupancyMap]] = None
        self._unique_objects: Optional[Dict[str, TrackedObject]] = None
        self._occupancy_maps_tl: Optional[List[Tuple[List[str], np.ndarray]]] = None

        self._initialized: bool = False

    def __getitem__(self, time_idx) -> PDMOccupancyMap:
        """
        Retrieves occupancy map for time_idx and adapt temporal resolution.
        :param time_idx: index for future simulation iterations [10Hz]
        :return: occupancy map
        """
        assert self._initialized, "PDMObservation: Has not been updated yet!"
        assert (
            0 <= time_idx < len(self._global_to_local_idcs)
        ), f"PDMObservation: index {time_idx} out of range!"

        local_idx = self._global_to_local_idcs[time_idx]
        return self._occupancy_maps[local_idx]

    @property
    def collided_track_ids(self) -> List[str]:
        """
        Getter for past collided track tokens.
        :return: list of tokens
        """
        assert self._initialized, "PDMObservation: Has not been updated yet!"
        return self._collided_track_ids

    @property
    def red_light_token(self) -> str:
        """
        Getter for red light token indicator
        :return: string
        """
        return self._red_light_token

    @property
    def unique_objects(self) -> Dict[str, TrackedObject]:
        """
        Getter for unique tracked objects
        :return: dictionary of tokens, tracked objects
        """
        assert self._initialized, "PDMObservation: Has not been updated yet!"
        return self._unique_objects

    @property
    def detections_tracks(self) -> List[DetectionsTracks]:
        """
        Getter for detections tracks
        :return: list of detections tracks
        """
        assert self._initialized, "PDMObservation: Has not been updated yet!"
        return self._detections_tracks

    def update(
        self,
        ego_state: EgoState,
        observation: Observation,
        traffic_light_data: List[TrafficLightStatusData],
        route_lane_dict: Dict[str, LaneGraphEdgeMapObject],
    ) -> None:
        """
        Update & lazy loads information  of PDMObservation.
        :param ego_state: state of ego vehicle
        :param observation: input observation of nuPlan
        :param traffic_light_data: list of traffic light states
        :param route_lane_dict: dictionary of on-route lanes
        :param map_api: map object of nuPlan
        """

        self._occupancy_maps: List[PDMOccupancyMap] = []
        object_manager = self._get_object_manager(ego_state, observation)

        (
            traffic_light_tokens,
            traffic_light_polygons,
        ) = self._get_traffic_light_geometries(traffic_light_data, route_lane_dict)

        (
            static_object_tokens,
            static_object_coords,
            dynamic_object_tokens,
            dynamic_object_coords,
            dynamic_object_dxy,
        ) = object_manager.get_nearest_objects(ego_state.center.point)

        has_static_object, has_dynamic_object = (
            len(static_object_tokens) > 0,
            len(dynamic_object_tokens) > 0,
        )

        if has_static_object and static_object_coords.ndim == 1:
            static_object_coords = static_object_coords[None, ...]

        if has_dynamic_object and dynamic_object_coords.ndim == 1:
            dynamic_object_coords = dynamic_object_coords[None, ...]
            dynamic_object_dxy = dynamic_object_dxy[None, ...]

        if has_static_object:
            static_object_coords[..., BBCoordsIndex.CENTER, :] = static_object_coords[
                ..., BBCoordsIndex.FRONT_LEFT, :
            ]
            static_object_polygons = shapely.creation.polygons(static_object_coords)

        else:
            static_object_polygons = np.array([], dtype=np.object_)

        if has_dynamic_object:
            dynamic_object_coords[..., BBCoordsIndex.CENTER, :] = dynamic_object_coords[
                ..., BBCoordsIndex.FRONT_LEFT, :
            ]
        else:
            dynamic_object_polygons = np.array([], dtype=np.object_)
            dynamic_object_tokens = []

        traffic_light_polygons = np.array(traffic_light_polygons, dtype=np.object_)

        for sample in np.arange(
            0,
            self._observation_samples + self._observation_sample_res,
            self._observation_sample_res,
        ):
            if has_dynamic_object:
                delta_t = float(sample) * self._sample_interval
                dynamic_object_coords_t = (
                    dynamic_object_coords + delta_t * dynamic_object_dxy[:, None]
                )
                dynamic_object_polygons = shapely.creation.polygons(
                    dynamic_object_coords_t
                )

            all_polygons = np.concatenate(
                [
                    static_object_polygons,
                    dynamic_object_polygons,
                    traffic_light_polygons,
                ],
                axis=0,
            )

            occupancy_map = PDMOccupancyMap(
                static_object_tokens + dynamic_object_tokens + traffic_light_tokens,
                all_polygons,
            )
            self._occupancy_maps.append(occupancy_map)

        # save collided objects to ignore in the future
        ego_polygon: Polygon = ego_state.car_footprint.geometry
        intersecting_obstacles = self._occupancy_maps[0].intersects(ego_polygon)
        new_collided_track_ids = []

        for intersecting_obstacle in intersecting_obstacles:
            if self._red_light_token in intersecting_obstacle:
                within = ego_polygon.within(
                    self._occupancy_maps[0][intersecting_obstacle]
                )
                if not within:
                    continue
            new_collided_track_ids.append(intersecting_obstacle)

        # TODO: these are only the current tracks. Other update functions also add future tracks
        self._detections_tracks = [DetectionsTracks(observation.tracked_objects)]
        self._collided_track_ids = self._collided_track_ids + new_collided_track_ids
        self._unique_objects = object_manager.unique_objects
        self._initialized = True

    def update_replay(self, scenario: AbstractScenario, iteration_index: int) -> None:
        detection_tracks = scenario.get_future_tracked_objects(
            iteration_index, self._observation_samples * self._sample_interval
        )
        occupancy_maps = []
        unique_objects = {}

        for detection_track in detection_tracks:
            tokens, polygons = [], []
            for tracked_object in detection_track.tracked_objects:
                token, polygon = tracked_object.track_token, tracked_object.box.geometry
                tokens.append(token)
                polygons.append(polygon)

                if token not in unique_objects.keys():
                    unique_objects[token] = tracked_object

            occupancy_map = PDMOccupancyMap(tokens, polygons)
            occupancy_maps.append(occupancy_map)

        assert (
            len(occupancy_maps) == self._observation_samples + 1
        ), f"Expected observation length {self._observation_samples + 1}, but got {len(occupancy_maps)}"

        self._detections_tracks = detection_tracks
        self._occupancy_maps: List[PDMOccupancyMap] = occupancy_maps
        self._collided_track_ids = []
        self._unique_objects = unique_objects
        self._initialized = True

    def update_detections_tracks(
        self,
        detection_tracks: List[DetectionsTracks],
        traffic_light_data: Optional[List[List[TrafficLightStatusData]]] = None,
        route_lane_dict: Optional[Dict[str, LaneGraphEdgeMapObject]] = None,
        compute_traffic_light_data: bool = False,
    ) -> None:
        """
        Updates detection tracks and update traffic light from `traffic_light_data` or existing `_occupancy_maps_tl`.
        By default, it uses the existing `_occupancy_maps_tl` if `_occupancy_maps_tl` is not `None`.

        Args:
            detection_tracks: List of detection tracks.
            traffic_light_data: Optional traffic light data corresponding to detection tracks.
            route_lane_dict: Optional mapping of route lanes to lane graph edge objects.
            compute_traffic_light_data: If 'True', the traffic light data provided in parameter 'traffic_light_data'
                is used to compute the traffic light data occupancy map and overwrite the existing traffic light occupancy maps.
                If 'False', the existing traffic light occupancy maps are kept. (default: False)

        Logic of processing traffic light data:
            1. If `compute_traffic_light_data` is True:
                - Validate that both `traffic_light_data` and `route_lane_dict` are provided.
                - Extract traffic light tokens and polygons from `traffic_light_data` for the current detection track.
                - Append the extracted traffic light tokens and polygons to `occupancy_maps_tl`.
            2. Otherwise (if `compute_traffic_light_data` is False):
                - Check if `_occupancy_maps_tl` is not `None`:
                    - If `_occupancy_maps_tl` is available, extract traffic light tokens and polygons from it for the current detection track.
            3. Append the extracted tokens and polygons (from either source) to the current track's occupancy map.
        """
        occupancy_maps = []
        occupancy_maps_tl = (
            [] if compute_traffic_light_data else self._occupancy_maps_tl
        )
        unique_objects = {}

        for idx, detection_track in enumerate(detection_tracks):
            tokens, polygons = [], []
            for tracked_object in detection_track.tracked_objects:
                token, polygon = tracked_object.track_token, tracked_object.box.geometry
                tokens.append(token)
                polygons.append(polygon)

                if token not in unique_objects.keys():
                    unique_objects[token] = tracked_object

            if compute_traffic_light_data:
                if traffic_light_data is not None and route_lane_dict is not None:
                    assert idx < len(
                        traffic_light_data
                    ), f"Length of traffic_light_data ({len(traffic_light_data)}) does not match detection_tracks ({len(detection_tracks)})."

                    (
                        traffic_light_tokens,
                        traffic_light_polygons,
                    ) = self._get_traffic_light_geometries(
                        traffic_light_data[idx], route_lane_dict
                    )
                    traffic_light_polygons = np.array(
                        traffic_light_polygons, dtype=np.object_
                    )

                    tokens += traffic_light_tokens
                    polygons = np.concatenate(
                        [polygons, traffic_light_polygons], axis=0
                    )

                    occupancy_map_tl = (traffic_light_tokens, traffic_light_polygons)
                    occupancy_maps_tl.append(occupancy_map_tl)
                else:
                    warnings.warn(
                        "compute_traffic_light_data is True, but traffic_light_data or route_lane_dict is not provided. Skipping traffic light processing."
                    )
            else:
                if self._occupancy_maps_tl is not None:
                    assert idx < len(
                        self._occupancy_maps_tl
                    ), f"Index {idx} exceeds the length of _occupancy_maps_tl ({len(self._occupancy_maps_tl)})."
                    (
                        traffic_light_tokens,
                        traffic_light_polygons,
                    ) = self._occupancy_maps_tl[idx]
                    tokens += traffic_light_tokens
                    polygons = np.concatenate(
                        [polygons, traffic_light_polygons], axis=0
                    )
                else:
                    warnings.warn(
                        "compute_traffic_light_data is False, and _occupancy_maps_tl is None. Keeping it as None."
                    )

            occupancy_map = PDMOccupancyMap(tokens, polygons)
            occupancy_maps.append(occupancy_map)

        # Validate occupancy_maps length
        if len(occupancy_maps) != self._observation_samples + 1:
            warnings.warn(
                f"Expected length of detections_tracks {self._observation_samples + 1}, but got {len(occupancy_maps)}.\
                Observation will be shorter than expected. This can happen if your metric cache is longer than the future horizon of the traffic agents policy.\
                The observation will be truncated to the length of the length of the detections tracks"
            )

        # Update class state
        self._detections_tracks = detection_tracks
        self._occupancy_maps: List[PDMOccupancyMap] = occupancy_maps
        self._occupancy_maps_tl = (
            occupancy_maps_tl  # Retain or update _occupancy_maps_tl
        )
        self._collided_track_ids = []
        self._unique_objects = unique_objects
        self._initialized = True

    def _get_object_manager(
        self, ego_state: EgoState, observation: Observation
    ) -> PDMObjectManager:
        """
        Creates object manager class, but adding valid tracked objects.
        :param ego_state: state of ego-vehicle
        :param observation: input observation of nuPlan
        :return: PDMObjectManager class
        """
        object_manager = PDMObjectManager()

        for object in observation.tracked_objects:
            if (
                (object.tracked_object_type == TrackedObjectType.EGO)
                or (
                    self._map_radius
                    and ego_state.center.distance_to(object.center) > self._map_radius
                )
                or (object.track_token in self._collided_track_ids)
            ):
                continue

            object_manager.add_object(object)

        return object_manager

    def _get_traffic_light_geometries(
        self,
        traffic_light_data: List[TrafficLightStatusData],
        route_lane_dict: Dict[str, LaneGraphEdgeMapObject],
    ) -> Tuple[List[str], List[Polygon]]:
        """
        Collects red traffic lights along ego's route.
        :param traffic_light_data: list of traffic light states
        :param route_lane_dict: dictionary of on-route lanes
        :return: tuple of tokens and polygons of red traffic lights
        """
        traffic_light_tokens, traffic_light_polygons = [], []

        for data in traffic_light_data:
            lane_connector_id = str(data.lane_connector_id)

            if (data.status == TrafficLightStatusType.RED) and (
                lane_connector_id in route_lane_dict.keys()
            ):
                lane_connector = route_lane_dict[lane_connector_id]
                traffic_light_tokens.append(
                    f"{self._red_light_token}_{lane_connector_id}"
                )
                traffic_light_polygons.append(lane_connector.polygon)

        return traffic_light_tokens, traffic_light_polygons
