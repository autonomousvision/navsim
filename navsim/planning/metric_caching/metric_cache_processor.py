import pathlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES
from nuplan.common.geometry.convert import absolute_to_relative_poses
from nuplan.common.maps.abstract_map_objects import (
    LaneGraphEdgeMapObject,
    RoadBlockGraphEdgeMapObject,
)
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusData
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history_buffer import (
    SimulationHistoryBuffer,
)
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import (
    SimulationIteration,
)
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.experiments.cache_metadata_entry import CacheMetadataEntry

from navsim.common.dataclasses import Trajectory
from navsim.common.enums import SceneFrameType
from navsim.planning.metric_caching.metric_cache import MapParameters, MetricCache
from navsim.planning.metric_caching.metric_caching_utils import StateInterpolator
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_observation import (
    PDMObservation,
)
from navsim.planning.simulation.planner.pdm_planner.pdm_closed_planner import (
    PDMClosedPlanner,
)
from navsim.planning.simulation.planner.pdm_planner.proposal.batch_idm_policy import (
    BatchIDMPolicy,
)


class MetricCacheProcessor:
    """Class for creating metric cache in NAVSIM."""

    def __init__(
        self,
        cache_path: Optional[str],
        force_feature_computation: bool,
        proposal_sampling: TrajectorySampling,
    ):
        """
        Initialize class.
        :param cache_path: Whether to cache features.
        :param force_feature_computation: If true, even if cache exists, it will be overwritten.
        """
        self._cache_path = pathlib.Path(cache_path) if cache_path else None
        self._force_feature_computation = force_feature_computation

        # 1s additional observation for ttc metric
        future_poses = proposal_sampling.num_poses + int(
            1.0 / proposal_sampling.interval_length
        )
        future_sampling = TrajectorySampling(
            num_poses=future_poses, interval_length=proposal_sampling.interval_length
        )
        self._proposal_sampling = proposal_sampling
        self._map_radius = 100

        self._pdm_closed = PDMClosedPlanner(
            trajectory_sampling=future_sampling,
            proposal_sampling=self._proposal_sampling,
            idm_policies=BatchIDMPolicy(
                speed_limit_fraction=[0.2, 0.4, 0.6, 0.8, 1.0],
                fallback_target_velocity=15.0,
                min_gap_to_lead_agent=1.0,
                headway_time=1.5,
                accel_max=1.5,
                decel_max=3.0,
            ),
            lateral_offsets=[-1.0, 1.0],
            map_radius=self._map_radius,
        )

    def _get_planner_inputs(
        self, scenario: AbstractScenario
    ) -> Tuple[PlannerInput, PlannerInitialization]:
        """
        Creates planner input arguments from scenario object.
        :param scenario: scenario object of nuPlan
        :return: tuple of planner input and initialization objects
        """

        # Initialize Planner
        planner_initialization = PlannerInitialization(
            route_roadblock_ids=scenario.get_route_roadblock_ids(),
            mission_goal=scenario.get_mission_goal(),
            map_api=scenario.map_api,
        )

        history = SimulationHistoryBuffer.initialize_from_list(
            buffer_size=1,
            ego_states=[scenario.initial_ego_state],
            observations=[scenario.initial_tracked_objects],
        )

        planner_input = PlannerInput(
            iteration=SimulationIteration(index=0, time_point=scenario.start_time),
            history=history,
            traffic_light_data=list(scenario.get_traffic_light_status_at_iteration(0)),
        )

        return planner_input, planner_initialization

    def _interpolate_gt_observation(
        self, scenario: NavSimScenario
    ) -> List[DetectionsTracks]:
        """
        Helper function to interpolate detections tracks to higher temporal resolution.
        :param scenario: scenario interface of nuPlan framework
        :return: interpolated detection tracks
        """

        # TODO: add to config
        state_size = 6  # (time, x, y, heading, velo_x, velo_y)

        time_horizon = self._proposal_sampling.time_horizon  # [s]
        resolution_step = 0.5  # [s]
        interpolate_step = self._proposal_sampling.interval_length  # [s]

        scenario_step = scenario.database_interval  # [s]

        # sample detection tracks a 2Hz
        relative_time_s = (
            np.arange(0, (time_horizon * 1 / resolution_step) + 1, 1, dtype=float)
            * resolution_step
        )

        gt_indices = np.arange(
            0,
            int(time_horizon / scenario_step) + 1,
            int(resolution_step / scenario_step),
        )
        gt_detection_tracks = [
            scenario.get_tracked_objects_at_iteration(iteration=iteration)
            for iteration in gt_indices
        ]

        detection_tracks_states: Dict[str, Any] = {}
        unique_detection_tracks: Dict[str, Any] = {}

        for time_s, detection_track in zip(relative_time_s, gt_detection_tracks):

            for tracked_object in detection_track.tracked_objects:
                # log detection track
                token = tracked_object.track_token

                # extract states for dynamic and static objects
                tracked_state = np.zeros(state_size, dtype=np.float64)
                tracked_state[:4] = (
                    time_s,
                    tracked_object.center.x,
                    tracked_object.center.y,
                    tracked_object.center.heading,
                )

                if tracked_object.tracked_object_type in AGENT_TYPES:
                    # extract additional states for dynamic objects
                    tracked_state[4:] = (
                        tracked_object.velocity.x,
                        tracked_object.velocity.y,
                    )

                # found new object
                if token not in detection_tracks_states.keys():
                    detection_tracks_states[token] = [tracked_state]
                    unique_detection_tracks[token] = tracked_object

                # object already existed
                else:
                    detection_tracks_states[token].append(tracked_state)

        # create time interpolators
        detection_interpolators: Dict[str, StateInterpolator] = {}
        for token, states_list in detection_tracks_states.items():
            states = np.array(states_list, dtype=np.float64)
            detection_interpolators[token] = StateInterpolator(states)

        # interpolate at 10Hz
        interpolated_time_s = (
            np.arange(0, int(time_horizon / interpolate_step) + 1, 1, dtype=float)
            * interpolate_step
        )

        interpolated_detection_tracks = []
        for time_s in interpolated_time_s:
            interpolated_tracks = []
            for token, interpolator in detection_interpolators.items():
                initial_detection_track = unique_detection_tracks[token]
                interpolated_state = interpolator.interpolate(time_s)

                if interpolator.start_time == interpolator.end_time:
                    interpolated_tracks.append(initial_detection_track)

                elif interpolated_state is not None:

                    tracked_type = initial_detection_track.tracked_object_type
                    metadata = (
                        initial_detection_track.metadata
                    )  # copied since time stamp is ignored

                    oriented_box = OrientedBox(
                        StateSE2(*interpolated_state[:3]),
                        initial_detection_track.box.length,
                        initial_detection_track.box.width,
                        initial_detection_track.box.height,
                    )

                    if tracked_type in AGENT_TYPES:
                        velocity = StateVector2D(*interpolated_state[3:])

                        detection_track = Agent(
                            tracked_object_type=tracked_type,
                            oriented_box=oriented_box,
                            velocity=velocity,
                            metadata=initial_detection_track.metadata,  # simply copy
                        )
                    else:
                        detection_track = StaticObject(
                            tracked_object_type=tracked_type,
                            oriented_box=oriented_box,
                            metadata=metadata,
                        )

                    interpolated_tracks.append(detection_track)
            interpolated_detection_tracks.append(
                DetectionsTracks(TrackedObjects(interpolated_tracks))
            )
        return interpolated_detection_tracks

    def _build_pdm_observation(
        self,
        interpolated_detection_tracks: List[DetectionsTracks],
        interpolated_traffic_light_data: List[List[TrafficLightStatusData]],
        route_lane_dict: Dict[str, LaneGraphEdgeMapObject],
    ):
        # convert to pdm observation
        pdm_observation = PDMObservation(
            self._proposal_sampling,
            self._proposal_sampling,
            self._map_radius,
            observation_sample_res=1,
            extend_observation_for_ttc=False,
        )
        pdm_observation.update_detections_tracks(
            interpolated_detection_tracks,
            interpolated_traffic_light_data,
            route_lane_dict,
            compute_traffic_light_data=True,
        )
        return pdm_observation

    def _interpolate_traffic_light_status(
        self, scenario: NavSimScenario
    ) -> List[List[TrafficLightStatusData]]:

        time_horizon = self._proposal_sampling.time_horizon  # [s]
        interpolate_step = self._proposal_sampling.interval_length  # [s]

        scenario_step = scenario.database_interval  # [s]
        gt_indices = np.arange(0, int(time_horizon / scenario_step) + 1, 1, dtype=int)

        traffic_light_status = []
        for iteration in gt_indices:
            current_status_list = list(
                scenario.get_traffic_light_status_at_iteration(iteration=iteration)
            )
            for _ in range(int(scenario_step / interpolate_step)):
                traffic_light_status.append(current_status_list)

        if scenario_step == interpolate_step:
            return traffic_light_status
        else:
            return traffic_light_status[: -int(scenario_step / interpolate_step) + 1]

    def _load_route_dicts(
        self, scenario: NavSimScenario, route_roadblock_ids: List[str]
    ) -> Tuple[
        Dict[str, RoadBlockGraphEdgeMapObject], Dict[str, LaneGraphEdgeMapObject]
    ]:
        route_roadblock_ids = list(dict.fromkeys(route_roadblock_ids))

        route_roadblock_dict = {}
        route_lane_dict = {}

        for id_ in route_roadblock_ids:
            block = scenario.map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            block = block or scenario.map_api.get_map_object(
                id_, SemanticMapLayer.ROADBLOCK_CONNECTOR
            )

            route_roadblock_dict[block.id] = block

            for lane in block.interior_edges:
                route_lane_dict[lane.id] = lane

        return route_roadblock_dict, route_lane_dict

    def _build_file_path(self, scenario: NavSimScenario) -> pathlib.Path:
        return (
            self._cache_path
            / scenario.log_name
            / scenario.scenario_type
            / scenario.token
            / "metric_cache.pkl"
        ) if self._cache_path else None

    def compute_and_save_metric_cache(
        self, scenario: NavSimScenario
    ) -> Optional[CacheMetadataEntry]:
        file_name = self._build_file_path(scenario)
        assert file_name is not None, "Cache path can not be None for saving cache."
        if file_name.exists() and not self._force_feature_computation:
            return CacheMetadataEntry(file_name)
        metric_cache = self.compute_metric_cache(scenario)
        metric_cache.dump()
        return CacheMetadataEntry(metric_cache.file_path)

    def _extract_ego_future_trajectory(self, scenario: NavSimScenario) -> Trajectory:
        ego_trajectory_sampling = TrajectorySampling(
            time_horizon=self._proposal_sampling.time_horizon,
            interval_length=scenario.database_interval,
        )
        future_ego_states = list(
            scenario.get_ego_future_trajectory(
                iteration=0,
                time_horizon=ego_trajectory_sampling.time_horizon,
                num_samples=ego_trajectory_sampling.num_poses,
            )
        )
        initial_ego_state = scenario.get_ego_state_at_iteration(0)
        if future_ego_states[0].time_point != initial_ego_state.time_point:
            # nuPlan does not return the initial state while navsim does
            # make sure to add the initial state before transforming to relative poses
            future_ego_states = [initial_ego_state] + future_ego_states

        future_ego_poses = [state.rear_axle for state in future_ego_states]
        relative_future_states = absolute_to_relative_poses(future_ego_poses)[1:]
        return Trajectory(
            poses=np.array([[pose.x, pose.y, pose.heading] for pose in relative_future_states]),
            trajectory_sampling=ego_trajectory_sampling,
        )

    def compute_metric_cache(
        self, scenario: NavSimScenario
    ) -> MetricCache:
        file_name = self._build_file_path(scenario)

        # TODO: we should infer this from the scene metadata
        is_synthetic_scene = len(scenario.token) == 17

        # init and run PDM-Closed
        planner_input, planner_initialization = self._get_planner_inputs(scenario)
        self._pdm_closed.initialize(planner_initialization)
        pdm_closed_trajectory = self._pdm_closed.compute_planner_trajectory(
            planner_input
        )

        route_roadblock_dict, route_lane_dict = self._load_route_dicts(
            scenario, planner_initialization.route_roadblock_ids
        )

        interpolated_detection_tracks = self._interpolate_gt_observation(scenario)
        interpolated_traffic_light_status = self._interpolate_traffic_light_status(
            scenario
        )

        observation = self._build_pdm_observation(
            interpolated_detection_tracks=interpolated_detection_tracks,
            interpolated_traffic_light_data=interpolated_traffic_light_status,
            route_lane_dict=route_lane_dict,
        )
        future_tracked_objects = interpolated_detection_tracks[1:]

        past_human_trajectory = InterpolatedTrajectory(
            [ego_state for ego_state in scenario.get_ego_past_trajectory(0, 1.5)]
        )

        if not is_synthetic_scene:
            human_trajectory = self._extract_ego_future_trajectory(scenario)
        else:
            human_trajectory = None

        # save and dump features
        return MetricCache(
            file_path=file_name,
            log_name=scenario.log_name,
            scene_type=SceneFrameType.SYNTHETIC if is_synthetic_scene else SceneFrameType.ORIGINAL,
            timepoint=scenario.start_time,
            trajectory=pdm_closed_trajectory,
            human_trajectory=human_trajectory,
            past_human_trajectory=past_human_trajectory,
            ego_state=scenario.initial_ego_state,
            observation=observation,
            centerline=self._pdm_closed._centerline,
            route_lane_ids=list(self._pdm_closed._route_lane_dict.keys()),
            drivable_area_map=self._pdm_closed._drivable_area_map,
            past_detections_tracks=[
                dt
                for dt in scenario.get_past_tracked_objects(
                    iteration=0, time_horizon=1.5, num_samples=3
                )
            ][:-1],
            current_tracked_objects=[scenario.initial_tracked_objects],
            future_tracked_objects=future_tracked_objects,
            map_parameters=MapParameters(
                map_root=scenario.map_root,
                map_version=scenario.map_version,
                map_name=scenario.map_api.map_name,
            ),
        )
