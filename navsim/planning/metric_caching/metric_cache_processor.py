import pathlib
from typing import Any, Dict, Optional, Tuple

import numpy as np

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
from nuplan.planning.training.experiments.cache_metadata_entry import CacheMetadataEntry
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer

from navsim.planning.simulation.planner.pdm_planner.pdm_closed_planner import PDMClosedPlanner
from navsim.planning.simulation.planner.pdm_planner.proposal.batch_idm_policy import BatchIDMPolicy
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_observation import PDMObservation
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.metric_caching.metric_caching_utils import StateInterpolator


class MetricCacheProcessor:
    """Class for creating metric cache in NAVSIM."""

    def __init__(
        self,
        cache_path: Optional[str],
        force_feature_computation: bool,
    ):
        """
        Initialize class.
        :param cache_path: Whether to cache features.
        :param force_feature_computation: If true, even if cache exists, it will be overwritten.
        """
        self._cache_path = pathlib.Path(cache_path) if cache_path else None
        self._force_feature_computation = force_feature_computation

        # TODO: Add to some config
        self._future_sampling = TrajectorySampling(num_poses=50, interval_length=0.1)
        self._proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
        self._map_radius = 100

        self._pdm_closed = PDMClosedPlanner(
            trajectory_sampling=self._future_sampling,
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

    def _get_planner_inputs(self, scenario: AbstractScenario) -> Tuple[PlannerInput, PlannerInitialization]:
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

    def _interpolate_gt_observation(self, scenario: AbstractScenario) -> PDMObservation:
        """
        Helper function to interpolate detections tracks to higher temporal resolution.
        :param scenario: scenario interface of nuPlan framework
        :return: observation object of PDM-Closed
        """

        # TODO: add to config
        state_size = 6  # (time, x, y, heading, velo_x, velo_y)

        time_horizon = 5.0  # [s]
        resolution_step = 0.5  # [s]
        interpolate_step = 0.1  # [s]

        scenario_step = scenario.database_interval  # [s]

        # sample detection tracks a 2Hz
        relative_time_s = np.arange(0, (time_horizon * 1 / resolution_step) + 1, 1, dtype=float) * resolution_step

        gt_indices = np.arange(0, int(time_horizon / scenario_step) + 1, int(resolution_step / scenario_step))
        gt_detection_tracks = [
            scenario.get_tracked_objects_at_iteration(iteration=iteration) for iteration in gt_indices
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
        interpolated_time_s = np.arange(0, int(time_horizon / interpolate_step) + 1, 1, dtype=float) * interpolate_step

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
                    metadata = initial_detection_track.metadata  # copied since time stamp is ignored

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
            interpolated_detection_tracks.append(DetectionsTracks(TrackedObjects(interpolated_tracks)))

        # convert to pdm observation
        pdm_observation = PDMObservation(
            self._future_sampling,
            self._proposal_sampling,
            self._map_radius,
            observation_sample_res=1,
        )
        pdm_observation.update_detections_tracks(interpolated_detection_tracks)
        return pdm_observation

    def compute_metric_cache(self, scenario: AbstractScenario) -> Optional[CacheMetadataEntry]:

        file_name = self._cache_path / scenario.log_name / scenario.scenario_type / scenario.token / "metric_cache.pkl"

        if file_name.exists() and not self._force_feature_computation:
            return CacheMetadataEntry(file_name)

        # init and run PDM-Closed
        planner_input, planner_initialization = self._get_planner_inputs(scenario)
        self._pdm_closed.initialize(planner_initialization)
        pdm_closed_trajectory = self._pdm_closed.compute_planner_trajectory(planner_input)

        observation = self._interpolate_gt_observation(scenario)

        # save and dump features
        MetricCache(
            file_name,
            pdm_closed_trajectory,
            scenario.initial_ego_state,
            observation,
            self._pdm_closed._centerline,
            list(self._pdm_closed._route_lane_dict.keys()),
            self._pdm_closed._drivable_area_map,
        ).dump()

        # return metadata
        return CacheMetadataEntry(file_name)
