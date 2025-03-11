import hashlib
import uuid
from typing import Dict, List, Optional

import numpy as np
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer
from nuplan.common.maps.maps_datatypes import TrafficLightStatusType
from nuplan.planning.simulation.observation.idm_agents import IDMAgents
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks

from navsim.planning.simulation.observation.navsim_idm.navsim_idm_agent_manager import NavsimIDMAgentManager
from navsim.planning.simulation.observation.navsim_idm.navsim_idm_agents_builder import (
    build_idm_agents_on_map_rails,
    get_starting_segment,
)


class NavsimIDMAgents(IDMAgents):
    """
    Subclass of IDMAgents that overrides methods to support dynamic input and flexible behavior.
    """

    def __init__(
        self,
        target_velocity: float,
        min_gap_to_lead_agent: float,
        headway_time: float,
        accel_max: float,
        decel_max: float,
        open_loop_detections_types: List[str],
        minimum_path_length: float = 20,
        planned_trajectory_samples: Optional[int] = None,
        planned_trajectory_sample_interval: Optional[float] = None,
        radius: float = 100,
        add_open_loop_parked_vehicles: bool = False,
        idm_snap_threshold: float = 1.5,
    ):
        """
        Initialize the subclass by reusing the parent's constructor and injecting custom behavior.
        """
        # Call the parent class constructor
        super().__init__(
            target_velocity,
            min_gap_to_lead_agent,
            headway_time,
            accel_max,
            decel_max,
            open_loop_detections_types,
            scenario=None,  # Scenario dependency removed in the subclass
            minimum_path_length=minimum_path_length,
            planned_trajectory_samples=planned_trajectory_samples,
            planned_trajectory_sample_interval=planned_trajectory_sample_interval,
            radius=radius,
        )
        self._add_open_loop_parked_vehicles = add_open_loop_parked_vehicles
        self._idm_snap_threshold = idm_snap_threshold

    def _get_idm_agent_manager(
        self, ego_state: EgoState, vehicle_current_tracks: DetectionsTracks, map_api
    ) -> NavsimIDMAgentManager:
        """
        Override the parent's method to dynamically initialize the IDM Agent Manager.
        :param ego_state: Current state of the ego vehicle.
        :param vehicle_current_tracks: Tracks of nearby vehicles.
        :param map_api: Map API for accessing map-related data.
        :return: NavsimIDMAgentManager instance for managing agents.
        """
        if not self._idm_agent_manager:
            # Build IDM agents dynamically based on inputs
            agents, agent_occupancy = build_idm_agents_on_map_rails(
                self._target_velocity,
                self._min_gap_to_lead_agent,
                self._headway_time,
                self._accel_max,
                self._decel_max,
                self._minimum_path_length,
                self._idm_snap_threshold,
                self._open_loop_detections_types,
                ego_state,
                vehicle_current_tracks,
                map_api,
            )
            # Initialize the IDM Agent Manager
            self._idm_agent_manager = NavsimIDMAgentManager(agents, agent_occupancy, map_api)
        return self._idm_agent_manager

    def get_observation(
        self,
        ego_state: EgoState,
        vehicle_current_tracks: DetectionsTracks,
        map_api: AbstractMap,
        objects_future_tracks: TrackedObjects,
    ) -> DetectionsTracks:
        """
        Override the parent's method to generate observations using dynamic inputs.
        :param ego_state: Current state of the ego vehicle.
        :param vehicle_current_tracks: Tracks of nearby vehicles.
        :param map_api: Map API for accessing map-related data.
        :param objects_future_tracks: Future tracked objects for open-loop detections.
        :return: DetectionsTracks object containing active agents and open-loop detections.
        """
        detections = self._get_idm_agent_manager(ego_state, vehicle_current_tracks, map_api,).get_active_agents(
            self.current_iteration,
            self._planned_trajectory_samples,
            self._planned_trajectory_sample_interval,
        )

        for object in detections.tracked_objects.tracked_objects:
            new_metadata = SceneObjectMetadata(
                timestamp_us=ego_state.time_us,
                token=hashlib.md5(str(uuid.uuid4()).encode()).hexdigest()[: len(object.metadata.token)],
                track_token=object.metadata.track_token,
                track_id=object.metadata.track_id,
                category_name=object.metadata.category_name,
            )
            object._metadata = new_metadata

        inactive_vehicle_agents = [
            agent
            for agent in vehicle_current_tracks.tracked_objects.tracked_objects
            if agent.track_token
            not in {active_agent.track_token for active_agent in detections.tracked_objects.tracked_objects}
        ]

        if self._add_open_loop_parked_vehicles and inactive_vehicle_agents:
            for agent in inactive_vehicle_agents:
                is_stationary = agent.velocity.magnitude() < 0.1
                is_in_lanes = map_api.is_in_layer(agent.center, SemanticMapLayer.LANE) or map_api.is_in_layer(
                    agent.center, SemanticMapLayer.INTERSECTION
                )

                # lateral deviation
                route, _ = get_starting_segment(agent, map_api)
                lateral_deviation = None
                if route:
                    state_on_path = route.baseline_path.get_nearest_pose_from_position(agent.center)
                    lateral_deviation = np.hypot(
                        state_on_path.x - agent.center.x,
                        state_on_path.y - agent.center.y,
                    )

                # collision check
                collides_with_active_agents = any(
                    [
                        agent.box.geometry.intersects(active_agent.box.geometry)
                        for active_agent in detections.tracked_objects.tracked_objects
                    ]
                )

                # case1: stationary and not in lanes and not collides with active agents
                if is_stationary and not is_in_lanes and not collides_with_active_agents:
                    detections.tracked_objects.tracked_objects.append(agent)
                    continue

                # case2: lateral deviation > threshold and not collides with active agents
                if (
                    lateral_deviation is not None
                    and lateral_deviation > self._idm_snap_threshold
                    and not collides_with_active_agents
                ):
                    detections.tracked_objects.tracked_objects.append(agent)

        if self._open_loop_detections_types:
            # Add open-loop tracked objects
            open_loop_detections = objects_future_tracks.get_tracked_objects_of_types(self._open_loop_detections_types)
            detections.tracked_objects.tracked_objects.extend(open_loop_detections)
        return detections

    def update_observation(
        self,
        iteration: int,
        ego_state: EgoState,
        vehicle_current_tracks: DetectionsTracks,
        map_api,
        objects_future_tracks: TrackedObjects,
        traffic_light_status: Optional[Dict[TrafficLightStatusType, List[str]]] = None,
    ) -> None:
        """
        Override the parent's method to update agent states with dynamic inputs.

        :param iteration: Current simulation iteration.
        :param ego_state: Current state of the ego vehicle.
        :param vehicle_current_tracks: Tracks of nearby vehicles.
        :param map_api: Map API for accessing map-related data.
        :param objects_future_tracks: Future tracked objects for open-loop detections.
        :param traffic_light_status: Optional traffic light data to influence agent behavior.
        """
        self.current_iteration = iteration
        tspan = 0.1  # Fixed time step (e.g., 0.1 seconds)

        self._get_idm_agent_manager(ego_state, vehicle_current_tracks, map_api,).propagate_agents(
            ego_state,
            tspan,
            self.current_iteration,
            objects_future_tracks.get_tracked_objects_of_types(self._open_loop_detections_types),
            self._radius,
            traffic_light_status,
        )
