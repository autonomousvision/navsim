import logging
from typing import List, Tuple

import numpy as np
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.idm.idm_agent import (
    IDMAgent,
    IDMInitialState,
)
from nuplan.planning.simulation.observation.idm.idm_agent_manager import UniqueIDMAgents
from nuplan.planning.simulation.observation.idm.idm_agents_builder import (
    get_starting_segment,
)
from nuplan.planning.simulation.observation.idm.idm_policy import IDMPolicy
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.occupancy_map.abstract_occupancy_map import OccupancyMap
from nuplan.planning.simulation.occupancy_map.strtree_occupancy_map import (
    STRTreeOccupancyMap,
    STRTreeOccupancyMapFactory,
)

logger = logging.getLogger(__name__)


def build_idm_agents_on_map_rails(
    target_velocity: float,
    min_gap_to_lead_agent: float,
    headway_time: float,
    accel_max: float,
    decel_max: float,
    minimum_path_length: float,
    idm_snap_threshold: float,
    open_loop_detections_types: List[TrackedObjectType],
    ego_agent: EgoState,
    vehicle_current_tracks: DetectionsTracks,
    map_api: AbstractScenario,
) -> Tuple[UniqueIDMAgents, OccupancyMap]:
    """
    Build unique agents from a scenario. InterpolatedPaths are created for each agent according to their driven path

    :param target_velocity: Desired velocity in free traffic [m/s]
    :param min_gap_to_lead_agent: Minimum relative distance to lead vehicle [m]
    :param headway_time: Desired time headway. The minimum possible time to the vehicle in front [s]
    :param accel_max: maximum acceleration [m/s^2]
    :param decel_max: maximum deceleration (positive value) [m/s^2]
    :param minimum_path_length: [m] The minimum path length
    :param scenario: scenario
    :param open_loop_detections_types: The open-loop detection types to include.
    :return: a dictionary of IDM agent uniquely identified by a track_token
    """
    unique_agents: UniqueIDMAgents = {}

    detections = vehicle_current_tracks
    map_api = map_api
    ego_agent = ego_agent.agent

    open_loop_detections = detections.tracked_objects.get_tracked_objects_of_types(
        open_loop_detections_types
    )
    # An occupancy map used only for collision checking
    init_agent_occupancy = STRTreeOccupancyMapFactory.get_from_boxes(
        open_loop_detections
    )
    init_agent_occupancy.insert(ego_agent.token, ego_agent.box.geometry)

    # Initialize occupancy map
    occupancy_map = STRTreeOccupancyMap({})

    agent: Agent
    for agent in detections.tracked_objects.get_tracked_objects_of_type(
        TrackedObjectType.VEHICLE
    ):
        # filter for only vehicles
        if agent.track_token not in unique_agents:

            route, progress = get_starting_segment(agent, map_api)

            # Ignore agents that a baseline path cannot be built for
            if route is None:
                continue

            # Snap agent to baseline path
            state_on_path = route.baseline_path.get_nearest_pose_from_position(
                agent.center.point
            )

            # Ignore agents that far away from baseline
            lateral_deviation = np.hypot(
                state_on_path.x - agent.center.x, state_on_path.y - agent.center.y
            )

            if lateral_deviation > idm_snap_threshold:
                continue

            box_on_baseline = OrientedBox.from_new_pose(
                agent.box,
                StateSE2(state_on_path.x, state_on_path.y, state_on_path.heading),
            )

            # Check for collision
            if not init_agent_occupancy.intersects(box_on_baseline.geometry).is_empty():
                continue

            # Add to init_agent_occupancy for collision checking
            init_agent_occupancy.insert(agent.track_token, box_on_baseline.geometry)

            # Add to occupancy_map to pass on to IDMAgentManger
            occupancy_map.insert(agent.track_token, box_on_baseline.geometry)

            # Project velocity into local frame
            if np.isnan(agent.velocity.array).any():
                ego_state = ego_agent
                logger.debug(
                    f"Agents has nan velocity. Setting velocity to ego's velocity of "
                    f"{ego_state.dynamic_car_state.speed}"
                )
                velocity = StateVector2D(ego_state.dynamic_car_state.speed, 0.0)
            else:
                velocity = StateVector2D(
                    np.hypot(agent.velocity.x, agent.velocity.y), 0
                )

            initial_state = IDMInitialState(
                metadata=agent.metadata,
                tracked_object_type=agent.tracked_object_type,
                box=box_on_baseline,
                velocity=velocity,
                path_progress=progress,
                predictions=agent.predictions,
            )
            target_velocity = route.speed_limit_mps or target_velocity
            unique_agents[agent.track_token] = IDMAgent(
                start_iteration=0,
                initial_state=initial_state,
                route=[route],
                policy=IDMPolicy(
                    target_velocity,
                    min_gap_to_lead_agent,
                    headway_time,
                    accel_max,
                    decel_max,
                ),
                minimum_path_length=minimum_path_length,
            )

    return unique_agents, occupancy_map
