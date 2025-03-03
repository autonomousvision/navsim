from typing import Dict, List, Optional

import numpy as np
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects import TrackedObject
from nuplan.common.geometry.transform import rotate_angle
from nuplan.common.maps.abstract_map_objects import StopLine
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusType
from nuplan.planning.metrics.utils.expert_comparisons import principal_value
from nuplan.planning.simulation.observation.idm.idm_agent import IDMAgent
from nuplan.planning.simulation.observation.idm.idm_agent_manager import IDMAgentManager
from nuplan.planning.simulation.observation.idm.idm_states import IDMLeadAgentState
from nuplan.planning.simulation.observation.idm.utils import path_to_linestring
from shapely.geometry.base import CAP_STYLE

UniqueIDMAgents = Dict[str, IDMAgent]


def simplify_occupancy_map_geometries(occupancy_map, tolerance=1e-5):
    """
    Simplify the geometries in the occupancy map.

    """
    all_ids = occupancy_map.get_all_ids()
    for geometry_id in all_ids:
        geom = occupancy_map.get(geometry_id)
        if geom is not None:
            geom_simplified = geom.simplify(tolerance, preserve_topology=True)
            occupancy_map.set(geometry_id, geom_simplified)
    return occupancy_map


class NavsimIDMAgentManager(IDMAgentManager):
    """IDM agent manager with optional traffic light status."""

    def propagate_agents(
        self,
        ego_state: EgoState,
        tspan: float,
        iteration: int,
        open_loop_detections: List[TrackedObject],
        radius: float,
        traffic_light_status: Optional[Dict[TrafficLightStatusType, List[str]]] = None,
    ) -> None:
        """
        Propagate each active agent forward in time.

        :param ego_state: the ego's current state in the simulation.
        :param tspan: the interval of time to simulate.
        :param iteration: the simulation iteration.
        :param traffic_light_status: {traffic_light_status: lane_connector_ids} A dictionary containing traffic light information.
        :param open_loop_detections: A list of open loop detections the IDM agents should be responsive to.
        :param radius: [m] The radius around the ego state
        """
        self.agent_occupancy.set("ego", ego_state.car_footprint.geometry)
        track_ids = []
        for track in open_loop_detections:
            track_ids.append(track.track_token)
            self.agent_occupancy.insert(track.track_token, track.box.geometry)

        self._filter_agents_out_of_range(ego_state, radius)

        for agent_token, agent in self.agents.items():
            if agent.is_active(iteration) and agent.has_valid_path():
                if traffic_light_status is not None:
                    agent.plan_route(traffic_light_status)
                    # Add stop lines into occupancy map if they are impacting the agent
                    stop_lines = self._get_relevant_stop_lines(
                        agent, traffic_light_status
                    )
                    # Keep track of the stop lines that were inserted. This is to remove them for each agent
                    inactive_stop_line_tokens = (
                        self._insert_stop_lines_into_occupancy_map(stop_lines)
                    )

                # Check for agents that intersects THIS agent's path
                agent_path = path_to_linestring(agent.get_path_to_go())

                # simplify the shapes of the occupancy map to filter out subsequent
                # points in a geometry that are almost identical as they will lead to an
                # exception when checking if a point is inside or outside the geometry
                self.agent_occupancy = simplify_occupancy_map_geometries(
                    self.agent_occupancy, tolerance=1e-5
                )

                intersecting_agents = self.agent_occupancy.intersects(
                    agent_path.buffer((agent.width / 2), cap_style=CAP_STYLE.flat)
                )
                assert intersecting_agents.contains(
                    agent_token
                ), "Agent's baseline does not intersect the agent itself"

                # Checking if there are agents intersecting THIS agent's baseline.
                # Hence, we are checking for at least 2 intersecting agents.
                if intersecting_agents.size > 1:
                    (
                        nearest_id,
                        nearest_agent_polygon,
                        relative_distance,
                    ) = intersecting_agents.get_nearest_entry_to(agent_token)
                    agent_heading = agent.to_se2().heading

                    if "ego" in nearest_id:
                        ego_velocity = ego_state.dynamic_car_state.rear_axle_velocity_2d
                        longitudinal_velocity = np.hypot(ego_velocity.x, ego_velocity.y)
                        relative_heading = ego_state.rear_axle.heading - agent_heading
                    elif "stop_line" in nearest_id:
                        longitudinal_velocity = 0.0
                        relative_heading = 0.0
                    elif nearest_id in self.agents:
                        nearest_agent = self.agents[nearest_id]
                        longitudinal_velocity = nearest_agent.velocity
                        relative_heading = (
                            nearest_agent.to_se2().heading - agent_heading
                        )
                    else:
                        longitudinal_velocity = 0.0
                        relative_heading = 0.0

                    # Wrap angle to [-pi, pi]
                    relative_heading = principal_value(relative_heading)
                    # take the longitudinal component of the projected velocity
                    projected_velocity = rotate_angle(
                        StateSE2(longitudinal_velocity, 0, 0), relative_heading
                    ).x

                    # relative_distance already takes the vehicle dimension into account.
                    # Therefore there is no need to pass in the length_rear.
                    length_rear = 0
                else:
                    # Free road case: no leading vehicle
                    projected_velocity = 0.0
                    relative_distance = agent.get_progress_to_go()
                    length_rear = agent.length / 2

                agent.propagate(
                    IDMLeadAgentState(
                        progress=relative_distance,
                        velocity=projected_velocity,
                        length_rear=length_rear,
                    ),
                    tspan,
                )
                self.agent_occupancy.set(agent_token, agent.projected_footprint)
                if traffic_light_status is not None:
                    self.agent_occupancy.remove(inactive_stop_line_tokens)
        self.agent_occupancy.remove(track_ids)

    def _get_relevant_stop_lines(
        self,
        agent: IDMAgent,
        traffic_light_status: Optional[Dict[TrafficLightStatusType, List[str]]],
    ) -> List[StopLine]:
        """
        Retrieve the stop lines that are affecting the given agent.

        :param agent: The IDM agent of interest.
        :param traffic_light_status: A dictionary containing traffic light information.
        :return: A list of stop lines associated with the given traffic light status.
        """
        if traffic_light_status is None:
            return []

        relevant_lane_connectors = list(
            {segment.id for segment in agent.get_route()}
            & set(traffic_light_status.get(TrafficLightStatusType.RED, []))
        )
        lane_connectors = [
            self._map_api.get_map_object(lc_id, SemanticMapLayer.LANE_CONNECTOR)
            for lc_id in relevant_lane_connectors
        ]
        return [
            stop_line for lc in lane_connectors if lc for stop_line in lc.stop_lines
        ]
