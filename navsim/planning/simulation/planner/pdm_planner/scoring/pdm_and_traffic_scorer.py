from copy import deepcopy
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters, get_pacifica_parameters
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.common.maps.nuplan_map.nuplan_map import NuPlanMap
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from shapely import Point

from navsim.common.dataclasses import PDMResults
from navsim.planning.metric_caching.metric_cache import MapParameters
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_observation import PDMObservation
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
from navsim.traffic_agents_policies.abstract_traffic_agents_policy import (
    extract_vehicle_trajectories_from_detections_tracks,
)


class PaddingTrackedObject:
    def __init__(self, track_token):
        self.track_token = track_token
        self.box = OrientedBox(
            center=StateSE2(x=np.inf, y=np.inf, heading=0),
            length=0,
            width=0,
            height=0,
        )
        self.velocity = StateVector2D(x=0, y=0)
        self.tracked_object_type = TrackedObjectType.VEHICLE
        self.center = self.box.center


class PDMTrafficScorer(PDMScorer):
    def __init__(
        self,
        proposal_sampling: TrajectorySampling,
        config: PDMScorerConfig = PDMScorerConfig(),
        vehicle_parameters: VehicleParameters = get_pacifica_parameters(),
    ):
        super().__init__(proposal_sampling, config, vehicle_parameters)

    def extract_centerline_for_agent(
        self, agent_states: List[npt.NDArray[np.float64]], map_api: NuPlanMap
    ) -> Tuple[PDMPath, List[int]]:
        # find starting lane for agent
        proximal_objects = map_api.get_proximal_map_objects(
            point=Point(agent_states[0][StateIndex.X], agent_states[0][StateIndex.Y]),
            layers=[SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR],
            radius=10.0,
        )
        nearby_lanes = [map_obj for map_objects in proximal_objects.values() for map_obj in map_objects]
        if len(nearby_lanes) == 0:
            return None, None
        elif len(nearby_lanes) == 1:
            starting_lane = nearby_lanes[0]
        else:
            agent_point = Point(agent_states[0][StateIndex.X], agent_states[0][StateIndex.Y])
            dist_to_lanes = [lane.polygon.distance(agent_point) for lane in nearby_lanes]
            starting_lane = nearby_lanes[np.argmin(dist_to_lanes)]

        # match trajectory onto successor lanes
        agent_lanes = [starting_lane]
        for agent_state in agent_states[1:]:
            # try to match state to one of the outgoing lanes
            outgoing_lanes = starting_lane.outgoing_edges
            candidate_lanes = outgoing_lanes + agent_lanes
            agent_point = Point(agent_state[StateIndex.X], agent_state[StateIndex.Y])
            dist_to_lanes = [lane.polygon.distance(agent_point) for lane in candidate_lanes]
            closest_lane = candidate_lanes[np.argmin(dist_to_lanes)]
            agent_lanes.append(closest_lane)

        # return PDM-Path and lane ids
        agent_centerline = PDMPath(
            discrete_path=[state for lane in agent_lanes for state in lane.baseline_path.discrete_path],
        )
        lane_ids = [lane.id for lane in agent_lanes]

        return agent_centerline, lane_ids

    def build_ego_tracked_object_states(
        self,
        states: npt.NDArray[np.float64],
    ) -> List[Agent]:
        vehicle_parameters = get_pacifica_parameters()
        return [
            Agent(
                tracked_object_type=TrackedObjectType.VEHICLE,
                oriented_box=OrientedBox(
                    center=StateSE2(x=state[StateIndex.X], y=state[StateIndex.Y], heading=state[StateIndex.HEADING]),
                    length=vehicle_parameters.length,
                    width=vehicle_parameters.width,
                    height=vehicle_parameters.height,
                ),
                velocity=StateVector2D(x=state[StateIndex.VELOCITY_X], y=state[StateIndex.VELOCITY_Y]),
                metadata=SceneObjectMetadata(
                    timestamp_us=0,
                    token=None,
                    track_id=None,
                    track_token="ego",
                ),
                angular_velocity=state[StateIndex.ANGULAR_VELOCITY],
            )
            for state in states
        ]

    def build_agent_centric_observation(
        self,
        observation: PDMObservation,
        traffic_agent_detections_tracks: List[DetectionsTracks],
        ego_tracked_objects: List[Agent],
        agent_token: str,
    ) -> PDMObservation:

        agent_centric_observation = deepcopy(observation)

        # remove target agent tracks from observation
        # and add ego agent to it
        agent_centric_detection_tracks = [
            DetectionsTracks(
                tracked_objects=TrackedObjects(
                    [o for o in tracks.tracked_objects if o.track_token != agent_token] + [ego_tracked_objects[t]]
                )
            )
            for t, tracks in enumerate(traffic_agent_detections_tracks)
        ]

        agent_centric_observation.update_detections_tracks(
            detection_tracks=agent_centric_detection_tracks,
        )
        return agent_centric_observation

    def build_ego_centric_observation(
        self,
        observation: PDMObservation,
        traffic_agent_detections_tracks: List[DetectionsTracks],
    ) -> PDMObservation:
        ego_centric_observation = deepcopy(observation)
        ego_centric_observation.update_detections_tracks(
            detection_tracks=traffic_agent_detections_tracks,
        )
        return ego_centric_observation

    def score_proposals(
        self,
        states: npt.NDArray[np.float64],
        observation: PDMObservation,
        centerline: PDMPath,
        route_lane_ids: List[str],
        drivable_area_map: PDMDrivableMap,
        map_parameters: MapParameters,
        simulated_agent_detections_tracks: List[DetectionsTracks],
    ) -> List[pd.DataFrame]:

        map_api = get_maps_api(map_parameters.map_root, map_parameters.map_version, map_parameters.map_name)

        # Observations need to be one second longer than the ego-trajectory to calculate ego ttc metrics
        # Thus, we slice the traffic agents trajectories to only evaluate the first four seconds
        trajectory_length = states.shape[1]
        (
            logreplay_agent_trajectories,
            logreplay_agent_masks,
            logreplay_agent_tokens,
        ) = extract_vehicle_trajectories_from_detections_tracks(
            detections_tracks=observation.detections_tracks[:trajectory_length],
            reverse_padding=False,
        )
        (
            simulated_agent_trajectories,
            simulated_agent_masks,
            simulated_agent_tokens,
        ) = extract_vehicle_trajectories_from_detections_tracks(
            detections_tracks=simulated_agent_detections_tracks[:trajectory_length],
            reverse_padding=False,
        )
        # The log-replay agents might contain more agents than the simulated agents
        # since the log-replay also covers agents not visible at the current state, which are
        # interpolated to the current timestamp. The simulated agents only contain the agents
        # that are visible at the current state. We need to make sure to access the correct
        # log-replay trajectory for each agent, so we store them in a dictionary with the agent token as key
        simulated_agents = {
            token: (simulated_agent_trajectory, simulated_agent_mask)
            for (token, simulated_agent_trajectory, simulated_agent_mask) in zip(
                simulated_agent_tokens, simulated_agent_trajectories, simulated_agent_masks
            )
        }
        logreplay_agents = {
            token: (logreplay_agent_trajectory, logreplay_agent_mask)
            for (token, logreplay_agent_trajectory, logreplay_agent_mask) in zip(
                logreplay_agent_tokens, logreplay_agent_trajectories, logreplay_agent_masks
            )
        }

        # We evaluate traffic agents with respect to the ego proposal at index 1
        # which refers to the model outut. Note: This is very hacky and should be refactored
        pred_idx = 1
        ego_tracked_objects = self.build_ego_tracked_object_states(states[pred_idx])

        agent_scores = []
        for agent_token in simulated_agent_tokens:
            agent_mask = simulated_agents[agent_token][1]
            agent_trajectory = simulated_agents[agent_token][0]
            logreplay_agent_trajectory = logreplay_agents[agent_token][0]
            if not np.all(agent_mask):
                # agent is not observed the whole time, so we skip it
                continue

            # extract centerline for agent
            agent_centerline, lane_ids = self.extract_centerline_for_agent(
                agent_states=agent_trajectory, map_api=map_api
            )
            if agent_centerline is None:
                # agent is too far away from the map, so we skip it
                continue

            # for each agent, we need to remove its own tracks from the observation and add ego
            agent_centric_observation = self.build_agent_centric_observation(
                observation=observation,
                ego_tracked_objects=ego_tracked_objects,
                traffic_agent_detections_tracks=simulated_agent_detections_tracks,
                agent_token=agent_token,
            )

            # we also evaluate the log-replay future for each agent, otherwise progress metric is meaningless
            agent_proposals = np.stack([agent_trajectory, logreplay_agent_trajectory], axis=0)

            # evaluate the trajectory of the agent
            agent_score = super().score_proposals(
                states=agent_proposals,
                observation=agent_centric_observation,
                centerline=agent_centerline,
                route_lane_ids=lane_ids,
                drivable_area_map=drivable_area_map,
            )[0]
            agent_scores.append(agent_score)

        if len(agent_scores) > 0:
            agent_scores = pd.concat(agent_scores)
            agent_scores = agent_scores.mean(axis=0).to_frame().T.add_prefix("traffic_")
        else:
            agent_scores = pd.DataFrame([PDMResults.get_empty_results()])
            agent_scores = agent_scores.add_prefix("traffic_")

        # make sure to use the reactive agents here
        ego_centric_observation = self.build_ego_centric_observation(
            observation=observation,
            traffic_agent_detections_tracks=simulated_agent_detections_tracks,
        )

        ego_scores = super().score_proposals(
            states=states,
            observation=ego_centric_observation,
            centerline=centerline,
            route_lane_ids=route_lane_ids,
            drivable_area_map=drivable_area_map,
        )

        # we have exactly one future traffic scenario in the metric cache,
        # but potentially multiple ego-proposals
        # (usually at least two, i.e. one from PDM and one from the Agent).
        # Thus, we append the traffic scores to each ego score result.
        results = [pd.concat([ego_score, agent_scores], axis=1) for ego_score in ego_scores]

        return results
