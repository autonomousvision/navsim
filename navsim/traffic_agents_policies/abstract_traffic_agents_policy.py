from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import numpy.typing as npt

from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import (
    filter_agents,
    extract_and_pad_agent_states
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.planning.metric_caching.metric_cache import MetricCache

def extract_vehicle_trajectories_from_detections_tracks(
    detections_tracks: List[DetectionsTracks],
    reverse_padding: bool=False,
) -> Tuple[List[npt.NDArray[np.float64]], List[npt.NDArray[np.bool]], List[str]]:
    """
    Extract the agent states as an array and pads it with the most recent available states. 
    Note: agents that don't appear in the current time step are ignored.
    See nuplan's extract_and_pad_agent_states for details
    :param detections_tracks: list of DetectionsTracks objects
    :param reverse_padding: if True, the last element in the list will be used as the filter. 
        Set to true when filtering past trajectories
    :returns 
        agent_trajectories: list of length num agents of agent trajectories of shape (num timesteps, 6)
        agent_trajectory_masks: list of length num agents of agent trajectory masks of shape (num timesteps, 1)
        agent_tokens: list of length num agents of agent tokens
    """
    def _state_extractor(scene_object: Agent) -> npt.NDArray[np.float64]:
        return np.array([
            scene_object.center.x,
            scene_object.center.y,
            scene_object.center.heading,
            scene_object.velocity.x,
            scene_object.velocity.y,
            0.0, # acceleration.x,
            0.0, # acceleration.y,
            0.0, # steering angle
            0.0, # steering rate
            0.0, # angular_velocity,
            0.0, # angular_acceleration,
    ])

    def _token_extractor(scene_object: Agent) -> str:
        return scene_object.track_token
    
    vehicle_tracks = filter_tracked_objects_by_type(
    detections_tracks, TrackedObjectType.VEHICLE
    )
    tracked_objects: List[TrackedObjects] = [
        t.tracked_objects for t in vehicle_tracks
    ]
    filtered_agents = filter_agents(tracked_objects, reverse=reverse_padding)

    # agent_states and masks are nested lists of [num_frames, num_agents]
    (
        agent_states_horizon,
        agent_states_horizon_masks,
    ) = extract_and_pad_agent_states(
        agent_trajectories=filtered_agents,
        state_extractor=_state_extractor,
        reverse=reverse_padding,
    )
    agent_trajectories = [
        np.array([
            agent_states_horizon[t][a]
            for t in range(len(agent_states_horizon))
        ])
        for a in range(len(agent_states_horizon[0]))
    ]
    agent_trajectory_masks = [
        np.array([
            agent_states_horizon_masks[t][a]
            for t in range(len(agent_states_horizon_masks))
        ])
        for a in range(len(agent_states_horizon_masks[0]))
    ]

    # agent_states and masks are nested lists of [num_frames, num_agents]
    (
        agent_tokens_horizon,
        _,
    ) = extract_and_pad_agent_states(
        agent_trajectories=filtered_agents,
        state_extractor=_token_extractor,
        reverse=reverse_padding,
    )
    # tokens remain constant over time so we only need the first frame
    agent_tokens = agent_tokens_horizon[0]
    return agent_trajectories, agent_trajectory_masks, agent_tokens

def filter_tracked_objects_by_type(
    tracked_objects: List[DetectionsTracks], object_type: TrackedObjectType
) -> List[DetectionsTracks]:
    return [
        DetectionsTracks(
            TrackedObjects(p.tracked_objects.get_tracked_objects_of_type(object_type))
        )
        for p in tracked_objects
    ]

def filter_tracked_objects_by_types(
    tracked_objects: List[DetectionsTracks], object_types: List[TrackedObjectType]
) -> List[DetectionsTracks]:
    return [
        DetectionsTracks(
            TrackedObjects(p.tracked_objects.get_tracked_objects_of_types(object_types))
        )
        for p in tracked_objects
    ]

class AbstractTrafficAgentsPolicy(ABC):    

    @abstractmethod
    def __init__(self, future_trajectory_sampling: TrajectorySampling) -> None:
        pass

    @abstractmethod
    def get_list_of_simulated_object_types(self) -> List[TrackedObjectType]:
        """
        Returns the list of object types that the policy simulates.
        For all remaining objects, the ground truth future tracks are used.
        The policy should only return the tracks for the object types it simulates.
        The remaining objects are automatically merged to the DetectionsTracks.
        """
        pass

    def simulate_environment(
        self,
        simulated_ego_states: npt.NDArray[np.float64],
        metric_cache: MetricCache
    ) -> List[DetectionsTracks]:
        """
        Simulates the environment, including the ego agent and traffic agents.
        :param simulated_ego_states: trajectory the ego-vehicle will follow
        :param metric_cache: general metric cache with describing the state of all agents and their environment
        :return: DetectionsTracks object containing the simulated traffic agents
        """

        simulated_detections_tracks = self.simulate_traffic_agents(simulated_ego_states, metric_cache)

        # assert that the simulated detectionstracks only include the object types that the policy simulates
        assert all(
            all(
                obj.tracked_object_type in self.get_list_of_simulated_object_types()
                for obj in detections_tracks.tracked_objects
            )
            for detections_tracks in simulated_detections_tracks
        ), "Traffic agents policy must only return detections tracks of the object types it simulates (see get_list_of_simulated_object_types)"

        remaining_object_detections_tracks = filter_tracked_objects_by_types(
            metric_cache.future_tracked_objects,
            [t for t in TrackedObjectType if t not in self.get_list_of_simulated_object_types()]
        )
        # the metric cache might contain longer tracks than we simulate, so we truncate the remaining objects' tracks
        remaining_object_detections_tracks = remaining_object_detections_tracks[:len(simulated_detections_tracks)]
        assert len(simulated_detections_tracks) + 1 == simulated_ego_states.shape[0], (
            f"""
                Traffic agents policy returned trajectories of invalid length:
                {len(simulated_detections_tracks) + 1} != {simulated_ego_states.shape[0]}
            """
        )

        # merge simulated and log-replay object tracks
        future_detections_tracks = [
            DetectionsTracks(
                TrackedObjects(
                    [obj for obj in simulated_detections_tracks[i].tracked_objects]
                    + [obj for obj in remaining_object_detections_tracks[i].tracked_objects]
                )
            )
            for i in range(len(simulated_detections_tracks))
        ]

        return metric_cache.current_tracked_objects + future_detections_tracks


    @abstractmethod
    def simulate_traffic_agents(
        self,
        simulated_ego_states: npt.NDArray[np.float64],
        metric_cache: MetricCache
    ) -> List[DetectionsTracks]:
        """
        Simulates the (reactive) behavior of traffic agents, 
        given that the ego agent follows the trajectory provided.
            :param simulated_ego_states: trajectory the ego-vehicle will follow
            :param metric_cache: general metric cache with describing the state of all agents and their environment
            :return: DetectionsTracks object containing the simulated traffic agents
        """
        pass