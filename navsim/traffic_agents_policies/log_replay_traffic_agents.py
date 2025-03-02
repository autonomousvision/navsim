from typing import List

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.traffic_agents_policies.abstract_traffic_agents_policy import (
    AbstractTrafficAgentsPolicy,
)


class LogReplayTrafficAgents(AbstractTrafficAgentsPolicy):
    def __init__(self, future_trajectory_sampling: TrajectorySampling):
        self.future_trajectory_sampling = future_trajectory_sampling

    def get_list_of_simulated_object_types(self) -> List[TrackedObjectType]:
        """
        inherited. See Superclass
        """
        return [t for t in TrackedObjectType]

    def simulate_traffic_agents(
        self, simulated_ego_states: npt.NDArray[np.float64], metric_cache: MetricCache
    ) -> List[DetectionsTracks]:
        """
        inherited. See Superclass
        """
        raise NotImplementedError("Use simulate_environment instead for this policy")

    def simulate_environment(
        self, simulated_ego_states: npt.NDArray[np.float64], metric_cache: MetricCache
    ) -> List[DetectionsTracks]:
        """
        Simulate traffic agents, while ensuring we remove agents that collide with `ego` in the first frame across all frames.

        :param simulated_ego_states: Ego vehicle states over time.
        :param metric_cache: The metric cache containing ego state and detected objects.
        :return: A list of `DetectionsTracks`, with colliding agents removed across all frames.
        """

        ego_box = metric_cache.ego_state.car_footprint.oriented_box.geometry

        num_poses = self.future_trajectory_sampling.num_poses + 1
        detections_tracks = metric_cache.observation.detections_tracks[:num_poses]

        first_frame_detections = detections_tracks[0]

        colliding_agents = {
            agent.metadata.track_token
            for agent in first_frame_detections.tracked_objects.tracked_objects
            if agent.box.geometry.intersects(ego_box)
        }

        if not colliding_agents:
            return detections_tracks

        cleaned_detections_tracks = []
        for frame_detections in detections_tracks:
            filtered_agents = [
                agent
                for agent in frame_detections.tracked_objects.tracked_objects
                if agent.metadata.track_token not in colliding_agents
            ]
            cleaned_detections_tracks.append(
                DetectionsTracks(TrackedObjects(filtered_agents))
            )

        return cleaned_detections_tracks
