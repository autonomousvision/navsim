from typing import List

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.traffic_agents_policies.abstract_traffic_agents_policy import (
    AbstractTrafficAgentsPolicy,
    filter_tracked_objects_by_type,
)


class ConstantVelocityTrafficAgents(AbstractTrafficAgentsPolicy):
    """Naive background traffic agents with constant velocity and constant heading."""

    def __init__(self, future_trajectory_sampling: TrajectorySampling):
        self.future_trajectory_sampling = future_trajectory_sampling

    def get_list_of_simulated_object_types(self) -> List[TrackedObjectType]:
        """Inherited, see superclass."""
        return [TrackedObjectType.VEHICLE]

    def simulate_traffic_agents(
        self, simulated_ego_states: npt.NDArray[np.float64], metric_cache: MetricCache
    ) -> List[DetectionsTracks]:
        """Inherited, see superclass."""
        # extract all vehicle agents in the current frame
        vehicle_current_tracks = filter_tracked_objects_by_type(
            metric_cache.current_tracked_objects, TrackedObjectType.VEHICLE
        )[0]

        # simulate the future trajectory of the vehicle agents
        future_tracked_objects: List[DetectionsTracks] = []
        for timestep in range(1, self.future_trajectory_sampling.num_poses + 1):
            dt = timestep * self.future_trajectory_sampling.interval_length
            future_tracked_objects_at_dt = []

            agent: Agent
            for agent in vehicle_current_tracks.tracked_objects:
                future_oriented_box = OrientedBox.from_new_pose(
                    box=agent.box,
                    pose=StateSE2(
                        x=agent.center.x + agent.velocity.x * dt,
                        y=agent.center.y + agent.velocity.y * dt,
                        heading=agent.center.heading,
                    ),
                )

                future_agent_detection = Agent(
                    tracked_object_type=TrackedObjectType.VEHICLE,
                    oriented_box=future_oriented_box,
                    velocity=agent.velocity,
                    metadata=agent.metadata,
                    angular_velocity=agent.angular_velocity,
                )
                future_tracked_objects_at_dt.append(future_agent_detection)

            future_tracked_objects.append(
                DetectionsTracks(tracked_objects=TrackedObjects(tracked_objects=future_tracked_objects_at_dt))
            )
        return future_tracked_objects
