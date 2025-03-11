from copy import deepcopy
from typing import List

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimeDuration, TimePoint
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.observation.navsim_idm_agents import NavsimIDMAgents
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
from navsim.traffic_agents_policies.abstract_traffic_agents_policy import (
    AbstractTrafficAgentsPolicy,
    filter_tracked_objects_by_type,
)


def build_ego_state_from_simulated_ego_states(
    ego_state_arr: npt.NDArray[np.float64],
    initial_ego_state: EgoState,
    time_point: TimePoint,
) -> EgoState:
    car_footprint = CarFootprint(
        center=StateSE2(
            x=ego_state_arr[StateIndex.X],
            y=ego_state_arr[StateIndex.Y],
            heading=ego_state_arr[StateIndex.HEADING],
        ),
        vehicle_parameters=initial_ego_state.car_footprint.vehicle_parameters,
    )
    rear_axle_acceleration_2d = StateVector2D(
        x=ego_state_arr[StateIndex.ACCELERATION_X],
        y=ego_state_arr[StateIndex.ACCELERATION_Y],
    )
    rear_axle_velocity_2d = StateVector2D(
        x=ego_state_arr[StateIndex.VELOCITY_X],
        y=ego_state_arr[StateIndex.VELOCITY_Y],
    )
    dynamic_car_state = DynamicCarState(
        rear_axle_to_center_dist=initial_ego_state.car_footprint.vehicle_parameters.rear_axle_to_center,
        rear_axle_acceleration_2d=rear_axle_acceleration_2d,
        rear_axle_velocity_2d=rear_axle_velocity_2d,
        angular_acceleration=ego_state_arr[StateIndex.ANGULAR_ACCELERATION],
        tire_steering_rate=ego_state_arr[StateIndex.STEERING_RATE],
    )
    return EgoState(
        car_footprint=car_footprint,
        dynamic_car_state=dynamic_car_state,
        tire_steering_angle=ego_state_arr[StateIndex.STEERING_ANGLE],
        is_in_auto_mode=initial_ego_state.is_in_auto_mode,
        time_point=time_point,
    )


class NavsimIDMTrafficAgents(AbstractTrafficAgentsPolicy):
    def __init__(
        self,
        future_trajectory_sampling: TrajectorySampling,
        idm_agents_observation: NavsimIDMAgents,
    ):
        self.future_trajectory_sampling = future_trajectory_sampling
        self._idm_agents_observation: NavsimIDMAgents = idm_agents_observation

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

        # egostate
        initial_ego_state = metric_cache.ego_state
        # map api
        map_api = get_maps_api(
            metric_cache.map_parameters.map_root,
            metric_cache.map_parameters.map_version,
            metric_cache.map_parameters.map_name,
        )
        # extract future tracked objects
        objects_future_tracks = metric_cache.future_tracked_objects
        # traffic light status
        traffic_light_status = getattr(metric_cache, "traffic_light_status", None)

        # we need to make a fresh copy of the idm_agents_observation
        # otherwise its state will leak into other simulations
        idm_agents_observation = deepcopy(self._idm_agents_observation)

        # current observation
        idm_agents_observation.get_observation(
            initial_ego_state, vehicle_current_tracks, map_api, objects_future_tracks[0].tracked_objects
        )

        # build EgoStates from the simulated_ego_states array
        ego_future_trajectory: List[EgoState] = []
        for t, ego_state_arr in enumerate(simulated_ego_states):
            ego_future_trajectory.append(
                build_ego_state_from_simulated_ego_states(
                    ego_state_arr=ego_state_arr,
                    initial_ego_state=initial_ego_state,
                    time_point=initial_ego_state.time_point + TimeDuration.from_s((t + 1) * 0.1),
                )
            )

        # simulate the future trajectory of the vehicle agents
        future_tracked_objects: List[DetectionsTracks] = []
        for timestep in range(1, self.future_trajectory_sampling.num_poses + 1):

            ego_state = ego_future_trajectory[timestep - 1]

            idm_agents_observation.update_observation(
                timestep,
                ego_state,
                vehicle_current_tracks,
                map_api,
                objects_future_tracks[timestep - 1].tracked_objects,
                traffic_light_status[timestep] if traffic_light_status is not None else None,
            )
            future_tracked_objects.append(
                idm_agents_observation.get_observation(
                    ego_state, vehicle_current_tracks, map_api, objects_future_tracks[timestep - 1].tracked_objects
                )
            )

        return future_tracked_objects
