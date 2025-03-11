import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects import TrackedObject
from nuplan.planning.metrics.utils.collision_utils import CollisionType
from nuplan.planning.simulation.observation.idm.utils import is_agent_behind, is_track_stopped
from shapely import LineString, Polygon

from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex


def get_collision_type(
    state: npt.NDArray[np.float64],
    ego_polygon: Polygon,
    tracked_object: TrackedObject,
    tracked_object_polygon: Polygon,
    stopped_speed_threshold: float = 5e-02,
) -> CollisionType:
    """
    Classify collision between ego and the track.
    :param ego_state: Ego's state at the current timestamp.
    :param tracked_object: Tracked object.
    :param stopped_speed_threshold: Threshold for 0 speed due to noise.
    :return Collision type.
    """

    ego_speed = np.hypot(
        state[StateIndex.VELOCITY_X],
        state[StateIndex.VELOCITY_Y],
    )

    is_ego_stopped = float(ego_speed) <= stopped_speed_threshold

    center_point = tracked_object_polygon.centroid
    tracked_object_center = StateSE2(center_point.x, center_point.y, tracked_object.box.center.heading)

    ego_rear_axle_pose: StateSE2 = StateSE2(*state[StateIndex.STATE_SE2])

    # Collisions at (close-to) zero ego speed
    if is_ego_stopped:
        collision_type = CollisionType.STOPPED_EGO_COLLISION

    # Collisions at (close-to) zero track speed
    elif is_track_stopped(tracked_object):
        collision_type = CollisionType.STOPPED_TRACK_COLLISION

    # Rear collision when both ego and track are not stopped
    elif is_agent_behind(ego_rear_axle_pose, tracked_object_center):
        collision_type = CollisionType.ACTIVE_REAR_COLLISION

    # Front bumper collision when both ego and track are not stopped
    elif LineString(
        [
            ego_polygon.exterior.coords[0],
            ego_polygon.exterior.coords[3],
        ]
    ).intersects(tracked_object_polygon):
        collision_type = CollisionType.ACTIVE_FRONT_COLLISION

    # Lateral collision when both ego and track are not stopped
    else:
        collision_type = CollisionType.ACTIVE_LATERAL_COLLISION

    return collision_type
