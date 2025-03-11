from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES, TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataclasses import Annotations, EgoStatus
from navsim.common.enums import BoundingBoxIndex
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import normalize_angle

# TODO: Refactor this file
tracked_object_types: Dict[str, TrackedObjectType] = {
    "vehicle": TrackedObjectType.VEHICLE,
    "pedestrian": TrackedObjectType.PEDESTRIAN,
    "bicycle": TrackedObjectType.BICYCLE,
    "traffic_cone": TrackedObjectType.TRAFFIC_CONE,
    "barrier": TrackedObjectType.BARRIER,
    "czone_sign": TrackedObjectType.CZONE_SIGN,
    "generic_object": TrackedObjectType.GENERIC_OBJECT,
    "ego": TrackedObjectType.EGO,
}


def ego_status_to_ego_state(
    ego_status: EgoStatus, vehicle_parameters: VehicleParameters, time_point: TimePoint
) -> EgoState:
    rear_axle_velocity_2d = StateVector2D(*ego_status.ego_velocity)
    rear_axle_acceleration_2d = StateVector2D(*ego_status.ego_acceleration)
    return EgoState.build_from_rear_axle(
        StateSE2(*ego_status.ego_pose),
        tire_steering_angle=0.0,
        vehicle_parameters=vehicle_parameters,
        time_point=time_point,
        rear_axle_velocity_2d=rear_axle_velocity_2d,
        rear_axle_acceleration_2d=rear_axle_acceleration_2d,
    )


def annotations_to_detection_tracks(annotations: Annotations, ego_state: EgoState) -> DetectionsTracks:
    """
    Convert annotations dataclass (NAVSIM) too detection tracks (nuPlan).
    :param annotations: dataclass for bounding box annotations
    :param ego_state: object of ego vehicle state
    :return: detection tracks dataclass
    """

    detection_tracks: List[TrackedObject] = []

    time_point = ego_state.time_point
    track_boxes = gt_boxes_oriented_box(annotations.boxes, ego_state)

    for track_idx, track_box in enumerate(track_boxes):
        track_type = tracked_object_types[annotations.names[track_idx]]
        track_metadata = SceneObjectMetadata(
            time_point.time_us,
            token=annotations.instance_tokens[track_idx],
            track_id=None,
            track_token=annotations.track_tokens[track_idx],
        )

        if track_type in AGENT_TYPES:
            vx, vy = (
                annotations.velocity_3d[track_idx][0],
                annotations.velocity_3d[track_idx][1],
            )
            velocity = StateVector2D(vx, vy)

            detection_track = Agent(
                tracked_object_type=track_type,
                oriented_box=track_box,
                velocity=rotate_vector(velocity, ego_state.rear_axle.heading),
                metadata=track_metadata,
            )
        else:
            detection_track = StaticObject(
                tracked_object_type=track_type,
                oriented_box=track_box,
                metadata=track_metadata,
            )

        detection_tracks.append(detection_track)

    return DetectionsTracks(TrackedObjects(detection_tracks))


def gt_boxes_oriented_box(gt_boxes: List[npt.NDArray[np.float32]], ego_state: EgoState) -> List[OrientedBox]:
    """
    Concerts bounding box extends from NAVSIM to oriented box objects in nuPlan.
    :param gt_boxes: numpy array contained box extend
    :param ego_state: object of ego vehicle state
    :return: list of oriented object instances
    """

    oriented_boxes: List[OrientedBox] = []
    for gt_box in gt_boxes:
        local_box_x, local_box_y, local_box_heading = (
            gt_box[BoundingBoxIndex.X],
            gt_box[BoundingBoxIndex.Y],
            gt_box[BoundingBoxIndex.HEADING],
        )
        local_box_se2 = rotate_state_se2(
            StateSE2(local_box_x, local_box_y, local_box_heading),
            angle=ego_state.rear_axle.heading,
        )

        global_box_x, global_box_y, global_box_heading = (
            local_box_se2.x + ego_state.rear_axle.x,
            local_box_se2.y + ego_state.rear_axle.y,
            normalize_angle(local_box_se2.heading),
        )
        box_length, box_width, box_height = gt_box[3], gt_box[4], gt_box[5]
        oriented_box = OrientedBox(
            StateSE2(global_box_x, global_box_y, global_box_heading),
            box_length,
            box_width,
            box_height,
        )
        oriented_boxes.append(oriented_box)

    return oriented_boxes


def rotate_state_se2(state_se2: StateSE2, angle: float = np.deg2rad(0)) -> StateSE2:
    """
    Rotate (x,y,θ) state given angle
    :param state_se2: object containing SE(2) information
    :param angle: angle in rad to rotate, defaults to np.deg2rad(0)
    :return: rotated SE(2) object
    """
    sin, cos = np.sin(angle), np.cos(angle)
    x_rotated = state_se2.x * cos - state_se2.y * sin
    y_rotated = state_se2.x * sin + state_se2.y * cos
    heading_rotated = normalize_angle(state_se2.heading + angle)
    return StateSE2(x_rotated, y_rotated, heading_rotated)


def rotate_vector(vector: StateVector2D, angle: float) -> StateVector2D:
    """
    Rotate 2D vector given angle
    :param vector: object describing 2D vector
    :param angle: angle in rad to rotate
    :return: rotated 2D vector
    """
    sin, cos = np.sin(angle), np.cos(angle)
    x_rotated = vector.x * cos - vector.y * sin
    y_rotated = vector.x * sin + vector.y * cos
    return StateVector2D(x_rotated, y_rotated)


def sample_future_indices(
    future_sampling: TrajectorySampling, iteration: int, time_horizon: float, num_samples: Optional[int]
) -> List[int]:
    """
    Helper function to sample future time indices.
    :param future_sampling: dataclass describing future sampling specification.
    :param iteration: starting iteration to sample
    :param time_horizon: horizon to sample for [s]
    :param num_samples: number of future sample to extract
    :raises ValueError: invalid input arguments
    :return: list of integers
    """
    time_interval = future_sampling.interval_length
    if time_horizon <= 0.0 or time_interval <= 0.0 or time_horizon < time_interval:
        raise ValueError(
            f"Time horizon {time_horizon} must be greater or equal than target time interval {time_interval}"
            " and both must be positive."
        )

    num_samples = num_samples if num_samples else int(time_horizon / time_interval)

    num_intervals = int(time_horizon / time_interval) + 1
    step_size = int(num_intervals // num_samples)
    time_idcs = np.arange(iteration, num_intervals, step_size)

    return list(time_idcs)


def sample_past_indices(
    past_sampling: TrajectorySampling, iteration: int, time_horizon: float, num_samples: Optional[int]
) -> List[int]:
    """
    Helper function to sample past time indices.
    :param past_sampling: dataclass describing past sampling specification.
    :param iteration: starting iteration to sample
    :param time_horizon: horizon to sample for [s]
    :param num_samples: number of past sample to extract
    :raises ValueError: invalid input arguments
    :return: list of integers
    """
    time_idcs = sample_future_indices(
        future_sampling=past_sampling, iteration=iteration, time_horizon=time_horizon, num_samples=num_samples
    )
    return [-idx for idx in time_idcs[::-1]]
