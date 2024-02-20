from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import numpy.typing as npt
from PIL import Image

from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.common.actor_state.state_representation import StateSE2

from pyquaternion import Quaternion
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import warnings

NAVSIM_INTERVAL_LENGTH: float = 0.5
OPENSCENE_DATA_ROOT = os.environ.get("OPENSCENE_DATA_ROOT")

@dataclass
class Camera:
    image: npt.NDArray[np.float32]

    # TODO: Refactor parameters
    sensor2lidar_rotation: npt.NDArray[np.float32]
    sensor2lidar_translation: npt.NDArray[np.float32]
    intrinsics: npt.NDArray[np.float32]
    distortion: npt.NDArray[np.float32]


@dataclass
class Cameras:

    f0: Camera
    l0: Camera
    l1: Camera
    l2: Camera
    r0: Camera
    r1: Camera
    r2: Camera
    b0: Camera

    @classmethod
    def from_camera_dict(cls, camera_dict: Dict[str, Any]) -> Cameras:

        data_dict: Dict[str, Camera] = {}
        for camera_name in camera_dict.keys():
            # TODO: adapt for complete OpenScenes data
            image_path = (
                Path(OPENSCENE_DATA_ROOT)
                / "sensor_blobs/mini"
                / camera_dict[camera_name]["data_path"]
            )
            data_dict[camera_name] = Camera(
                image=np.array(Image.open(image_path)),
                sensor2lidar_rotation=camera_dict[camera_name]["sensor2lidar_rotation"],
                sensor2lidar_translation=camera_dict[camera_name]["sensor2lidar_translation"],
                intrinsics=camera_dict[camera_name]["cam_intrinsic"],
                distortion=camera_dict[camera_name]["distortion"],
            )

        return Cameras(
            f0=data_dict["CAM_F0"],
            l0=data_dict["CAM_L0"],
            l1=data_dict["CAM_L1"],
            l2=data_dict["CAM_L2"],
            r0=data_dict["CAM_R0"],
            r1=data_dict["CAM_R1"],
            r2=data_dict["CAM_R2"],
            b0=data_dict["CAM_B0"],
        )


@dataclass
class Lidar:
    merged_pc: npt.NDArray[np.float32]

    # TODO: add lidar parameters
    parameters: Optional[Any] = None


@dataclass
class EgoStatus:

    ego_pose: npt.NDArray[np.float64]
    ego_velocity: npt.NDArray[np.float32]
    ego_acceleration: npt.NDArray[np.float32]
    driving_command: npt.NDArray[np.int]
    in_global_frame: bool = False  # False for AgentInput


@dataclass
class AgentInput:

    ego_statuses: List[EgoStatus]
    cameras: Optional[List[Cameras]] = None 
    lidars: Optional[List[Lidar]] = None 

    def __post_init__(self):
        # TODO: add assertions
        pass

    @classmethod
    def from_scene_dict_list(
        cls,
        scene_dict_list: List[Dict],
        num_history_frames: int,
        sensor_modalities: List[str] = ["lidar", "camera"],
    ) -> AgentInput:
        assert len(scene_dict_list) > 0, "Scene list is empty!"

        include_cameras = "camera" in sensor_modalities
        include_lidar = "lidar" in sensor_modalities

        ego_statuses: List[EgoStatus] = []
        cameras: List[EgoStatus] = [] if include_cameras else None
        lidars: List[Lidar] = [] if include_lidar else None

        for frame_idx in range(num_history_frames):

            # 1. Ego statuses
            ego_translation = scene_dict_list[frame_idx]["ego2global_translation"]
            ego_quaternion = Quaternion(*scene_dict_list[frame_idx]["ego2global_rotation"])
            ego_pose = np.array(
                [ego_translation[0], ego_translation[1], ego_quaternion.yaw_pitch_roll[0]],
                dtype=np.float64,
            )
            ego_dynamic_state = scene_dict_list[frame_idx]["ego_dynamic_state"]
            ego_status = EgoStatus(
                ego_pose=ego_pose,
                ego_velocity=np.array(ego_dynamic_state[:2], dtype=np.float32),
                ego_acceleration=np.array(ego_dynamic_state[2:], dtype=np.float32),
                driving_command=scene_dict_list[frame_idx]["driving_command"],
            )
            ego_statuses.append(ego_status)

            if include_cameras:
                cameras.append(Cameras.from_camera_dict(scene_dict_list[frame_idx]["cams"]))

            if include_lidar:
                # TODO: Add lidar data
                warnings.warn(f"Lidar currently not available in OpenScenes!")
                lidars = None

        return AgentInput(ego_statuses, cameras, lidars)


@dataclass
class Annotations:

    boxes: npt.NDArray[np.float32]
    names: List[str]
    velocity_3d: npt.NDArray[np.float32]
    instance_tokens: List[str]
    track_tokens: List[str]

    def __post_init__(self):
        annotation_lengths: Dict[str, int] = {
            attribute_name: len(attribute) for attribute_name, attribute in vars(self).items()
        }
        assert (
            len(set(annotation_lengths.values())) == 1
        ), f"Annotations expects all attributes to have equal length, but got {annotation_lengths}"


@dataclass
class Trajectory:
    poses: npt.NDArray[np.float32]  # local coordinates
    trajectory_sampling: TrajectorySampling = TrajectorySampling(
        time_horizon=5, interval_length=0.5
    )

    def __post_init__(self):
        assert (
            self.poses.ndim == 2
        ), "Trajectory poses should have two dimensions for samples and poses."
        assert (
            self.poses.shape[0] == self.trajectory_sampling.num_poses
        ), "Trajectory poses and sampling have unequal number of poses."
        assert self.poses.shape[1] == 3, "Trajectory requires (x, y, heading) at last dim."


@dataclass
class SceneMetadata:
    log_name: str
    scene_token: str
    map_name: str
    initial_token: str

    num_history_frames: int
    num_future_frames: int


@dataclass
class Frame:

    token: str
    timestamp: int
    roadblock_ids: List[str]
    traffic_lights: List[Tuple[str, bool]]
    annotations: Annotations

    ego_status: EgoStatus
    lidar: Optional[Lidar] = None
    cameras: Optional[Cameras] = None


@dataclass
class Scene:

    # Ground truth information
    scene_metadata: SceneMetadata
    frames: List[Frame]

    def get_future_trajectory(self, num_trajectory_frames: Optional[int] = None) -> Trajectory:
        
        if num_trajectory_frames is None:
            num_trajectory_frames = self.scene_metadata.num_future_frames
        
        start_frame_idx = self.scene_metadata.num_history_frames - 1

        global_ego_poses = []
        for frame_idx in range(start_frame_idx, start_frame_idx + num_trajectory_frames + 1):
            global_ego_poses.append(self.frames[frame_idx].ego_status.ego_pose)

        origin = StateSE2(*global_ego_poses[0])
        local_ego_poses = convert_absolute_to_relative_se2_array(
            origin, np.array(global_ego_poses[1:], dtype=np.float32)
        )

        return Trajectory(
            local_ego_poses,
            TrajectorySampling(
                num_poses=len(local_ego_poses),
                interval_length=NAVSIM_INTERVAL_LENGTH,
            ),
        )

    def get_history_trajectory(self, num_trajectory_frames: Optional[int] = None) -> Trajectory:

        if num_trajectory_frames is None:
            num_trajectory_frames = self.scene_metadata.num_history_frames

        global_ego_poses = []
        for frame_idx in range(num_trajectory_frames):
            global_ego_poses.append(self.frames[frame_idx].ego_status.ego_pose)

        origin = StateSE2(*global_ego_poses[-1])
        local_ego_poses = convert_absolute_to_relative_se2_array(
            origin, np.array(global_ego_poses, dtype=np.float32)
        )

        return Trajectory(
            local_ego_poses,
            TrajectorySampling(
                num_poses=len(local_ego_poses),
                interval_length=NAVSIM_INTERVAL_LENGTH,
            ),
        )

    def get_agent_input(
        self,
        sensor_modalities: List[str] = ["lidar", "camera"],
    ) -> AgentInput:

        local_ego_poses = self.get_history_trajectory().poses

        include_cameras = "camera" in sensor_modalities
        include_lidar = "lidar" in sensor_modalities

        ego_statuses: List[EgoStatus] = []
        cameras: List[EgoStatus] = [] if include_cameras else None
        lidars: List[Lidar] = [] if include_lidar else None

        for frame_idx in range(self.scene_metadata.num_history_frames):
            frame_ego_status = self.frames[frame_idx].ego_status

            ego_statuses.append(
                EgoStatus(
                    ego_pose=local_ego_poses[frame_idx],
                    ego_velocity=frame_ego_status.ego_velocity,
                    ego_acceleration=frame_ego_status.ego_acceleration,
                    driving_command=frame_ego_status.driving_command,
                )
            )

            if include_cameras:
                cameras.append(self.frames[frame_idx].cameras)

            if include_lidar:
                # TODO: Add lidar data
                warnings.warn(f"Lidar currently not available in OpenScenes!")
                lidars = None

        return AgentInput(ego_statuses, cameras, lidars)

    @classmethod
    def from_scene_dict_list(
        cls,
        scene_dict_list: List[Dict],
        num_history_frames: int,
        num_future_frames: int,
        sensor_modalities: List[str] = ["lidar", "camera"],
    ) -> Scene:
        assert len(scene_dict_list) >= 0, "Scene list is empty!"
        
        # NOTE: potentially needs adaption in future
        scene_metadata = SceneMetadata(
            log_name=scene_dict_list[num_history_frames - 1]["log_name"],
            scene_token=scene_dict_list[num_history_frames - 1]["scene_token"],
            map_name=scene_dict_list[num_history_frames - 1]["map_location"],
            initial_token=scene_dict_list[num_history_frames - 1]["token"],
            num_history_frames=num_history_frames,
            num_future_frames=num_future_frames,
        )

        frames: List[Frame] = []
        for frame_idx in range(len(scene_dict_list)):

            ego_translation = scene_dict_list[frame_idx]["ego2global_translation"]
            ego_quaternion = Quaternion(*scene_dict_list[frame_idx]["ego2global_rotation"])
            global_ego_pose = np.array(
                [ego_translation[0], ego_translation[1], ego_quaternion.yaw_pitch_roll[0]],
                dtype=np.float64,
            )
            ego_dynamic_state = scene_dict_list[frame_idx]["ego_dynamic_state"]
            global_ego_status = EgoStatus(
                ego_pose=global_ego_pose,
                ego_velocity=np.array(ego_dynamic_state[:2], dtype=np.float32),
                ego_acceleration=np.array(ego_dynamic_state[2:], dtype=np.float32),
                driving_command=scene_dict_list[frame_idx]["driving_command"],
                in_global_frame=True,
            )

            annotations = Annotations(
                boxes=scene_dict_list[frame_idx]["anns"]["gt_boxes"],
                names=scene_dict_list[frame_idx]["anns"]["gt_names"],
                velocity_3d=scene_dict_list[frame_idx]["anns"]["gt_velocity_3d"],
                instance_tokens=scene_dict_list[frame_idx]["anns"]["instance_tokens"],
                track_tokens=scene_dict_list[frame_idx]["anns"]["track_tokens"],
            )

            if "camera" in sensor_modalities:
                cameras = Cameras.from_camera_dict(scene_dict_list[frame_idx]["cams"])
            else:
                cameras = None

            if "lidar" in sensor_modalities:
                # TODO: Add lidar data
                warnings.warn(f"Lidar currently not available in OpenScenes!")
                lidar = None
            else:
                lidar = None

            frame = Frame(
                token=scene_dict_list[frame_idx]["token"],
                timestamp=scene_dict_list[frame_idx]["timestamp"],
                roadblock_ids=scene_dict_list[frame_idx]["roadblock_ids"],
                traffic_lights=scene_dict_list[frame_idx]["traffic_lights"],
                annotations=annotations,
                ego_status=global_ego_status,
                lidar=lidar,
                cameras=cameras,
            )
            frames.append(frame)

        return Scene(scene_metadata=scene_metadata, frames=frames)


@dataclass
class SceneFilter:
    
    num_history_frames: int = 4
    num_future_frames: int = 10
    has_route: bool = True
    
    max_scenes: Optional[int] = None 
    log_names: Optional[List[str]] = None
    
    # NOTE: Not implemented
    tokens: Optional[List[str]] = None
    map_names: Optional[List[str]] = None
    
    @property
    def num_frames(self):
        return self.num_history_frames + self.num_future_frames
    

@dataclass
class PDMResults:

    no_at_fault_collisions: float
    drivable_area_compliance: float
    driving_direction_compliance: float

    ego_progress: float
    time_to_collision_within_bound: float
    comfort: float

    score: float
