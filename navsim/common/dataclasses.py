from __future__ import annotations

import io
import os
import pickle
import warnings
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import TrafficLightStatuses
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.database.maps_db.gpkg_mapsdb import MAP_LOCATIONS
from nuplan.database.utils.pointclouds.lidar import LidarPointCloud
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from PIL import Image
from pyquaternion import Quaternion

from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)

NAVSIM_INTERVAL_LENGTH: float = 0.5
OPENSCENE_DATA_ROOT = os.environ.get("OPENSCENE_DATA_ROOT")
NUPLAN_MAPS_ROOT = os.environ.get("NUPLAN_MAPS_ROOT")


@dataclass
class Camera:
    """Camera dataclass for image and parameters."""

    image: Optional[npt.NDArray[np.float32]] = None

    sensor2lidar_rotation: Optional[npt.NDArray[np.float32]] = None
    sensor2lidar_translation: Optional[npt.NDArray[np.float32]] = None
    intrinsics: Optional[npt.NDArray[np.float32]] = None
    distortion: Optional[npt.NDArray[np.float32]] = None

    camera_path: Optional[Path] = None


@dataclass
class Cameras:
    """Multi-camera dataclass."""

    cam_f0: Camera
    cam_l0: Camera
    cam_l1: Camera
    cam_l2: Camera
    cam_r0: Camera
    cam_r1: Camera
    cam_r2: Camera
    cam_b0: Camera

    @classmethod
    def from_camera_dict(
        cls,
        sensor_blobs_path: Path,
        camera_dict: Dict[str, Any],
        sensor_names: List[str],
    ) -> Cameras:
        """
        Load camera dataclass from dictionary.
        :param sensor_blobs_path: root directory of sensor data.
        :param camera_dict: dictionary containing camera specifications.
        :param sensor_names: list of camera identifiers to include.
        :return: Cameras dataclass.
        """

        data_dict: Dict[str, Camera] = {}
        for camera_name in camera_dict.keys():
            camera_identifier = camera_name.lower()
            if camera_identifier in sensor_names:
                image_path = sensor_blobs_path / camera_dict[camera_name]["data_path"]
                data_dict[camera_identifier] = Camera(
                    image=np.array(Image.open(image_path)),
                    sensor2lidar_rotation=camera_dict[camera_name]["sensor2lidar_rotation"],
                    sensor2lidar_translation=camera_dict[camera_name]["sensor2lidar_translation"],
                    intrinsics=camera_dict[camera_name]["cam_intrinsic"],
                    distortion=camera_dict[camera_name]["distortion"],
                    camera_path=camera_dict[camera_name]["data_path"],
                )
            else:
                data_dict[camera_identifier] = Camera()  # empty camera

        return Cameras(
            cam_f0=data_dict["cam_f0"],
            cam_l0=data_dict["cam_l0"],
            cam_l1=data_dict["cam_l1"],
            cam_l2=data_dict["cam_l2"],
            cam_r0=data_dict["cam_r0"],
            cam_r1=data_dict["cam_r1"],
            cam_r2=data_dict["cam_r2"],
            cam_b0=data_dict["cam_b0"],
        )


@dataclass
class Lidar:
    """Lidar point cloud dataclass."""

    # NOTE:
    # merged lidar point cloud as (6,n) float32 array with n points
    # first axis: (x, y, z, intensity, ring, lidar_id), see LidarIndex
    lidar_pc: Optional[npt.NDArray[np.float32]] = None
    lidar_path: Optional[Path] = None

    @staticmethod
    def _load_bytes(lidar_path: Path) -> BinaryIO:
        """Helper static method to load lidar point cloud stream."""
        with open(lidar_path, "rb") as fp:
            return io.BytesIO(fp.read())

    @classmethod
    def from_paths(cls, sensor_blobs_path: Path, lidar_path: Path, sensor_names: List[str]) -> Lidar:
        """
        Loads lidar point cloud dataclass in log loading.
        :param sensor_blobs_path: root directory to sensor data
        :param lidar_path: relative lidar path from logs.
        :param sensor_names: list of sensor identifiers to load`
        :return: lidar point cloud dataclass
        """

        # NOTE: this could be extended to load specific LiDARs in the merged pc
        if "lidar_pc" in sensor_names:
            global_lidar_path = sensor_blobs_path / lidar_path
            lidar_pc = LidarPointCloud.from_buffer(cls._load_bytes(global_lidar_path), "pcd").points
            return Lidar(lidar_pc, lidar_path)
        return Lidar()  # empty lidar


@dataclass
class EgoStatus:
    """Ego vehicle status dataclass."""

    ego_pose: npt.NDArray[np.float64]
    ego_velocity: npt.NDArray[np.float32]
    ego_acceleration: npt.NDArray[np.float32]
    driving_command: npt.NDArray[np.int]
    in_global_frame: bool = False  # False for AgentInput


@dataclass
class AgentInput:
    """Dataclass for agent inputs with current and past ego statuses and sensors."""

    ego_statuses: List[EgoStatus]
    cameras: List[Cameras]
    lidars: List[Lidar]

    @classmethod
    def from_scene_dict_list(
        cls,
        scene_dict_list: List[Dict],
        sensor_blobs_path: Path,
        num_history_frames: int,
        sensor_config: SensorConfig,
    ) -> AgentInput:
        """
        Load agent input from scene dictionary.
        :param scene_dict_list: list of scene frames (in logs).
        :param sensor_blobs_path: root directory of sensor data
        :param num_history_frames: number of agent input frames
        :param sensor_config: sensor config dataclass
        :return: agent input dataclass
        """
        assert len(scene_dict_list) > 0, "Scene list is empty!"

        global_ego_poses = []
        for frame_idx in range(num_history_frames):
            ego_translation = scene_dict_list[frame_idx]["ego2global_translation"]
            ego_quaternion = Quaternion(*scene_dict_list[frame_idx]["ego2global_rotation"])
            global_ego_pose = np.array(
                [
                    ego_translation[0],
                    ego_translation[1],
                    ego_quaternion.yaw_pitch_roll[0],
                ],
                dtype=np.float64,
            )
            global_ego_poses.append(global_ego_pose)

        local_ego_poses = convert_absolute_to_relative_se2_array(
            StateSE2(*global_ego_poses[-1]),
            np.array(global_ego_poses, dtype=np.float64),
        )

        ego_statuses: List[EgoStatus] = []
        cameras: List[EgoStatus] = []
        lidars: List[Lidar] = []

        for frame_idx in range(num_history_frames):

            ego_dynamic_state = scene_dict_list[frame_idx]["ego_dynamic_state"]
            ego_status = EgoStatus(
                ego_pose=np.array(local_ego_poses[frame_idx], dtype=np.float32),
                ego_velocity=np.array(ego_dynamic_state[:2], dtype=np.float32),
                ego_acceleration=np.array(ego_dynamic_state[2:], dtype=np.float32),
                driving_command=scene_dict_list[frame_idx]["driving_command"],
            )
            ego_statuses.append(ego_status)

            sensor_names = sensor_config.get_sensors_at_iteration(frame_idx)
            cameras.append(
                Cameras.from_camera_dict(
                    sensor_blobs_path=sensor_blobs_path,
                    camera_dict=scene_dict_list[frame_idx]["cams"],
                    sensor_names=sensor_names,
                )
            )

            lidars.append(
                Lidar.from_paths(
                    sensor_blobs_path=sensor_blobs_path,
                    lidar_path=Path(scene_dict_list[frame_idx]["lidar_path"]) if scene_dict_list[frame_idx]["lidar_path"] is not None else None,
                    sensor_names=sensor_names,
                )
            )

        return AgentInput(ego_statuses, cameras, lidars)

    @classmethod
    def from_scene_dict_list_private(
        cls,
        scene_dict_list: List[Dict],
        sensor_blobs_path: Path,
        num_history_frames: int,
        sensor_config: SensorConfig,
    ) -> AgentInput:
        """
        Load agent input from scene dictionary.
        :param scene_dict_list: list of scene frames (in logs).
        :param sensor_blobs_path: root directory of sensor data
        :param num_history_frames: number of agent input frames
        :param sensor_config: sensor config dataclass
        :return: agent input dataclass
        """
        assert len(scene_dict_list) > 0, "Scene list is empty!"

        ego_statuses: List[EgoStatus] = []
        cameras: List[EgoStatus] = []
        lidars: List[Lidar] = []

        for frame_idx in range(num_history_frames):
            ego_statuses.append(scene_dict_list[frame_idx].ego_status)
            cameras.append(
                    scene_dict_list[frame_idx].cameras
            )
            lidars.append(
                    scene_dict_list[frame_idx].lidar
            )

        return AgentInput(ego_statuses, cameras, lidars)

@dataclass
class Annotations:
    """Dataclass of annotations (e.g. bounding boxes) per frame."""

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
    """Trajectory dataclass in NAVSIM."""

    poses: npt.NDArray[np.float32]  # local coordinates
    trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5)

    def __post_init__(self):
        assert self.poses.ndim == 2, "Trajectory poses should have two dimensions for samples and poses."
        assert (
            self.poses.shape[0] == self.trajectory_sampling.num_poses
        ), "Trajectory poses and sampling have unequal number of poses."
        assert self.poses.shape[1] == 3, "Trajectory requires (x, y, heading) at last dim."


@dataclass
class SceneMetadata:
    """Dataclass of scene metadata (e.g. location) per scene."""

    log_name: str
    scene_token: str
    map_name: str
    initial_token: str

    num_history_frames: int
    num_future_frames: int

    #  maps between synthetic scenes and the corresponding original scene
    #  with the same timestamp in the same log.
    #  NOTE: this is not the corresponding first stage scene token
    #  for original scenes this is None
    corresponding_original_scene: str = None

    # maps to the initial frame token (at 0.0s) of the corresponding original scene
    # for original scenes this is None
    corresponding_original_initial_token: str = None


@dataclass
class Frame:
    """Frame dataclass with privileged information."""

    token: str
    timestamp: int
    roadblock_ids: List[str]
    traffic_lights: List[Tuple[str, bool]]
    annotations: Annotations

    ego_status: EgoStatus
    lidar: Lidar
    cameras: Cameras


@dataclass
class Scene:
    """Scene dataclass defining a single sample in NAVSIM."""

    # Ground truth information
    scene_metadata: SceneMetadata
    map_api: AbstractMap
    frames: List[Frame]
    extended_traffic_light_data: Optional[List[TrafficLightStatuses]] = None
    extended_detections_tracks: Optional[List[DetectionsTracks]] = None
    """
    scene_metadata (SceneMetadata): Metadata describing the scene, including its unique identifiers and attributes.
    map_api (AbstractMap): Map API interface providing access to map-related information such as lane geometry and topology.
    frames (List[Frame]): A sequence of frames describing the state of the ego-vehicle and its surroundings.
    extended_traffic_light_data (Optional[List[TrafficLightStatuses]], optional):
        A list containing traffic light status information for each future frame after the scene ends.
        Each `TrafficLightStatuses` entry includes a `TrafficLightStatusData` object for every lane connector
        controlled by a traffic light. Defaults to None.
    extended_detections_tracks (Optional[List[DetectionsTracks]], optional):
        A list containing detection tracks for each future frame after the scene ends.
        This can be used to provide future detections of pedestrians and objects in synthetic scenarios
        where future frames are unavailable. Defaults to None.
    """

    def get_future_trajectory(self, num_trajectory_frames: Optional[int] = None) -> Trajectory:
        """
        Extracts future trajectory of the human operator in local coordinates (ie. ego rear-axle).
        :param num_trajectory_frames: optional number frames to extract poses, defaults to None
        :return: trajectory dataclass
        """

        if num_trajectory_frames is None:
            num_trajectory_frames = self.scene_metadata.num_future_frames

        start_frame_idx = self.scene_metadata.num_history_frames - 1

        global_ego_poses = []
        for frame_idx in range(start_frame_idx, start_frame_idx + num_trajectory_frames + 1):
            global_ego_poses.append(self.frames[frame_idx].ego_status.ego_pose)

        local_ego_poses = convert_absolute_to_relative_se2_array(
            StateSE2(*global_ego_poses[0]),
            np.array(global_ego_poses[1:], dtype=np.float64),
        )

        return Trajectory(
            local_ego_poses,
            TrajectorySampling(
                num_poses=len(local_ego_poses),
                interval_length=NAVSIM_INTERVAL_LENGTH,
            ),
        )

    def get_history_trajectory(self, num_trajectory_frames: Optional[int] = None) -> Trajectory:
        """
        Extracts past trajectory of ego vehicles in local coordinates (ie. ego rear-axle).
        :param num_trajectory_frames: optional number frames to extract poses, defaults to None
        :return: trajectory dataclass
        """

        if num_trajectory_frames is None:
            num_trajectory_frames = self.scene_metadata.num_history_frames

        global_ego_poses = []
        for frame_idx in range(num_trajectory_frames):
            global_ego_poses.append(self.frames[frame_idx].ego_status.ego_pose)

        origin = StateSE2(*global_ego_poses[-1])
        local_ego_poses = convert_absolute_to_relative_se2_array(origin, np.array(global_ego_poses, dtype=np.float64))

        return Trajectory(
            local_ego_poses,
            TrajectorySampling(
                num_poses=len(local_ego_poses),
                interval_length=NAVSIM_INTERVAL_LENGTH,
            ),
        )

    def get_agent_input(self) -> AgentInput:
        """
        Extracts agents input dataclass (without privileged information) from scene.
        :return: agent input dataclass
        """

        local_ego_poses = self.get_history_trajectory().poses
        ego_statuses: List[EgoStatus] = []
        cameras: List[Cameras] = []
        lidars: List[Lidar] = []

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
            cameras.append(self.frames[frame_idx].cameras)
            lidars.append(self.frames[frame_idx].lidar)

        return AgentInput(ego_statuses, cameras, lidars)

    @classmethod
    def _build_map_api(cls, map_name: str) -> AbstractMap:
        """Helper classmethod to load map api from name."""
        assert map_name in MAP_LOCATIONS, f"The map name {map_name} is invalid, must be in {MAP_LOCATIONS}"
        return get_maps_api(NUPLAN_MAPS_ROOT, "nuplan-maps-v1.0", map_name)

    @classmethod
    def _build_annotations(cls, scene_frame: Dict) -> Annotations:
        """Helper classmethod to load annotation dataclass from logs."""
        return Annotations(
            boxes=scene_frame["anns"]["gt_boxes"],
            names=scene_frame["anns"]["gt_names"],
            velocity_3d=scene_frame["anns"]["gt_velocity_3d"],
            instance_tokens=scene_frame["anns"]["instance_tokens"],
            track_tokens=scene_frame["anns"]["track_tokens"],
        )

    @classmethod
    def _build_ego_status(cls, scene_frame: Dict) -> EgoStatus:
        """Helper classmethod to load ego status dataclass from logs."""
        ego_translation = scene_frame["ego2global_translation"]
        ego_quaternion = Quaternion(*scene_frame["ego2global_rotation"])
        global_ego_pose = np.array(
            [ego_translation[0], ego_translation[1], ego_quaternion.yaw_pitch_roll[0]],
            dtype=np.float64,
        )
        ego_dynamic_state = scene_frame["ego_dynamic_state"]
        return EgoStatus(
            ego_pose=global_ego_pose,
            ego_velocity=np.array(ego_dynamic_state[:2], dtype=np.float32),
            ego_acceleration=np.array(ego_dynamic_state[2:], dtype=np.float32),
            driving_command=scene_frame["driving_command"],
            in_global_frame=True,
        )

    @classmethod
    def from_scene_dict_list(
        cls,
        scene_dict_list: List[Dict],
        sensor_blobs_path: Path,
        num_history_frames: int,
        num_future_frames: int,
        sensor_config: SensorConfig,
    ) -> Scene:
        """
        Load scene dataclass from scene dictionary list (for log loading).
        :param scene_dict_list: list of scene frames (in logs)
        :param sensor_blobs_path: root directory of sensor data
        :param num_history_frames: number of past and current frames to load
        :param num_future_frames: number of future frames to load
        :param sensor_config: sensor config dataclass
        :return: scene dataclass
        """
        assert len(scene_dict_list) >= 0, "Scene list is empty!"
        scene_metadata = SceneMetadata(
            log_name=scene_dict_list[num_history_frames - 1]["log_name"],
            scene_token=scene_dict_list[num_history_frames - 1]["scene_token"],
            map_name=scene_dict_list[num_history_frames - 1]["map_location"],
            initial_token=scene_dict_list[num_history_frames - 1]["token"],
            num_history_frames=num_history_frames,
            num_future_frames=num_future_frames,
        )
        map_api = cls._build_map_api(scene_metadata.map_name)

        frames: List[Frame] = []
        for frame_idx in range(len(scene_dict_list)):
            global_ego_status = cls._build_ego_status(scene_dict_list[frame_idx])
            annotations = cls._build_annotations(scene_dict_list[frame_idx])

            sensor_names = sensor_config.get_sensors_at_iteration(frame_idx)

            cameras = Cameras.from_camera_dict(
                sensor_blobs_path=sensor_blobs_path,
                camera_dict=scene_dict_list[frame_idx]["cams"],
                sensor_names=sensor_names,
            )

            lidar = Lidar.from_paths(
                sensor_blobs_path=sensor_blobs_path,
                lidar_path=Path(scene_dict_list[frame_idx]["lidar_path"]),
                sensor_names=sensor_names,
            )

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

        return Scene(scene_metadata=scene_metadata, map_api=map_api, frames=frames)
    
    @classmethod
    def from_scene_dict_list_private(
        cls,
        scene_dict_list: List[Dict],
        sensor_blobs_path: Path,
        num_history_frames: int,
        num_future_frames: int,
        sensor_config: SensorConfig,
    ) -> Scene:
        """
        Load scene dataclass from scene dictionary list (for log loading).
        :param scene_dict_list: list of scene frames (in logs)
        :param sensor_blobs_path: root directory of sensor data
        :param num_history_frames: number of past and current frames to load
        :param num_future_frames: number of future frames to load
        :param sensor_config: sensor config dataclass
        :return: scene dataclass
        """
        assert len(scene_dict_list) >= 0, "Scene list is empty!"
        scene_metadata = SceneMetadata(
            log_name=scene_dict_list[num_history_frames - 1]["log_name"],
            scene_token=scene_dict_list[num_history_frames - 1]["scene_token"],
            map_name=scene_dict_list[num_history_frames - 1]["map_location"],
            initial_token=scene_dict_list[num_history_frames - 1]["token"],
            num_history_frames=num_history_frames,
            num_future_frames=num_future_frames,
        )

        global_ego_poses = []
        for frame_idx in range(num_history_frames):
            ego_translation = scene_dict_list[frame_idx]["ego2global_translation"]
            ego_quaternion = Quaternion(*scene_dict_list[frame_idx]["ego2global_rotation"])
            global_ego_pose = np.array(
                [
                    ego_translation[0],
                    ego_translation[1],
                    ego_quaternion.yaw_pitch_roll[0],
                ],
                dtype=np.float64,
            )
            global_ego_poses.append(global_ego_pose)

        local_ego_poses = convert_absolute_to_relative_se2_array(
            StateSE2(*global_ego_poses[-1]),
            np.array(global_ego_poses, dtype=np.float64),
        )

        frames: List[Frame] = []
        for frame_idx in range(len(scene_dict_list)):
            ego_dynamic_state = scene_dict_list[frame_idx]["ego_dynamic_state"]
            ego_status = EgoStatus(
                ego_pose=np.array(local_ego_poses[frame_idx], dtype=np.float32),
                ego_velocity=np.array(ego_dynamic_state[:2], dtype=np.float32),
                ego_acceleration=np.array(ego_dynamic_state[2:], dtype=np.float32),
                driving_command=scene_dict_list[frame_idx]["driving_command"],
            )

            sensor_names = sensor_config.get_sensors_at_iteration(frame_idx)
            cameras = Cameras.from_camera_dict(
                sensor_blobs_path=sensor_blobs_path,
                camera_dict=scene_dict_list[frame_idx]["cams"],
                sensor_names=sensor_names,
            )

            frame = Frame(
                token=scene_dict_list[frame_idx]["token"],
                timestamp=scene_dict_list[frame_idx]["timestamp"],
                roadblock_ids=scene_dict_list[frame_idx]["roadblock_ids"],
                traffic_lights=scene_dict_list[frame_idx]["traffic_lights"],
                annotations=None,
                ego_status=ego_status,
                lidar=None,
                cameras=cameras,
            )
            frames.append(frame)
            
        return Scene(scene_metadata=scene_metadata, map_api=None, frames=frames)

    def save_to_disk(self, data_path: Path):
        """
        Save scene dataclass to disk.
        Note: this will NOT save the images or point clouds.
        :param data_path: root directory to save scene data
        :param sensor_blobs_path: root directory to sensor data
        """

        assert self.scene_metadata.scene_token is not None, "Scene token cannot be 'None', when saving to disk."
        assert data_path.is_dir(), f"Data path {data_path} is not a directory."

        # collect all the relevant data for the frames
        frames_data = []
        for frame in self.frames:
            camera_dict = {}
            for camera_field in fields(frame.cameras):
                camera_name = camera_field.name
                camera: Camera = getattr(frame.cameras, camera_name)
                if camera.image is not None:
                    camera_dict[camera_name] = {
                        "data_path": camera.camera_path,
                        "sensor2lidar_rotation": camera.sensor2lidar_rotation,
                        "sensor2lidar_translation": camera.sensor2lidar_translation,
                        "cam_intrinsic": camera.intrinsics,
                        "distortion": camera.distortion,
                    }
                else:
                    camera_dict[camera_name] = {}

            if frame.lidar.lidar_pc is not None:
                lidar_path = frame.lidar.lidar_path
            else:
                lidar_path = None

            frames_data.append(
                {
                    "token": frame.token,
                    "timestamp": frame.timestamp,
                    "roadblock_ids": frame.roadblock_ids,
                    "traffic_lights": frame.traffic_lights,
                    "annotations": asdict(frame.annotations),
                    "ego_status": asdict(frame.ego_status),
                    "lidar_path": lidar_path,
                    "camera_dict": camera_dict,
                }
            )

        # collect all the relevant data for the scene
        scene_dict = {
            "scene_metadata": asdict(self.scene_metadata),
            "frames": frames_data,
            "extended_traffic_light_data": self.extended_traffic_light_data,
            "extended_detections_tracks": self.extended_detections_tracks,
        }

        # save the scene_dict to disk
        save_path = data_path / f"{self.scene_metadata.scene_token}.pkl"

        with open(save_path, "wb") as f:
            pickle.dump(scene_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_from_disk(
        cls,
        file_path: Path,
        sensor_blobs_path: Path,
        sensor_config: SensorConfig = None,
    ) -> Scene:
        """
        Load scene dataclass from disk. Only used for synthesized views.
        Regular scenes are loaded from logs.
        :return: scene dataclass
        """
        if sensor_config is None:
            sensor_config = SensorConfig.build_no_sensors()
        # Load the metadata
        with open(file_path, "rb") as f:
            scene_data = pickle.load(f)

        scene_metadata = SceneMetadata(**scene_data["scene_metadata"])
        # build the map from the map_path
        map_api = cls._build_map_api(scene_metadata.map_name)

        scene_frames: List[Frame] = []
        for frame_idx, frame_data in enumerate(scene_data["frames"]):
            sensor_names = sensor_config.get_sensors_at_iteration(frame_idx)
            lidar_path = Path(frame_data["lidar_path"]) if frame_data["lidar_path"] else None
            lidar = Lidar.from_paths(
                sensor_blobs_path=sensor_blobs_path,
                lidar_path=lidar_path,
                sensor_names=sensor_names,
            )

            cameras = Cameras.from_camera_dict(
                sensor_blobs_path=sensor_blobs_path,
                camera_dict=frame_data["camera_dict"],
                sensor_names=sensor_names,
            )

            scene_frames.append(
                Frame(
                    token=frame_data["token"],
                    timestamp=frame_data["timestamp"],
                    roadblock_ids=frame_data["roadblock_ids"],
                    traffic_lights=frame_data["traffic_lights"],
                    annotations=Annotations(**frame_data["annotations"]),
                    ego_status=EgoStatus(**frame_data["ego_status"]),
                    lidar=lidar,
                    cameras=cameras,
                )
            )

        return Scene(
            scene_metadata=scene_metadata,
            map_api=map_api,
            frames=scene_frames,
            extended_traffic_light_data=scene_data["extended_traffic_light_data"],
            extended_detections_tracks=scene_data["extended_detections_tracks"],
        )


@dataclass
class SceneFilter:
    """Scene filtering configuration for scene loading."""

    num_history_frames: int = 4
    num_future_frames: int = 10
    frame_interval: Optional[int] = None
    has_route: bool = True

    max_scenes: Optional[int] = None
    log_names: Optional[List[str]] = None
    tokens: Optional[List[str]] = None
    include_synthetic_scenes: bool = False
    all_mapping: Optional[Dict[Tuple[str, str], List[Tuple[str, str]]]] = None
    synthetic_scene_tokens: Optional[List[str]] = None

    # for reactive and non_reactive
    reactive_synthetic_initial_tokens: Optional[List[str]] = None
    non_reactive_synthetic_initial_tokens: Optional[List[str]] = None

    # TODO: expand filter options

    def __post_init__(self):

        if self.frame_interval is None:
            self.frame_interval = self.num_frames

        assert self.num_history_frames >= 1, "SceneFilter: num_history_frames must greater equal one."
        assert self.num_future_frames >= 0, "SceneFilter: num_future_frames must greater equal zero."
        assert self.frame_interval >= 1, "SceneFilter: frame_interval must greater equal one."

        if (
            not self.include_synthetic_scenes
            and self.synthetic_scene_tokens is not None
            and len(self.synthetic_scene_tokens) > 0
        ):
            warnings.warn(
                "SceneFilter: synthetic_scene_tokens are provided but include_synthetic_scenes is False. No synthetic scenes will be loaded."
            )

    @property
    def num_frames(self) -> int:
        """
        :return: total number for frames for scenes to extract.
        """
        return self.num_history_frames + self.num_future_frames


@dataclass
class SensorConfig:
    """Configuration dataclass of agent sensors for memory management."""

    # Config values of sensors are either
    # - bool: Whether to load history or not
    # - List[int]: For loading specific history steps
    cam_f0: Union[bool, List[int]]
    cam_l0: Union[bool, List[int]]
    cam_l1: Union[bool, List[int]]
    cam_l2: Union[bool, List[int]]
    cam_r0: Union[bool, List[int]]
    cam_r1: Union[bool, List[int]]
    cam_r2: Union[bool, List[int]]
    cam_b0: Union[bool, List[int]]
    lidar_pc: Union[bool, List[int]]

    def get_sensors_at_iteration(self, iteration: int) -> List[str]:
        """
        Creates a list of sensor identifiers given iteration.
        :param iteration: integer indicating the history iteration.
        :return: list of sensor identifiers to load.
        """
        sensors_at_iteration: List[str] = []
        for sensor_name, sensor_include in asdict(self).items():
            if isinstance(sensor_include, bool) and sensor_include:
                sensors_at_iteration.append(sensor_name)
            elif isinstance(sensor_include, list) and iteration in sensor_include:
                sensors_at_iteration.append(sensor_name)
        return sensors_at_iteration

    @classmethod
    def build_all_sensors(cls, include: Union[bool, List[int]] = True) -> SensorConfig:
        """
        Classmethod to load all sensors with the same specification.
        :param include: boolean or integers for sensors to include, defaults to True
        :return: sensor configuration dataclass
        """
        return SensorConfig(
            cam_f0=include,
            cam_l0=include,
            cam_l1=include,
            cam_l2=include,
            cam_r0=include,
            cam_r1=include,
            cam_r2=include,
            cam_b0=include,
            lidar_pc=include,
        )

    @classmethod
    def build_no_sensors(cls) -> SensorConfig:
        """
        Classmethod to load no sensors.
        :return: sensor configuration dataclass
        """
        return cls.build_all_sensors(include=False)


@dataclass
class PDMResults:
    """Helper dataclass to record PDM results."""

    no_at_fault_collisions: float
    drivable_area_compliance: float
    driving_direction_compliance: float
    traffic_light_compliance: float

    ego_progress: float
    time_to_collision_within_bound: float
    lane_keeping: float
    history_comfort: float

    multiplicative_metrics_prod: float
    weighted_metrics: npt.NDArray[np.float64]
    weighted_metrics_array: npt.NDArray[np.float64]

    pdm_score: float

    @classmethod
    def get_empty_results(cls) -> PDMResults:
        """
        Returns an instance of the class where all values are NaN.
        :return: empty PDM results dataclass.
        """
        return PDMResults(
            no_at_fault_collisions=np.nan,
            drivable_area_compliance=np.nan,
            driving_direction_compliance=np.nan,
            traffic_light_compliance=np.nan,
            ego_progress=np.nan,
            time_to_collision_within_bound=np.nan,
            lane_keeping=np.nan,
            history_comfort=np.nan,
            multiplicative_metrics_prod=np.nan,
            weighted_metrics=np.nan,
            weighted_metrics_array=np.nan,
            pdm_score=np.nan,
        )
