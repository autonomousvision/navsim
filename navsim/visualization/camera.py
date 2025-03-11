from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from PIL import ImageColor
from pyquaternion import Quaternion

from navsim.common.dataclasses import Annotations, Camera, Lidar
from navsim.common.enums import BoundingBoxIndex, LidarIndex
from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types
from navsim.visualization.config import AGENT_CONFIG
from navsim.visualization.lidar import filter_lidar_pc, get_lidar_pc_color


def add_camera_ax(ax: plt.Axes, camera: Camera) -> plt.Axes:
    """
    Adds camera image to matplotlib ax object
    :param ax: matplotlib ax object
    :param camera: navsim camera dataclass
    :return: ax object with image
    """
    ax.imshow(camera.image)
    return ax


def add_lidar_to_camera_ax(ax: plt.Axes, camera: Camera, lidar: Lidar) -> plt.Axes:
    """
    Adds camera image with lidar point cloud on matplotlib ax object
    :param ax: matplotlib ax object
    :param camera: navsim camera dataclass
    :param lidar: navsim lidar dataclass
    :return: ax object with image
    """

    image, lidar_pc = camera.image.copy(), lidar.lidar_pc.copy()
    image_height, image_width = image.shape[:2]

    lidar_pc = filter_lidar_pc(lidar_pc)
    lidar_pc_colors = np.array(get_lidar_pc_color(lidar_pc))

    pc_in_cam, pc_in_fov_mask = _transform_pcs_to_images(
        lidar_pc,
        camera.sensor2lidar_rotation,
        camera.sensor2lidar_translation,
        camera.intrinsics,
        img_shape=(image_height, image_width),
    )

    for (x, y), color in zip(pc_in_cam[pc_in_fov_mask], lidar_pc_colors[pc_in_fov_mask]):
        color = (int(color[0]), int(color[1]), int(color[2]))
        cv2.circle(image, (int(x), int(y)), 5, color, -1)

    ax.imshow(image)
    return ax


def add_annotations_to_camera_ax(ax: plt.Axes, camera: Camera, annotations: Annotations) -> plt.Axes:
    """
    Adds camera image with bounding boxes on matplotlib ax object
    :param ax: matplotlib ax object
    :param camera: navsim camera dataclass
    :param annotations: navsim annotations dataclass
    :return: ax object with image
    """

    box_labels = annotations.names
    boxes = _transform_annotations_to_camera(
        annotations.boxes,
        camera.sensor2lidar_rotation,
        camera.sensor2lidar_translation,
    )
    box_positions, box_dimensions, box_heading = (
        boxes[:, BoundingBoxIndex.POSITION],
        boxes[:, BoundingBoxIndex.DIMENSION],
        boxes[:, BoundingBoxIndex.HEADING],
    )
    corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array([0.5, 0.5, 0.5])
    corners = box_dimensions.reshape([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])
    corners = _rotation_3d_in_axis(corners, box_heading, axis=1)
    corners += box_positions.reshape(-1, 1, 3)

    # Then draw project corners to image.
    box_corners, corners_pc_in_fov = _transform_points_to_image(corners.reshape(-1, 3), camera.intrinsics)
    box_corners = box_corners.reshape(-1, 8, 2)
    corners_pc_in_fov = corners_pc_in_fov.reshape(-1, 8)
    valid_corners = corners_pc_in_fov.any(-1)

    box_corners, box_labels = box_corners[valid_corners], box_labels[valid_corners]
    image = _plot_rect_3d_on_img(camera.image.copy(), box_corners, box_labels)

    ax.imshow(image)
    return ax


def _transform_annotations_to_camera(
    boxes: npt.NDArray[np.float32],
    sensor2lidar_rotation: npt.NDArray[np.float32],
    sensor2lidar_translation: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """
    Helper function to transform bounding boxes into camera frame
    TODO: Refactor
    :param boxes: array representation of bounding boxes
    :param sensor2lidar_rotation: camera rotation
    :param sensor2lidar_translation: camera translation
    :return: bounding boxes in camera coordinates
    """

    locs, rots = (
        boxes[:, BoundingBoxIndex.POSITION],
        boxes[:, BoundingBoxIndex.HEADING :],
    )
    dims_cam = boxes[
        :, [BoundingBoxIndex.LENGTH, BoundingBoxIndex.HEIGHT, BoundingBoxIndex.WIDTH]
    ]  # l, w, h -> l, h, w

    rots_cam = np.zeros_like(rots)
    for idx, rot in enumerate(rots):
        rot = Quaternion(axis=[0, 0, 1], radians=rot)
        rot = Quaternion(matrix=sensor2lidar_rotation).inverse * rot
        rots_cam[idx] = -rot.yaw_pitch_roll[0]

    lidar2cam_r = np.linalg.inv(sensor2lidar_rotation)
    lidar2cam_t = sensor2lidar_translation @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t

    locs_cam = np.concatenate([locs, np.ones_like(locs)[:, :1]], -1)  # -1, 4
    locs_cam = lidar2cam_rt.T @ locs_cam.T
    locs_cam = locs_cam.T
    locs_cam = locs_cam[:, :-1]
    return np.concatenate([locs_cam, dims_cam, rots_cam], -1)


def _rotation_3d_in_axis(points: npt.NDArray[np.float32], angles: npt.NDArray[np.float32], axis: int = 0):
    """
    Rotate 3D points by angles according to axis.
    TODO: Refactor
    :param points: array of points
    :param angles: array of angles
    :param axis: axis to perform rotation, defaults to 0
    :raises value: _description_
    :raises ValueError: if axis invalid
    :return: rotated points
    """
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = np.stack(
            [
                np.stack([rot_cos, zeros, -rot_sin]),
                np.stack([zeros, ones, zeros]),
                np.stack([rot_sin, zeros, rot_cos]),
            ]
        )
    elif axis == 2 or axis == -1:
        rot_mat_T = np.stack(
            [
                np.stack([rot_cos, -rot_sin, zeros]),
                np.stack([rot_sin, rot_cos, zeros]),
                np.stack([zeros, zeros, ones]),
            ]
        )
    elif axis == 0:
        rot_mat_T = np.stack(
            [
                np.stack([zeros, rot_cos, -rot_sin]),
                np.stack([zeros, rot_sin, rot_cos]),
                np.stack([ones, zeros, zeros]),
            ]
        )
    else:
        raise ValueError(f"axis should in range [0, 1, 2], got {axis}")
    return np.einsum("aij,jka->aik", points, rot_mat_T)


def _plot_rect_3d_on_img(
    image: npt.NDArray[np.float32],
    box_corners: npt.NDArray[np.float32],
    box_labels: List[str],
    thickness: int = 3,
) -> npt.NDArray[np.uint8]:
    """
    Plot the boundary lines of 3D rectangular on 2D images.
    TODO: refactor
    :param image:  The numpy array of image.
    :param box_corners: Coordinates of the corners of 3D, shape of [N, 8, 2].
    :param box_labels: labels of boxes for coloring
    :param thickness: pixel width of liens, defaults to 3
    :return: image with 3D bounding boxes
    """
    line_indices = (
        (0, 1),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 5),
        (3, 2),
        (3, 7),
        (4, 5),
        (4, 7),
        (2, 6),
        (5, 6),
        (6, 7),
    )
    for i in range(len(box_corners)):
        layer = tracked_object_types[box_labels[i]]
        color = ImageColor.getcolor(AGENT_CONFIG[layer]["fill_color"], "RGB")
        corners = box_corners[i].astype(np.int)
        for start, end in line_indices:
            cv2.line(
                image,
                (corners[start, 0], corners[start, 1]),
                (corners[end, 0], corners[end, 1]),
                color,
                thickness,
                cv2.LINE_AA,
            )
    return image.astype(np.uint8)


def _transform_points_to_image(
    points: npt.NDArray[np.float32],
    intrinsic: npt.NDArray[np.float32],
    image_shape: Optional[Tuple[int, int]] = None,
    eps: float = 1e-3,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """
    Transforms points in camera frame to image pixel coordinates
    TODO: refactor
    :param points: points in camera frame
    :param intrinsic: camera intrinsics
    :param image_shape: shape of image in pixel
    :param eps: lower threshold of points, defaults to 1e-3
    :return: points in pixel coordinates, mask of values in frame
    """
    points = points[:, :3]

    viewpad = np.eye(4)
    viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic

    pc_img = np.concatenate([points, np.ones_like(points)[:, :1]], -1)
    pc_img = viewpad @ pc_img.T
    pc_img = pc_img.T

    cur_pc_in_fov = pc_img[:, 2] > eps
    pc_img = pc_img[..., 0:2] / np.maximum(pc_img[..., 2:3], np.ones_like(pc_img[..., 2:3]) * eps)
    if image_shape is not None:
        img_h, img_w = image_shape
        cur_pc_in_fov = (
            cur_pc_in_fov
            & (pc_img[:, 0] < (img_w - 1))
            & (pc_img[:, 0] > 0)
            & (pc_img[:, 1] < (img_h - 1))
            & (pc_img[:, 1] > 0)
        )
    return pc_img, cur_pc_in_fov


def _transform_pcs_to_images(
    lidar_pc: npt.NDArray[np.float32],
    sensor2lidar_rotation: npt.NDArray[np.float32],
    sensor2lidar_translation: npt.NDArray[np.float32],
    intrinsic: npt.NDArray[np.float32],
    img_shape: Optional[Tuple[int, int]] = None,
    eps: float = 1e-3,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """
    Transforms points in camera frame to image pixel coordinates
    TODO: refactor
    :param lidar_pc: lidar point cloud
    :param sensor2lidar_rotation: camera rotation
    :param sensor2lidar_translation: camera translation
    :param intrinsic: camera intrinsics
    :param img_shape: image shape in pixels, defaults to None
    :param eps: threshold for lidar pc height, defaults to 1e-3
    :return: lidar pc in pixel coordinates, mask of values in frame
    """
    pc_xyz = lidar_pc[LidarIndex.POSITION, :].T

    lidar2cam_r = np.linalg.inv(sensor2lidar_rotation)
    lidar2cam_t = sensor2lidar_translation @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t

    viewpad = np.eye(4)
    viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
    lidar2img_rt = viewpad @ lidar2cam_rt.T

    cur_pc_xyz = np.concatenate([pc_xyz, np.ones_like(pc_xyz)[:, :1]], -1)
    cur_pc_cam = lidar2img_rt @ cur_pc_xyz.T
    cur_pc_cam = cur_pc_cam.T
    cur_pc_in_fov = cur_pc_cam[:, 2] > eps
    cur_pc_cam = cur_pc_cam[..., 0:2] / np.maximum(cur_pc_cam[..., 2:3], np.ones_like(cur_pc_cam[..., 2:3]) * eps)

    if img_shape is not None:
        img_h, img_w = img_shape
        cur_pc_in_fov = (
            cur_pc_in_fov
            & (cur_pc_cam[:, 0] < (img_w - 1))
            & (cur_pc_cam[:, 0] > 0)
            & (cur_pc_cam[:, 1] < (img_h - 1))
            & (cur_pc_cam[:, 1] > 0)
        )
    return cur_pc_cam, cur_pc_in_fov
