from typing import Any, List

import numpy as np
import numpy.typing as npt
import matplotlib
import matplotlib.pyplot as plt

from navsim.visualization.config import LIDAR_CONFIG
from navsim.common.enums import LidarIndex


def filter_lidar_pc(lidar_pc: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """
    Filter lidar point cloud according to global configuration
    :param lidar_pc: numpy array of shape (6,n)
    :return: filtered point cloud
    """

    pc = lidar_pc.T
    mask = (
        np.ones((len(pc)), dtype=bool)
        & (pc[:, LidarIndex.X] > LIDAR_CONFIG["x_lim"][0])
        & (pc[:, LidarIndex.X] < LIDAR_CONFIG["x_lim"][1])
        & (pc[:, LidarIndex.Y] > LIDAR_CONFIG["y_lim"][0])
        & (pc[:, LidarIndex.Y] < LIDAR_CONFIG["y_lim"][1])
        & (pc[:, LidarIndex.Z] > LIDAR_CONFIG["z_lim"][0])
        & (pc[:, LidarIndex.Z] < LIDAR_CONFIG["z_lim"][1])
    )
    pc = pc[mask]
    return pc.T


def get_lidar_pc_color(lidar_pc: npt.NDArray[np.float32], as_hex: bool = False) -> List[Any]:
    """
    Compute color map of lidar point cloud according to global configuration
    :param lidar_pc: numpy array of shape (6,n)
    :param as_hex: whether to return hex values, defaults to False
    :return: list of RGB or hex values
    """

    pc = lidar_pc.T
    if LIDAR_CONFIG["color_element"] == "none":
        colors_rgb = np.zeros((len(pc), 3), dtype=np.uin8)

    else:
        if LIDAR_CONFIG["color_element"] == "distance":
            color_intensities = np.linalg.norm(pc[:, LidarIndex.POSITION], axis=-1)
        else:
            color_element_map = {
                "x": LidarIndex.X,
                "y": LidarIndex.Y,
                "z": LidarIndex.Z,
                "intensity": LidarIndex.INTENSITY,
                "ring": LidarIndex.RING,
                "id": LidarIndex.ID,
            }
            color_intensities = pc[:, color_element_map[LIDAR_CONFIG["color_element"]]]

        min, max = color_intensities.min(), color_intensities.max()
        norm_intensities = [(value - min) / (max - min) for value in color_intensities]
        colormap = plt.get_cmap("viridis")
        colors_rgb = np.array([colormap(value) for value in norm_intensities])
        colors_rgb = (colors_rgb[:, :3] * 255).astype(np.uint8)

    assert len(colors_rgb) == len(pc)
    if as_hex:
        return [matplotlib.colors.to_hex(tuple(c / 255.0 for c in rgb)) for rgb in colors_rgb]

    return [tuple(value) for value in colors_rgb]
