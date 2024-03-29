from typing import Any, Dict, List
import matplotlib.pyplot as plt

import numpy as np
from shapely import affinity
from shapely.geometry import Polygon, LineString

from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.geometry.transform import translate_longitudinally

from navsim.common.dataclasses import Frame, Annotations, Trajectory, Lidar
from navsim.common.enums import BoundingBoxIndex, LidarIndex

from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types
from navsim.visualization.lidar import filter_lidar_pc, get_lidar_pc_color
from navsim.visualization.config import (
    BEV_PLOT_CONFIG,
    MAP_LAYER_CONFIG,
    AGENT_CONFIG,
    LIDAR_CONFIG,
)


def add_configured_bev_on_ax(ax: plt.Axes, map_api: AbstractMap, frame: Frame) -> plt.Axes:
    """
    Adds birds-eye-view visualization optionally with map, annotations, or lidar
    :param ax: matplotlib ax object
    :param map_api: nuPlans map interface
    :param frame: navsim frame dataclass
    :return: ax with plot
    """    

    if "map" in BEV_PLOT_CONFIG["layers"]:
        add_map_to_bev_ax(ax, map_api, StateSE2(*frame.ego_status.ego_pose))

    if "annotations" in BEV_PLOT_CONFIG["layers"]:
        add_annotations_to_bev_ax(ax, frame.annotations)

    if "lidar" in BEV_PLOT_CONFIG["layers"]:
        add_lidar_to_bev_ax(ax, frame.lidar)

    return ax


def add_annotations_to_bev_ax(
    ax: plt.Axes, annotations: Annotations, add_ego: bool = True
) -> plt.Axes:
    """
    Adds birds-eye-view visualization of annotations (ie. bounding boxes)
    :param ax: matplotlib ax object
    :param annotations: navsim annotations dataclass
    :param add_ego: boolean weather to add ego bounding box, defaults to True
    :return: ax with plot
    """    

    for name_value, box_value in zip(annotations.names, annotations.boxes):
        agent_type = tracked_object_types[name_value]

        x, y, heading = (
            box_value[BoundingBoxIndex.X],
            box_value[BoundingBoxIndex.Y],
            box_value[BoundingBoxIndex.HEADING],
        )
        box_length, box_width, box_height = box_value[3], box_value[4], box_value[5]
        agent_box = OrientedBox(StateSE2(x, y, heading), box_length, box_width, box_height)

        add_oriented_box_to_bev_ax(ax, agent_box, AGENT_CONFIG[agent_type])

    if add_ego:
        car_footprint = CarFootprint.build_from_rear_axle(
            rear_axle_pose=StateSE2(0, 0, 0),
            vehicle_parameters=get_pacifica_parameters(),
        )
        add_oriented_box_to_bev_ax(
            ax, car_footprint.oriented_box, AGENT_CONFIG[TrackedObjectType.EGO], add_heading=False
        )
    return ax


def add_map_to_bev_ax(ax: plt.Axes, map_api: AbstractMap, origin: StateSE2) -> plt.Axes:
    """
    Adds birds-eye-view visualization of map (ie. polygons / lines)
    TODO: add more layers for visualizations (or flags in config)
    :param ax: matplotlib ax object
    :param map_api: nuPlans map interface
    :param origin: (x,y,θ) dataclass of global ego frame
    :return: ax with plot
    """    

    # layers for plotting complete layers
    polygon_layers: List[SemanticMapLayer] = [
        SemanticMapLayer.LANE,
        SemanticMapLayer.WALKWAYS,
        SemanticMapLayer.CARPARK_AREA,
        SemanticMapLayer.INTERSECTION,
        SemanticMapLayer.STOP_LINE,
        SemanticMapLayer.CROSSWALK,
    ]

    # layers for plotting complete layers
    polyline_layers: List[SemanticMapLayer] = [
        SemanticMapLayer.LANE,
        SemanticMapLayer.LANE_CONNECTOR,
    ]

    # query map api with interesting layers
    map_object_dict = map_api.get_proximal_map_objects(
        point=origin.point,
        radius=max(BEV_PLOT_CONFIG["figure_margin"]),
        layers=list(set(polygon_layers + polyline_layers)),
    )

    def _geometry_local_coords(geometry: Any, origin: StateSE2) -> Any:
        """ Helper for transforming shapely geometry in coord-frame """
        a = np.cos(origin.heading)
        b = np.sin(origin.heading)
        d = -np.sin(origin.heading)
        e = np.cos(origin.heading)
        xoff = -origin.x
        yoff = -origin.y
        translated_geometry = affinity.affine_transform(geometry, [1, 0, 0, 1, xoff, yoff])
        rotated_geometry = affinity.affine_transform(translated_geometry, [a, b, d, e, 0, 0])
        return rotated_geometry

    for polygon_layer in polygon_layers:
        for map_object in map_object_dict[polygon_layer]:
            polygon: Polygon = _geometry_local_coords(map_object.polygon, origin)
            add_polygon_to_bev_ax(ax, polygon, MAP_LAYER_CONFIG[polygon_layer])

    for polyline_layer in polyline_layers:
        for map_object in map_object_dict[polyline_layer]:
            linestring: LineString = _geometry_local_coords(
                map_object.baseline_path.linestring, origin
            )
            add_linestring_to_bev_ax(
                ax, linestring, MAP_LAYER_CONFIG[SemanticMapLayer.BASELINE_PATHS]
            )
    return ax


def add_lidar_to_bev_ax(ax: plt.Axes, lidar: Lidar) -> plt.Axes:
    """
    Add lidar point cloud in birds-eye-view
    :param ax: matplotlib ax object
    :param lidar: navsim lidar dataclass
    :return: ax with plot
    """    

    lidar_pc = filter_lidar_pc(lidar.lidar_pc)
    lidar_pc_colors = get_lidar_pc_color(lidar_pc, as_hex=True)
    ax.scatter(
        lidar_pc[LidarIndex.Y],
        lidar_pc[LidarIndex.X],
        c=lidar_pc_colors,
        alpha=LIDAR_CONFIG["alpha"],
        s=LIDAR_CONFIG["size"],
        zorder=LIDAR_CONFIG["zorder"],
    )
    return ax


def add_trajectory_to_bev_ax(
    ax: plt.Axes, trajectory: Trajectory, config: Dict[str, Any]
) -> plt.Axes:
    """
    Add trajectory poses as lint to plot
    :param ax: matplotlib ax object
    :param trajectory: navsim trajectory dataclass
    :param config: dictionary with plot parameters
    :return: ax with plot
    """    
    poses = np.concatenate([np.array([[0, 0]]), trajectory.poses[:, :2]])
    ax.plot(
        poses[:, 1],
        poses[:, 0],
        color=config["line_color"],
        alpha=config["line_color_alpha"],
        linewidth=config["line_width"],
        linestyle=config["line_style"],
        marker=config["marker"],
        markersize=config["marker_size"],
        markeredgecolor=config["marker_edge_color"],
        zorder=config["zorder"],
    )
    return ax


def add_oriented_box_to_bev_ax(
    ax: plt.Axes, box: OrientedBox, config: Dict[str, Any], add_heading: bool = True
) -> plt.Axes:
    """
    Adds birds-eye-view visualization of surrounding bounding boxes
    :param ax: matplotlib ax object
    :param box: nuPlan dataclass for 2D bounding boxes
    :param config: dictionary with plot parameters
    :param add_heading: whether to add a heading line, defaults to True
    :return: ax with plot
    """    

    box_corners = box.all_corners()
    corners = [[corner.x, corner.y] for corner in box_corners]
    corners = np.asarray(corners + [corners[0]])

    ax.fill(
        corners[:, 1],
        corners[:, 0],
        color=config["fill_color"],
        alpha=config["fill_color_alpha"],
        zorder=config["zorder"],
    )
    ax.plot(
        corners[:, 1],
        corners[:, 0],
        color=config["line_color"],
        alpha=config["line_color_alpha"],
        linewidth=config["line_width"],
        linestyle=config["line_style"],
        zorder=config["zorder"],
    )

    if add_heading:
        future = translate_longitudinally(box.center, distance=box.length / 2 + 1)
        line = np.array([[box.center.x, box.center.y], [future.x, future.y]])
        ax.plot(
            line[:, 1],
            line[:, 0],
            color=config["line_color"],
            alpha=config["line_color_alpha"],
            linewidth=config["line_width"],
            linestyle=config["line_style"],
            zorder=config["zorder"],
        )

    return ax


def add_polygon_to_bev_ax(ax: plt.Axes, polygon: Polygon, config: Dict[str, Any]) -> plt.Axes:
    """
    Adds shapely polygon to birds-eye-view visualization
    :param ax: matplotlib ax object
    :param polygon: shapely Polygon 
    :param config: dictionary containing plot parameters
    :return: ax with plot
    """    

    def _add_element_helper(element: Polygon):
        """ Helper to add single polygon to ax """
        exterior_x, exterior_y = element.exterior.xy
        ax.fill(
            exterior_y,
            exterior_x,
            color=config["fill_color"],
            alpha=config["fill_color_alpha"],
            zorder=config["zorder"],
        )
        ax.plot(
            exterior_y,
            exterior_x,
            color=config["line_color"],
            alpha=config["line_color_alpha"],
            linewidth=config["line_width"],
            linestyle=config["line_style"],
            zorder=config["zorder"],
        )
        for interior in element.interiors:
            x_interior, y_interior = interior.xy
            ax.fill(
                y_interior,
                x_interior,
                color=BEV_PLOT_CONFIG["background_color"],
                zorder=config["zorder"],
            )
            ax.plot(
                y_interior,
                x_interior,
                color=config["line_color"],
                alpha=config["line_color_alpha"],
                linewidth=config["line_width"],
                linestyle=config["line_style"],
                zorder=config["zorder"],
            )

    if isinstance(polygon, Polygon):
        _add_element_helper(polygon)
    else:
        # NOTE: in rare cases, a map polygon has several sub-polygons.
        for element in polygon:
            _add_element_helper(element)

    return ax


def add_linestring_to_bev_ax(
    ax: plt.Axes, linestring: LineString, config: Dict[str, Any]
) -> plt.Axes:
    """
    Adds shapely linestring (polyline) to birds-eye-view visualization
    :param ax: matplotlib ax object
    :param linestring: shapely LineString
    :param config: dictionary containing plot parameters
    :return: ax with plot
    """    

    x, y = linestring.xy
    ax.plot(
        y,
        x,
        color=config["line_color"],
        alpha=config["line_color_alpha"],
        linewidth=config["line_width"],
        linestyle=config["line_style"],
        zorder=config["zorder"],
    )

    return ax
