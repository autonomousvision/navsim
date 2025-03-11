from typing import Any, Dict

from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import SemanticMapLayer

LIGHT_GREY: str = "#D3D3D3"

TAB_10: Dict[int, str] = {
    0: "#1f77b4",  # blue
    1: "#ff7f0e",  # orange
    2: "#2ca02c",  # green
    3: "#d62728",  # red
    4: "#9467bd",  # violet
    5: "#8c564b",  # brown
    6: "#e377c2",  # pink
    7: "#7f7f7f",  # grey
    8: "#bcbd22",  # yellow
    9: "#17becf",  # cyan
}


NEW_TAB_10: Dict[int, str] = {
    0: "#4e79a7",  # blue
    1: "#f28e2b",  # orange
    2: "#e15759",  # red
    3: "#76b7b2",  # cyan
    4: "#59a14f",  # green
    5: "#edc948",  # yellow
    6: "#b07aa1",  # violet
    7: "#ff9da7",  # pink
    8: "#9c755f",  # brown
    9: "#bab0ac",  # grey
}


ELLIS_5: Dict[int, str] = {
    0: "#DE7061",  # red
    1: "#B0E685",  # green
    2: "#4AC4BD",  # cyan
    3: "#E38C47",  # orange
    4: "#699CDB",  # blue
}


BEV_PLOT_CONFIG: Dict[str, Any] = {
    "figure_size": (5, 5),
    "figure_margin": (64, 64),
    "background_color": "white",
    "layers": ["map", "annotations"],  # "map", "annotations", "lidar"
}

CAMERAS_PLOT_CONFIG: Dict[str, Any] = {
    "figure_size": (12, 7),
}


LIDAR_CONFIG: Dict[str, Any] = {
    "color_element": "distance",  # ["none", "distance", "x", "y", "z", "intensity", "ring", "id"]
    "color_map": "viridis",
    "x_lim": [-32, 32],
    "y_lim": [-32, 32],
    "z_lim": [-4, 64],
    "alpha": 0.5,
    "size": 0.1,
    "zorder": 3,
}

MAP_LAYER_CONFIG: Dict[SemanticMapLayer, Any] = {
    SemanticMapLayer.LANE: {
        "fill_color": LIGHT_GREY,
        "fill_color_alpha": 1.0,
        "line_color": LIGHT_GREY,
        "line_color_alpha": 0.0,
        "line_width": 1.0,
        "line_style": "-",
        "zorder": 1,
    },
    SemanticMapLayer.WALKWAYS: {
        "fill_color": "#d4d19e",
        "fill_color_alpha": 1.0,
        "line_color": "#d4d19e",
        "line_color_alpha": 0.0,
        "line_width": 1.0,
        "line_style": "-",
        "zorder": 1,
    },
    SemanticMapLayer.CARPARK_AREA: {
        "fill_color": "#b9d3b4",
        "fill_color_alpha": 1.0,
        "line_color": "#b9d3b4",
        "line_color_alpha": 0.0,
        "line_width": 0.0,
        "line_style": "-",
        "zorder": 1,
    },
    SemanticMapLayer.PUDO: {
        "fill_color": "#AF75A7",
        "fill_color_alpha": 0.3,
        "line_color": "#AF75A7",
        "line_color_alpha": 1.0,
        "line_width": 1.0,
        "line_style": "-",
        "zorder": 1,
    },
    SemanticMapLayer.INTERSECTION: {
        "fill_color": "#D3D3D3",
        "fill_color_alpha": 1.0,
        "line_color": "#D3D3D3",
        "line_color_alpha": 1.0,
        "line_width": 1.0,
        "line_style": "-",
        "zorder": 1,
    },
    SemanticMapLayer.STOP_LINE: {
        "fill_color": "#FF0101",
        "fill_color_alpha": 0.0,
        "line_color": "#FF0101",
        "line_color_alpha": 0.0,
        "line_width": 1.0,
        "line_style": "-",
        "zorder": 1,
    },
    SemanticMapLayer.CROSSWALK: {
        "fill_color": NEW_TAB_10[6],
        "fill_color_alpha": 0.3,
        "line_color": NEW_TAB_10[6],
        "line_color_alpha": 0.0,
        "line_width": 1.0,
        "line_style": "-",
        "zorder": 1,
    },
    SemanticMapLayer.ROADBLOCK: {
        "fill_color": "#0000C0",
        "fill_color_alpha": 0.2,
        "line_color": "#0000C0",
        "line_color_alpha": 1.0,
        "line_width": 1.0,
        "line_style": "-",
        "zorder": 1,
    },
    SemanticMapLayer.BASELINE_PATHS: {
        "line_color": "#666666",
        "line_color_alpha": 1.0,
        "line_width": 1.0,
        "line_style": "--",
        "zorder": 1,
    },
    SemanticMapLayer.LANE_CONNECTOR: {
        "line_color": "#CBCBCB",
        "line_color_alpha": 1.0,
        "line_width": 1.0,
        "line_style": "-",
        "zorder": 1,
    },
}

AGENT_CONFIG: Dict[SemanticMapLayer, Any] = {
    TrackedObjectType.VEHICLE: {
        "fill_color": ELLIS_5[4],
        "fill_color_alpha": 1.0,
        "line_color": "black",
        "line_color_alpha": 1.0,
        "line_width": 1.0,
        "line_style": "-",
        "zorder": 2,
    },
    TrackedObjectType.PEDESTRIAN: {
        "fill_color": NEW_TAB_10[6],
        "fill_color_alpha": 1.0,
        "line_color": "black",
        "line_color_alpha": 1.0,
        "line_width": 1.0,
        "line_style": "-",
        "zorder": 2,
    },
    TrackedObjectType.BICYCLE: {
        "fill_color": ELLIS_5[3],
        "fill_color_alpha": 1.0,
        "line_color": "black",
        "line_color_alpha": 1.0,
        "line_width": 1.0,
        "line_style": "-",
        "zorder": 2,
    },
    TrackedObjectType.TRAFFIC_CONE: {
        "fill_color": NEW_TAB_10[5],
        "fill_color_alpha": 1.0,
        "line_color": "black",
        "line_color_alpha": 1.0,
        "line_width": 1.0,
        "line_style": "-",
        "zorder": 2,
    },
    TrackedObjectType.BARRIER: {
        "fill_color": NEW_TAB_10[5],
        "fill_color_alpha": 1.0,
        "line_color": "black",
        "line_color_alpha": 1.0,
        "line_width": 1.0,
        "line_style": "-",
        "zorder": 2,
    },
    TrackedObjectType.CZONE_SIGN: {
        "fill_color": NEW_TAB_10[5],
        "fill_color_alpha": 1.0,
        "line_color": "black",
        "line_color_alpha": 1.0,
        "line_width": 1.0,
        "line_style": "-",
        "zorder": 2,
    },
    TrackedObjectType.GENERIC_OBJECT: {
        "fill_color": NEW_TAB_10[5],
        "fill_color_alpha": 1.0,
        "line_color": "black",
        "line_color_alpha": 1.0,
        "line_width": 1.0,
        "line_style": "-",
        "zorder": 2,
    },
    TrackedObjectType.EGO: {
        "fill_color": ELLIS_5[0],
        "fill_color_alpha": 1.0,
        "line_color": "black",
        "line_color_alpha": 1.0,
        "line_width": 1.0,
        "line_style": "-",
        "zorder": 2,
    },
}

TRAJECTORY_CONFIG: Dict[str, Any] = {
    "human": {
        "fill_color": NEW_TAB_10[4],
        "fill_color_alpha": 1.0,
        "line_color": NEW_TAB_10[4],
        "line_color_alpha": 1.0,
        "line_width": 2.0,
        "line_style": "-",
        "marker": "o",
        "marker_size": 5,
        "marker_edge_color": "black",
        "zorder": 3,
    },
    "agent": {
        "fill_color": ELLIS_5[0],
        "fill_color_alpha": 1.0,
        "line_color": ELLIS_5[0],
        "line_color_alpha": 1.0,
        "line_width": 2.0,
        "line_style": "-",
        "marker": "o",
        "marker_size": 5,
        "marker_edge_color": "black",
        "zorder": 3,
    },
}
