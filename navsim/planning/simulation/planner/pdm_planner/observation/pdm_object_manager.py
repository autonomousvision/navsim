import copy
from typing import Dict, Tuple

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.actor_state.tracked_objects import TrackedObject
from nuplan.common.actor_state.tracked_objects_types import (
    AGENT_TYPES,
    TrackedObjectType,
)

from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    BBCoordsIndex,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    normalize_angle,
)

MAX_DYNAMIC_OBJECTS: Dict[TrackedObjectType, int] = {
    TrackedObjectType.VEHICLE: 50,
    TrackedObjectType.PEDESTRIAN: 25,
    TrackedObjectType.BICYCLE: 10,
}

MAX_STATIC_OBJECTS: int = 50


class PDMObjectManager:
    """Class that stores and sorts tracked objects around the ego-vehicle."""

    def __init__(
        self,
    ):
        """Constructor of PDMObjectManager."""

        # all objects
        self._unique_objects: Dict[str, TrackedObject] = {}

        # dynamic objects
        self._dynamic_object_tokens = {key: [] for key in MAX_DYNAMIC_OBJECTS.keys()}
        self._dynamic_object_coords = {key: [] for key in MAX_DYNAMIC_OBJECTS.keys()}
        self._dynamic_object_dxy = {key: [] for key in MAX_DYNAMIC_OBJECTS.keys()}

        # static objects
        self._static_object_tokens = []
        self._static_object_coords = []

    @property
    def unique_objects(self) -> Dict[str, TrackedObject]:
        """
        Getter of unique_objects
        :return: Dictionary of uniquely tracked objects
        """
        return self._unique_objects

    def add_object(self, object: TrackedObject) -> None:
        """
        Add object to manager and sort category (dynamic/static)
        :param object: any tracked object
        """
        self._unique_objects[object.track_token] = object

        coords_list = [
            [corner.x, corner.y] for corner in copy.deepcopy(object.box.all_corners())
        ]
        coords_list.append([object.center.x, object.center.y])

        coords: np.ndarray = np.array(coords_list, dtype=np.float64)

        if object.tracked_object_type in AGENT_TYPES:
            velocity = object.velocity
            velocity_angle = np.arctan2(velocity.y, velocity.x)
            agent_drives_forward = (
                np.abs(normalize_angle(object.center.heading - velocity_angle))
                < np.pi / 2
            )

            track_heading = (
                object.center.heading
                if agent_drives_forward
                else normalize_angle(object.center.heading + np.pi)
            )

            dxy = np.array(
                [
                    np.cos(track_heading) * velocity.magnitude(),
                    np.sin(track_heading) * velocity.magnitude(),
                ],
                dtype=np.float64,
            ).T  # x,y velocity [m/s]

            self._add_dynamic_object(
                object.tracked_object_type, object.track_token, coords, dxy
            )

        else:
            self._add_static_object(
                object.tracked_object_type, object.track_token, coords
            )

    def get_nearest_objects(self, position: Point2D) -> Tuple:
        """
        Retrieve nearest k objects depending on category.
        :param position: global map position
        :return: tuple containing tokens, coords, and dynamic information of objects
        """
        dynamic_object_tokens, dynamic_object_coords_list, dynamic_object_dxy_list = (
            [],
            [],
            [],
        )

        for dynamic_object_type in MAX_DYNAMIC_OBJECTS.keys():
            (
                dynamic_object_tokens_,
                dynamic_object_coords_,
                dynamic_object_dxy_,
            ) = self._get_nearest_dynamic_objects(position, dynamic_object_type)

            if dynamic_object_coords_.ndim != 3:
                continue

            dynamic_object_tokens.extend(dynamic_object_tokens_)
            dynamic_object_coords_list.append(dynamic_object_coords_)
            dynamic_object_dxy_list.append(dynamic_object_dxy_)

        if len(dynamic_object_coords_list) > 0:
            dynamic_object_coords = np.concatenate(
                dynamic_object_coords_list, axis=0, dtype=np.float64
            )
            dynamic_object_dxy = np.concatenate(
                dynamic_object_dxy_list, axis=0, dtype=np.float64
            )
        else:
            dynamic_object_coords = np.array([], dtype=np.float64)
            dynamic_object_dxy = np.array([], dtype=np.float64)

        static_object_tokens, static_object_coords = self._get_nearest_static_objects(
            position, None
        )

        return (
            static_object_tokens,
            static_object_coords,
            dynamic_object_tokens,
            dynamic_object_coords,
            dynamic_object_dxy,
        )

    def _add_dynamic_object(
        self,
        type: TrackedObjectType,
        token: str,
        coords: npt.NDArray[np.float64],
        dxy: npt.NDArray[np.float64],
    ) -> None:
        """
        Adds dynamic obstacle to the manager.
        :param type: Object type (vehicle, pedestrian, etc.)
        :param token: Temporally consistent object identifier
        :param coords: Bounding-box coordinates
        :param dxy: velocity (x,y) [m/s]
        """
        self._dynamic_object_tokens[type].append(token)
        self._dynamic_object_coords[type].append(coords)
        self._dynamic_object_dxy[type].append(dxy)

    def _add_static_object(
        self,
        type: TrackedObjectType,
        token: str,
        coords: npt.NDArray[np.float64],
    ) -> None:
        """
        Adds static obstacle to manager.
        :param type: Object type (e.g. generic, traffic cone, etc.), currently ignored
        :param token: Temporally consistent object identifier
        :param coords: Bounding-box coordinates
        """
        self._static_object_tokens.append(token)
        self._static_object_coords.append(coords)

    def _get_nearest_dynamic_objects(
        self, position: Point2D, type: TrackedObjectType
    ) -> Tuple:
        """
        Retrieves nearest k dynamic objects depending on type
        :param position: Ego-vehicle position
        :param type: Object type to sort
        :return: Tuple of tokens, coords, and velocity of nearest objects.
        """
        position_coords = position.array[None, ...]  # shape: (1,2)

        object_tokens = self._dynamic_object_tokens[type]
        object_coords = np.array(self._dynamic_object_coords[type], dtype=np.float64)
        object_dxy = np.array(self._dynamic_object_dxy[type], dtype=np.float64)

        if len(object_tokens) > 0:
            # add axis if single object found
            if object_coords.ndim == 1:
                object_coords = object_coords[None, ...]
                object_dxy = object_dxy[None, ...]

            position_to_center_dist = (
                (object_coords[..., BBCoordsIndex.CENTER, :] - position_coords) ** 2.0
            ).sum(axis=-1) ** 0.5

            object_argsort = np.argsort(position_to_center_dist)

            object_tokens = [object_tokens[i] for i in object_argsort][
                : MAX_DYNAMIC_OBJECTS[type]
            ]
            object_coords = object_coords[object_argsort][: MAX_DYNAMIC_OBJECTS[type]]
            object_dxy = object_dxy[object_argsort][: MAX_DYNAMIC_OBJECTS[type]]

        return (object_tokens, object_coords, object_dxy)

    def _get_nearest_static_objects(
        self, position: Point2D, type: TrackedObjectType
    ) -> Tuple:
        """
        Retrieves nearest k static obstacles around ego's position.
        :param position: ego's position
        :param type: type of static obstacle (currently ignored)
        :return: tuple of tokens and coords of nearest objects
        """
        position_coords = position.array[None, ...]  # shape: (1,2)

        object_tokens = self._static_object_tokens
        object_coords = np.array(self._static_object_coords, dtype=np.float64)

        if len(object_tokens) > 0:
            # add axis if single object found
            if object_coords.ndim == 1:
                object_coords = object_coords[None, ...]

            position_to_center_dist = (
                (object_coords[..., BBCoordsIndex.CENTER, :] - position_coords) ** 2.0
            ).sum(axis=-1) ** 0.5

            object_argsort = np.argsort(position_to_center_dist)

            object_tokens = [object_tokens[i] for i in object_argsort][
                :MAX_STATIC_OBJECTS
            ]
            object_coords = object_coords[object_argsort][:MAX_STATIC_OBJECTS]

        return (object_tokens, object_coords)
