from __future__ import annotations

from typing import Any, Dict, List, Tuple, Type

import numpy as np
import numpy.typing as npt
from nuplan.planning.simulation.occupancy_map.abstract_occupancy_map import Geometry
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

import shapely.vectorized
from shapely.strtree import STRtree
from shapely.geometry import Point

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import AbstractMap, MapObject


class PDMOccupancyMap:
    """Occupancy map class of PDM, based on shapely's str-tree."""

    def __init__(
        self,
        tokens: List[str],
        geometries: npt.NDArray[np.object_],
        node_capacity: int = 10,
    ):
        """
        Constructor of PDMOccupancyMap
        :param tokens: list of tracked tokens
        :param geometries: list/array of polygons
        :param node_capacity: max number of child nodes in str-tree, defaults to 10
        """
        assert len(tokens) == len(
            geometries
        ), f"PDMOccupancyMap: Tokens/Geometries ({len(tokens)}/{len(geometries)}) have unequal length!"

        # attribute
        self._tokens = tokens
        self._geometries = geometries
        self._node_capacity = node_capacity

        # loaded during initialization
        self._token_to_idx: Dict[str, int] = {token: idx for idx, token in enumerate(tokens)}
        self._str_tree = STRtree(self._geometries, node_capacity)

    def __reduce__(self) -> Tuple[Type[PDMOccupancyMap], Tuple[Any, ...]]:
        """Helper for pickling."""
        return self.__class__, (self._tokens, self._geometries, self._node_capacity)

    def __getitem__(self, token) -> Geometry:
        """
        Retrieves geometry of token.
        :param token: geometry identifier
        :return: Geometry of token
        """
        return self._geometries[self._token_to_idx[token]]

    def __len__(self) -> int:
        """
        Number of geometries in the occupancy map
        :return: int
        """
        return len(self._tokens)

    @property
    def tokens(self) -> List[str]:
        """
        Getter for track tokens in occupancy map
        :return: list of strings
        """
        return self._tokens

    @property
    def token_to_idx(self) -> Dict[str, int]:
        """
        Getter for track tokens in occupancy map
        :return: dictionary of tokens and indices
        """
        return self._token_to_idx

    def intersects(self, geometry: Geometry) -> List[str]:
        """
        Searches for intersecting geometries in the occupancy map
        :param geometry: geometries to query
        :return: list of tokens for intersecting geometries
        """
        indices = self.query(geometry, predicate="intersects")
        return [self._tokens[idx] for idx in indices]

    def query(self, geometry: Geometry, predicate=None):
        """
        Function to directly calls shapely's query function on str-tree
        :param geometry: geometries to query
        :param predicate: see shapely, defaults to None
        :return: query output
        """
        return self._str_tree.query(geometry, predicate=predicate)


class PDMDrivableMap(PDMOccupancyMap):
    def __init__(
        self,
        tokens: List[str],
        map_types: List[SemanticMapLayer],
        geometries: npt.NDArray[np.object_],
        node_capacity: int = 10,
    ):
        assert (
            len(tokens) == len(geometries) == len(map_types)
        ), f"PDMDrivableMap: Tokens/Geometries/Types ({len(tokens)}/{len(geometries)}/{len(map_types)}) have unequal length!"

        super().__init__(tokens=tokens, geometries=geometries, node_capacity=node_capacity)

        # attribute
        self._map_types = map_types

    def __reduce__(self) -> Tuple[Type[PDMDrivableMap], Tuple[Any, ...]]:
        """Helper for pickling."""
        return self.__class__, (
            self._tokens,
            self._map_types,
            self._geometries,
            self._node_capacity
        )

    @property
    def map_types(self) -> List[SemanticMapLayer]:
        """
        Getter for SemanticMapLayer types of polygons in occupancy map
        :return: list of SemanticMapLayer
        """
        return self._map_types

    @classmethod
    def from_simulation(
        cls, map_api: AbstractMap, ego_state: EgoState, map_radius: float = 50
    ) -> PDMDrivableMap:
        """ """

        # TODO: Fix SemanticMapLayer.DRIVABLE_AREA problems
        roadblock_layers = [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]

        drivable_map_layers = [
            SemanticMapLayer.INTERSECTION,
            SemanticMapLayer.CARPARK_AREA,
        ]

        # query all drivable map elements around ego position
        position: Point2D = ego_state.center.point
        drivable_area = map_api.get_proximal_map_objects(
            position, map_radius, roadblock_layers + drivable_map_layers
        )

        # collect lane polygons in list, save on-route indices
        polygons: List[Geometry] = []
        polygon_tokens: List[str] = []
        polygon_types: List[SemanticMapLayer] = []

        def extract_map_layer(map_objects: List[MapObject]) -> Tuple[List[Geometry], List[str]]:
            polygons_: List[Geometry] = []
            polygon_tokens_: List[str] = []

            for map_object in map_objects:
                polygons_.append(map_object.polygon)
                polygon_tokens_.append(map_object.id)

            return polygons_, polygon_tokens_

        # 1. Roadblock Polygons
        polygons_, polygon_tokens_ = extract_map_layer(drivable_area[SemanticMapLayer.ROADBLOCK])
        polygons.extend(polygons_)
        polygon_tokens.extend(polygon_tokens_)
        polygon_types.extend(len(polygons_) * [SemanticMapLayer.ROADBLOCK])

        # 2. Lane & Lane-Connector Polygons
        for map_layer in roadblock_layers:
            for roadblock in drivable_area[map_layer]:
                # extract roadblocks
                polygons_, polygon_tokens_ = extract_map_layer(roadblock.interior_edges)
                polygons.extend(polygons_)
                polygon_tokens.extend(polygon_tokens_)

                if map_layer == SemanticMapLayer.ROADBLOCK:
                    polygon_types.extend(len(polygons_) * [SemanticMapLayer.LANE])
                else:
                    polygon_types.extend(len(polygons_) * [SemanticMapLayer.LANE_CONNECTOR])

        # 3. Other drivable area polygons
        for map_layer in drivable_map_layers:
            polygons_, polygon_tokens_ = extract_map_layer(drivable_area[map_layer])
            polygons.extend(polygons_)
            polygon_tokens.extend(polygon_tokens_)
            polygon_types.extend(len(polygons_) * [map_layer])

        return PDMDrivableMap(polygon_tokens, polygon_types, polygons)

    def get_indices_of_map_type(self, map_types: List[SemanticMapLayer]) -> List[int]:
        """
        Getter for indices of a particular SemanticMapLayer
        :return: list of integers
        """
        indices_of_type = [
            idx for idx, map_type_ in enumerate(self._map_types) if map_type_ in map_types
        ]
        return indices_of_type

    def points_in_polygons(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
        """
        Determines whether input-points are in polygons of the occupancy map
        :param points: input-points
        :return: boolean array of shape (polygons, input-points)
        """
        assert points.shape[-1] == 2, "Points array must have shape (...,2) for x, y coordinates!"

        input_shape = points.shape[:-1]
        flattened_points = points.reshape(-1, 2)

        output = np.zeros((len(self._geometries), len(flattened_points)), dtype=bool)
        for i, polygon in enumerate(self._geometries):
            output[i] = shapely.vectorized.contains(
                polygon, flattened_points[:, 0], flattened_points[:, 1]
            )

        output_shape = (len(self._geometries),) + input_shape
        return output.reshape(output_shape)

    def is_in_layer(self, point: Point2D, layer: SemanticMapLayer) -> bool:
        """
        Checks if point is in map layer
        :param point: Point2D of nuPlan
        :param layer: semantic map layer
        :return: boolean
        """
        polygons_indices = self._str_tree.query(Point(point.x, point.y), predicate="within")
        polygons_types = [self._map_types[polygon_idx] for polygon_idx in polygons_indices]
        return layer in polygons_types
