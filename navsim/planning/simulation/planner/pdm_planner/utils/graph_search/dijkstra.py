from typing import Dict, List, Optional, Tuple

import numpy as np
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject, RoadBlockGraphEdgeMapObject


class Dijkstra:
    """
    A class that performs dijkstra's shortest path. The class operates on lane level graph search.
    The goal condition is specified to be if the lane can be found at the target roadblock or roadblock connector.
    """

    def __init__(self, start_edge: LaneGraphEdgeMapObject, candidate_lane_edge_ids: List[str]):
        """
        Constructor for the Dijkstra class.
        :param start_edge: The starting edge for the search
        :param candidate_lane_edge_ids: The candidates lane ids that can be included in the search.
        """
        self._queue = list([start_edge])
        self._parent: Dict[str, Optional[LaneGraphEdgeMapObject]] = dict()
        self._candidate_lane_edge_ids = candidate_lane_edge_ids

    def search(self, target_roadblock: RoadBlockGraphEdgeMapObject) -> Tuple[List[LaneGraphEdgeMapObject], bool]:
        """
        Performs dijkstra's shortest path to find a route to the target roadblock.
        :param target_roadblock: The target roadblock the path should end at.
        :return:
            - A route starting from the given start edge
            - A bool indicating if the route is successfully found. Successful means that there exists a path
              from the start edge to an edge contained in the end roadblock.
              If unsuccessful the shortest deepest path is returned.
        """
        start_edge = self._queue[0]

        # Initial search states
        path_found: bool = False
        end_edge: LaneGraphEdgeMapObject = start_edge

        self._parent[start_edge.id] = None
        self._frontier = [start_edge.id]
        self._dist = [1]
        self._depth = [1]

        self._expanded = []
        self._expanded_id = []
        self._expanded_dist = []
        self._expanded_depth = []

        while len(self._queue) > 0:
            dist, idx = min((val, idx) for (idx, val) in enumerate(self._dist))
            current_edge = self._queue[idx]
            current_depth = self._depth[idx]

            del self._dist[idx], self._queue[idx], self._frontier[idx], self._depth[idx]

            if self._check_goal_condition(current_edge, target_roadblock):
                end_edge = current_edge
                path_found = True
                break

            self._expanded.append(current_edge)
            self._expanded_id.append(current_edge.id)
            self._expanded_dist.append(dist)
            self._expanded_depth.append(current_depth)

            # Populate queue
            for next_edge in current_edge.outgoing_edges:
                if next_edge.id not in self._candidate_lane_edge_ids:
                    continue

                alt = dist + self._edge_cost(next_edge)
                if next_edge.id not in self._expanded_id and next_edge.id not in self._frontier:
                    self._parent[next_edge.id] = current_edge
                    self._queue.append(next_edge)
                    self._frontier.append(next_edge.id)
                    self._dist.append(alt)
                    self._depth.append(current_depth + 1)
                    end_edge = next_edge

                elif next_edge.id in self._frontier:
                    next_edge_idx = self._frontier.index(next_edge.id)
                    current_cost = self._dist[next_edge_idx]
                    if alt < current_cost:
                        self._parent[next_edge.id] = current_edge
                        self._dist[next_edge_idx] = alt
                        self._depth[next_edge_idx] = current_depth + 1

        if not path_found:
            # filter max depth
            max_depth = max(self._expanded_depth)
            idx_max_depth = list(np.where(np.array(self._expanded_depth) == max_depth)[0])
            dist_at_max_depth = [self._expanded_dist[i] for i in idx_max_depth]

            dist, _idx = min((val, idx) for (idx, val) in enumerate(dist_at_max_depth))
            end_edge = self._expanded[idx_max_depth[_idx]]

        return self._construct_path(end_edge), path_found

    @staticmethod
    def _edge_cost(lane: LaneGraphEdgeMapObject) -> float:
        """
        Edge cost of given lane.
        :param lane: lane class
        :return: length of lane
        """
        return lane.baseline_path.length

    @staticmethod
    def _check_end_condition(depth: int, target_depth: int) -> bool:
        """
        Check if the search should end regardless if the goal condition is met.
        :param depth: The current depth to check.
        :param target_depth: The target depth to check against.
        :return: True if:
            - The current depth exceeds the target depth.
        """
        return depth > target_depth

    @staticmethod
    def _check_goal_condition(
        current_edge: LaneGraphEdgeMapObject,
        target_roadblock: RoadBlockGraphEdgeMapObject,
    ) -> bool:
        """
        Check if the current edge is at the target roadblock at the given depth.
        :param current_edge: The edge to check.
        :param target_roadblock: The target roadblock the edge should be contained in.
        :return: whether the current edge is in the target roadblock
        """
        return current_edge.get_roadblock_id() == target_roadblock.id

    def _construct_path(self, end_edge: LaneGraphEdgeMapObject) -> List[LaneGraphEdgeMapObject]:
        """
        :param end_edge: The end edge to start back propagating back to the start edge.
        :param depth: The depth of the target edge.
        :return: The constructed path as a list of LaneGraphEdgeMapObject
        """
        path = [end_edge]
        while self._parent[end_edge.id] is not None:
            node = self._parent[end_edge.id]
            path.append(node)
            end_edge = node
        path.reverse()

        return path
