import copy
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from nuplan.common.actor_state.state_representation import StateSE2, TimePoint
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES
from nuplan.common.actor_state.vehicle_parameters import (
    VehicleParameters,
    get_pacifica_parameters,
)
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.metrics.utils.collision_utils import CollisionType
from nuplan.planning.simulation.observation.idm.utils import (
    is_agent_ahead,
    is_agent_behind,
)
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from shapely import Point, creation

from navsim.common.dataclasses import PDMResults
from navsim.planning.metric_caching.metric_cache import MapParameters
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_observation import (
    PDMObservation,
)
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import (
    PDMDrivableMap,
)
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import (
    ego_is_comfortable,
)
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer_utils import (
    get_collision_type,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    coords_array_to_polygon_array,
    ego_states_to_state_array,
    state_array_to_coords_array,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    BBCoordsIndex,
    EgoAreaIndex,
    MultiMetricIndex,
    StateIndex,
    WeightedMetricIndex,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath


@dataclass
class PDMScorerConfig:

    # weighted metric weights
    progress_weight: float = 5.0
    ttc_weight: float = 5.0
    lane_keeping_weight: float = 2.0
    history_comfort_weight: float = 2.0
    two_frame_extended_comfort_weight: float = 2.0

    # thresholds
    # comfort related config in navsim/planning/simulation/planner/pdm_planner/scoring/pdm_comfort_metrics.py
    driving_direction_horizon: float = 1.0  # [s] (driving direction) (nuplan)
    driving_direction_compliance_threshold: float = (
        2.0  # [m] (driving direction) (nuplan)
    )
    driving_direction_violation_threshold: float = (
        6.0  # [m] (driving direction) (nuplan)
    )

    stopped_speed_threshold: float = 5e-03  # [m/s] (ttc)
    future_collision_horizon_window: float = 1.0  # [s] (ttc)
    progress_distance_threshold: float = 5.0  # [m] (progress)
    lane_keeping_deviation_limit: float = 0.5  # [m] (lane keeping) (hydraMDP++)
    lane_keeping_horizon_window: float = 2.0  # [s] (lane keeping) (hydraMDP++)

    # human flag
    human_penalty_filter: Optional[bool] = None

    @property
    def weighted_metrics_array(self) -> npt.NDArray[np.float64]:
        weighted_metrics = np.zeros(len(WeightedMetricIndex), dtype=np.float64)
        weighted_metrics[WeightedMetricIndex.PROGRESS] = self.progress_weight
        weighted_metrics[WeightedMetricIndex.TTC] = self.ttc_weight
        weighted_metrics[WeightedMetricIndex.LANE_KEEPING] = self.lane_keeping_weight
        weighted_metrics[
            WeightedMetricIndex.HISTORY_COMFORT
        ] = self.history_comfort_weight
        weighted_metrics[
            WeightedMetricIndex.TWO_FRAME_EXTENDEND_COMFORT
        ] = self.two_frame_extended_comfort_weight
        return weighted_metrics


class PDMScorer:
    """Class to score proposals in PDM pipeline. Re-implements nuPlan's closed-loop metrics."""

    def __init__(
        self,
        proposal_sampling: TrajectorySampling,
        config: PDMScorerConfig = PDMScorerConfig(),
        vehicle_parameters: VehicleParameters = get_pacifica_parameters(),
    ):
        """
        Constructor of PDMScorer
        :param proposal_sampling: Sampling parameters for proposals
        """
        self.proposal_sampling = proposal_sampling
        self._config = config
        self._vehicle_parameters = vehicle_parameters

        # lazy loaded
        self._observation: Optional[PDMObservation] = None
        self._centerline: Optional[PDMPath] = None
        self._route_lane_ids: Optional[List[str]] = None
        self._drivable_area_map: Optional[PDMDrivableMap] = None
        self._human_past_trajectory: Optional[InterpolatedTrajectory] = None

        self._num_proposals: Optional[int] = None
        self._states: Optional[npt.NDArray[np.float64]] = None
        self._ego_coords: Optional[npt.NDArray[np.float64]] = None
        self._ego_polygons: Optional[npt.NDArray[np.object_]] = None

        self._ego_areas: Optional[npt.NDArray[np.bool_]] = None

        self._multi_metrics: Optional[npt.NDArray[np.float64]] = None
        self._weighted_metrics: Optional[npt.NDArray[np.float64]] = None
        self._progress_raw: Optional[npt.NDArray[np.float64]] = None

        self._collision_time_idcs: Optional[npt.NDArray[np.float64]] = None
        self._ttc_time_idcs: Optional[npt.NDArray[np.float64]] = None

    def time_to_at_fault_collision(self, proposal_idx: int) -> float:
        """
        Returns time to at-fault collision for given proposal
        :param proposal_idx: index for proposal
        :return: time to infraction
        """
        return (
            self._collision_time_idcs[proposal_idx]
            * self.proposal_sampling.interval_length
        )

    def time_to_ttc_infraction(self, proposal_idx: int) -> float:
        """
        Returns time to ttc infraction for given proposal
        :param proposal_idx: index for proposal
        :return: time to infraction
        """
        return (
            self._ttc_time_idcs[proposal_idx] * self.proposal_sampling.interval_length
        )

    def score_proposals(
        self,
        states: npt.NDArray[np.float64],
        observation: PDMObservation,
        centerline: PDMPath,
        route_lane_ids: List[str],
        drivable_area_map: PDMDrivableMap,
        map_parameters: Optional[MapParameters] = None,
        simulated_agent_detections_tracks=None,
        human_past_trajectory: Optional[InterpolatedTrajectory] = None,
    ) -> List[pd.DataFrame]:
        """
        Scores proposal similar to nuPlan's closed-loop metrics
        :param states: array representation of simulated proposals
        :param observation: PDM's observation class
        :param centerline: path of the centerline
        :param route_lane_ids: list containing on-route lane ids
        :param drivable_area_map: Occupancy map of drivable are polygons
        :return: A List containing the PDMResult for each proposal
        """
        if simulated_agent_detections_tracks is not None:
            observation.update_detections_tracks(
                detection_tracks=simulated_agent_detections_tracks,
            )

        # initialize & lazy load class values
        self._reset(
            states,
            observation,
            centerline,
            route_lane_ids,
            drivable_area_map,
            human_past_trajectory,
        )

        # fill value ego-area array (used in multiple metrics)
        self._calculate_ego_area()

        # 1. multiplicative metrics
        self._calculate_no_at_fault_collision()
        self._calculate_drivable_area_compliance()
        self._calculate_traffic_light_compliance()
        self._calculate_driving_direction_compliance()

        # 2. weighted metrics
        self._calculate_progress()
        self._calculate_ttc()
        self._calculate_lane_keeping()
        self._calculate_history_comfort()

        multiplicative_metrics_prods, weighted_metrics_all = self._aggregate_scores()
        pdm_scores = self._aggregate_pdm_scores()

        results: List[pd.DataFrame] = []
        for proposal_idx in range(self._num_proposals):

            no_at_fault_collisions = self._multi_metrics[
                MultiMetricIndex.NO_COLLISION, proposal_idx
            ]
            drivable_area_compliance = self._multi_metrics[
                MultiMetricIndex.DRIVABLE_AREA, proposal_idx
            ]
            driving_direction_compliance = self._multi_metrics[
                MultiMetricIndex.DRIVING_DIRECTION, proposal_idx
            ]
            traffic_light_compliance = self._multi_metrics[
                MultiMetricIndex.TRAFFIC_LIGHT_COMPLIANCE, proposal_idx
            ]

            ego_progress = self._weighted_metrics[
                WeightedMetricIndex.PROGRESS, proposal_idx
            ]
            time_to_collision_within_bound = self._weighted_metrics[
                WeightedMetricIndex.TTC, proposal_idx
            ]
            lane_keeping = self._weighted_metrics[
                WeightedMetricIndex.LANE_KEEPING, proposal_idx
            ]
            history_comfort = self._weighted_metrics[
                WeightedMetricIndex.HISTORY_COMFORT, proposal_idx
            ]

            multiplicative_metrics_prod = multiplicative_metrics_prods[proposal_idx]
            weighted_metrics = weighted_metrics_all[:, proposal_idx]
            pdm_score = pdm_scores[proposal_idx]

            results.append(
                pd.DataFrame(
                    [
                        PDMResults(
                            no_at_fault_collisions=no_at_fault_collisions,
                            drivable_area_compliance=drivable_area_compliance,
                            driving_direction_compliance=driving_direction_compliance,
                            traffic_light_compliance=traffic_light_compliance,
                            ego_progress=ego_progress,
                            time_to_collision_within_bound=time_to_collision_within_bound,
                            lane_keeping=lane_keeping,
                            history_comfort=history_comfort,
                            multiplicative_metrics_prod=multiplicative_metrics_prod,
                            weighted_metrics=weighted_metrics,
                            weighted_metrics_array=self._config.weighted_metrics_array,
                            pdm_score=pdm_score,
                        )
                    ]
                )
            )
        return results

    def _aggregate_scores(self) -> npt.NDArray[np.float64]:
        """
        Aggregates metrics with multiplicative and weighted average.
        :return: array containing score of each proposal
        """

        # accumulate multiplicative metrics
        multiplicate_metric_scores = self._multi_metrics.prod(axis=0)

        # normalize and fill progress values
        raw_progress = self._progress_raw * multiplicate_metric_scores
        max_raw_progress = np.max(raw_progress)
        if max_raw_progress > self._config.progress_distance_threshold:
            normalized_progress = raw_progress / max_raw_progress
        else:
            normalized_progress = np.ones(len(raw_progress), dtype=np.float64)
            normalized_progress[multiplicate_metric_scores == 0.0] = 0.0
        self._weighted_metrics[WeightedMetricIndex.PROGRESS] = normalized_progress

        return self._multi_metrics.prod(axis=0), self._weighted_metrics

    def _aggregate_pdm_scores(self) -> npt.NDArray[np.float64]:
        """
        Score for PDM proposals, ignoring two-frame extended comfort.
        """

        # accumulate multiplicative metrics
        multiplicate_metric_scores = self._multi_metrics.prod(axis=0)

        # normalize and fill progress values
        raw_progress = self._progress_raw * multiplicate_metric_scores
        max_raw_progress = np.max(raw_progress)
        if max_raw_progress > self._config.progress_distance_threshold:
            normalized_progress = raw_progress / max_raw_progress
        else:
            normalized_progress = np.ones(len(raw_progress), dtype=np.float64)
            normalized_progress[multiplicate_metric_scores == 0.0] = 0.0
        self._weighted_metrics[WeightedMetricIndex.PROGRESS] = normalized_progress

        # Exclude the two-frame extended comfort metric from the weighted metrics calculation.
        mask = np.ones_like(self._config.weighted_metrics_array, dtype=bool)
        mask[WeightedMetricIndex.TWO_FRAME_EXTENDEND_COMFORT] = False

        weighted_metrics_array = self._config.weighted_metrics_array
        weighted_metric_scores = (
            self._weighted_metrics[mask] * weighted_metrics_array[mask, None]
        ).sum(axis=0)
        weighted_metric_scores /= weighted_metrics_array[mask].sum()

        # calculate final scores
        final_scores = multiplicate_metric_scores * weighted_metric_scores

        return final_scores

    def _reset(
        self,
        states: npt.NDArray[np.float64],
        observation: PDMObservation,
        centerline: PDMPath,
        route_lane_ids: List[str],
        drivable_area_map: PDMDrivableMap,
        human_past_trajectory: Optional[InterpolatedTrajectory],
    ) -> None:
        """
        Resets metric values and lazy loads input classes.
        :param states: array representation of simulated proposals
        :param observation: PDM's observation class
        :param centerline: path of the centerline
        :param route_lane_ids: list containing on-route lane ids
        :param drivable_area_map: Occupancy map of drivable are polygons
        """
        assert states.ndim == 3
        assert states.shape[1] == self.proposal_sampling.num_poses + 1
        assert states.shape[2] == StateIndex.size()

        self._observation = observation
        self._centerline = centerline
        self._route_lane_ids = route_lane_ids
        self._drivable_area_map = drivable_area_map
        self._human_past_trajectory = human_past_trajectory

        self._num_proposals = states.shape[0]

        # save ego state values
        self._states = states

        # calculate coordinates of ego corners and center
        self._ego_coords = state_array_to_coords_array(states, self._vehicle_parameters)

        # initialize all ego polygons from corners
        self._ego_polygons = coords_array_to_polygon_array(self._ego_coords)

        # zero initialize all remaining arrays.
        self._ego_areas = np.zeros(
            (
                self._num_proposals,
                self.proposal_sampling.num_poses + 1,
                len(EgoAreaIndex),
            ),
            dtype=np.bool_,
        )
        self._multi_metrics = np.zeros(
            (len(MultiMetricIndex), self._num_proposals), dtype=np.float64
        )
        self._weighted_metrics = np.zeros(
            (len(WeightedMetricIndex), self._num_proposals), dtype=np.float64
        )
        self._progress_raw = np.zeros(self._num_proposals, dtype=np.float64)

        # initialize infraction arrays with infinity (meaning no infraction occurs)
        self._collision_time_idcs = np.zeros(self._num_proposals, dtype=np.float64)
        self._ttc_time_idcs = np.zeros(self._num_proposals, dtype=np.float64)
        self._collision_time_idcs.fill(np.inf)
        self._ttc_time_idcs.fill(np.inf)

    def _calculate_ego_area(self) -> None:
        """
        Determines the area of proposals over time.
        Areas are (1) in multiple lanes, (2) non-drivable area, or (3) oncoming traffic
        """

        n_proposals, n_horizon, n_points, _ = self._ego_coords.shape

        in_polygons = self._drivable_area_map.points_in_polygons(self._ego_coords)
        in_polygons = in_polygons.transpose(
            1, 2, 0, 3
        )  # shape: n_proposals, n_horizon, n_polygons, n_points

        drivable_area_idcs = self._drivable_area_map.get_indices_of_map_type(
            [
                SemanticMapLayer.ROADBLOCK,
                SemanticMapLayer.INTERSECTION,
                SemanticMapLayer.DRIVABLE_AREA,
                SemanticMapLayer.CARPARK_AREA,
            ]
        )

        drivable_lane_idcs = self._drivable_area_map.get_indices_of_map_type(
            [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
        )

        drivable_on_route_idcs: List[int] = [
            idx
            for idx in drivable_lane_idcs
            if self._drivable_area_map.tokens[idx] in self._route_lane_ids
        ]  # index mask for on-route lanes

        corners_in_polygon = in_polygons[..., :-1]  # ignore center coordinate
        center_in_polygon = in_polygons[..., -1]  # only center

        # in_multiple_lanes: if
        # - more than one drivable polygon contains at least one corner
        # - no polygon contains all corners
        batch_multiple_lanes_mask = np.zeros((n_proposals, n_horizon), dtype=np.bool_)
        batch_multiple_lanes_mask = (
            corners_in_polygon[:, :, drivable_lane_idcs].sum(axis=-1) > 0
        ).sum(axis=-1) > 1

        batch_not_single_lanes_mask = np.zeros((n_proposals, n_horizon), dtype=np.bool_)
        batch_not_single_lanes_mask = np.all(
            corners_in_polygon[:, :, drivable_lane_idcs].sum(axis=-1) != 4, axis=-1
        )

        multiple_lanes_mask = np.logical_and(
            batch_multiple_lanes_mask, batch_not_single_lanes_mask
        )
        self._ego_areas[multiple_lanes_mask, EgoAreaIndex.MULTIPLE_LANES] = True

        # in_nondrivable_area: if at least one corner is not within any drivable polygon
        batch_nondrivable_area_mask = np.zeros((n_proposals, n_horizon), dtype=np.bool_)
        batch_nondrivable_area_mask = (
            corners_in_polygon[:, :, drivable_area_idcs].sum(axis=-2) > 0
        ).sum(axis=-1) < 4
        self._ego_areas[
            batch_nondrivable_area_mask, EgoAreaIndex.NON_DRIVABLE_AREA
        ] = True

        # in_oncoming_traffic: if center not in any drivable polygon that is on-route
        batch_oncoming_traffic_mask = np.zeros((n_proposals, n_horizon), dtype=np.bool_)
        batch_oncoming_traffic_mask = (
            center_in_polygon[..., drivable_on_route_idcs].sum(axis=-1) == 0
        )
        self._ego_areas[
            batch_oncoming_traffic_mask, EgoAreaIndex.ONCOMING_TRAFFIC
        ] = True

    def _calculate_no_at_fault_collision(self) -> None:
        """
        Re-implementation of nuPlan's at-fault collision metric.
        """
        no_at_fault_collision_scores = np.ones(self._num_proposals, dtype=np.float64)

        proposal_collided_track_ids = {
            proposal_idx: copy.deepcopy(self._observation.collided_track_ids)
            for proposal_idx in range(self._num_proposals)
        }

        for time_idx in range(self.proposal_sampling.num_poses + 1):
            ego_polygons = self._ego_polygons[:, time_idx]
            intersecting = self._observation[time_idx].query(
                ego_polygons, predicate="intersects"
            )

            if len(intersecting) == 0:
                continue

            for proposal_idx, geometry_idx in zip(intersecting[0], intersecting[1]):
                token = self._observation[time_idx].tokens[geometry_idx]
                if (self._observation.red_light_token in token) or (
                    token in proposal_collided_track_ids[proposal_idx]
                ):
                    continue

                ego_in_multiple_lanes_or_nondrivable_area = (
                    self._ego_areas[proposal_idx, time_idx, EgoAreaIndex.MULTIPLE_LANES]
                    or self._ego_areas[
                        proposal_idx, time_idx, EgoAreaIndex.NON_DRIVABLE_AREA
                    ]
                )

                tracked_object = self._observation.unique_objects[token]

                # classify collision
                collision_type: CollisionType = get_collision_type(
                    self._states[proposal_idx, time_idx],
                    self._ego_polygons[proposal_idx, time_idx],
                    tracked_object,
                    self._observation[time_idx][token],
                )
                collisions_at_stopped_track_or_active_front: bool = collision_type in [
                    CollisionType.ACTIVE_FRONT_COLLISION,
                    CollisionType.STOPPED_TRACK_COLLISION,
                ]
                collision_at_lateral: bool = (
                    collision_type == CollisionType.ACTIVE_LATERAL_COLLISION
                )

                # 1. at fault collision
                if collisions_at_stopped_track_or_active_front or (
                    ego_in_multiple_lanes_or_nondrivable_area and collision_at_lateral
                ):
                    no_at_fault_collision_score = (
                        0.0
                        if tracked_object.tracked_object_type in AGENT_TYPES
                        else 0.5
                    )
                    no_at_fault_collision_scores[proposal_idx] = np.minimum(
                        no_at_fault_collision_scores[proposal_idx],
                        no_at_fault_collision_score,
                    )
                    self._collision_time_idcs[proposal_idx] = min(
                        time_idx, self._collision_time_idcs[proposal_idx]
                    )

                else:  # 2. no at fault collision
                    proposal_collided_track_ids[proposal_idx].append(token)

        self._multi_metrics[
            MultiMetricIndex.NO_COLLISION
        ] = no_at_fault_collision_scores

    def _calculate_drivable_area_compliance(self) -> None:
        """
        Re-implementation of nuPlan's drivable area compliance metric
        """
        drivable_area_compliance_scores = np.ones(self._num_proposals, dtype=np.float64)
        off_road_mask = self._ego_areas[:, :, EgoAreaIndex.NON_DRIVABLE_AREA].any(
            axis=-1
        )
        drivable_area_compliance_scores[off_road_mask] = 0.0
        self._multi_metrics[
            MultiMetricIndex.DRIVABLE_AREA
        ] = drivable_area_compliance_scores

    def _calculate_driving_direction_compliance(self) -> None:
        """
        Re-implementation of nuPlan's driving direction compliance metric
        """
        center_coordinates = self._ego_coords[:, :, BBCoordsIndex.CENTER]
        oncoming_progress = np.zeros(
            (self._num_proposals, self.proposal_sampling.num_poses + 1),
            dtype=np.float64,
        )
        oncoming_progress[:, 1:] = np.linalg.norm(
            center_coordinates[:, 1:] - center_coordinates[:, :-1], axis=-1
        )

        # mask out progress along the driving direction
        oncoming_traffic_masks = self._ego_areas[:, :, EgoAreaIndex.ONCOMING_TRAFFIC]
        oncoming_progress[~oncoming_traffic_masks] = 0.0

        # aggregate
        driving_direction_compliance_scores = np.ones(
            self._num_proposals, dtype=np.float64
        )
        horizon = int(
            self._config.driving_direction_horizon
            / self.proposal_sampling.interval_length
        )

        oncoming_progress_over_horizon = np.concatenate(
            [
                oncoming_progress[:, max(0, time_idx - horizon) : time_idx + 1].sum(
                    axis=-1
                )[..., None]
                for time_idx in range(oncoming_progress.shape[-1])
            ],
            dtype=np.float64,
            axis=-1,
        )

        for proposal_idx, progress in enumerate(
            oncoming_progress_over_horizon.max(axis=-1)
        ):
            if progress < self._config.driving_direction_compliance_threshold:
                driving_direction_compliance_scores[proposal_idx] = 1.0
            elif progress < self._config.driving_direction_violation_threshold:
                driving_direction_compliance_scores[proposal_idx] = 0.5
            else:
                driving_direction_compliance_scores[proposal_idx] = 0.0

        self._multi_metrics[
            MultiMetricIndex.DRIVING_DIRECTION
        ] = driving_direction_compliance_scores

    def _calculate_progress(self) -> None:
        """
        Re-implementation of nuPlan's progress metric (non-normalized).
        Calculates progress along the centerline.
        """

        # calculate raw progress in meter
        progress_in_meter = np.zeros(self._num_proposals, dtype=np.float64)
        for proposal_idx in range(self._num_proposals):
            start_point = Point(
                *self._ego_coords[proposal_idx, 0, BBCoordsIndex.CENTER]
            )
            end_point = Point(*self._ego_coords[proposal_idx, -1, BBCoordsIndex.CENTER])
            progress = self._centerline.project([start_point, end_point])
            progress_in_meter[proposal_idx] = progress[1] - progress[0]

        self._progress_raw = np.clip(progress_in_meter, a_min=0, a_max=None)

    def _calculate_ttc(self):
        """
        Re-implementation of nuPlan's time-to-collision metric.
        """

        ttc_scores = np.ones(self._num_proposals, dtype=np.float64)
        temp_collided_track_ids = {
            proposal_idx: copy.deepcopy(self._observation.collided_track_ids)
            for proposal_idx in range(self._num_proposals)
        }

        # calculate TTC for specific time horizon (default:1s) in the future with less temporal resolution.
        future_time_idcs = np.arange(
            0, int(self._config.future_collision_horizon_window * 10), 3
        )
        n_future_steps = len(future_time_idcs)

        # create polygons for each ego position and specific time horizon (default:1s) future projection
        coords_exterior = self._ego_coords.copy()
        coords_exterior[:, :, BBCoordsIndex.CENTER, :] = coords_exterior[
            :, :, BBCoordsIndex.FRONT_LEFT, :
        ]
        coords_exterior_time_steps = np.repeat(
            coords_exterior[:, :, None], n_future_steps, axis=2
        )

        speeds = np.hypot(
            self._states[..., StateIndex.VELOCITY_X],
            self._states[..., StateIndex.VELOCITY_Y],
        )

        dxy_per_s = np.stack(
            [
                np.cos(self._states[..., StateIndex.HEADING]) * speeds,
                np.sin(self._states[..., StateIndex.HEADING]) * speeds,
            ],
            axis=-1,
        )

        for idx, future_time_idx in enumerate(future_time_idcs):
            delta_t = float(future_time_idx) * self.proposal_sampling.interval_length
            coords_exterior_time_steps[:, :, idx] = (
                coords_exterior_time_steps[:, :, idx] + dxy_per_s[:, :, None] * delta_t
            )

        polygons = creation.polygons(coords_exterior_time_steps)

        # ttc needs to look future_time_idcs into the future,
        # so we can only calculate it for n_proposal_steps_to_evaluate steps

        n_proposal_steps_to_evaluate = self.proposal_sampling.num_poses - max(
            future_time_idcs
        )
        # check collision for each proposal and projection
        for time_idx in range(n_proposal_steps_to_evaluate + 1):
            for step_idx, future_time_idx in enumerate(future_time_idcs):
                current_time_idx = time_idx + future_time_idx
                polygons_at_time_step = polygons[:, time_idx, step_idx]
                intersecting = self._observation[current_time_idx].query(
                    polygons_at_time_step, predicate="intersects"
                )

                if len(intersecting) == 0:
                    continue

                for proposal_idx, geometry_idx in zip(intersecting[0], intersecting[1]):
                    token = self._observation[current_time_idx].tokens[geometry_idx]
                    if (
                        (self._observation.red_light_token in token)
                        or (token in temp_collided_track_ids[proposal_idx])
                        or (
                            speeds[proposal_idx, time_idx]
                            < self._config.stopped_speed_threshold
                        )
                    ):
                        continue

                    ego_in_multiple_lanes_or_nondrivable_area = (
                        self._ego_areas[
                            proposal_idx, time_idx, EgoAreaIndex.MULTIPLE_LANES
                        ]
                        or self._ego_areas[
                            proposal_idx, time_idx, EgoAreaIndex.NON_DRIVABLE_AREA
                        ]
                    )
                    ego_rear_axle: StateSE2 = StateSE2(
                        *self._states[proposal_idx, time_idx, StateIndex.STATE_SE2]
                    )

                    centroid = self._observation[current_time_idx][token].centroid
                    track_heading = self._observation.unique_objects[
                        token
                    ].box.center.heading
                    track_state = StateSE2(centroid.x, centroid.y, track_heading)
                    # TODO: fix ego_area for intersection
                    if is_agent_ahead(ego_rear_axle, track_state) or (
                        (
                            ego_in_multiple_lanes_or_nondrivable_area
                            or self._drivable_area_map.is_in_layer(
                                ego_rear_axle.point, layer=SemanticMapLayer.INTERSECTION
                            )
                        )
                        and not is_agent_behind(ego_rear_axle, track_state)
                    ):
                        ttc_scores[proposal_idx] = np.minimum(
                            ttc_scores[proposal_idx], 0.0
                        )
                        self._ttc_time_idcs[proposal_idx] = min(
                            time_idx, self._ttc_time_idcs[proposal_idx]
                        )
                    else:
                        temp_collided_track_ids[proposal_idx].append(token)

        self._weighted_metrics[WeightedMetricIndex.TTC] = ttc_scores

    def _calculate_traffic_light_compliance(self) -> None:
        """
        Re-implementation of hydraMDP++'s traffic light compliance metric.
        """
        # Initialize scores for all proposals to 1 (compliant by default)
        traffic_light_compliance_scores = np.ones(self._num_proposals, dtype=np.float64)

        # Iterate over each time step within the horizon
        for time_idx in range(self.proposal_sampling.num_poses + 1):
            # Get ego polygons (vehicle shapes) at the current time step
            ego_polygons = self._ego_polygons[:, time_idx]
            # Query objects intersecting with the ego polygons
            intersecting = self._observation[time_idx].query(
                ego_polygons, predicate="intersects"
            )

            # If no intersections, skip this time step
            if len(intersecting) == 0:
                continue

            # Iterate over each intersecting object
            for proposal_idx, geometry_idx in zip(intersecting[0], intersecting[1]):
                # Skip if the score is already 0
                if traffic_light_compliance_scores[proposal_idx] == 0.0:
                    continue

                token = self._observation[time_idx].tokens[geometry_idx]

                # Check if the intersecting object is a red light
                if token.startswith(self._observation.red_light_token):
                    traffic_light_compliance_scores[proposal_idx] = 0.0

        # Store the scores in the multi-metrics system for later evaluation
        self._multi_metrics[
            MultiMetricIndex.TRAFFIC_LIGHT_COMPLIANCE
        ] = traffic_light_compliance_scores

    def _calculate_lane_keeping(self) -> None:
        """
        Revised implementation of hydraMDP++'s lane keeping metric.
        The trajectory is considered failing lane-keeping only if it deviates beyond
        the lateral threshold continuously for at least certain seconds.

        """
        # Initialize lane-keeping scores to 1.0
        lane_keeping_scores = np.ones(self._num_proposals, dtype=np.float64)
        lateral_deviation_limit = self._config.lane_keeping_deviation_limit

        interval_length = self.proposal_sampling.interval_length
        continuous_steps_required = int(
            np.ceil(self._config.lane_keeping_horizon_window / interval_length)
        )

        centerline = self._centerline.linestring

        for proposal_idx in range(self._num_proposals):
            consecutive_exceeds = 0
            for time_idx in range(self.proposal_sampling.num_poses + 1):
                ego_position = Point(
                    *self._ego_coords[proposal_idx, time_idx, BBCoordsIndex.CENTER]
                )

                is_in_intersection = self._drivable_area_map.is_in_layer(
                    ego_position, layer=SemanticMapLayer.INTERSECTION
                )

                if is_in_intersection:
                    continue

                lateral_deviation = ego_position.distance(centerline)

                if lateral_deviation > lateral_deviation_limit:
                    consecutive_exceeds += 1
                else:
                    consecutive_exceeds = 0

                if consecutive_exceeds >= continuous_steps_required:
                    lane_keeping_scores[proposal_idx] = 0.0
                    break

        self._weighted_metrics[WeightedMetricIndex.LANE_KEEPING] = lane_keeping_scores

    def _calculate_history_comfort(self) -> None:
        """
        Implementation of comfort metric, padded with past history states.
        """

        is_history_comfortable = np.ones(self._num_proposals, dtype=np.float64)

        if self._human_past_trajectory is not None:

            # interpolate human past trajectory
            history_start_time_us = self._human_past_trajectory.start_time.time_us
            history_end_time_us = self._human_past_trajectory.end_time.time_us
            time_interval_us = int(0.1 * 1e6)

            history_time_us = np.arange(
                history_start_time_us,
                history_end_time_us,
                time_interval_us,
                dtype=np.int64,
            )
            history_time_us = np.clip(
                history_time_us, history_start_time_us, history_end_time_us
            )

            history_timepoints = [
                TimePoint(time_us) for time_us in history_time_us[:-1]
            ]

            history_state_array = ego_states_to_state_array(
                self._human_past_trajectory.get_state_at_times(history_timepoints)
            )

            # Create state array padded with past human states
            num_padded_poses = len(history_state_array) + self._states.shape[1]
            padded_states_array = np.zeros(
                (self._num_proposals, num_padded_poses, StateIndex.size()),
                dtype=np.float64,
            )

            padded_states_array[:, : len(history_state_array)] = history_state_array
            padded_states_array[:, len(history_state_array) :] = self._states

            # create new timepoints with padding and compute comfort scores
            time_point_s: npt.NDArray[np.float64] = (
                np.arange(0, num_padded_poses).astype(np.float64)
                * self.proposal_sampling.interval_length
            )
            is_history_comfortable = ego_is_comfortable(
                padded_states_array, time_point_s
            ).all(axis=-1)

        self._weighted_metrics[
            WeightedMetricIndex.HISTORY_COMFORT
        ] = is_history_comfortable
