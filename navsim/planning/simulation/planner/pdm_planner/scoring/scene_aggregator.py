from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import ego_is_two_frame_extended_comfort


@dataclass
class SceneAggregator:
    now_frame: str
    previous_frame: str
    score_df: pd.DataFrame
    proposal_sampling: TrajectorySampling
    second_stage: Optional[List[Tuple[str, str]]] = None

    def calculate_pseudo_closed_loop_weights(self, first_stage_row, second_stage_scores) -> pd.Series:

        pd.options.mode.copy_on_write = True

        def _calc_distance(x1, y1, x2, y2):
            return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        second_stage_scores = second_stage_scores.copy()
        second_stage_scores["distance"] = second_stage_scores.apply(
            lambda x: _calc_distance(
                first_stage_row["endpoint_x"],
                first_stage_row["endpoint_y"],
                x["start_point_x"],
                x["start_point_y"],
            ),
            axis=1,
        )
        second_stage_scores["weight"] = np.exp(-second_stage_scores["distance"])
        second_stage_scores["weight"] /= second_stage_scores["weight"].sum()

        assert np.isclose(
            second_stage_scores["weight"].sum(), 1.0, atol=1e-6
        ), f"Second-stage weights do not sum to 1. Got {second_stage_scores['weight'].sum()}"

        return second_stage_scores[["weight"]].reset_index()

    def _compute_two_frame_comfort(self, current_token: str, previous_token: str) -> float:
        try:
            current_row = self.score_df.loc[current_token]
            prev_row = self.score_df.loc[previous_token]
        except KeyError as e:
            raise ValueError(f"Missing token in score_df: {e}")

        current_states = current_row.get("ego_simulated_states")
        prev_states = prev_row.get("ego_simulated_states")

        interval_length = self.proposal_sampling.interval_length
        observation_interval = current_row["start_time"] - prev_row["start_time"]

        assert 0 < observation_interval < 0.55, f"Invalid interval {observation_interval}"

        overlap_start = round(observation_interval / interval_length)
        current_states_overlap = current_states[:-overlap_start]
        prev_states_overlap = prev_states[overlap_start:]

        n_overlap = current_states_overlap.shape[0]
        time_point_s = np.arange(n_overlap) * interval_length

        two_frame_comfort = ego_is_two_frame_extended_comfort(
            current_states_overlap[None, :],
            prev_states_overlap[None, :],
            time_point_s,
        )[0].astype(np.float64)

        return two_frame_comfort

    def aggregate_scores(self, one_stage_only=False) -> pd.DataFrame:
        updates = []

        if one_stage_only:
            main_comfort = self._compute_two_frame_comfort(self.now_frame, self.previous_frame)
            updates.append({"token": self.now_frame, "two_frame_extended_comfort": main_comfort})

        else:
            # =====First stage=====
            main_comfort = self._compute_two_frame_comfort(self.now_frame, self.previous_frame)
            updates.append({"token": self.now_frame, "two_frame_extended_comfort": main_comfort, "weight": 1.0})
            updates.append({"token": self.previous_frame, "two_frame_extended_comfort": main_comfort, "weight": 1.0})

            # =====Second stage=====
            # t = 0s and t = 4s
            second_stage_now_tokens = [pair[0] for pair in self.second_stage]
            second_stage_now_scores = self.score_df.loc[second_stage_now_tokens]

            first_stage_now_row = self.score_df.loc[self.now_frame]
            weights_now = self.calculate_pseudo_closed_loop_weights(first_stage_now_row, second_stage_now_scores)

            # t = -0.5s and t = 3.5s
            second_stage_prev_tokens = [pair[1] for pair in self.second_stage]
            second_stage_prev_scores = self.score_df.loc[second_stage_prev_tokens]

            first_stage_prev_row = self.score_df.loc[self.previous_frame]
            weights_prev = self.calculate_pseudo_closed_loop_weights(first_stage_prev_row, second_stage_prev_scores)

            weights = pd.concat([weights_now, weights_prev], ignore_index=True)

            weight_map = dict(zip(weights["token"], weights["weight"]))

            for (now_token, prev_token) in self.second_stage:

                two_frame_comfort = self._compute_two_frame_comfort(now_token, prev_token)
                weight_now = weight_map[now_token]
                weight_prev = weight_map[prev_token]

                updates.append(
                    {"token": now_token, "two_frame_extended_comfort": two_frame_comfort, "weight": weight_now}
                )
                updates.append(
                    {"token": prev_token, "two_frame_extended_comfort": two_frame_comfort, "weight": weight_prev}
                )

        return pd.DataFrame(updates)
