from __future__ import annotations

import lzma
import pickle

from pathlib import Path
from typing import Any, Dict, List
from tqdm import tqdm

from navsim.common.dataclasses import AgentInput, Scene, SceneFilter, SensorConfig
from navsim.planning.metric_caching.metric_cache import MetricCache


def filter_scenes(data_path: Path, scene_filter: SceneFilter) -> Dict[str, List[Dict[str, Any]]]:

    def split_list(input_list: List[Any], n: int) -> List[List[Any]]:
        return [input_list[i : i + n] for i in range(0, len(input_list), n)]

    filtered_scenes: Dict[str, Scene] = {}
    stop_loading: bool = False

    # filter logs
    log_files = list(data_path.iterdir())
    if scene_filter.log_names is not None:
        log_files = [
            log_file
            for log_file in log_files
            if log_file.name.replace(".pkl", "") in scene_filter.log_names
        ]

    for log_pickle_path in tqdm(log_files, desc="Loading logs"):

        scene_dict_list = pickle.load(open(log_pickle_path, "rb"))
        for frame_list in split_list(scene_dict_list, scene_filter.num_frames):
            # Filter scenes which are too short
            if len(frame_list) < scene_filter.num_frames:
                continue

            # Filter scenes with no route
            if (
                scene_filter.has_route
                and len(frame_list[scene_filter.num_history_frames - 1]["roadblock_ids"]) == 0
            ):
                continue

            # Filter by token
            if (
                scene_filter.tokens is not None
                and frame_list[scene_filter.num_history_frames - 1]["token"]
                not in scene_filter.tokens
            ):
                continue

            token = frame_list[scene_filter.num_history_frames - 1]["token"]
            filtered_scenes[token] = frame_list

            if (scene_filter.max_scenes is not None) and (
                len(filtered_scenes) >= scene_filter.max_scenes
            ):
                stop_loading = True
                break

        if stop_loading:
            break

    return filtered_scenes


class SceneLoader:

    def __init__(
        self,
        data_path: Path,
        sensor_blobs_path: Path,
        scene_filter: SceneFilter,
        sensor_config: SensorConfig = SensorConfig.build_no_sensors(),
    ):

        self._filtered_tokens = filter_scenes(data_path, scene_filter)
        self._sensor_blobs_path = sensor_blobs_path
        self._scene_filter = scene_filter
        self._sensor_config = sensor_config

    @property
    def tokens(self) -> List[str]:
        return list(self._filtered_tokens.keys())

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx) -> str:
        return self.tokens[idx]

    def get_scene_from_token(self, token: str) -> Scene:
        assert token in self.tokens
        return Scene.from_scene_dict_list(
            self._filtered_tokens[token],
            self._sensor_blobs_path,
            num_history_frames=self._scene_filter.num_history_frames,
            num_future_frames=self._scene_filter.num_future_frames,
            sensor_config=self._sensor_config,
        )

    def get_agent_input_from_token(self, token: str) -> AgentInput:
        assert token in self.tokens
        return AgentInput.from_scene_dict_list(
            self._filtered_tokens[token],
            self._sensor_blobs_path,
            num_history_frames=self._scene_filter.num_history_frames,
            sensor_config=self._sensor_config,
        )


class MetricCacheLoader:

    def __init__(
        self,
        cache_path: Path,
        file_name: str = "metric_cache.pkl",
    ):

        self._file_name = file_name
        self._metric_cache_paths = self._load_metric_cache_paths(cache_path)

    def _load_metric_cache_paths(self, cache_path: Path) -> Dict[str, Path]:

        # This is ugly lol
        metric_cache_dict: Dict[str, Path] = {}
        for log_path in cache_path.iterdir():
            if "metadata" in str(log_path):
                continue
            for scenario_path in log_path.iterdir():
                for token_path in scenario_path.iterdir():
                    metric_cache_path = token_path / self._file_name
                    assert (
                        metric_cache_path.is_file()
                    ), f"Metric cache at {metric_cache_path} is missing!"
                    token = str(token_path).split("/")[-1]
                    metric_cache_dict[token] = metric_cache_path

        return metric_cache_dict

    @property
    def tokens(self) -> List[str]:
        return list(self._metric_cache_paths.keys())

    def __len__(self):
        return len(self._metric_cache_paths)

    def __getitem__(self, idx: int) -> MetricCache:
        return self.get_from_token(self.tokens[idx])

    def get_from_token(self, token: str) -> MetricCache:

        with lzma.open(self._metric_cache_paths[token], "rb") as f:
            metric_cache: MetricCache = pickle.load(f)

        return metric_cache

    def to_pickle(self, path: Path) -> None:
        full_metric_cache = {}
        for token in tqdm(self.tokens):
            full_metric_cache[token] = self.get_from_token(token)
        with open(path, "wb") as f:
            pickle.dump(full_metric_cache, f)
