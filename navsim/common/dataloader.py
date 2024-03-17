from __future__ import annotations

import lzma
import pickle

from pathlib import Path
from typing import Any, Dict, List
from tqdm import tqdm

from navsim.common.dataclasses import Scene, AgentInput, SceneFilter
from navsim.planning.metric_caching.metric_cache import MetricCache


def filter_scenes(
    data_path: Path, sensor_blobs_path: Path, scene_filter: SceneFilter, sensor_modalities: List[str] = ["lidar", "camera"]
) -> Dict[str, Scene]:

    def split_list(input_list: List[Any], n: int) -> List[List[Any]]:
        return [input_list[i : i + n] for i in range(0, len(input_list), n)]

    filtered_scenes: Dict[str, Scene] = {}
    stop_loading: bool = False

    for log_pickle_path in tqdm(list(data_path.iterdir()), desc="Loading logs"):
        
        scene_dict_list = pickle.load(open(log_pickle_path, "rb"))
        for frame_list in split_list(scene_dict_list, scene_filter.num_frames):
            # Filter scenes which are too short
            if len(frame_list) < scene_filter.num_frames:
                continue

            # Filter scenes with no route
            if scene_filter.has_route and len(frame_list[0]["roadblock_ids"]) == 0:
                continue

            # TODO: Filter by token
            # TODO: Implement temporally overlapping scenes
            token = frame_list[scene_filter.num_history_frames - 1]["token"]
            filtered_scenes[token] = Scene.from_scene_dict_list(
                frame_list,
                sensor_blobs_path,
                num_history_frames=scene_filter.num_history_frames,
                num_future_frames=scene_filter.num_future_frames,
                sensor_modalities=sensor_modalities,
            )

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
        scene_filter: SceneFilter = SceneFilter(),
        sensor_modalities: List[str] = ["lidar", "camera"],
    ):

        self._filtered_scenes = filter_scenes(data_path, sensor_blobs_path, scene_filter, sensor_modalities)
        self._scene_filter = scene_filter
        self._sensor_modalities = sensor_modalities

    @property
    def tokens(self) -> List[str]:
        return list(self._filtered_scenes.keys())

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx) -> Scene:
        token = self.tokens[idx]
        return self.get_from_token(token)

    def get_from_token(self, token: str) -> Scene:
        assert token in self.tokens
        return self._filtered_scenes[token]


class AgentInputLoader:

    def __init__(
        self,
        data_path: Path,
        sensor_blobs_path: Path,
        scene_filter: SceneFilter = SceneFilter(),
        sensor_modalities: List[str] = ["lidar", "camera"],
    ):

        self._filtered_scenes = filter_scenes(data_path, sensor_blobs_path, scene_filter, sensor_modalities)
        self._scene_filter = scene_filter
        self._sensor_modalities = sensor_modalities

    @property
    def tokens(self) -> List[str]:
        return list(self._filtered_scenes.keys())

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx) -> AgentInput:
        token = self.tokens[idx]
        return self.get_from_token(token)

    def get_from_token(self, token: str) -> AgentInput:
        assert token in self.tokens
        return self._filtered_scenes[token].get_agent_input(self._sensor_modalities)


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
        return len(self._pickle_dict_keys)

    def __getitem__(self, idx: int) -> MetricCache:
        return self.get_from_token(self.tokens[idx])

    def get_from_token(self, token: str) -> MetricCache:

        with lzma.open(self._metric_cache_paths[token], "rb") as f:
            metric_cache: MetricCache = pickle.load(f)

        return metric_cache
