# dataloader only for private test
from __future__ import annotations

import lzma
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm

from navsim.common.dataclasses import AgentInput, Scene, SceneFilter, SensorConfig
from navsim.planning.metric_caching.metric_cache import MetricCache

FrameList = List[Dict[str, Any]]


import pickle
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def group_frames_by_sequence(file_path, sequence_length=4):

    with open(file_path, 'rb') as f:
        scene_dict_list = pickle.load(f)
    scene_groups = defaultdict(list)
    for frame in scene_dict_list:
        scene_token = frame['scene_token']
        scene_groups[scene_token].append(frame)
    
    for scene_token in scene_groups:
        scene_groups[scene_token].sort(key=lambda x: x['frame_idx'])
    
    sequence_groups = []
    for scene_token, frames in tqdm(scene_groups.items(), desc="Loading original scenes"):
        if len(frames) == 5:
            sequence_groups.append(frames[0:4])
            sequence_groups.append(frames[1:5])
    
    return sequence_groups

def filter_scenes(data_path: Path, scene_filter: SceneFilter) -> Tuple[Dict[str, FrameList], List[str]]:
    """
    Load a set of scenes from dataset, while applying scene filter configuration.
    :param data_path: root directory of log folder
    :param scene_filter: scene filtering configuration class
    :return: dictionary of raw logs format, and list of final frame tokens that can be used to filter synthetic scenes
    """

    filtered_scenes: Dict[str, Scene] = {}

    if scene_filter.tokens is not None:
        filter_tokens = True
        tokens = set(scene_filter.tokens)

    log_files = list(data_path.iterdir())
    scene_dict_list = group_frames_by_sequence(log_files[0])

    for frame_list in scene_dict_list:

        # Filter by token
        token = frame_list[scene_filter.num_history_frames - 1]["token"]
        if filter_tokens and token not in tokens:
            continue

        filtered_scenes[token] = frame_list

    return filtered_scenes



def filter_synthetic_scenes(
    data_path: Path, 
    scene_filter: SceneFilter, 
    sensor_config: SensorConfig = SensorConfig.build_no_sensors(),
    sensor_blobs_path: Path = None,
) -> Dict[str, Tuple[Path, str]]:
    # Load all the synthetic scenes that belong to the original scenes already loaded
    loaded_scenes: Dict[str, Tuple[Path, str, int]] = {}
    synthetic_scenes_paths = list(data_path.iterdir())

    filter_initial_tokens = scene_filter.reactive_synthetic_initial_tokens is not None

    for scene_path in tqdm(synthetic_scenes_paths, desc="Loading synthetic scenes"):

        with open(scene_path, "rb") as f:
            scene_dict_list = pickle.load(f)

        synthetic_scene_pre = Scene.from_scene_dict_list_private(
            scene_dict_list[:-1],
            sensor_blobs_path,
            scene_filter.num_history_frames, 
            scene_filter.num_future_frames,
            sensor_config=sensor_config)
        
        synthetic_scene_now = Scene.from_scene_dict_list_private(
            scene_dict_list[1:],
            sensor_blobs_path,
            scene_filter.num_history_frames, 
            scene_filter.num_future_frames,
            sensor_config=sensor_config)


        if filter_initial_tokens and synthetic_scene_pre.scene_metadata.initial_token not in scene_filter.reactive_synthetic_initial_tokens:
            continue

        loaded_scenes.update({synthetic_scene_pre.scene_metadata.initial_token: synthetic_scene_pre})
        loaded_scenes.update({synthetic_scene_now.scene_metadata.initial_token: synthetic_scene_now})

    return loaded_scenes


class SceneLoader:
    """Simple data loader of scenes from logs."""

    def __init__(
        self,
        data_path: Path,
        original_sensor_path: Path,
        scene_filter: SceneFilter,
        synthetic_sensor_path: Path = None,
        synthetic_scenes_path: Path = None,
        sensor_config: SensorConfig = SensorConfig.build_no_sensors(),
    ):
        """
        Initializes the scene data loader.
        :param data_path: root directory of log folder
        :param synthetic_sensor_path: root directory of sensor  (synthetic)
        :param original_sensor_path: root directory of sensor  (original)
        :param scene_filter: dataclass for scene filtering specification
        :param sensor_config: dataclass for sensor loading specification, defaults to no sensors
        """

        self.scene_frames_dicts = filter_scenes(data_path, scene_filter)
        self._synthetic_sensor_path = synthetic_sensor_path
        self._original_sensor_path = original_sensor_path
        self._scene_filter = scene_filter
        self._sensor_config = sensor_config

        if scene_filter.include_synthetic_scenes:
            assert (
                synthetic_scenes_path is not None
            ), "Synthetic scenes path cannot be None, when synthetic scenes_filter.include_synthetic_scenes is set to True."
            self.synthetic_scenes = filter_synthetic_scenes(
                data_path=synthetic_scenes_path,
                scene_filter=scene_filter,
                sensor_config=self._sensor_config,
                sensor_blobs_path=self._synthetic_sensor_path,
            )
            self.synthetic_scenes_tokens = set(self.synthetic_scenes.keys())
        else:
            self.synthetic_scenes = {}
            self.synthetic_scenes_tokens = set()

    @property
    def tokens(self) -> List[str]:
        """
        :return: list of scene identifiers for loading.
        """
        return list(self.scene_frames_dicts.keys()) + list(self.synthetic_scenes.keys())

    @property
    def tokens_stage_one(self) -> List[str]:
        """
        original scenes
        :return: list of scene identifiers for loading.
        """
        return list(self.scene_frames_dicts.keys())

    @property
    def reactive_tokens_stage_two(self) -> List[str]:
        """
        reactive synthetic scenes
        :return: list of scene identifiers for loading.
        """
        reactive_synthetic_initial_tokens = self._scene_filter.reactive_synthetic_initial_tokens
        if reactive_synthetic_initial_tokens is None:
            return None
        return list(set(self.synthetic_scenes_tokens) & set(reactive_synthetic_initial_tokens))

    @property
    def non_reactive_tokens_stage_two(self) -> List[str]:
        """
        non reactive synthetic scenes
        :return: list of scene identifiers for loading.
        """
        non_reactive_synthetic_initial_tokens = self._scene_filter.non_reactive_synthetic_initial_tokens
        if non_reactive_synthetic_initial_tokens is None:
            return None
        return list(set(self.synthetic_scenes_tokens) & set(non_reactive_synthetic_initial_tokens))

    @property
    def reactive_tokens(self) -> List[str]:
        """
        original scenes and reactive synthetic scenes
        :return: list of scene identifiers for loading.
        """
        reactive_synthetic_initial_tokens = self._scene_filter.reactive_synthetic_initial_tokens
        if reactive_synthetic_initial_tokens is None:
            return list(self.scene_frames_dicts.keys())
        return list(self.scene_frames_dicts.keys()) + list(
            set(self.synthetic_scenes_tokens) & set(reactive_synthetic_initial_tokens)
        )

    @property
    def non_reactive_tokens(self) -> List[str]:
        """
        original scenes and non reactive synthetic scenes
        :return: list of scene identifiers for loading.
        """
        non_reactive_synthetic_initial_tokens = self._scene_filter.non_reactive_synthetic_initial_tokens
        if non_reactive_synthetic_initial_tokens is None:
            return list(self.scene_frames_dicts.keys())
        return list(self.scene_frames_dicts.keys()) + list(
            set(self.synthetic_scenes_tokens) & set(non_reactive_synthetic_initial_tokens)
        )

    def __len__(self) -> int:
        """
        :return: number for scenes possible to load.
        """
        return len(self.tokens)

    def __getitem__(self, idx) -> str:
        """
        :param idx: index of scene
        :return: unique scene identifier
        """
        return self.tokens[idx]

    def get_scene_from_token(self, token: str) -> Scene:
        """
        Loads scene given a scene identifier string (token).
        :param token: scene identifier string.
        :return: scene dataclass
        """
        assert token in self.tokens
        if token in self.synthetic_scenes:
            return Scene.load_from_disk(
                file_path=self.synthetic_scenes[token][0],
                sensor_blobs_path=self._synthetic_sensor_path,
                sensor_config=self._sensor_config,
            )
        else:
            return Scene.from_scene_dict_list(
                self.scene_frames_dicts[token],
                self._original_sensor_path,
                num_history_frames=self._scene_filter.num_history_frames,
                num_future_frames=self._scene_filter.num_future_frames,
                sensor_config=self._sensor_config,
            )

    def get_agent_input_from_token(self, token: str) -> AgentInput:
        """
        Loads agent input given a scene identifier string (token).
        :param token: scene identifier string.
        :return: agent input dataclass
        """
        assert token in self.tokens
        if token in self.synthetic_scenes:
            return AgentInput.from_scene_dict_list_private(
                self.synthetic_scenes[token].frames,
                self._original_sensor_path,
                num_history_frames=self._scene_filter.num_history_frames,
                sensor_config=self._sensor_config,
            )
        else:
            return AgentInput.from_scene_dict_list(
                self.scene_frames_dicts[token],
                self._original_sensor_path,
                num_history_frames=self._scene_filter.num_history_frames,
                sensor_config=self._sensor_config,
            )

    def get_tokens_list_per_log(self) -> Dict[str, List[str]]:
        """
        Collect tokens for each logs file given filtering.
        :return: dictionary of logs names and tokens
        """
        # generate a dict that contains a list of tokens for each log-name
        tokens_per_logs: Dict[str, List[str]] = {}
        for token, scene_dict_list in self.scene_frames_dicts.items():
            log_name = scene_dict_list[0]["log_name"]
            if tokens_per_logs.get(log_name):
                tokens_per_logs[log_name].append(token)
            else:
                tokens_per_logs.update({log_name: [token]})

        for scene_path, log_name in self.synthetic_scenes.values():
            if tokens_per_logs.get(log_name):
                tokens_per_logs[log_name].append(scene_path.stem)
            else:
                tokens_per_logs.update({log_name: [scene_path.stem]})

        return tokens_per_logs


class MetricCacheLoader:
    """Simple dataloader for metric cache."""

    def __init__(self, cache_path: Path, file_name: str = "metric_cache.pkl"):
        """
        Initializes the metric cache loader.
        :param cache_path: directory of cache folder
        :param file_name: file name of cached files, defaults to "metric_cache.pkl"
        """

        self._file_name = file_name
        self.metric_cache_paths = self._load_metric_cache_paths(cache_path)

    def _load_metric_cache_paths(self, cache_path: Path) -> Dict[str, Path]:
        """
        Helper function to load all cache file paths from folder.
        :param cache_path: directory of cache folder
        :return: dictionary of token and file path
        """
        metadata_dir = cache_path / "metadata"
        metadata_file = [file for file in metadata_dir.iterdir() if ".csv" in str(file)][0]
        with open(str(metadata_file), "r") as f:
            cache_paths = f.read().splitlines()[1:]
        metric_cache_dict = {cache_path.split("/")[-2]: cache_path for cache_path in cache_paths}
        return metric_cache_dict

    @property
    def tokens(self) -> List[str]:
        """
        :return: list of scene identifiers for loading.
        """
        return list(self.metric_cache_paths.keys())

    def __len__(self):
        """
        :return: number for scenes possible to load.
        """
        return len(self.metric_cache_paths)

    def __getitem__(self, idx: int) -> MetricCache:
        """
        :param idx: index of cache to cache to load
        :return: metric cache dataclass
        """
        return self.get_from_token(self.tokens[idx])

    def get_from_token(self, token: str) -> MetricCache:
        """
        Load metric cache from scene identifier
        :param token: unique identifier of scene
        :return: metric cache dataclass
        """
        with lzma.open(self.metric_cache_paths[token], "rb") as f:
            metric_cache: MetricCache = pickle.load(f)
        return metric_cache

    def to_pickle(self, path: Path) -> None:
        """
        Dumps complete metric cache into pickle.
        :param path: directory of cache folder
        """
        full_metric_cache = {}
        for token in tqdm(self.tokens):
            full_metric_cache[token] = self.get_from_token(token)
        with open(path, "wb") as f:
            pickle.dump(full_metric_cache, f)
