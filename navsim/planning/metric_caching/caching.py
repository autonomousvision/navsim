import gc
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from hydra.utils import instantiate
from nuplan.planning.training.experiments.cache_metadata_entry import (
    CacheMetadataEntry,
    CacheResult,
    save_cache_metadata,
)
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.utils.multithreading.worker_utils import worker_map
from omegaconf import DictConfig

from navsim.common.dataclasses import Scene, SensorConfig
from navsim.common.dataloader import SceneFilter, SceneLoader
from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario

logger = logging.getLogger(__name__)


def cache_scenarios(
    args: List[Dict[str, Union[List[str], DictConfig]]]
) -> List[CacheResult]:
    """
    Performs the caching of scenario DB files in parallel.
    :param args: A list of dicts containing the following items:
        "scenario": the scenario as built by scenario_builder
        "cfg": the DictConfig to use to process the file.
    :return: A dict with the statistics of the job. Contains the following keys:
        "successes": The number of successfully processed scenarios.
        "failures": The number of scenarios that couldn't be processed.
    """

    # Define a wrapper method to help with memory garbage collection.
    # This way, everything will go out of scope, allowing the python GC to clean up after the function.
    #
    # This is necessary to save memory when running on large datasets.
    def cache_scenarios_internal(args: List[Dict[str, Union[Path, DictConfig]]]) -> List[CacheResult]:
        def cache_single_scenario(
            scene_dict: Dict[str, Any], processor: MetricCacheProcessor
        ) -> Optional[CacheMetadataEntry]:
            scene = Scene.from_scene_dict_list(
                scene_dict,
                None,
                num_history_frames=cfg.train_test_split.scene_filter.num_history_frames,
                num_future_frames=cfg.train_test_split.scene_filter.num_future_frames,
                sensor_config=SensorConfig.build_no_sensors(),
            )
            scenario = NavSimScenario(
                scene,
                map_root=os.environ["NUPLAN_MAPS_ROOT"],
                map_version="nuplan-maps-v1.0",
            )

            return processor.compute_and_save_metric_cache(scenario)

        def cache_single_synthetic_scenario(
            scene_path: Path, processor: MetricCacheProcessor
        ) -> Optional[CacheMetadataEntry]:
            scene = Scene.load_from_disk(scene_path, None, SensorConfig.build_no_sensors())
            scenario = NavSimScenario(scene, map_root=os.environ["NUPLAN_MAPS_ROOT"], map_version="nuplan-maps-v1.0")

            return processor.compute_and_save_metric_cache(scenario)

        node_id = int(os.environ.get("NODE_RANK", 0))
        thread_id = str(uuid.uuid4())

        log_names = [a["log_file"] for a in args]
        tokens = [t for a in args for t in a["tokens"]]
        cfg: DictConfig = args[0]["cfg"]

        scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
        scene_filter.log_names = log_names
        scene_filter.tokens = tokens
        scene_loader = SceneLoader(
            synthetic_sensor_path=None,
            original_sensor_path=None,
            data_path=Path(cfg.navsim_log_path),
            synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
            scene_filter=scene_filter,
            sensor_config=SensorConfig.build_no_sensors(),
        )

        # Create feature preprocessor
        assert (
            cfg.metric_cache_path is not None
        ), f"Cache path cannot be None when caching, got {cfg.metric_cache_path}"

        processor = MetricCacheProcessor(
            cache_path=cfg.metric_cache_path,
            force_feature_computation=cfg.force_feature_computation,
            proposal_sampling=instantiate(cfg.proposal_sampling),
        )

        logger.info(
            f"Extracted {len(scene_loader)} scenarios for thread_id={thread_id}, node_id={node_id}."
        )
        num_failures = 0
        num_successes = 0
        all_file_cache_metadata: List[Optional[CacheMetadataEntry]] = []
        for idx, scene_dict in enumerate(scene_loader.scene_frames_dicts.values()):
            logger.info(
                f"Processing scenario {idx + 1} / {len(scene_loader.scene_frames_dicts)} in thread_id={thread_id}, node_id={node_id}"
            )
            file_cache_metadata = cache_single_scenario(scene_dict, processor)
            gc.collect()

            num_failures += 0 if file_cache_metadata else 1
            num_successes += 1 if file_cache_metadata else 0
            all_file_cache_metadata += [file_cache_metadata]

        for idx, (scene_path, _) in enumerate(scene_loader.synthetic_scenes.values()):
            logger.info(
                f"Processing synthetic scenario {idx + 1} / {len(scene_loader.synthetic_scenes)} in thread_id={thread_id}, node_id={node_id}"
            )
            file_cache_metadata = cache_single_synthetic_scenario(scene_path, processor)
            gc.collect()

            num_failures += 0 if file_cache_metadata else 1
            num_successes += 1 if file_cache_metadata else 0
            all_file_cache_metadata += [file_cache_metadata]

        logger.info(
            f"Finished processing scenarios for thread_id={thread_id}, node_id={node_id}"
        )
        return [
            CacheResult(
                failures=num_failures,
                successes=num_successes,
                cache_metadata=all_file_cache_metadata,
            )
        ]

    result = cache_scenarios_internal(args)

    # Force a garbage collection to clean up any unused resources
    gc.collect()

    return result


def cache_data(cfg: DictConfig, worker: WorkerPool) -> None:
    """
    Build the lightning datamodule and cache all samples.
    :param cfg: omegaconf dictionary
    :param worker: Worker to submit tasks which can be executed in parallel
    """
    assert (
        cfg.metric_cache_path is not None
    ), f"Cache path cannot be None when caching, got {cfg.metric_cache_path}"

    # Extract scenes based on scene-loader to know which tokens to distribute across workers
    # TODO: infer the tokens per log from metadata, to not have to load metric cache and scenes here
    scene_loader = SceneLoader(
        synthetic_sensor_path=None,
        original_sensor_path=None,
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )

    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]
    logger.info("Starting metric caching of %s files...", str(len(data_points)))

    cache_results = worker_map(worker, cache_scenarios, data_points)

    num_success = sum(result.successes for result in cache_results)
    num_fail = sum(result.failures for result in cache_results)
    num_total = num_success + num_fail
    if num_fail == 0:
        logger.info(
            "Completed dataset caching! All %s features and targets were cached successfully.",
            str(num_total),
        )
    else:
        logger.info(
            "Completed dataset caching! Failed features and targets: %s out of %s",
            str(num_fail),
            str(num_total),
        )

    cached_metadata = [
        cache_metadata_entry
        for cache_result in cache_results
        for cache_metadata_entry in cache_result.cache_metadata
        if cache_metadata_entry is not None
    ]

    node_id = int(os.environ.get("NODE_RANK", 0))
    logger.info(
        f"Node {node_id}: Storing metadata csv file containing cache paths for valid features and targets..."
    )
    save_cache_metadata(cached_metadata, Path(cfg.metric_cache_path), node_id)
    logger.info("Done storing metadata csv file.")
