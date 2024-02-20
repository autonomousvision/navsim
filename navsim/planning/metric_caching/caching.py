import gc
import itertools
import logging
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union


import pickle

from omegaconf import DictConfig

from nuplan.common.utils.distributed_scenario_filter import (
    DistributedMode,
    DistributedScenarioFilter,
)
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.abstract_scenario_builder import (
    AbstractScenarioBuilder,
    RepartitionStrategy,
)
from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
from nuplan.planning.script.builders.scenario_filter_builder import build_scenario_filter
from nuplan.planning.training.experiments.cache_metadata_entry import (
    CacheMetadataEntry,
    CacheResult,
    save_cache_metadata,
)
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.utils.multithreading.worker_utils import chunk_list, worker_map

from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter

logger = logging.getLogger(__name__)


def build_scenarios_from_config(
    cfg: DictConfig, scenario_builder: AbstractScenarioBuilder, worker: WorkerPool
) -> List[AbstractScenario]:
    """
    Build scenarios from config file.
    :param cfg: Omegaconf dictionary
    :param scenario_builder: Scenario builder.
    :param worker: Worker to submit tasks which can be executed in parallel
    :return: A list of scenarios
    """
    scenario_filter = build_scenario_filter(cfg.scenario_filter)
    return scenario_builder.get_scenarios(scenario_filter, worker)  # type: ignore


def cache_scenarios(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[CacheResult]:
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
    def cache_scenarios_internal(
        args: List[Dict[str, Union[List[AbstractScenario], DictConfig]]]
    ) -> List[CacheResult]:
        node_id = int(os.environ.get("NODE_RANK", 0))
        thread_id = str(uuid.uuid4())

        scenarios: List[AbstractScenario] = [a["scenario"] for a in args]
        cfg: DictConfig = args[0]["cfg"]

        # Create feature preprocessor
        assert (
            cfg.cache.cache_path is not None
        ), f"Cache path cannot be None when caching, got {cfg.cache.cache_path}"

        processor = MetricCacheProcessor(
            cache_path=cfg.cache.cache_path,
            force_feature_computation=cfg.cache.force_feature_computation,
        )

        logger.info(
            f"Extracted {len(scenarios)} scenarios for thread_id={thread_id}, node_id={node_id}."
        )
        num_failures = 0
        num_successes = 0
        all_file_cache_metadata: List[Optional[CacheMetadataEntry]] = []
        for idx, scenario in enumerate(scenarios):
            logger.info(
                f"Processing scenario {idx + 1} / {len(scenarios)} in thread_id={thread_id}, node_id={node_id}"
            )

            file_cache_metadata = processor.compute_metric_cache(scenario)

            num_failures += 0 if file_cache_metadata else 1
            num_successes += 1 if file_cache_metadata else 0
            all_file_cache_metadata += [file_cache_metadata]

        logger.info(f"Finished processing scenarios for thread_id={thread_id}, node_id={node_id}")
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
        cfg.cache.cache_path is not None
    ), f"Cache path cannot be None when caching, got {cfg.cache.cache_path}"

    if not cfg.cache.use_open_scene:
        scenario_builder = build_scenario_builder(cfg)
        if int(os.environ.get("NUM_NODES", 1)) > 1 and cfg.distribute_by_scenario:
            # Partition differently based on how the scenario builder loads the data
            repartition_strategy = scenario_builder.repartition_strategy
            if repartition_strategy == RepartitionStrategy.REPARTITION_FILE_DISK:
                scenario_filter = DistributedScenarioFilter(
                    cfg=cfg,
                    worker=worker,
                    node_rank=int(os.environ.get("NODE_RANK", 0)),
                    num_nodes=int(os.environ.get("NUM_NODES", 1)),
                    synchronization_path=cfg.cache.cache_path,
                    timeout_seconds=cfg.get("distributed_timeout_seconds", 3600),
                    distributed_mode=cfg.get("distributed_mode", DistributedMode.LOG_FILE_BASED),
                )
                scenarios = scenario_filter.get_scenarios()
            elif repartition_strategy == RepartitionStrategy.INLINE:
                scenarios = build_scenarios_from_config(cfg, scenario_builder, worker)
                num_nodes = int(os.environ.get("NUM_NODES", 1))
                node_id = int(os.environ.get("NODE_RANK", 0))
                scenarios = chunk_list(scenarios, num_nodes)[node_id]
            else:
                expected_repartition_strategies = [e.value for e in RepartitionStrategy]
                raise ValueError(
                    f"Expected repartition strategy to be in {expected_repartition_strategies}, got {repartition_strategy}."
                )
        else:
            logger.debug(
                "Building scenarios without distribution, if you're running on a multi-node system, make sure you aren't"
                "accidentally caching each scenario multiple times!"
            )
            scenarios = build_scenarios_from_config(cfg, scenario_builder, worker)
    else:
        
        NUPLAN_MAPS_ROOT = os.environ["NUPLAN_MAPS_ROOT"]
        NUPLAN_MAP_VERSION = "nuplan-maps-v1.0"

        scenarios: List[AbstractScenario] = []
        
        # TODO: add scene filter settings to config
        scene_loader = SceneLoader(
            data_path=Path(cfg.navsim_log_path),
            scene_filter=SceneFilter(),
            sensor_modalities=[],
        )

        for idx in range(len(scene_loader)):
            scene = scene_loader[idx]

            scenario = NavSimScenario(
                scene, map_root=NUPLAN_MAPS_ROOT, map_version=NUPLAN_MAP_VERSION
            )
            scenarios.append(scenario)

    data_points = [{"scenario": scenario, "cfg": cfg} for scenario in scenarios]
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
    save_cache_metadata(cached_metadata, Path(cfg.cache.cache_path), node_id)
    logger.info("Done storing metadata csv file.")
