import logging
import os
from pathlib import Path
from typing import List, Union, Optional
from omegaconf import DictConfig, OmegaConf
from shutil import rmtree
import random
import numpy as np
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from nuplan.common.utils.s3_utils import is_s3_path
from nuplan.planning.script.builders.utils.utils_type import is_target_type, validate_type
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool, WorkerResources
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.utils.multithreading.worker_sequential import Sequential
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.callback.simulation_log_callback import SimulationLogCallback
from nuplan.planning.simulation.callback.metric_callback import MetricCallback
from nuplan.planning.simulation.callback.serialization_callback import SerializationCallback
from nuplan.planning.simulation.callback.abstract_callback import AbstractCallback

from navsim.planning.script.utils import run_runners, set_default_path, set_up_common_builder
from navsim.planning.script.builders.simulation_builder import build_simulations



def build_simulation_callbacks(
    cfg: DictConfig, output_dir: Path, worker: Optional[WorkerPool] = None
) -> List[AbstractCallback]:
    """
    Builds callback.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :param output_dir: directory for all experiment results.
    :param worker: to run certain callbacks in the background (everything runs in main process if None).
    :return: List of callbacks.
    """
    logger.info('Building AbstractCallback...')
    callbacks = []
    for config in cfg.callback.values():
        if is_target_type(config, SerializationCallback):
            callback: SerializationCallback = instantiate(config, output_directory=output_dir)
        
        elif is_target_type(config, SimulationLogCallback) or is_target_type(config, MetricCallback):
            # SimulationLogCallback and MetricCallback store state (futures) from each runner, so they are initialized
            # in the simulation builder
            continue
        else:
            callback = instantiate(config)
        validate_type(callback, AbstractCallback)
        callbacks.append(callback)
    logger.info(f'Building AbstractCallback: {len(callbacks)}...DONE!')
    return callbacks

def build_callbacks_worker(cfg: DictConfig) -> Optional[WorkerPool]:
    """
    Builds workerpool for callbacks.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return: Workerpool, or None if we'll run without one.
    """
    if not is_target_type(cfg.worker, Sequential) or cfg.disable_callback_parallelization:
        return None

    if cfg.number_of_cpus_allocated_per_simulation not in [None, 1]:
        raise ValueError("Expected `number_of_cpus_allocated_per_simulation` to be set to 1 with Sequential worker.")

    max_workers = min(
        WorkerResources.current_node_cpu_count() - (cfg.number_of_cpus_allocated_per_simulation or 1),
        cfg.max_callback_workers,
    )
    callbacks_worker_pool = SingleMachineParallelExecutor(use_process_pool=True, max_workers=max_workers)
    return callbacks_worker_pool

def run_simulation(cfg: DictConfig, planners: Optional[Union[AbstractPlanner, List[AbstractPlanner]]] = None) -> None:
    """
    Execute all available challenges simultaneously on the same scenario. Helper function for main to allow planner to
    be specified via config or directly passed as argument.
    :param cfg: Configuration that is used to run the experiment.
        Already contains the changes merged from the experiment's config to default config.
    :param planners: Pre-built planner(s) to run in simulation. Can either be a single planner or list of planners.
    """
    # Fix random seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    profiler_name = 'building_simulation'
    common_builder = set_up_common_builder(cfg=cfg, profiler_name=profiler_name)

    # Build simulation callbacks
    callbacks_worker_pool = build_callbacks_worker(cfg)

    callbacks = build_simulation_callbacks(cfg=cfg, output_dir=common_builder.output_dir, worker=callbacks_worker_pool)

    # Remove planner from config to make sure run_simulation does not receive multiple planner specifications.
    if planners and 'planner' in cfg.keys():
        logger.info('Using pre-instantiated planner. Ignoring planner in config')
        OmegaConf.set_struct(cfg, False)
        cfg.pop('planner')
        OmegaConf.set_struct(cfg, True)

    # Construct simulations
    if isinstance(planners, AbstractPlanner):
        planners = [planners]

    runners = build_simulations(
        cfg=cfg,
        callbacks=callbacks,
        worker=common_builder.worker,
        pre_built_planners=planners,
        callbacks_worker=callbacks_worker_pool,
    )

    if common_builder.profiler:
        # Stop simulation construction profiling
        common_builder.profiler.save_profiler(profiler_name)

    logger.info('Running simulation...')
    run_runners(runners=runners, common_builder=common_builder, cfg=cfg, profiler_name='running_simulation')
    logger.info('Finished running simulation!')

def clean_up_s3_artifacts() -> None:
    """
    Cleanup lingering s3 artifacts that are written locally.
    This happens because some minor write-to-s3 functionality isn't yet implemented.
    """
    # Lingering artifacts get written locally to a 's3:' directory. Hydra changes
    # the working directory to a subdirectory of this, so we serach the working
    # path for it.
    working_path = os.getcwd()
    s3_dirname = "s3:"
    s3_ind = working_path.find(s3_dirname)
    if s3_ind != -1:
        local_s3_path = working_path[: working_path.find(s3_dirname) + len(s3_dirname)]
        rmtree(local_s3_path)



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# If set, use the env. variable to overwrite the default dataset and experiment paths
set_default_path()

# If set, use the env. variable to overwrite the Hydra config
CONFIG_PATH = os.getenv('NUPLAN_HYDRA_CONFIG_PATH', 'config/simulation')

if os.environ.get('NUPLAN_HYDRA_CONFIG_PATH') is not None:
    CONFIG_PATH = os.path.join('../../../../', CONFIG_PATH)

if os.path.basename(CONFIG_PATH) != 'simulation':
    CONFIG_PATH = os.path.join(CONFIG_PATH, 'simulation')
CONFIG_NAME = 'default_simulation'

@hydra.main(config_path="config/simulation", config_name="default_simulation")
def main(cfg: DictConfig=None) -> None:
    """
    Execute all available challenges simultaneously on the same scenario. Calls run_simulation to allow planner to
    be specified via config or directly passed as argument.
    :param cfg: Configuration that is used to run the experiment.
        Already contains the changes merged from the experiment's config to default config.
    """
    # assert cfg.simulation_log_main_path is None, 'Simulation_log_main_path must not be set when running simulation.'
    # Execute simulation with preconfigured planner(s).
    run_simulation(cfg=cfg)

    if is_s3_path(Path(cfg.output_dir)):
        clean_up_s3_artifacts()

if __name__ == '__main__':
    main()
