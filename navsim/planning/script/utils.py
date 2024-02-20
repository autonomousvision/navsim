import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from nuplan.common.utils.io_utils import safe_path_to_string
from nuplan.common.utils.file_backed_barrier import distributed_sync
from nuplan.planning.script.builders.folder_builder import build_simulation_experiment_folder
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.script.builders.main_callback_builder import build_main_multi_callback
from nuplan.planning.simulation.main_callback.multi_main_callback import MultiMainCallback
from nuplan.planning.simulation.runner.abstract_runner import AbstractRunner
from nuplan.planning.simulation.runner.executor import execute_runners
from nuplan.planning.simulation.runner.runner_report import RunnerReport
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool

from navsim.planning.script.builders.worker_pool_builder import build_worker


logger = logging.getLogger(__name__)

@dataclass
class CommonBuilder:
    """Common builder data."""

    worker: WorkerPool
    multi_main_callback: MultiMainCallback
    output_dir: Path
    profiler: Any


def update_config_for_simulation(cfg: DictConfig) -> None:
    """
    Updates the config based on some conditions.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    """
    # Make the configuration editable.
    OmegaConf.set_struct(cfg, False)
    if cfg.max_number_of_workers:
        # In case simulation is running in multi-threaded way perform the following
        # Remove the locking bottleneck
        cfg.callbacks = [callback for callback in cfg.callback.values()]

    # Save all interpolations and remove keys that were only used for interpolation and have no further use.
    OmegaConf.resolve(cfg)

    # Finalize the configuration and make it non-editable.
    OmegaConf.set_struct(cfg, True)

    # Log the final configuration after all overrides, interpolations and updates.
    if cfg.log_config:
        logger.info(f"Creating experiment: {cfg.experiment}")
        logger.info("\n" + OmegaConf.to_yaml(cfg))

def set_up_common_builder(cfg: DictConfig, profiler_name: str) -> CommonBuilder:
    """
    Set up a common builder when running simulations.
    :param cfg: Hydra configuration.
    :param profiler_name: Profiler name.
    :return A data classes with common builders.
    """
    # Build multi main callback
    multi_main_callback = build_main_multi_callback(cfg)

    # After run_simulation start
    multi_main_callback.on_run_simulation_start()

    # Update and override configs for simulation
    update_config_for_simulation(cfg=cfg)

    # Configure logger
    build_logger(cfg)

    # Construct builder
    worker = build_worker(cfg)

    # Create output storage folder
    build_simulation_experiment_folder(cfg=cfg)

    # Simulation Callbacks
    output_dir = Path(cfg.output_dir)

    return CommonBuilder(
        worker=worker,
        multi_main_callback=multi_main_callback,
        output_dir=output_dir,
        profiler=None,
    )

def set_default_path() -> None:
    """
    This function sets the default paths as environment variables if none are set.
    These can then be used by Hydra, unless the user overwrites them from the command line.
    """
    DEFAULT_DATA_ROOT = os.path.expanduser('~/nuplan/dataset')
    DEFAULT_EXP_ROOT = os.path.expanduser('~/nuplan/exp')

    if 'NUPLAN_DATA_ROOT' not in os.environ:
        logger.info(f'Setting default NUPLAN_DATA_ROOT: {DEFAULT_DATA_ROOT}')
        os.environ['NUPLAN_DATA_ROOT'] = DEFAULT_DATA_ROOT

    if 'NUPLAN_EXP_ROOT' not in os.environ:
        logger.info(f'Setting default NUPLAN_EXP_ROOT: {DEFAULT_EXP_ROOT}')
        os.environ['NUPLAN_EXP_ROOT'] = DEFAULT_EXP_ROOT

def run_runners(
    runners: List[AbstractRunner], common_builder: CommonBuilder, profiler_name: str, cfg: DictConfig
) -> None:
    """
    Run a list of runners.
    :param runners: A list of runners.
    :param common_builder: Common builder.
    :param profiler_name: Profiler name.
    :param cfg: Hydra config.
    """
    assert len(runners) > 0, 'No scenarios found to simulate!'
    if common_builder.profiler:
        # Start simulation running profiling
        common_builder.profiler.start_profiler(profiler_name)

    logger.info('Executing runners...')
    reports = execute_runners(
        runners=runners,
        worker=common_builder.worker,
        num_gpus=cfg.number_of_gpus_allocated_per_simulation,
        num_cpus=cfg.number_of_cpus_allocated_per_simulation,
        exit_on_failure=cfg.exit_on_failure,
        verbose=cfg.verbose,
    )
    logger.info('Finished executing runners!')

    # Save RunnerReports as parquet file
    save_runner_reports(reports, common_builder.output_dir, cfg.runner_report_file)

    # Sync up nodes when running distributed simulation
    distributed_sync(Path(cfg.output_dir / Path("barrier")), cfg.distributed_timeout_seconds)

    # Only run on_run_simulation_end callbacks on master node
    if int(os.environ.get('NODE_RANK', 0)) == 0:
        common_builder.multi_main_callback.on_run_simulation_end()

    # Save profiler
    if common_builder.profiler:
        common_builder.profiler.save_profiler(profiler_name)

def save_runner_reports(reports: List[RunnerReport], output_dir: Path, report_name: str) -> None:
    """
    Save runner reports to a parquet file in the output directory.
    Output directory can be local or s3.
    :param reports: Runner reports returned from each simulation.
    :param output_dir: Output directory to save the report.
    :param report_name: Report name.
    """
    report_dicts = []
    for report in map(lambda x: x.__dict__, reports):  # type: ignore
        if (planner_report := report["planner_report"]) is not None:
            planner_report_statistics = planner_report.compute_summary_statistics()
            del report["planner_report"]
            report.update(planner_report_statistics)
        report_dicts.append(report)
    df = pd.DataFrame(report_dicts)
    df['duration'] = df['end_time'] - df['start_time']

    save_path = output_dir / report_name
    df.to_parquet(safe_path_to_string(save_path))
    logger.info(f'Saved runner reports to {save_path}')