import logging
import os
from typing import List, Optional

from hydra.utils import instantiate
from nuplan.common.utils.distributed_scenario_filter import DistributedMode, DistributedScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.script.builders.metric_builder import build_metrics_engines
from nuplan.planning.script.builders.utils.utils_type import is_target_type
from nuplan.planning.simulation.callback.abstract_callback import AbstractCallback
from nuplan.planning.simulation.callback.metric_callback import MetricCallback
from nuplan.planning.simulation.callback.multi_callback import MultiCallback
from nuplan.planning.simulation.controller.abstract_controller import AbstractEgoController
from nuplan.planning.simulation.observation.abstract_observation import AbstractObservation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.runner.simulations_runner import SimulationRunner
from nuplan.planning.simulation.simulation import Simulation
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.simulation.simulation_time_controller.abstract_simulation_time_controller import (
    AbstractSimulationTimeController,
)
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from omegaconf import DictConfig

from navsim.planning.script.builders.observation_builder import build_observations
from navsim.planning.script.builders.planner_builder import build_planners

logger = logging.getLogger(__name__)


def build_simulations(
    cfg: DictConfig,
    worker: WorkerPool,
    callbacks: List[AbstractCallback],
    callbacks_worker: Optional[WorkerPool] = None,
    pre_built_planners: Optional[List[AbstractPlanner]] = None,
) -> List[SimulationRunner]:
    """
    Build simulations.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :param callbacks: Callbacks for simulation.
    :param worker: Worker for job execution.
    :param callbacks_worker: worker pool to use for callbacks from sim
    :param pre_built_planners: List of pre-built planners to run in simulation.
    :return A dict of simulation engines with challenge names.
    """
    logger.info("Building simulations...")

    # Create Simulation object container
    simulations = list()

    # Retrieve scenarios

    logger.info("Extracting scenarios...")

    # Only allow simulation with NuPlanScenarioBuilder except when the NUPLAN_SIMULATION_ALLOW_ANY_BUILDER environment variable is set to a non-zero value.
    if not int(os.environ.get("NUPLAN_SIMULATION_ALLOW_ANY_BUILDER", "0")) and not is_target_type(
        cfg.scenario_builder, NuPlanScenarioBuilder
    ):
        raise ValueError(f"Simulation framework only runs with NuPlanScenarioBuilder. Got {cfg.scenario_builder}")

    scenario_filter = DistributedScenarioFilter(
        cfg=cfg,
        worker=worker,
        node_rank=int(os.environ.get("NODE_RANK", 0)),
        num_nodes=int(os.environ.get("NUM_NODES", 1)),
        synchronization_path=cfg.output_dir,
        timeout_seconds=cfg.distributed_timeout_seconds,
        distributed_mode=DistributedMode[cfg.distributed_mode],
    )
    scenarios = scenario_filter.get_scenarios()

    metric_engines_map = {}
    if cfg.run_metric:
        logger.info("Building metric engines...")
        metric_engines_map = build_metrics_engines(cfg=cfg, scenarios=scenarios)
        logger.info("Building metric engines...DONE")
    else:
        logger.info("Metric engine is disable")

    logger.info("Building simulations from %d scenarios...", len(scenarios))

    # Build a metric metadata file
    for scenario in scenarios:

        # Build planners
        if pre_built_planners is None:
            if "planner" not in cfg.keys():
                raise KeyError('Planner not specified in config. Please specify a planner using "planner" field.')

            planners = build_planners(cfg.planner, scenario)
        else:
            planners = pre_built_planners

        for planner in planners:
            # Ego Controller
            ego_controller: AbstractEgoController = instantiate(cfg.ego_controller, scenario=scenario)

            # Simulation Manager
            simulation_time_controller: AbstractSimulationTimeController = instantiate(
                cfg.simulation_time_controller, scenario=scenario
            )

            # Perception
            observations: AbstractObservation = build_observations(cfg.observation, scenario=scenario)

            # Metric Engine
            metric_engine = metric_engines_map.get(scenario.scenario_type, None)
            if metric_engine is not None:
                stateful_callbacks = [MetricCallback(metric_engine=metric_engine, worker_pool=callbacks_worker)]
            else:
                stateful_callbacks = []

            if "simulation_log_callback" in cfg.callback:
                stateful_callbacks.append(
                    instantiate(cfg.callback["simulation_log_callback"], worker_pool=callbacks_worker)
                )

            # Construct simulation and manager
            simulation_setup = SimulationSetup(
                time_controller=simulation_time_controller,
                observations=observations,
                ego_controller=ego_controller,
                scenario=scenario,
            )

            simulation = Simulation(
                simulation_setup=simulation_setup,
                callback=MultiCallback(callbacks + stateful_callbacks),
                simulation_history_buffer_duration=cfg.simulation_history_buffer_duration,
            )
            simulations.append(SimulationRunner(simulation, planner))

    logger.info("Building simulations...DONE!")
    return simulations
