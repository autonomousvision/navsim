from typing import List, Optional, Type, cast

from hydra._internal.utils import _locate
from hydra.utils import instantiate
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from omegaconf import DictConfig


def _build_planner(planner_cfg: DictConfig, scenario: Optional[AbstractScenario]) -> AbstractPlanner:
    """
    Instantiate planner
    :param planner_cfg: config of a planner
    :param scenario: scenario
    :return AbstractPlanner
    """
    config = planner_cfg.copy()

    planner_cls: Type[AbstractPlanner] = _locate(config._target_)

    if planner_cls.requires_scenario:
        assert scenario is not None, (
            "Scenario was not provided to build the planner. " f"Planner {config} can not be build!"
        )
        planner = cast(AbstractPlanner, instantiate(config, scenario=scenario))
    else:
        planner = cast(AbstractPlanner, instantiate(config))

    return planner


def build_planners(planner_cfg: DictConfig, scenario: Optional[AbstractScenario]) -> List[AbstractPlanner]:
    """
    Instantiate multiple planners by calling build_planner
    :param planners_cfg: planners config
    :param scenario: scenario
    :return planners: List of AbstractPlanners
    """
    return [_build_planner(planner, scenario) for planner in planner_cfg.values()]
