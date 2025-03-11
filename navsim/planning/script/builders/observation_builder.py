from typing import cast

from hydra.utils import instantiate
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.abstract_observation import AbstractObservation
from omegaconf import DictConfig


def build_observations(observation_cfg: DictConfig, scenario: AbstractScenario) -> AbstractObservation:
    """
    Instantiate observations
    :param observation_cfg: config of a planner
    :param scenario: scenario
    :return AbstractObservation
    """
    observation = cast(AbstractObservation, instantiate(observation_cfg, scenario=scenario))
    return observation
