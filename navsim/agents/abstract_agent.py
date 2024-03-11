from __future__ import annotations

import abc

from abc import abstractmethod
from typing import Any, List

from navsim.common.dataclasses import AgentInput, Trajectory


class AbstractAgent(abc.ABC):
    """
    Interface for a generic end-to-end agent.
    """
    requires_scene = False

    def __new__(cls, *args: Any, **kwargs: Any) -> AbstractAgent:
        """
        Define attributes needed by all agents, take care when overriding.
        :param cls: class being constructed.
        :param args: arguments to constructor.
        :param kwargs: keyword arguments to constructor.
        """
        instance: AbstractAgent = super().__new__(cls)
        instance._compute_trajectory_runtimes = []
        return instance

    @abstractmethod
    def name(self) -> str:
        """
        :return: string describing name of this agent.
        """
        pass
    
    @abstractmethod
    def get_sensor_modalities(self) -> List[str]:
        """
        :return: List of strings describing the required sensors, e.g. ["lidar", "camera"].
        """
        pass

    @abc.abstractmethod
    def initialize(self) -> None:
        """
        Initialize agent
        :param initialization: Initialization class.
        """
        pass

    @abc.abstractmethod
    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        """
        Computes the ego vehicle trajectory.
        :param current_input: Dataclass with agent inputs.
        :return: Trajectory representing the predicted ego's position in future
        """
        pass
