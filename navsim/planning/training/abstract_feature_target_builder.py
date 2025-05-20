from abc import abstractmethod
from typing import Dict

from torch import Tensor

from navsim.common.dataclasses import AgentInput, Scene


class AbstractFeatureBuilder:
    """Abstract class of feature builder for agent training."""

    def __init__(self):
        pass

    @abstractmethod
    def get_unique_name(self) -> str:
        """
        :return: Unique name of created feature.
        """

    @abstractmethod
    def compute_features(self, agent_input: AgentInput) -> Dict[str, Tensor]:
        """
        Computes features from the AgentInput object, i.e., without access to ground-truth.
        Outputs a dictionary where each item has a unique identifier and maps to a single feature tensor.
        One FeatureBuilder can return a dict with multiple FeatureTensors.
        """


class AbstractTargetBuilder:
    def __init__(self):
        pass

    @abstractmethod
    def get_unique_name(self) -> str:
        """
        :return: Unique name of created target.
        """

    @abstractmethod
    def compute_targets(self, scene: Scene) -> Dict[str, Tensor]:
        """
        Computes targets from the Scene object, i.e., with access to ground-truth.
        Outputs a dictionary where each item has a unique identifier and maps to a single target tensor.
        One TargetBuilder can return a dict with multiple TargetTensors.
        """
        pass
