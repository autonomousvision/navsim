from abc import abstractmethod, ABC
from typing import Dict, Union, List
import torch
import pytorch_lightning as pl
from typing import Tuple
import numpy as np
from navsim.common.dataclasses import AgentInput, Trajectory, SensorConfig, Scene
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder


class AbstractAgent(torch.nn.Module, ABC):
    """Interface for an agent in NAVSIM."""

    def __init__(
        self,
        requires_scene: bool = False,
    ):
        super().__init__()
        self.requires_scene = requires_scene

    @abstractmethod
    def name(self) -> str:
        """
        :return: string describing name of this agent.
        """
        pass

    @abstractmethod
    def get_sensor_config(self) -> SensorConfig:
        """
        :return: Dataclass defining the sensor configuration for lidar and cameras.
        """
        pass

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize agent
        :param initialization: Initialization class.
        """
        pass

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the agent.
        :param features: Dictionary of features.
        :return: Dictionary of predictions.
        """
        raise NotImplementedError

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """
        :return: List of target builders.
        """
        raise NotImplementedError("No feature builders. Agent does not support training.")

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """
        :return: List of feature builders.
        """
        raise NotImplementedError("No target builders. Agent does not support training.")

    def compute_trajectory(self, agent_input: AgentInput) -> Tuple[np.ndarray,np.ndarray]:
        """
        Computes the ego vehicle trajectory.
        :param current_input: Dataclass with agent inputs.
        :return: Trajectory representing the predicted ego's position in future
        """
        self.eval()
        features: Dict[str, torch.Tensor] = {}
        # targets: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        # for builder in self.get_target_builders():
            # targets.update(builder.compute_targets(scene))

        
        # "trajectory": trajectory,
        # "agent_states": agent_states,
        # "agent_labels": agent_labels,
        # "bev_semantic_map": bev_semantic_map,`
        
        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}
        # targets = {k: v.unsqueeze(0) for k, v in targets.items()}
        # print(targets)
        # print('<<<<<<<<<<<<<<<')
        # poses1 = targets["trajectory"].squeeze(0).numpy()
        # forward pass
        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions['trajectory'].squeeze(0).numpy()# 20 8 3     20 64 8 3
            anchor_poses = predictions['anchor_trajectories'].squeeze(0).numpy()  # 确保转换为numpy
        # extract trajectory
        # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        # print(anchor_trajectories.shape)
        return poses,anchor_poses
        # return Trajectory(poses)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Computes the loss used for backpropagation based on the features, targets and model predictions.
        """
        raise NotImplementedError("No loss. Agent does not support training.")

    def get_optimizers(
        self,
    ) -> Union[torch.optim.Optimizer, Dict[str, Union[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]]]:
        """
        Returns the optimizers that are used by thy pytorch-lightning trainer.
        Has to be either a single optimizer or a dict of optimizer and lr scheduler.
        """
        raise NotImplementedError("No optimizers. Agent does not support training.")

    def get_training_callbacks(self) -> List[pl.Callback]:
        """
        Returns a list of pytorch-lightning callbacks that are used during training.
        See navsim.planning.training.callbacks for examples.
        """
        return []