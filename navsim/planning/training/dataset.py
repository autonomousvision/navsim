from typing import Dict, List, Tuple
import torch

from navsim.common.dataloader import SceneLoader
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        scene_loader: SceneLoader,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder]
    ):
        super().__init__()
        self._scene_loader = scene_loader
        self._feature_builders = feature_builders
        self._target_builders = target_builders

    def __len__(self):
        return len(self._scene_loader)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        scene = self._scene_loader.get_scene_from_token(self._scene_loader.tokens[idx])
        features: Dict[str, torch.Tensor] = {}
        for builder in self._feature_builders:
            features.update(builder.compute_features(scene.get_agent_input()))
        targets: Dict[str, torch.Tensor] = {}
        for builder in self._target_builders:
            targets.update(builder.compute_targets(scene))
        return (features, targets)