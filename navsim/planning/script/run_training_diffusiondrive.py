import logging
from pathlib import Path
from typing import Tuple

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent_diffusiondrive import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name
            for log_name in train_scene_filter.log_names
            if log_name in cfg.train_logs
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [
            log_name
            for log_name in val_scene_filter.log_names
            if log_name in cfg.val_logs
        ]
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)
    navsim_blobs_path = Path(cfg.navsim_blobs_path)
    synthetic_scenes_path = Path(cfg.synthetic_scenes_path)

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        navsim_blobs_path=navsim_blobs_path,
        data_path=data_path,
        synthetic_scenes_path=synthetic_scenes_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        navsim_blobs_path=navsim_blobs_path,
        data_path=data_path,
        synthetic_scenes_path=synthetic_scenes_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """
    # print(">>>>>")
    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)
    # import pdb;pdb.set_trace()
    # print("dzx")
    logger.info("Building Lightning Module")
    # if cfg.agent.checkpoint_path:
    #     lightning_module = AgentLightningModule.load_from_checkpoint(
    #         cfg.agent.checkpoint_path,agent=agent
    #     )
    # else:    
    lightning_module = AgentLightningModule(
        agent=agent,
    )
    # import pdb;pdb.set_trace()
    # print("dzx")
    # print("dzx")    
    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    logger.info("Building Datasets")
    train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=True)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False)
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=agent.get_training_callbacks())

    # from pytorch_lightning.strategies import DDPStrategy
    # trainer = pl.Trainer(
    #     **{
    #         **cfg.trainer.params,
    #         "strategy": DDPStrategy(find_unused_parameters=True),
    #     },
    #     callbacks=agent.get_training_callbacks()
    # )
    # trainer = pl.Trainer(
    #     **cfg.trainer.params,callbacks=agent.get_training_callbacks()
    # )
    logger.info("Starting Training")
    # import pdb;pdb.set_trace()
    
    # checkpoint_path = "/data/hdd01/dingzx/navsim_exp/training_diffusiondrive_agent/2025.04.30.20.01.08/lightning_logs/version_0/checkpoints/89.ckpt"
    # 初始化 Trainer 并恢复训练
    # trainer = pl.Trainer(
    #     resume_from_checkpoint=checkpoint_path,  # PyTorch Lightning <2.0 的写法
    #     # ckpt_path=checkpoint_path,   
    # )

    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    # if cfg.agent.checkpoint_path:
    #     # trainer.max_epochs = 134 #70个
    #     trainer.fit(
    #         model=lightning_module,
    #         train_dataloaders=train_dataloader,
    #         val_dataloaders=val_dataloader,
    #         ckpt_path=cfg.agent.checkpoint_path
    #     )
    # else:
    #     trainer.fit(
    #         model=lightning_module,
    #         train_dataloaders=train_dataloader,
    #         val_dataloaders=val_dataloader
    #     )
    # trainer.fit(
    #     model=lightning_module,
    #     train_dataloaders=train_dataloader,
    #     val_dataloaders=val_dataloader,
    #     ckpt_path=checkpoint_path
    # )

if __name__ == "__main__":
    main()

