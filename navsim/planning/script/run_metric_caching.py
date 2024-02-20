import logging
import os

import hydra
from omegaconf import DictConfig

from nuplan.planning.script.builders.logging_builder import build_logger

from navsim.planning.script.utils import set_default_path
from navsim.planning.metric_caching.caching import cache_data
from navsim.planning.script.builders.worker_pool_builder import build_worker


logging.getLogger("numba").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# If set, use the env. variable to overwrite the default dataset and experiment paths
set_default_path()

# If set, use the env. variable to overwrite the Hydra config
CONFIG_PATH = os.getenv("NUPLAN_HYDRA_CONFIG_PATH", "config/metric_caching")

if os.environ.get("NUPLAN_HYDRA_CONFIG_PATH") is not None:
    CONFIG_PATH = os.path.join("../../../../", CONFIG_PATH)

if os.path.basename(CONFIG_PATH) != "metric_caching":
    CONFIG_PATH = os.path.join(CONFIG_PATH, "metric_caching")
CONFIG_NAME = "default_metric_caching"



@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training/validation experiments.
    :param cfg: omegaconf dictionary
    """
    if cfg.enable_profiling:
        raise ValueError("Profiling is not supported to remove pytorch-lightning dependency")

    # Fix random seed
    # pl.seed_everything(cfg.seed, workers=True)


    # Configure logger
    build_logger(cfg)
    
    # Build worker
    worker = build_worker(cfg)

    # Precompute and cache all features
    logger.info("Starting Metric Caching...")
    if cfg.worker == "ray_distributed" and cfg.worker.use_distributed:
        raise AssertionError("ray in distributed mode will not work with this job")
    cache_data(cfg=cfg, worker=worker)

if __name__ == "__main__":
    main()
