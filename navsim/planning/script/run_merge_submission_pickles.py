import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from navsim.common.dataclasses import SensorConfig, Trajectory
from navsim.common.dataloader import SceneLoader

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_merge_submission_pickles"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for submission creation script.
    :param cfg: omegaconf dictionary
    """

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)
    navsim_blobs_path = Path(cfg.navsim_blobs_path)
    synthetic_scenes_path = Path(cfg.synthetic_scenes_path)
    save_path = Path(cfg.output_dir)
    scene_filter = instantiate(cfg.train_test_split.scene_filter)

    tokens = SceneLoader(
        data_path=data_path,
        scene_filter=scene_filter,
        sensor_blobs_path=sensor_blobs_path,
        navsim_blobs_path=navsim_blobs_path,
        synthetic_scenes_path=synthetic_scenes_path,
        sensor_config=SensorConfig.build_no_sensors(),
    ).tokens

    # submission_pickles: List[str] = instantiate(cfg.submission_pickles)

    merged_list: List[Dict[str, Trajectory]] = []
    for submission_pickle in cfg.submission_pickles:
        with open(submission_pickle, "rb") as f:
            output_list: List[Dict[str, Trajectory]] = pickle.load(f)["predictions"]
        for output in output_list:
            assert set(output.keys()) == set(tokens), f"Submission pickle {submission_pickle} has invalid scene tokens!"
        merged_list.extend(output_list)
    logger.info(f"Merged {len(merged_list)} submissions from {len(cfg.submission_pickles)} pickles")

    submission = {
        "team_name": cfg.team_name,
        "authors": cfg.authors,
        "email": cfg.email,
        "institution": cfg.institution,
        "country / region": cfg.country,
        "predictions": merged_list,
    }

    # pickle and save dict
    filename = os.path.join(save_path, "submission.pkl")
    with open(filename, "wb") as file:
        pickle.dump(submission, file)
    logger.info(f"Your submission filed was saved to {filename}")


if __name__ == "__main__":
    main()
