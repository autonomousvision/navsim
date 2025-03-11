import logging
import os
import pickle
import traceback
from pathlib import Path
from typing import Dict

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter, Trajectory
from navsim.common.dataloader import SceneLoader

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_create_submission_pickle"


def run_test_evaluation(
    agent: AbstractAgent,
    scene_filter: SceneFilter,
    data_path: Path,
    sensor_blobs_path: Path,
    navsim_blobs_path: Path,
    synthetic_scenes_path: Path,
) -> Dict[str, Trajectory]:
    """
    Function to create the output file for evaluation of an agent on the testserver
    :param agent: Agent object
    :param data_path: pathlib path to navsim logs
    :param sensor_blobs_path: pathlib path to sensor blobs
    :param synthetic_scenes_path: pathlib path to synthetic scenes
    :param save_path: pathlib path to folder where scores are stored as .csv
    """
    if agent.requires_scene:
        raise ValueError(
            """
            In evaluation, no access to the annotated scene is provided, but only to the AgentInput.
            Thus, agent.requires_scene has to be False for the agent that is to be evaluated.
            """
        )
    logger.info("Building Agent Input Loader")
    input_loader = SceneLoader(
        data_path=data_path,
        scene_filter=scene_filter,
        sensor_blobs_path=sensor_blobs_path,
        navsim_blobs_path=navsim_blobs_path,
        synthetic_scenes_path=synthetic_scenes_path,
        sensor_config=agent.get_sensor_config(),
    )
    agent.initialize()

    output: Dict[str, Trajectory] = {}
    for token in tqdm(input_loader, desc="Running evaluation"):
        try:
            agent_input = input_loader.get_agent_input_from_token(token)
            trajectory = agent.compute_trajectory(agent_input)
            output.update({token: trajectory})
        except Exception:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()

    return output


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for submission creation script.
    :param cfg: omegaconf dictionary
    """
    agent = instantiate(cfg.agent)
    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)
    navsim_blobs_path = Path(cfg.navsim_blobs_path)
    synthetic_scenes_path = Path(cfg.synthetic_scenes_path)
    save_path = Path(cfg.output_dir)
    scene_filter = instantiate(cfg.train_test_split.scene_filter)

    output = run_test_evaluation(
        agent=agent,
        scene_filter=scene_filter,
        data_path=data_path,
        synthetic_scenes_path=synthetic_scenes_path,
        sensor_blobs_path=sensor_blobs_path,
        navsim_blobs_path=navsim_blobs_path,
    )

    submission = {
        "team_name": cfg.team_name,
        "authors": cfg.authors,
        "email": cfg.email,
        "institution": cfg.institution,
        "country / region": cfg.country,
        "predictions": [output],
    }

    # pickle and save dict
    filename = os.path.join(save_path, "submission.pkl")
    with open(filename, "wb") as file:
        pickle.dump(submission, file)
    logger.info(f"Your submission filed was saved to {filename}")


if __name__ == "__main__":
    main()
