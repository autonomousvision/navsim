import pandas as pd
from tqdm import tqdm
import traceback

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from pathlib import Path
from typing import Any, Dict, List
from dataclasses import asdict
from datetime import datetime
import logging

from nuplan.planning.script.builders.logging_builder import build_logger

from navsim.common.dataloader import MetricCacheLoader
from navsim.agents.abstract_agent import AbstractAgent
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator
)
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    agent = instantiate(cfg.agent)
    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)
    metric_cache_path = Path(cfg.metric_cache_path)
    save_path = Path(cfg.output_dir)
    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    scene_filter: SceneFilter = instantiate(cfg.scene_filter)
    build_logger(cfg)
    assert simulator.proposal_sampling == scorer.proposal_sampling, "Simulator and scorer proposal sampling has to be identical"

    run_pdm_score(
        agent=agent,
        simulator=simulator,
        scorer=scorer,
        scene_filter=scene_filter,
        data_path=data_path,
        sensor_blobs_path=sensor_blobs_path,
        metric_cache_path=metric_cache_path,
        save_path=save_path,
    )

def run_pdm_score(
    agent: AbstractAgent,
    simulator: PDMSimulator,
    scorer: PDMScorer,
    scene_filter: SceneFilter,
    data_path: Path,
    sensor_blobs_path: Path,
    metric_cache_path: Path,
    save_path: Path
) -> None:
    """
    Function to evaluate an agent with the PDM-Score
    :param agent: Agent object
    :param data_path: pathlib path to navsim logs
    :param metric_cache_path: pathlib path to metric cache
    :param save_path: pathlib path to folder where scores are stored as .csv
    """    
    logger.info("Building SceneLoader")
    scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    metric_cache_loader = MetricCacheLoader(metric_cache_path)
    agent.initialize()

    # NOTE: This could be parallel
    score_rows: List[Dict[str, Any]] = []
    for token in tqdm(metric_cache_loader.tokens, desc="Compute PDM-Score"):
        score_row: Dict[str, Any] = {"token": token, "valid": True}

        try:
            agent_input = scene_loader.get_agent_input_from_token(token)
            metric_cache = metric_cache_loader.get_from_token(token)
            if agent.requires_scene:
                scene = scene_loader.get_scene_from_token(token)
                trajectory = agent.compute_trajectory(agent_input, scene)
            else:
                trajectory = agent.compute_trajectory(agent_input)

            pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            score_row.update(asdict(pdm_result))
        except Exception as e:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row["valid"] = False

        score_rows.append(score_row)

    pdm_score_df = pd.DataFrame(score_rows)
    average_row = pdm_score_df.drop(columns=["token", "valid"]).mean(skipna=True)
    average_row["token"] = "average"
    average_row["valid"] = pdm_score_df["valid"].all()
    pdm_score_df.loc[len(pdm_score_df)] = average_row

    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    pdm_score_df.to_csv(save_path / f"{agent.name()}_{timestamp}.csv")


if __name__ == "__main__":
    main()
