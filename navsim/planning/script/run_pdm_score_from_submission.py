import pandas as pd
from tqdm import tqdm
import traceback
import pickle

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from pathlib import Path
from typing import Any, Dict, List
from dataclasses import asdict
import logging

from nuplan.planning.script.builders.logging_builder import build_logger

from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator
)
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.common.dataclasses import Trajectory

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score_from_submission"

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    submission_file_path = Path(cfg.submission_file_path)
    metric_cache_path = Path(cfg.metric_cache_path)
    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    build_logger(cfg)
    assert simulator.proposal_sampling == scorer.proposal_sampling, "Simulator and scorer proposal sampling has to be identical"

    run_pdm_score(
        submission_file_path=submission_file_path,
        simulator=simulator,
        scorer=scorer,
        metric_cache_path=metric_cache_path,
    )

def run_pdm_score(
    submission_file_path: Path,
    simulator: PDMSimulator,
    scorer: PDMScorer,
    metric_cache_path: Path,
) -> None:
    """
    Function to evaluate an agent with the PDM-Score
    :param agent: Agent object
    :param data_path: pathlib path to navsim logs
    :param metric_cache_path: pathlib path to metric cache
    :param save_path: pathlib path to folder where scores are stored as .csv
    """    
    logger.info("Building SceneLoader")
    metric_cache_loader = MetricCacheLoader(metric_cache_path)
    with open(submission_file_path, "rb") as f:
        agent_output: Dict[str, Trajectory] = pickle.load(f)["predictions"]
    
    score_rows: List[Dict[str, Any]] = []
    for token in tqdm(metric_cache_loader.tokens, desc="Compute PDM-Score"):
        score_row: Dict[str, Any] = {"token": token, "valid": True}

        try:
            metric_cache = metric_cache_loader.get_from_token(token)
            trajectory = agent_output[token]
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
    if not pdm_score_df["valid"].all():
        logger.warning("Evaluation for some tokens failed. Check log for details")
    else:
        average_score = pdm_score_df["score"].mean()
        return average_score

if __name__ == "__main__":
    main()
