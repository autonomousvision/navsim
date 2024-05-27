from datetime import datetime
from typing import Any, Dict, List
from pathlib import Path
from dataclasses import asdict
import logging
import traceback
import pickle

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd
from tqdm import tqdm

from nuplan.planning.script.builders.logging_builder import build_logger

from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.common.dataclasses import Trajectory

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score_from_submission"


def run_pdm_score(
    submission_file_path: Path,
    simulator: PDMSimulator,
    scorer: PDMScorer,
    metric_cache_path: Path,
) -> pd.DataFrame:
    """
    Evaluate submission file with PDM score. 
    :param submission_file_path: path to submission pickle file
    :param simulator: internal simulator object of PDM
    :param scorer: internal scoring objected in PDM
    :param metric_cache_path: path to metric cache 
    :return: pandas data frame with PDMS results.
    """
    logger.info("Building SceneLoader")
    metric_cache_loader = MetricCacheLoader(metric_cache_path)
    with open(submission_file_path, "rb") as f:
        agent_output_list: List[Dict[str, Trajectory]] = pickle.load(f)["predictions"]

    assert len(agent_output_list) == 1, "Multi-seed evaluation currently not supported in run_pdm_score!"
    agent_output = agent_output_list[0]

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
        return pdm_score_df


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for running PDMS from submission pickle.
    :param cfg: omegaconf dictionary
    """
    submission_file_path = Path(cfg.submission_file_path)
    metric_cache_path = Path(cfg.metric_cache_path)
    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    build_logger(cfg)
    assert (
        simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"

    pdm_score_df = run_pdm_score(
        submission_file_path=submission_file_path,
        simulator=simulator,
        scorer=scorer,
        metric_cache_path=metric_cache_path,
    )

    average_row = pdm_score_df.drop(columns=["token", "valid"]).mean(skipna=True)
    average_row["token"] = "average"
    average_row["valid"] = pdm_score_df["valid"].all()
    pdm_score_df.loc[len(pdm_score_df)] = average_row

    save_path = Path(cfg.output_dir)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    pdm_score_df.to_csv(save_path / f"{timestamp}.csv")


if __name__ == "__main__":
    main()
