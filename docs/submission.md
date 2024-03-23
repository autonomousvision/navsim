# Submitting to the Leaderboard

Navsim comes with an official leaderboard on HuggingFace that remains open and prevents ambiguity in metric definitions between projects.
To submit to the evaluation server you need to create a pickle file that contains a trajectory for each test scenario.
Navsim provides a script to create such a pickle file. 

Have a look at `run_cv_submission_evaluation.sh`: this file creates the pickle file for a simplistic Constant-Velocity agent.
You can run it for your agent, by replacing the `agent` override.

**NOTE that you have to set the variables `TEAM_NAME`, `AUTHORS`, `EMAIL`, `INSTITUTION`, and `COUNTRY` for a valid submission**

### Warm-up phase
During the warm-up phase the evaluation on the submission server is based on the `mini` split.
This allows you to test your method and get familiar with the devkit and the submisison procedure.
You should be able to obtain the same evaluation results as on the server, by running the evaluation locally with the `warmup_test` scene-filter. To do so, use the override `scene_filter=warmup_test` when executing the script to run the pdm-scoring (e.g.,  `run_cv_pdm_score_evaluation.sh` for the constant-velocity agent).

### Test phase
> **More details and instructions soon**