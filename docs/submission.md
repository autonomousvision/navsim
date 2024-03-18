# Submitting to the Leaderboard

NAVSIM comes with official leaderboards on HuggingFace. The leaderboards prevent ambiguity in metric definitions between different projects, as all evaluation is performed on the server with the official evaluation script.

To submit to a leaderboard you need to create a pickle file that contains a trajectory for each test scenario. NAVSIM provides a script to create such a pickle file. 

Have a look at `run_cv_submission_evaluation.sh`: this file creates the pickle file for the ConstantVelocity agent. You can run it for your own agent by replacing the `agent` override.

**Note that you have to set the variables `TEAM_NAME`, `AUTHORS`, `EMAIL`, `INSTITUTION`, and `COUNTRY` for your submission to be valid.**

### Warm-up track
The warm-up track evaluates your submission on a [warm-up leaderboard](https://huggingface.co/spaces/AGC2024-P/e2e-driving-warmup) based on the `mini` split. This allows you to test your method and get familiar with the devkit and the submisison procedure, with a less restrictive submission budget (up to 5 submissions daily). Instructions on making a submission on HuggingFace are available in the HuggingFace space. Performance on the warm-up leaderboard is not taken into consideration for determining your team's ranking for the 2024 Autonomous Grand Challenge.

You should be able to obtain the same evaluation results as on the server, by running the evaluation locally with the `warmup_test` scene filter. To do so, use the override `scene_filter=warmup_test` when executing the script to run the PDM scoring (e.g.,  `run_cv_pdm_score_evaluation.sh` for the constant-velocity agent).

### Formal track
This is the [official challenge leaderboard](https://huggingface.co/spaces/AGC2024-P/e2e-driving-2024), based on secret held-out test frames. **Details and instructions for submission will be provided soon!**
