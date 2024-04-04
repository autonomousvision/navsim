# Submitting to the Leaderboard

NAVSIM comes with official leaderboards on HuggingFace. The leaderboards prevent ambiguity in metric definitions between different projects, as all evaluation is performed on the server with the official evaluation script.

To submit to a leaderboard you need to create a pickle file that contains a trajectory for each test scenario. NAVSIM provides a script to create such a pickle file. 

Have a look at `run_create_submission_pickle.sh`: this file creates the pickle file for the ConstantVelocity agent. You can run it for your own agent by replacing the `agent` override.
Follow the [submission instructions on huggingface](https://huggingface.co/spaces/AGC2024-P/e2e-driving-2024) to upload your submission.
**Note that you have to set the variables `TEAM_NAME`, `AUTHORS`, `EMAIL`, `INSTITUTION`, and `COUNTRY` in `run_create_submission_pickle.sh` to generate a valid submission file**

### Warm-up track
The warm-up track evaluates your submission on a [warm-up leaderboard](https://huggingface.co/spaces/AGC2024-P/e2e-driving-warmup) based on the `mini` split. This allows you to test your method and get familiar with the devkit and the submission procedure, with a less restrictive submission budget (up to 5 submissions daily). Instructions on making a submission on HuggingFace are available in the HuggingFace space. Performance on the warm-up leaderboard is not taken into consideration for determining your team's ranking for the 2024 Autonomous Grand Challenge.
Use the script `run_create_submission_pickle_warmup.sh` which already contains the overrides `scene_filter=warmup_test` and `split=mini` to generate the submission file for the warmup track.

You should be able to obtain the same evaluation results as on the server, by running the evaluation locally.
To do so, use the overrides `scene_filter=warmup_test` when executing the script to run the PDM scoring (e.g.,  `run_cv_pdm_score_evaluation.sh` for the constant-velocity agent).

### Formal track
This is the [official challenge leaderboard](https://huggingface.co/spaces/AGC2024-P/e2e-driving-2024), based on secret held-out test frames (see submission_test split on the install page). 
Use the script `run_create_submission_pickle.sh`. It will by default run with `scene_filter=competition_test` and `split=competition_test`.
You only need to set your own agent with the `agent` override.
