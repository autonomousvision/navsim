# Submitting to the leaderboard

NAVSIM comes with an [official leaderboard](https://huggingface.co/spaces/AGC2024-P/e2e-driving-navsim) on HuggingFace. The leaderboard prevents ambiguity in metric definitions between different projects, as all evaluation is performed on the server with the official evaluation script.

After the NAVSIM challenge 2024, we have re-opened the leaderboard with the `navtest` split. With 12k samples, the `navtest` split contains a larger set for comprehensive evaluations. In this guide, we describe how to create a valid submission and what rules apply for the new leaderboard.

### Rules
- **Technical Reports**:
  - We will periodically be removing all entries on the leaderboard which **do not provide an associated technical report**.
  - The technical report can be in any format (e.g. ArXiv paper, github readme, privately hosted pdf file, etc), as long as it clearly describes the methodology used in the submission, enabling reimplementation.
  - Technical reports must be provided by setting the `TEAM_NAME` variable of the submission file with `"<a href=Link/to/tech/report.pdf>Method name</a>"`. Note that this can also be edited on the leaderboard for an existing submission, if the report is created (or updated) after the initial submission.
- **Multi-seed submissions**:
  - Driving policies often differ significantly in performance when re-trained with different network initialization seeds.
  - Therefore, the leaderboard now supports (1) regular single-seed submissions and (2) multi-seed submission, which we **strongly encourage** (with a minimum of 3 training seeds).
  - The maximum, mean and standard deviations of our evaluation metrics will be displayed for multi-seed submissions.

### Regular navtest submission

To submit to a leaderboard you need to create a pickle file that contains a trajectory for each test scenario. NAVSIM provides a script to create such a pickle file. 

Have a look at `run_create_submission_pickle.sh`: this file creates the pickle file for the ConstantVelocity agent. You can run it for your own agent by replacing the `agent` override. Follow the [submission instructions on huggingface](https://huggingface.co/spaces/AGC2024-P/e2e-driving-navsim) to upload your submission.
**Note that you have to set the variables `TEAM_NAME`, `AUTHORS`, `EMAIL`, `INSTITUTION`, and `COUNTRY` in `run_create_submission_pickle.sh` to generate a valid submission file**

You should be able to obtain the same evaluation results as on the server by running the evaluation locally.
To do so, use the override `train_test_split=navtest` when executing the script to run the PDM scoring.

### Multi-seed navtest submission

For a multi-seed submission, you first have to create individual agents, i.e. trained on different seeds. Consequently, you can merge your entries to a single submission file with the `run_merge_submission_pickles.sh` bash script. Please set the override `train_test_split=navtest` to ensure all individual entries contain trajectories for the evaluation.
