# Submitting to the leaderboard

NAVSIM comes with an official leaderboard ([Leaderboard 2024](https://huggingface.co/spaces/AGC2024-P/e2e-driving-navsim), [Leaderboard 2025 Warmup](https://huggingface.co/spaces/AGC2025/e2e-driving-warmup)) on HuggingFace. The leaderboard prevents ambiguity in metric definitions between different projects, as all evaluation is performed on the server with the official evaluation script.

For the [NAVSIM challenge 2025 warmup](https://huggingface.co/spaces/AGC2025/e2e-driving-warmup), we now open the leaderboard with the `warmup_navsafe_two_stage_extended` split. In this guide, we describe how to create a valid submission and what rules apply for the new leaderboard.

### Rules

- **Open-source code and models**:
  - We will periodically (~every 6 months) be removing all entries on the leaderboard which **do not provide associated open-source training and inference code with the corresponding pre-trained checkpoints**. Even if removed for not having this information, an entry can be resubmitted once the code needed for reproducibility is made publicly available.
  - Code must be provided by setting the `TEAM_NAME` variable of the submission file as `"<a href=Link/to/repository>Method name</a>"`. Note that this can also be edited on the leaderboard for an existing submission, if the repo is created (or updated) after the initial submission.
- **Multi-seed submissions**:
  - Driving policies often differ significantly in performance when re-trained with different network initialization seeds.
  - Therefore, the leaderboard now supports (1) regular single-seed submissions and (2) multi-seed submission, which we **strongly encourage** (with a minimum of 3 training seeds).
  - The maximum, mean and standard deviations of our evaluation metrics will be displayed for multi-seed submissions.

### Regular warmup submission

To submit to a leaderboard you need to create a pickle file that contains a trajectory for each test scenario. NAVSIM provides a script to create such a pickle file.

Have a look at `run_cv_create_submission_pickle.sh` in the [NAVSIM](https://github.com/autonomousvision/navsim/blob/main/docs/install.md) repository: this file creates the pickle file for the ConstantVelocity agent. You can run it for your own agent by replacing the `agent` override. **Note that you have to set the variables `TEAM_NAME`, `AUTHORS`, `EMAIL`, `INSTITUTION`, and `COUNTRY` in `run_create_submission_pickle.sh` to generate a valid submission file**

You should be able to obtain the same evaluation results as on the server by running the evaluation locally. To do so, use the override `train_test_split=warmup_navsafe_two_stage_extended` when executing the script to run the PDM scoring.

### Multi-seed warmup submission

For a multi-seed submission, you first have to create individual agents, i.e. trained on different seeds. Consequently, you can merge your entries to a single submission file with the `run_merge_submission_pickles.sh` bash script. Please set the override `train_test_split=warmup_navsafe_two_stage_extended ` to ensure all individual entries contain trajectories for the evaluation.

## Submission Description

After that, upload your submission as **a HuggingFace model**. Note that private models are also acceptable by the competition space.

Specifically, click your profile picture on the top right of the Hugging Face website, and select `+New Model`.

Then, fill in the form and upload the `submission.pkl` file.

## Submission Process

1. Select `New Submission` in the left panel of the competition space. Paste the link of the Hugging Face model you created in the form. Then click `Submit` to make a new submission.
2. **Note that you can make up to 5 submissions per day.**

---

# FAQ

## How to View My Submissions?

You can check the status of your submissions in the **My Submissions** tab of the competition space. You can select a submission and click **Update Selected Submissions** at the bottom to refresh its evaluation status on the public leaderboard.

## Will My Evaluation Results Be Visible to Others?

The **public leaderboard** will display the best results of all teams at all times for this warm-up track.

**Note**: You can change your team name even after the competition ends.

Thus, if you want to stay anonymous on the public leaderboard, you can first use a temporary team name and change it to your real team name after the competition ends.

## Can I Submit Without Making My Submission File Public?

Of course. The competition space accepts Hugging Face private models. in fact, we recommend participants to submit as private models to keep their submissions private.

## My Evaluation Status Shows "Failed" — How Can I Get the Error Message?

First, make sure your submission is in the correct format as in submission preparation and you set the correct Hugging Face **model** link (in the format of Username/model) in New Submission. Please make sure that the name of the pickle file you uploaded is `submission.pkl`.

If you confirm that the submission format is correct, please contact [Wei Cao](mailto:dave.caowei@gmail.com) via email. Include the **Submission ID** of the corresponding submission (which can be found in the **My Submissions** tab).

```
Email Subject: [CVPR E2E] Failed submission - {Submission ID}
Body:
    Your Name: {}
    Team Name: {}
    Institution / Company: {}
    Email: {}
```

## The submission page shows Invalid Token, what should I do?

This means you are no longer logged in to the current competition space, or the space has automatically logged you out due to inactivity (more than a day).

Please refresh the page, click Login with Hugging Face at the bottom of the left panel, and resubmit.

## I could not visit My Submissions page, what should I do?

Chances are that you are not logged in to the current competition space.

Please refresh the page, click Login with Hugging Face at the bottom of the left panel.
