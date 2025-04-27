# Submitting to the leaderboard

NAVSIM comes with official leaderboards ([Leaderboard 2024](https://huggingface.co/spaces/AGC2024-P/e2e-driving-navsim), [Leaderboard 2025 Warmup](https://huggingface.co/spaces/AGC2025/e2e-driving-warmup), [Leaderboard 2025](https://huggingface.co/spaces/AGC2025/e2e-driving-2025)) on HuggingFace. The leaderboards prevent ambiguity in metric definitions between different projects, as all evaluation is performed on the server with the official evaluation script. Below, we provide the instructions relevant for the 2025 Leaderboards. In case you are interested in submitting to the Leaderboard 2024 (for NAVSIM v1, using the `navtest` evaluation split), please use the [NAVSIM v1.1 branch](https://github.com/autonomousvision/navsim/tree/v1.1).

For the [NAVSIM v2 End-to-End Driving Challenge 2025](https://huggingface.co/spaces/AGC2025/e2e-driving-2025), we will be using the leaderboard with the `private_test_hard_two_stage` split. In this guide, we describe how to create a valid submission.

## Rules
- [**General rules (for Leaderboard 2025)**](https://opendrivelab.com/challenge2025/#rule)
- **Open-source code and models (for Leaderboard 2024)**:

  - We will periodically (~every 6 months) be removing all entries on the leaderboard which **do not provide associated open-source training and inference code with the corresponding pre-trained checkpoints**. Even if removed for not having this information, an entry can be resubmitted once the code needed for reproducibility is made publicly available.
  - Code must be provided by setting the `TEAM_NAME` variable of the submission file as `"<a href=Link/to/repository>Method name</a>"`. Note that this can also be edited on the leaderboard for an existing submission, if the repo is created (or updated) after the initial submission.


## [Leaderboard 2025](https://huggingface.co/spaces/AGC2025/e2e-driving-2025)

⏰ **Note:** To ensure fairness in the competition, we do not provide datasets with annotations. Therefore, you can only obtain your score by submitting to the leaderboard. Based on the amount of data we use for evaluation, it should take **approximately 2 hours** to obtain your results after a succesful submission.

### Leaderboard submission

To submit to a leaderboard you need to create a pickle file that contains a trajectory for each test scenario. NAVSIM provides a script to create such a pickle file.

Have a look at `run_cv_create_submission_pickle.sh` in the [NAVSIM](https://github.com/autonomousvision/navsim/blob/main/docs/install.md) repository: this file creates the pickle file for the ConstantVelocity agent. You can run it for your own agent by replacing the `agent` override with `TRAIN_TEST_SPLIT=private_test_hard_two_stage`. **Note that you have to set the variables `TEAM_NAME`, `AUTHORS`, `EMAIL`, `INSTITUTION`, and `COUNTRY` in `run_create_submission_pickle.sh` to generate a valid submission file**

## Submission Description

After that, upload your submission as **a HuggingFace model**. Note that private models are also acceptable by the competition space.

Specifically, click your profile picture on the top right of the Hugging Face website, and select `+New Model`.

Then, fill in the form and upload the `submission.pkl` file.

## Submission Process

- Select `New Submission` in the left panel of the competition space. Paste the link of the Hugging Face model you created in the form. Then click `Submit` to make a new submission.
- **Note that you can only make one submission per day.**


## [Leaderboard 2025 Warmup](https://huggingface.co/spaces/AGC2025/e2e-driving-warmup)

The warmup leaderboard tests your model on a small set of scenes, to help ensure there are no technical issues before you attempt a submission to the formal leaderboard. We recommend the following steps:

### Run Warmup Score Locally

You can reproduce your warmup results locally in the [NAVSIM](https://github.com/autonomousvision/navsim/blob/main/docs/install.md) repository, and they should match the warmup results you obtain on Hugging Face. Follow the steps below:

1. **Download the dataset** — refer to the [dataset](install.md) for instructions.
2. **Cache the data** — follow the script `scripts/evaluation/run_metric_caching.sh`, and set `TRAIN_TEST_SPLIT=warmup_two_stage`
3. **Run the evaluation** — follow the script `scripts/evaluation/run_cv_pdm_score_evaluation.sh` with your own model.
   * If you specified `metric_cache_path` during caching, make sure to use the same path during evaluation.
   * Set `TRAIN_TEST_SPLIT=warmup_two_stage` to ensure that the score matches the one returned by Hugging Face.

### Warmup leaderboard submission

To submit to a leaderboard you need to create a pickle file that contains a trajectory for each test scenario. NAVSIM provides a script to create such a pickle file.

Have a look at `run_cv_create_submission_pickle_warmup.sh` in the [NAVSIM](https://github.com/autonomousvision/navsim/blob/main/docs/install.md) repository: this file creates the pickle file for the ConstantVelocity agent. You can run it for your own agent by replacing the `agent` override with `TRAIN_TEST_SPLIT=warmup_two_stage`. **Note that you have to set the variables `TEAM_NAME`, `AUTHORS`, `EMAIL`, `INSTITUTION`, and `COUNTRY` in `run_create_submission_pickle.sh` to generate a valid submission file**

You should be able to obtain the same evaluation results as on the server by running the evaluation locally.

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
