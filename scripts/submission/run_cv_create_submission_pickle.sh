TEAM_NAME="111"
AUTHORS="ddd"
EMAIL="11@qq.com"
INSTITUTION="1"
COUNTRY="CHINA"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=""
TRAIN_TEST_SPLIT=warmup_two_stage
# CHECKPOINT="/data/hdd01/dingzx/navsim_exp/training_diffusiondrive_agent/2025.03.28.23.58.29/lightning_logs/version_0/checkpoints/99.ckpt"
# CHECKPOINT="/data/hdd01/dingzx/navsim_exp/training_diffusiondrive_agent/2025.04.02.16.07.18/lightning_logs/version_0/checkpoints/99.ckpt"
CHECKPOINT='/data/hdd01/dingzx/navsim_exp/training_diffusiondrive_agent/2025.04.09.22.54.54/lightning_logs/version_0/checkpoints/99.ckpt'
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_create_submission_pickle.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=diffusiondrive_agent \
experiment_name=submission_cv_agent \
team_name=$TEAM_NAME \
authors=$AUTHORS \
email=$EMAIL \
agent.checkpoint_path="$CHECKPOINT" \
agent.config.latent=True \
institution=$INSTITUTION \
country=$COUNTRY
