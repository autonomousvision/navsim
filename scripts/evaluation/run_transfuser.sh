export HYDRA_FULL_ERROR=1
TRAIN_TEST_SPLIT=warmup_navsafe_two_stage_extended

# CHECKPOINT="/data/hdd01/dingzx/navsim_exp/training_diffusiondrive_agent/2025.03.28.23.58.29/lightning_logs/version_0/checkpoints/99.ckpt"
# 加上画图不要token限制 motion_blur数据增强
# 6个 分数一样
# CHECKPOINT="/data/hdd01/dingzx/navsim_exp/training_diffusiondrive_agent/2025.04.03.22.48.51/lightning_logs/version_0/checkpoints/99.ckpt"
# 1个 motion
# CHECKPOINT="/data/hdd01/dingzx/navsim_exp/training_diffusiondrive_agent/2025.04.02.16.07.18/lightning_logs/version_0/checkpoints/99.ckpt"
# 0.6
# CHECKPOINT='/data/hdd01/dingzx/navsim_exp/training_diffusiondrive_agent/2025.04.09.22.54.54/lightning_logs/version_0/checkpoints/99.ckpt'
# 2个
# CHECKPOINT='/data/hdd01/dingzx/navsim_exp/training_diffusiondrive_agent/2025.04.04.20.02.40/lightning_logs/version_0/checkpoints/99.ckpt'
# 无
# CHECKPOINT='/data/hdd01/dingzx/navsim_exp/training_diffusiondrive_agent/2025.03.28.08.55.12/lightning_logs/version_0/checkpoints/99.ckpt'
# rotate
CHECKPOINT='/data/hdd01/dingzx/navsim_exp/training_diffusiondrive_agent/2025.04.15.22.26.33/lightning_logs/version_0/checkpoints/99.ckpt'
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH=/data/hdd01/dingzx/workspace/navsim
cd /data/hdd01/dingzx/workspace/navsim

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=diffusiondrive_agent \
worker=single_machine_thread_pool \
agent.checkpoint_path="$CHECKPOINT" \
experiment_name=diffusiondrive_agent_eval \
agent.config.latent=True