export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
FEATURE_CACHE='/data/hdd01/xingzb/navsim_exp/training_cache'
# FEATURE_CACHE='/data/hdd01/dingzx/dataset/lpai/volumes/ad-e2e-vol-ga/zhengyupeng/socket/navsim_exp/metric_cache_syn_2048'
TRAIN_TEST_SPLIT=navtrain
# CHECKPOINT='/data/hdd01/dingzx/navsim_exp/training_diffusiondrive_agent/2025.04.30.20.01.08/lightning_logs/version_0/checkpoints/64.ckpt'
# CHECKPOINT='/data/hdd01/dingzx/navsim_exp/training_diffusiondrive_agent/2025.04.30.20.01.08/lightning_logs/version_0/checkpoints/59.ckpt'
# # FREEZE_PERCEPTION=False 
export NAVSIM_DEVKIT_ROOT="/data/hdd01/dingzx/workspace/navsim"
cd /data/hdd01/dingzx/workspace/navsim

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_diffusiondrive.py \
agent=diffusiondrive_agent \
experiment_name=training_diffusiondrive_agent \
train_test_split=$TRAIN_TEST_SPLIT \
cache_path=$FEATURE_CACHE \
force_cache_computation=False \
use_cache_without_dataset=True \
+trainer.params.devices=8 \
agent.config.latent=True







# export HYDRA_FULL_ERROR=1
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# FEATURE_CACHE='/data/hdd01/xingzb/navsim_exp/training_cache'
# TRAIN_TEST_SPLIT=navtrain
# export NAVSIM_DEVKIT_ROOT=/data/hdd01/dingzx/workspace/navsim
# python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
# agent=transfuser_agent \
# experiment_name=dzx_training_transfuser_agent \
# train_test_split=$TRAIN_TEST_SPLIT \
# cache_path=$FEATURE_CACHE \
# force_cache_computation=False \
# use_cache_without_dataset=True \
# +trainer.params.devices=8 \
# agent.config.latent=True















# agent.lr=6e-5 \
# agent.checkpoint_path="$CHECKPOINT" \


# # you lidar
# export HYDRA_FULL_ERROR=1
# TRAIN_TEST_SPLIT=navtrain
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9
# FEATURE_CACHE='/data/hdd01/xingzb/navsim_exp/training_cache'
# # # FREEZE_PERCEPTION=False 
# python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_diffusiondrive.py \
# agent=diffusiondrive_agent \
# experiment_name=training_diffusiondrive_agent \
# train_test_split=$TRAIN_TEST_SPLIT \
# cache_path=$FEATURE_CACHE \
# force_cache_computation=False \
# use_cache_without_dataset=True \
# +trainer.params.devices=10 \


# export HYDRA_FULL_ERROR=1
# TRAIN_TEST_SPLIT=navtrain
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9
# FEATURE_CACHE='/data/hdd01/xingzb/navsim_exp/training_cache'
# # # FREEZE_PERCEPTION=False 
# python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_diffusiondrive.py \
# agent=diffusiondrive_agent \
# experiment_name=training_diffusiondrive_agent \
# train_test_split=$TRAIN_TEST_SPLIT \
# cache_path=$FEATURE_CACHE \
# force_cache_computation=False \
# use_cache_without_dataset=True \
# +trainer.params.devices=10 \
# agent.config.latent=True 


## transfuser
# export HYDRA_FULL_ERROR=1
# TRAIN_TEST_SPLIT=navtrain
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9
# FEATURE_CACHE='/data/hdd01/xingzb/navsim_exp/training_cache'
# # # FREEZE_PERCEPTION=False 
# python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
# agent=transfuser_agent \
# experiment_name=training_transfuser_agent \
# train_test_split=$TRAIN_TEST_SPLIT \
# cache_path=$FEATURE_CACHE \
# force_cache_computation=False \
# use_cache_without_dataset=True \
# +trainer.params.devices=10 \
# agent.config.latent=True 
