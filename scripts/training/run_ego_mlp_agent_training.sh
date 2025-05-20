TRAIN_TEST_SPLIT=navtrain
export CUDA_VISIBLE_DEVICES=8,9
export HYDRA_FULL_ERROR=1
FEATURE_CACHE='/data/hdd01/xingzb/navsim_exp/training_cache'
export NAVSIM_DEVKIT_ROOT="/data/hdd01/dingzx/workspace/navsim"
cd /data/hdd01/dingzx/workspace/navsim
# /data/hdd01/dingzx/workspace/navsim/navsim/planning/script/run_training.py
python /data/hdd01/dingzx/workspace/navsim/navsim/planning/script/run_training.py \
experiment_name=debug_diffusiondriv_agent \
train_test_split=$TRAIN_TEST_SPLIT \
agent=ego_status_mlp_agent \
cache_path=$FEATURE_CACHE \
force_cache_computation=False \
use_cache_without_dataset=True

# ego_status_mlp_agent