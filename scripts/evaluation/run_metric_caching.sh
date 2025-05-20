# # navtest
# export HYDRA_FULL_ERROR=1
# TRAIN_TEST_SPLIT=navtest
# CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache_navtest

# python  $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
# train_test_split=$TRAIN_TEST_SPLIT \
# metric_cache_path=$CACHE_PATH


# export HYDRA_FULL_ERROR=1
# TRAIN_TEST_SPLIT=warmup_two_stage
# CACHE_PATH=$NAVSIM_EXP_ROOT/metric_warmup_two_stage

# python  $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
# train_test_split=$TRAIN_TEST_SPLIT \
# metric_cache_path=$CACHE_PATH


#!/bin/bash

# 进入checkpoints目录（根据你的实际路径修改）
cd /data/hdd01/dingzx/navsim_exp/training_diffusiondrive_agent/2025.05.18.22.59.47/lightning_logs/version_0/checkpoints/

# 循环处理所有符合格式的ckpt文件
for file in epoch=*-step=*.ckpt; do
    # 使用sed提取epoch数字并构建新文件名
    newname=$(echo "$file" | sed -E 's/epoch=([0-9]+)-step=.*\.ckpt/\1.ckpt/')
    # print(f"{newname")
    # print($file)
    echo "新文件名: $newname"
    echo "$file"
    
    # 执行重命名操作（测试时可以先在下一行加上echo）
    mv -- "$file" "$newname"
done