TRAIN_TEST_SPLIT=navtest
CACHE_PATH=YOUR_PATH_TO_METRIC_CACHE

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=human_agent \
experiment_name=human_agent \
traffic_agents_policy=non_reactive \
metric_cache_path=$CACHE_PATH \
