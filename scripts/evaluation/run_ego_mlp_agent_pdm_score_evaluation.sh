TRAIN_TEST_SPLIT=navtest
CHECKPOINT=/path/to/ego_status_mlp.ckpt

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=ego_status_mlp_agent \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=ego_mlp_agent
