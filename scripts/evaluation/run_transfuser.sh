TRAIN_TEST_SPLIT=navtest
CHECKPOINT=/path/to/transfuser.ckpt

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=transfuser_agent \
worker=single_machine_thread_pool \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=transfuser_agent_eval 
