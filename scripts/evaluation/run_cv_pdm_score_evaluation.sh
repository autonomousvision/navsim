TRAIN_TEST_SPLIT=warmup_two_stage

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
experiment_name=cv_agent


# If you want to run one stage only simulation, plesae uncomment and keep only the following lines:

# TRAIN_TEST_SPLIT=navtest

# python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_one_stage.py \
# train_test_split=$TRAIN_TEST_SPLIT \
# experiment_name=cv_agent \
# traffic_agents_policy=non_reactive \
