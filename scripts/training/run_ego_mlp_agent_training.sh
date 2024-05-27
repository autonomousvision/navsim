TRAIN_TEST_SPLIT=navtrain

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
experiment_name=training_ego_mlp_agent \
trainer.params.max_epochs=50 \
train_test_split=$TRAIN_TEST_SPLIT \