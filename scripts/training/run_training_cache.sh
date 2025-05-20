export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=8,9
TRAIN_TEST_SPLIT=navtrain
# # FREEZE_PERCEPTION=False 
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_caching.py \
agent=diffusiondrive_agent \
experiment_name=cache_agent \
train_test_split=$TRAIN_TEST_SPLIT \
+trainer.params.devices=2 \
agent.config.latent=True
