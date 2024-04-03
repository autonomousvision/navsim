SPLIT=mini
CHECKPOINT="TODO"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
agent=ego_status_mlp_agent \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=ego_mlp_agent \
split=$SPLIT \
