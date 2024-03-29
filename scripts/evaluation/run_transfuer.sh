SPLIT=mini
CHECKPOINT="TODO"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
agent=transfuser_agent \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=transfuser_agent_eval \
split=$SPLIT \
