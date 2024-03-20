SPLIT=mini

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_submission_evaluation.py \
agent=constant_velocity_agent \
split=$SPLIT \
experiment_name=submission_cv_agent \