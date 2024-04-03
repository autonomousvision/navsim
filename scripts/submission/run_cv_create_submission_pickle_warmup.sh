TEAM_NAME="MUST_SET"
AUTHORS="MUST_SET"
EMAIL="MUST_SET"
INSTITUTION="MUST_SET"
COUNTRY="MUST_SET"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_create_submission_pickle.py \
agent=constant_velocity_agent \
split=mini \
scene_filter=warmup_test \
experiment_name=submission_cv_agent_warmup \
team_name=$TEAM_NAME \
authors=$AUTHORS \
email=$EMAIL \
institution=$INSTITUTION \
country=$COUNTRY \
