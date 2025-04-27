TEAM_NAME="MUST_SET"
AUTHORS="MUST_SET"
EMAIL="MUST_SET"
INSTITUTION="MUST_SET"
COUNTRY="MUST_SET"

TRAIN_TEST_SPLIT=warmup_two_stage
SYNTHETIC_SENSOR_PATH=$OPENSCENE_DATA_ROOT/warmup_two_stage/sensor_blobs
SYNTHETIC_SCENES_PATH=$OPENSCENE_DATA_ROOT/warmup_two_stage/synthetic_scene_pickles

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_create_submission_pickle_warmup.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=constant_velocity_agent \
experiment_name=submission_cv_agent \
team_name=$TEAM_NAME \
authors=$AUTHORS \
email=$EMAIL \
institution=$INSTITUTION \
country=$COUNTRY \
synthetic_sensor_path=$SYNTHETIC_SENSOR_PATH \
synthetic_scenes_path=$SYNTHETIC_SCENES_PATH \
