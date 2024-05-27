TEAM_NAME="MUST_SET"
AUTHORS="MUST_SET"
EMAIL="MUST_SET"
INSTITUTION="MUST_SET"
COUNTRY="MUST_SET"

SUBMISSION_PICKLES="['/path/to/submission.pkl','/path/to/submission.pkl','/path/to/submission.pkl']"
TRAIN_TEST_SPLIT=navtest

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_merge_submission_pickles.py \
train_test_split=$TRAIN_TEST_SPLIT \
experiment_name=submission_merged_agent \
submission_pickles=$SUBMISSION_PICKLES \
team_name=$TEAM_NAME \
authors=$AUTHORS \
email=$EMAIL \
institution=$INSTITUTION \
country=$COUNTRY \
