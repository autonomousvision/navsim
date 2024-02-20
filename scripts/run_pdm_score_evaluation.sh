LOG_PATH=/path/to/navsim_logs
METRIC_CACHE_PATH=/path/to/metric_cache
OUTPUT_DIR=/path/to/output_dir

python $NAVSIM_DEVKIT_ROOT/navsim/evaluate/run_pdm_score.py \
navsim_log_path=$LOG_PATH \
metric_cache_path=$METRIC_CACHE_PATH \
output_dir=$OUTPUT_DIR \