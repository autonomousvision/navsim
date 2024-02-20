SPLIT=testing
LOG_PATH=/path/to/navsim_logs
METRIC_CACHE_PATH=/path/to/metric_cache

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
experiment_name=metric_caching \
job_name=metric_caching \
+splitter=nuplan \
navsim_log_path=$LOG_PATH \
cache.use_open_scene=true \
cache.cache_path=$METRIC_CACHE_PATH \