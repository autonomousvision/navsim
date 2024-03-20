SPLIT=test

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
cache.cache_path=/tmp/metric_cache \
split=$SPLIT
