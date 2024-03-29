SPLIT=test

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
split=$SPLIT \
cache.cache_path=/home/aah1si/openscene/exp/public_test_metric_cache \
scene_filter.frame_interval=1 \
