SPLIT=mini

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
experiment_name=metric_caching \
job_name=metric_caching \
+splitter=nuplan \
cache.use_open_scene=true \
worker=ray_distributed_no_torch \
split=$SPLIT