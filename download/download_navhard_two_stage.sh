wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/navsim-v2/navsim_v2.2_navhard_two_stage_curr_sensors.tar.gz
wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/navsim-v2/navsim_v2.2_navhard_two_stage_hist_sensors.tar.gz
wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/navsim-v2/navsim_v2.2_navhard_two_stage_scene_pickles.tar.gz
tar -xzvf navsim_v2.2_navhard_two_stage_curr_sensors.tar.gz
tar -xzvf navsim_v2.2_navhard_two_stage_hist_sensors.tar.gz
tar -xzvf navsim_v2.2_navhard_two_stage_scene_pickles.tar.gz
rm navsim_v2.2_navhard_two_stage_curr_sensors.tar.gz
rm navsim_v2.2_navhard_two_stage_hist_sensors.tar.gz
rm navsim_v2.2_navhard_two_stage_scene_pickles.tar.gz
