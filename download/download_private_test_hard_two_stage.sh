wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/navsim-v2/navsim_v2.2_private_test_hard_two_stage.tar.gz
tar -xzf navsim_v2.2_private_test_hard_two_stage.tar.gz
rm navsim_v2.2_private_test_hard_two_stage.tar.gz

wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/navsim-v2/navsim_v2.2_private_test_hard_openscene_metadata.tar.gz
tar -xzf navsim_v2.2_private_test_hard_two_stage_navsim_log.tar.gz
rm navsim_v2.2_private_test_hard_two_stage_navsim_log.tar.gz
mv openscene-v1.1/meta_datas/ private_test_hard_navsim_log
rm -r openscene-v1.1
