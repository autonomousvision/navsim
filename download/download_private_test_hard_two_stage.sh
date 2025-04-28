wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/navsim-v2/navsim_v2.2_private_test_hard_two_stage.tar.gz
tar -xzf navsim_v2.2_private_test_hard_two_stage.tar.gz
rm navsim_v2.2_private_test_hard_two_stage.tar.gz

wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_private_test_hard.tar.gz
tar -xzf openscene_metadata_private_test_hard.tar.gz
rm openscene_metadata_private_test_hard.tar.gz
mv openscene-v1.1/meta_datas/ private_test_hard_navsim_log
rm -r openscene-v1.1
