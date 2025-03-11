wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_private_test_e2e.tgz
tar -xzf openscene_metadata_private_test_e2e.tgz
rm openscene_metadata_private_test_e2e.tgz
mv openscene-v1.1/meta_datas/ private_test_e2e_navsim_logs
rm -r openscene-v1.1

wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_private_test_e2e.tgz
tar -xzf openscene_sensor_private_test_e2e.tgz
rm openscene_sensor_private_test_e2e.tgz
mv competition_test private_test_e2e_sensor_blobs
