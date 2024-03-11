wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_test.tgz
tar -xzf openscene_metadata_test.tgz
rm openscene_metadata_test.tgz

for split in {0..31}; do
    wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_test_camera/openscene_sensor_test_camera_${split}.tgz
    echo "Extracting file openscene_sensor_test_camera_${split}.tgz"
    tar -xzf openscene_sensor_test_camera_${split}.tgz
    rm openscene_sensor_test_camera_${split}.tgz
done

for split in {0..31}; do
    wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_test_lidar/openscene_sensor_test_lidar_${split}.tgz
    echo "Extracting file openscene_sensor_test_lidar_${split}.tgz"
    tar -xzf openscene_sensor_test_lidar_${split}.tgz
    rm openscene_sensor_test_lidar_${split}.tgz
done

mv openscene_v1.1/meta_datas test_navsim_logs
rm -r openscene_v1.1
mkdir 
mv openscene-v1.1/sensor_blobs test_sensor_blobs
rm -r openscene-v1.1