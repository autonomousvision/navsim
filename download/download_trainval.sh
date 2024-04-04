wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_trainval.tgz
tar -xzf openscene_metadata_trainval.tgz
rm openscene_metadata_trainval.tgz

for split in {0..199}; do
    wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_trainval_camera/openscene_sensor_trainval_camera_${split}.tgz
    echo "Extracting file openscene_sensor_trainval_camera_${split}.tgz"
    tar -xzf openscene_sensor_trainval_camera_${split}.tgz
    rm openscene_sensor_trainval_camera_${split}.tgz
done

for split in {0..199}; do
    wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_trainval_lidar/openscene_sensor_trainval_lidar_${split}.tgz
    echo "Extracting file openscene_sensor_trainval_lidar_${split}.tgz"
    tar -xzf openscene_sensor_trainval_lidar_${split}.tgz
    rm openscene_sensor_trainval_lidar_${split}.tgz
done

mv openscene-v1.1/meta_datas trainval_navsim_logs
mv openscene-v1.1/sensor_blobs trainval_sensor_blobs
rm -r openscene-v1.1