wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_trainval.tgz
tar -xzf openscene_metadata_trainval.tgz
rm openscene_metadata_trainval.tgz
mv openscene-v1.1/meta_datas trainval_navsim_logs
rm -r openscene-v1.1

mkdir -p trainval_sensor_blobs/trainval
for split in {1..32}; do
    wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/navsim/navtrain_current_${split}.tgz
    echo "Extracting file navtrain_current_${split}.tgz"
    tar -xzf navtrain_current_${split}.tgz
    rm navtrain_current_${split}.tgz

    rsync -rv navtrain_current_${split}/* trainval_sensor_blobs/trainval
    rm -r navtrain_current_${split}
done

for split in {1..32}; do
    wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/navsim/navtrain_history_${split}.tgz
    echo "Extracting file navtrain_history_${split}.tgz"
    tar -xzf navtrain_history_${split}.tgz
    rm navtrain_history_${split}.tgz

    rsync -rv navtrain_history_${split}/* trainval_sensor_blobs/trainval
    rm -r navtrain_history_${split}
done
