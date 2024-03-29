for split in {1..5}; do
    wget https://s3.eu-central-1.amazonaws.com/avg-projects-2/navsim/navtrain_current_${split}.tgz
    echo "Extracting file navtrain_current_${split}.tgz"
    tar -xzf navtrain_current_${split}.tgz
    rm navtrain_current_${split}.tgz

    mv navtrain_current_${split} navtrain
    rm -r navtrain_current_${split}
done

for split in {1..5}; do
    wget https://s3.eu-central-1.amazonaws.com/avg-projects-2/navsim/navtrain_history_${split}.tgz
    echo "Extracting file navtrain_history_${split}.tgz"
    tar -xzf navtrain_history_${split}.tgz
    rm navtrain_history_${split}.tgz

    mv navtrain_history_${split} navtrain
    rm -r navtrain_history_${split}
done
