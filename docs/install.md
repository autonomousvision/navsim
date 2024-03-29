# Download and installation

To get started with NAVSIM: 

### 1. Clone the navsim-devkit
Clone the repository
```
git clone https://github.com/autonomousvision/navsim.git
cd navsim
```
### 2. Download the demo data
You need to download the OpenScene logs and sensor blobs, as well as the nuPlan maps.
We provide scripts to download the nuplan maps, the mini split and the test split.
Navigate to the download directory and download the maps

**NOTE: Please check the [LICENSE file](https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/LICENSE) before downloading the data.**

```
cd download && ./download_maps
```

Next download the data splits you want to use.
Note that the dataset splits do not exactly map to the recommended standardized training / test splits-
Please refer to [splits](splits.md) for an overview on the standardized training and test splits including their size and check which dataset splits you need to download in order to be able to run them.

You can download the mini, trainval, test and private_test_e2e dataset split with the following scripts
```
./download_mini
./download_trainval
./download_test
./download_private_test_e2e
```
Also, the script `./download_navtrain` can be used to download a small portion of the  `trainval` dataset split which is needed for the `navtrain` training split. 

This will download the splits into the download directory. From there, move it to create the following structure.
```angular2html
~/navsim_workspace
├── navsim (containing the devkit)
├── exp
└── dataset
    ├── maps
    ├── navsim_logs
    |    ├── test
    |    ├── trainval
    |    ├── private_test_e2e
    │    └── mini
    └── sensor_blobs
         ├── test
         ├── trainval
         ├── private_test_e2e
         └── mini
```
Set the required environment variables, by adding the following to your `~/.bashrc` file
Based on the structure above, the environment variables need to be defined as:
```
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="$HOME/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="$HOME/navsim_workspace/navsim"
export OPENSCENE_DATA_ROOT="$HOME/navsim_workspace/dataset"
```

### 3. Install the navsim-devkit
Finally, install navsim.
To this end, create a new environment and install the required dependencies:
```
conda env create --name navsim -f environment.yml
conda activate navsim
pip install -e .
```