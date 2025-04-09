# Download and installation

To get started with NAVSIM:

### 1. Clone the navsim-devkit

Clone the repository

```bash
git clone https://github.com/autonomousvision/navsim.git
cd navsim
```

### 2. Download the dataset

You need to download the OpenScene logs and sensor blobs, as well as the nuPlan maps.
We provide scripts to download the nuplan maps, the mini split and the test split.
Navigate to the download directory and download the maps

**NOTE: Please check the [LICENSE file](https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/LICENSE) before downloading the data.**

```bash
cd download && ./download_maps
```

Next download the data splits you want to use.
Note that the dataset splits do not exactly map to the recommended standardized training / test splits-
Please refer to [splits](splits.md) for an overview on the standardized training and test splits including their size and check which dataset splits you need to download in order to be able to run them.

You can download the mini, trainval, test, private_test_e2e and warmup_synthetic_scenes dataset split with the following scripts

```bash
./download_mini
./download_trainval
./download_test
./download_private_test_e2e
./download_warmup_two_stage
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
    |    ├── test
    |    ├── trainval
    |    ├── private_test_e2e
    |    └── mini
    └── warmup_two_stage
         ├── openscene_meta_datas
	 ├── sensor_blobs
	 ├── synthetic_scene_pickles
         └── synthetic_scenes_attributes.csv

```

⚠️ **IMPORTANT:** If you have already downloaded the data for Navsim V2.0.1 and tried the Hugging Face Leaderboard, please replace the old `"synthetic_scenes"` folder with the new `"warmup_two_stage"` folder. In Navsim V2.1, the traffic agents' policy has been updated, and the old data is no longer compatible.

Set the required environment variables, by adding the following to your `~/.bashrc` file
Based on the structure above, the environment variables need to be defined as:

```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="$HOME/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="$HOME/navsim_workspace/navsim"
export OPENSCENE_DATA_ROOT="$HOME/navsim_workspace/dataset"
```

### 3. Install the navsim-devkit

Finally, install navsim.
To this end, create a new environment and install the required dependencies:

```bash
conda env create --name navsim -f environment.yml
conda activate navsim
pip install -e .
```
