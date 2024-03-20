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

Next download the mini split and the test split
```
./download_mini
./download_test
```

**The mini split and the test split take around ~160GB and ~220GB of memory respectively**

This will download the splits into the download directory. From there, move it to create the following structure.
```angular2html
~/navsim_workspace
├── navsim (containing the devkit)
├── exp
└── dataset
    ├── maps
    ├── navsim_logs
    |    ├── test
    │    └── mini
    └── sensor_blobs
         ├── test
         └── mini
```
Set the required environment variables, by adding the following to your `~/.bashrc` file
Based on the structure above, the environment variables need to be defined as:
```
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