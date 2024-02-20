# Download and installation

To get started with NAVSIM: 

### 1. Download the demo data
First, you need to download the OpenScene mini logs and sensor blobs, as well as the nuPlan maps.

**NOTE: Please check the [LICENSE file](https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/LICENSE) before downloading the data.**

```
wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip && unzip nuplan-maps-v1.1.zip
wget https://s3.eu-central-1.amazonaws.com/avg-projects-2/navsim/navsim_logs.zip && unzip navsim_logs.zip
wget https://s3.eu-central-1.amazonaws.com/avg-projects-2/navsim/sensor_blobs.zip && unzip sensor_blobs.zip
```
The `sensor_blobs` file is fairly large (90 GB). For understanding the metrics and testing the naive baselines in the demo, this is not strictly necessary.

### 2. Install the navsim-devkit
Next, setup the environment and install navsim.
Clone the repository
```
git clone https://github.com/kashyap7x/navsim.git
cd navsim
```
Then create a new environment and install the required dependencies:
```
conda env create --name navsim -f environment.yml
conda activate navsim
pip install -e .
```

Set the required environment variables, by adding the following to your `~/.bashrc` file
```
export NAVSIM_DEVKIT_ROOT=/path/to/navsim/devkit
export NUPLAN_EXP_ROOT=/path/to/navsim/exp
export NUPLAN_MAPS_ROOT=/path/to/nuplan/maps
export OPENSCENE_DATA_ROOT=/path/to/openscene
```
