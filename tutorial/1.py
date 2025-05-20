import os
from pathlib import Path

import hydra
from hydra.utils import instantiate
import numpy as np
import matplotlib.pyplot as plt

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig

SPLIT = "test"  # ["mini", "test", "trainval"]
FILTER = "warmup_navsafe_two_stage_extended"

hydra.initialize(config_path="/data/hdd01/dingzx/workspace/navsim/planning/script/config/common/train_test_split/scene_filter")
cfg = hydra.compose(config_name=FILTER)
scene_filter: SceneFilter = instantiate(cfg)
openscene_data_root = Path(os.getenv("OPENSCENE_DATA_ROOT"))
scene_loader = SceneLoader(
    "${oc.env:OPENSCENE_DATA_ROOT}/openscene-v1.1/meta_datas/${SPLIT}", # data_path
    "/data/hdd01/dingzx/dataset/synthetic_scenes/synthetic_sensor", # sensor_blobs_path
    "${oc.env:OPENSCENE_DATA_ROOT}/openscene-v1.1/sensor_blobs/${SPLIT}", # navsim_blobs_path
    "/data/hdd01/dingzx/dataset/synthetic_scenes/scene_pickles", # synthetic_scenes_path
    scene_filter,
    sensor_config=SensorConfig.build_all_sensors(),
)
token = np.random.choice(scene_loader.tokens)
scene = scene_loader.get_scene_from_token(token)


from navsim.visualization.plots import plot_bev_frame

frame_idx = scene.scene_metadata.num_history_frames - 1 # current frame
fig, ax = plot_bev_frame(scene, frame_idx)
plt.show()
from navsim.visualization.plots import plot_bev_with_agent
from navsim.agents.diffusiondrive.transfuser_agent import TransfuserAgent
from navsim.agents.abstract_agent_diffusiondrive import AbstractAgent

agent: AbstractAgent = instantiate()
# agent = ConstantVelocityAgent()
fig, ax = plot_bev_with_agent(scene, agent)
plt.show()
from navsim.visualization.plots import plot_cameras_frame

fig, ax = plot_cameras_frame(scene, frame_idx)
plt.show()
from navsim.visualization.plots import plot_cameras_frame_with_annotations

fig, ax = plot_cameras_frame_with_annotations(scene, frame_idx)
plt.show()

from navsim.visualization.plots import plot_cameras_frame_with_lidar

fig, ax = plot_cameras_frame_with_lidar(scene, frame_idx)
plt.show()

from navsim.visualization.plots import configure_bev_ax
from navsim.visualization.bev import add_annotations_to_bev_ax, add_lidar_to_bev_ax


fig, ax = plt.subplots(1, 1, figsize=(6, 6))

ax.set_title("Custom plot")

add_annotations_to_bev_ax(ax, scene.frames[frame_idx].annotations)
add_lidar_to_bev_ax(ax, scene.frames[frame_idx].lidar)

# configures frame to BEV view
configure_bev_ax(ax)

plt.show()

from navsim.visualization.plots import frame_plot_to_gif

frame_indices = [idx for idx in range(len(scene.frames))]  # all frames in scene
file_name = f"./{token}.gif"
images = frame_plot_to_gif(file_name, plot_cameras_frame_with_annotations, scene, frame_indices)