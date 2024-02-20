# Understanding and creating agents

Defining an agent starts by creating a new class that inherits from `navsim.agents.abstract_agent.AbstractAgent`.

Let’s dig deeper into this class.

## Inputs

`get_sensor_modalities()` can be overwritten to determine which sensors are accessible to the agent. The available sensors depend on the dataset. For OpenScene, this includes 8 cameras and a 360 degree LiDAR, available for a duration of 2 seconds into the past, at a frquency of 2Hz.

**Why LiDAR?** Recent literature on open-loop planning has opted away from LiDAR in favor of using all 8 cameras. This has significantly strained the compute requirements for training and testing SoTA planners. We hope that the availability of the LiDAR modality enables more computationally efficient submissions that use fewer (or low-resolution) camera inputs. 

**Ego Status.** Besides the sensor data, an agent also receives the ego pose, velocity and acceleration information in local coordinates. Finally, to disambiguate driver intention, we provide a discrete driving command, indicating whether the intended route is towards the left, straight or right direction. Importantly, the driving command in NAVSIM is based solely on the desired route, and does not entangle information regarding obstacles and traffic signs (as was prevalent on prior benchmarks such as nuScenes). Note that the left and right driving commands cover turns, lane changes and sharp curves.

## Output

Given this input, you will need to override the `compute_trajectory()` method and output a `Trajectory`. This is an array of BEV poses (with x, y and heading in local coordinates), as well as a `TrajectorySampling` config object that indicates the duration and frequency of the trajectory. The PDM score is evaluated for a horizon of 4 seconds at a frequency of 10Hz. The `TrajectorySampling` config facilitates interpolation when the output frequency is different from the one used during evaluation.

We provide a naive constant velocity agent as part of our demo, for reference:

https://github.com/autonomousvision/navsim/blob/51cecd51aa70b0e6bcfb3541b91ae88f2a78a25e/navsim/agents/constant_velocity_agent.py#L9

