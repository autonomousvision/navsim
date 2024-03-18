# Understanding and creating agents

Defining an agent starts by creating a new class that inherits from `navsim.agents.abstract_agent.AbstractAgent`.

Let’s dig deeper into this class. It has to implement the following methods:
- `__init__()`: 

    The constructor of the agent.
- `name()`

    This has to return the name of the agent. 
    The name will be used to define the filename of the evaluation csv.
    You can set this to an arbitrary value. 
- `initialize()`

    This will be called before inferring the agent for the first time.
    If multiple workers are used, every worker will call this method for its instance of the agent.
    If you need to load a state dict etc., you should do it here instead of in `__init__`.
- `get_sensor_config()`

    Has to return a `SensorConfig` (see `navsim.common.dataclasses.SensorConfig`) to define which sensor modalities should be loaded for the agent in each frame.
    The SensorConfig is a dataclass that stores for each sensor a List of indices of history frames for which the sensor should be loaded. Alternatively, a boolean can be used for each sensor, if all available frames should be loaded. 
    Moreover, you can return `SensorConfig.build_all_sensors()` if you want to have access to all available sensors.
    Details on the available sensors can be found below.
    
    **Loading the sensors has a big impact on runtime. If you don't need a sensor, consider to set it to `False`.**
- `compute_trajectory()`

    This is the main function of the agent. Given the `AgentInput` which contains the ego state as well as sensor modalities, it has to compute and return a future trajectory for the Agent.
    Details on the output format can be found below.
    
    **The future trajectory has to be returned as an object of type `from navsim.common.dataclasses.Trajectory`. For examples, see the constant velocity agent or the human agent.**
## Inputs

`get_sensor_config()` can be overwritten to determine which sensors are accessible to the agent. 

The available sensors depend on the dataset. For OpenScene, this includes 9 sensor modalities: 8 cameras and a merged point cloud (from 5 LiDARs). Each modality is available for a duration of 2 seconds into the past, at a frquency of 2Hz (i.e., 4 frames). Only this data will be released for the test frames (no maps/tracks/occupancy etc, which you may use during training but will not have access to for leaderboard submissions).

You can configure the set of sensor modalities to use and how much history you need for each frame with the `navsim.common.dataclasses.SensorConfig` dataclass.

**Why LiDAR?** Recent literature on open-loop planning has opted away from LiDAR in favor of using surround-view high-resolution cameras. This has significantly strained the compute requirements for training and testing SoTA planners. We hope that the availability of the LiDAR modality enables more computationally efficient submissions that use fewer (or low-resolution) camera inputs. 

**Ego Status.** Besides the sensor data, an agent also receives the ego pose, velocity and acceleration information in local coordinates. Finally, to disambiguate driver intention, we provide a discrete driving command, indicating whether the intended route is towards the left, straight or right direction. Importantly, the driving command in NAVSIM is based solely on the desired route, and does not entangle information regarding obstacles and traffic signs (as was prevalent on prior benchmarks such as nuScenes). Note that the left and right driving commands cover turns, lane changes and sharp curves.

## Output

Given this input, you will need to override the `compute_trajectory()` method and output a `Trajectory`. This is an array of BEV poses (with x, y and heading in local coordinates), as well as a `TrajectorySampling` config object that indicates the duration and frequency of the trajectory. The PDM score is evaluated for a horizon of 4 seconds at a frequency of 10Hz. The `TrajectorySampling` config facilitates interpolation when the output frequency is different from the one used during evaluation.

Check out this simple constant velocity agent for an example agent implementation:
https://github.com/autonomousvision/navsim/blob/51cecd51aa70b0e6bcfb3541b91ae88f2a78a25e/navsim/agents/constant_velocity_agent.py#L9
