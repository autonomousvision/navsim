import numpy as np
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory


class ConstantVelocityAgent(AbstractAgent):
    """Constant velocity baseline agent."""

    requires_scene = False

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(
            time_horizon=4, interval_length=0.5
        ),
    ):
        super().__init__(trajectory_sampling)

    def name(self) -> str:
        """Inherited, see superclass."""

        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig.build_no_sensors()

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        """Inherited, see superclass."""

        ego_velocity_2d = agent_input.ego_statuses[-1].ego_velocity
        ego_speed = (ego_velocity_2d**2).sum(-1) ** 0.5

        num_poses, dt = (
            self._trajectory_sampling.num_poses,
            self._trajectory_sampling.interval_length,
        )
        poses = np.array(
            [
                [(time_idx + 1) * dt * ego_speed, 0.0, 0.0]
                for time_idx in range(num_poses)
            ],
            dtype=np.float32,
        )

        return Trajectory(poses, self._trajectory_sampling)
