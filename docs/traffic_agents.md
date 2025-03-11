# Traffic Agent Policies

NAVSIM v2 introduces support for **reactive traffic agents**, allowing surrounding vehicles to respond to the ego-vehicle. However, the ego-vehicle itself remains **nonreactive**, no environment updates will be looped back into the planner. As in NAVSIM v1, the ego-agent must commit to a single plan, which is executed for the entire simulation horizon.

### Available Traffic Agent Policies

1. **Log-Replay** (Non-Reactive)
   - Identical to NAVSIM v1, traffic agents strictly follow recorded trajectories without reacting to the ego-vehicle.

2. **Constant-Velocity** (Debugging Only)
   - Traffic agents move in a straight line at a fixed velocity, providing a simple baseline for debugging.

3. **IDM (Intelligent Driver Model)**
   - Similar to nuPlan, this model simulates traffic agents with more realistic behavior, adjusting speed and spacing based on road conditions.
   - Pedestrians, static objects, and other non-vehicle agents still follow pre-recorded log data.

### Selecting a Traffic Agents Policy
Traffic agent policies can be selected by specifying an override when running the evaluation. For example:

`traffic_agents_policy=navsim_IDM_traffic_agents`

This allows to easily switch between different policies depending on your evaluation needs.

All available traffic agents policies can be found [here](navsim/planning/script/config/common/traffic_agents_policy.md)

### Learning-based Traffic Simulation

We aim to expand NAVSIM v2 with additional traffic policies, particularly learning-based models.
