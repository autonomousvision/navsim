# Traffic Agent Policies

NAVSIM v2 introduces support for **reactive traffic agents**, allowing surrounding vehicles to respond to the ego-vehicle. However, the ego-vehicle itself remains **nonreactive**, no environment updates will be looped back into the planner. As in NAVSIM v1, the ego-agent must commit to a single plan, which is executed for the entire simulation horizon.

### Available Traffic Agent Policies

1. **Log-Replay** (Non-Reactive)

   - Identical to NAVSIM v1, traffic agents strictly follow recorded trajectories without reacting to the ego-vehicle.
2. **Constant-Velocity** (Debugging Only)

   - Traffic agents move in a straight line at a fixed velocity, providing a simple baseline for debugging.
3. **IDM (Intelligent Driver Model)** (Reactive)

   - Similar to nuPlan, this model simulates traffic agents with more realistic behavior, adjusting speed and spacing based on road conditions.
   - Pedestrians, static objects, and other non-vehicle agents still follow pre-recorded log data.

### Selecting a Traffic Agents Policy

For single-stage simulation, you can specify the traffic agent policy by providing an override when running the evaluation script `navsim/planning/script/run_pdm_score_one_stage.py`.

An example can be found in the commented section of the script `run_cv_pdm_score_evaluation.sh`. For instance:

`traffic_agents=non_reactive` or `traffic_agents=reactive`

This makes it easy to switch between different traffic agent policies depending on your evaluation requirements.

In two-stage simulations (e.g., for Hugging Face submissions), reactive traffic agents are used by default.

All available traffic agents policies can be found [here](navsim/planning/script/config/common/traffic_agents_policy.md)

### Learning-based Traffic Simulation

We aim to expand NAVSIM v2 with additional traffic policies, particularly learning-based models.
