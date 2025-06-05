# The Predictive Driver Model Score (PDMS)

Fair comparisons are challenging in the open-loop planning literature, due to metrics of narrow scope or inconsistent definitions between different projects. The PDM Score is a combination of five sub-metrics, which provides a comprehensive analysis of different aspects of driving performance. Five of these sub-metrics are discrete-valued, and one is continuous. All metrics are computed after a 4-second non-reactive simulation of the planner output: background actors follow their recorded future trajectories, and the ego vehicle moves based on an LQR controller. The full composition of the PDM Ì is detailed below:

Metric | Weight | Range |
|---|---|---|
No at-fault Collisions (NC) | multiplier | {0, 1/2, 1} |
Drivable Area Compliance (DAC) | multiplier | {0, 1} |
Time to Collision (TTC) within bound | 5 | {0, 1} |
Ego Progress (EP) | 5 | [0, 1] |
Comfort (C) | 2 | {0, 1} |
Driving Direction Compliance (DDC) | 0 | {0, 1/2, 1} |

**Note:** The Driving Direction Compliance (DDC) metric is ignored in PDMS (with zero weight).

i.e., `PDMS = NC * DAC * (5*TTC + 5*EP + 2*C + 0*DDC) / 12`


To evaluate the PDM score for an agent you can run:
```bash
cd $NAVSIM_DEVKIT_ROOT/scripts/
./run_cv_pdm_score_evaluation.sh
```

By default, this will generate an evaluation csv for a simple constant velocity [planning baseline](https://github.com/autonomousvision/navsim/blob/main/docs/agents.md#output). You can modify the script to evaluate your own planning agent.

For instance, you can add a new config for your agent under `$NAVSIM_DEVKIT_ROOT/navsim/navsim/planning/script/config/pdm_scoring/agent/my_new_agent.yaml`. 
Then, running your own agent is as simple as adding an override `agent=my_new_agent` to the script.
You can find an example in `run_human_agent_pdm_score_evaluation.sh`
