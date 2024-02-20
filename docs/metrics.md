# Understanding the PDM Score

Fair comparisons are challenging in the open-loop planning literature, due to metrics of narrow scope or inconsistent definitions between different projects. The PDM Score is a combination of six sub-metrics, which provides a comprehensive analysis of different aspects of driving performance. Five of these sub-metrics are discrete-valued, and one is continuous. The final composition is detailed below:

Metric | Weight | Range |
|---|---|---|
No at-fault Collisions (NC) | multiplier | {0, 1/2, 1} |
Drivable Area Compliance (DAC) | multiplier | {0, 1} |
Driving Direction Compliance (DDC) | multiplier | {0, 1/2, 1} |
Time to Collision (TTC) within bound | 5 | {0, 1} |
Comfort (C) | 2 | {0, 1} |
Ego Progress (EP) | 5 | [0, 1] |

i.e., `PDM Score = NC * DAC * DDC * (5*TTC + 2*C + 5*EP) / 12`

To evaluate the PDM score for an agent you can run:
```
cd $NAVSIM_DEVKIT_ROOT/scripts/
./run_pdm_score_evaluation.sh
```
**Note: You have to adapt the variables `LOG_PATH` so that it points to the logs (annotations), `METRIC_CACHE_PATH` so that it points to the metric-cache and `OUTPUT_DIR` so that it points to a directory where the evaluation csv will be stored**

By default, this will generate an evaluation csv for a simple constant velocity planning baseline. You can modify the script to evaluate your own planning agent.
For instance, you can add a new config for your agent under `$NAVSIM_DEVKIT_ROOT/navsim/navsim/planning/script/config/pdm_scoring/agent/my_new_agent.yaml`. 
Then, running your own agent is as simple as adding an override `agent=my_new_agent` to the script.
