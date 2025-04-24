# The Extended Predictive Driver Model Score (EPDMS)

Fair comparisons are challenging in the open-loop planning literature, due to metrics of narrow scope or inconsistent definitions between different projects. To address this, NAVSIM v1 introduced the PDM Score (PDMS), a combination of several subscores and multiplicative penalties.

In NAVSIM v2, the EPDMS extends the PDMS, introducing:
- 2 new weighted metrics (LK and EC)
- 2 new multiplier metrics (DDC and TLC)
- False-positive penalty filtering

The Lane Keeping subscore (LK) penalizes driving too far from the centerline for an extended time. It is disabled on intersections where the centerline annotations often don't match the actual lane markings perceived by the sensors. Besides, NAVSIM v2 puts stronger emphasis on comfortable driving. The existing Comfort (C) subscore from NAVSIM v1 was slightly improved to also evaluate how the planned trajectory matches the vehicle's motion history, giving History Comfort (HC). Moreover, the new extended comfort (EC) subscore compares the trajectory outputs of subsequent frames and their resulting dynamic states. A discrepancy in acceleration, jerk etc. between subsequent frames results in uncomfortable behavior and thus a lower score.

The new Driving Direction Compliance (DDC) and Traffic Light Compliance (TLC) subscores extend the inadmissible behaviors detected and penalized in evaluation. Further, to reduce false positive penalties, we disable penalties when the human agent is also responsible for a violation. This ensures that the planner is not unfairly penalized in situations where breaking a rule is necessary to achieve a valid driving goal. For example, if the agent must briefly enter the oncoming lane to overtake a static obstacle.

The full composition of the EPDMS is detailed below (new metrics are marked in bold):

Metric | Weight | Range |
|---|---|---|
No at-fault Collisions (NC) | multiplier | {0, 1/2, 1} |
Drivable Area Compliance (DAC) | multiplier | {0, 1} |
**Driving Direction Compliance** (DDC) | multiplier | {0, 1/2, 1} |
**Traffic Light Compliance** (TLC) | multiplier | {0, 1} |
Ego Progress (EP) | 5 | [0, 1] |
Time to Collision (TTC) within bound | 5 | {0, 1} |
**Lane Keeping** (LK)  | 2 | {0, 1} |
**History Comfort** (HC) | 2 | {0, 1} |
**Extended Comfort** (EC) | 2 | {0, 1} |

The full EPDMS is defined as:

<br>


$$\text{EPDMS} = \left(\prod_{m\in\\{NC, DAC, DDC, TLC\\}} \text{filter}\_m(\text{agent}, \text{human})\right) \cdot  \left( \frac{\sum_{m \in \\{TTC, EP, HC, LK, EC\\}} w_m \cdot \text{filter}\_m(\text{agent}, \text{human}) }{\sum_{m\in \\{TTC, EP, HC, LK, EC\\}} w_m}\right)$$

$$\text{with}\quad \text{filter}_m(\text{agent}, \text{human}) = \begin{cases}
1.0 & \text{if } m(\text{human}) = 0 \\
m(\text{agent}) & \text{otherwise.}
\end{cases}$$
<!-- TODO: remove -->
<!-- Alternatively use: -->
<!-- $$\text{with}\quad \text{filter}_m(\text{agent}, \text{human}) = \mathbf{1}_{m(\text{human})\neq 0} \cdot m(\text{agent}) + 1.0 \cdot m(\text{human}).$$ -->
<br>

For reference, the PDMS used in NAVSIM v1 which used a slightly different version of HC called Comfort (C), was defined as:

<br>

$$ \text{PDMS} = \left(\prod_{m\in\\{NC, DAC\\}} m({\text{agent}})\right) \cdot \left(\frac{\sum_{m\in \\{TTC, EP, C\\}} w_m\cdot m(agent)}{\sum_{m\in \\{TTC, EP, C\\}} w_m}\right)$$

<br>
<br>

# Pseudo Closed-Loop Aggregation

In NAVSIM v1, all metrics were computed after a 4-second non-reactive simulation of the planner output: background actors followed their recorded future trajectories, and the ego vehicle moves based on an LQR controller.

The new NAVSIM v2 evaluation uses a two-stage aggregation process to approximate closed-loop behavior while keeping the setting open-loop. Here's how it works:

1. **First Stage Scoring:**
   - We evaluate an initial scene over a fixed horizon (4 seconds) using the EPDMS metric.
   - This stage follows a similar simulation procedure to NAVSIM v1.

2. **Second Stage Scoring:**
   - In addition to the initial scene, multiple potential follow-up scenes to this initial scene are included in the test set to be evaluated.
   - The follow-up scenes were pre-computed by rolling out several simulations starting from the initial scene, each with a different 4-second plan.
   - Therefore, each of these follow-up scenes starts from a state different to the endpoint of the initial scene, e.g. with a lateral offset or different speed.
   - We evaluate the submitted planner on each follow-up scene over a fixed horizon (4 seconds) using the EPDMS metric.

3. **Weighting and Aggregation:**
   - To emulate the effects of closed-loop simulation, the relevance of each follow-up scene to the overall score depends on how close its starting position is to where the submitted planner actually ended in the first stage.
   - We assign higher weights to follow-up scenes that start closer to the submitted planner's end position, with a gaussian kernel.
   - We first compute a weighted aggregation if all second-stage scores. Finally, we aggregate the scores of the first and second stage via a simple multiplication to produce the aggregated metric.

# Run an evaluation
To evaluate the PDM score for an agent you can run:
```bash
cd $NAVSIM_DEVKIT_ROOT/scripts/evaluation/
./run_cv_pdm_score_evaluation.sh
```

By default, this will generate an evaluation csv for a simple constant velocity [planning baseline](https://github.com/autonomousvision/navsim/blob/main/docs/agents.md#output). You can modify the script to evaluate your own planning agent.

For instance, you can add a new config for your agent under `$NAVSIM_DEVKIT_ROOT/navsim/navsim/planning/script/config/common/agent/my_new_agent.yaml`.
Then, running your own agent is as simple as adding an override `agent=my_new_agent` to the script.
You can find an example in `run_human_agent_pdm_score_evaluation.sh`
