# Independent candidate PDM diagnostics (NAVSIM v1)

This utility evaluates exported trajectories without loading the originating model.
It supports any positive number of candidates, including K=1 and K=64. It calls
`pdm_score` separately for each candidate using the same loaded scene cache.
It does **not** batch all candidates into `PDMScorer.score_proposals`: doing so
changes progress normalization and can make a candidate's score depend on the
other candidates. It is sequential, not an accelerated scoring implementation.

## Input and usage

Export a NumPy `.npz` file containing:

- `tokens`: Unicode array `[N]` of unique current-frame scene tokens.
- `poses`: finite array `[N, K, 8, 3]`, with local ego-frame `(x, y, heading)`
  at 0.5, 1.0, ..., 4.0 seconds, using NAVSIM's trajectory convention.
- `selected_index`: integer array `[N]`, containing the model-selected candidate
  index in `[0, K)`, recorded **before** looking at evaluation scores.

For example, after inference:

```python
np.savez_compressed(
    "predictions.npz",
    tokens=np.asarray(scene_tokens, dtype=str),
    poses=candidate_poses,
    selected_index=np.asarray(model_choices, dtype=np.int64),
)
```

Use a metric cache created for the same split in the NAVSIM v1 environment:

```bash
python -m navsim.planning.script.run_candidate_pdm_score \
  --predictions predictions.npz \
  --metric-cache-path /path/to/metric_cache \
  --output-dir /path/to/new/candidate_results
```

The cache uses the existing `MetricCacheLoader` metadata layout. Missing scene
caches or invalid inputs fail the run rather than silently reducing coverage.
The output directory must not already exist. Run once per model to keep outputs
separate. The CLI uses default v1 simulator/scorer settings (40 steps at 0.1 s).

## Output and interpretation

All scores are on **[0, 1]**, not [0, 100].

- `candidates.csv`: token, candidate index, six official submetrics and total score.
- `scenes.csv`: model-selected index and each scene's selected, mean, oracle and gap.
- `summary.json`: scene/candidate counts and the unweighted scene means of those
  diagnostics. This file is written only after every scene completes successfully;
  CSV files alone may be partial results of an interrupted run. There is no resume
  support in this initial utility.

Selected is the model's actual choice. Mean is the expected score of **uniform
random selection**, not a lower bound. Oracle is the maximum candidate score;
its gap is `oracle - selected`. Oracle uses evaluation information unavailable
at inference and **must not be reported as an agent's leaderboard result**.
These diagnostics do not change the official submission or evaluation protocol.

## Python API and tests

`navsim.evaluate.candidate_pdm_score.score_candidates` accepts a loaded cache,
a sequence of `Trajectory` objects, sampling, simulator and scorer. It returns
one `PDMResults` per candidate in input order. `summarize_candidates` summarizes
those results given the model's selected index. Parallel scene evaluation should
use worker-local simulators/scorers, not share these mutable objects across threads.

```bash
python -m pytest tests/evaluate/test_candidate_pdm_score.py -q
# Optional real-cache equivalence test; only load trusted pickle caches:
NAVSIM_TEST_METRIC_CACHE=/path/to/metric_cache.pkl \
  python -m pytest tests/evaluate/test_candidate_pdm_score.py -q
```

The optional test compares every submetric and total score against fresh
single-trajectory evaluations, checks reversed candidate order and checks K=1.
No dataset files or model predictions are included in this change.
