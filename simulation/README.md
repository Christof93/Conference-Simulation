# Policy Simulation - Quick Start

## Prerequisites
- Python 3.9+
- Use a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Simulation
Runs a single simulation with a balanced policy mix and writes logs and a summary.

```bash
python reputation-environment/run_policy_simulation.py
```

Outputs (written in the repo root and `log/`):
- `balanced_summary.json`: high-level results (steps, rewards, success rate, policy distribution)
- `log/balanced_actions.jsonl`: actions taken each step (JSONL)
- `log/balanced_observations.jsonl`: observations per step (JSONL)
- `log/balanced_projects.json`: final project states
- `log/balanced_area.pickle`: serialized environment state

Tune the call to `run_simulation_with_policies(...)` at the bottom of `reputation-environment/run_policy_simulation.py` to change:
- `n_agents`, `max_steps`, `n_groups`, `max_peer_group_size`
- `policy_distribution` (see `POLICY_CONFIGS` in the same file)
- `output_file_prefix` (affects filenames)

## Comparing Policy Distributions
`compare_policy_performances()` runs a batch across the predefined `POLICY_CONFIGS`. Uncomment its call at the bottom of the script to print a compact comparison summary.

## Agent Policies (in `reputation-environment/agent_policies.py`)

- Careerist (`careerist`)
  - Picks high-prestige opportunities above a threshold.
  - Collaborates with active peers at/above the active-peer average reputation.
  - Effort goes to the closest-deadline running project that still needs work.

- Orthodox Scientist (`orthodox_scientist`)
  - Prefers lowest-novelty opportunities; ties break toward higher prestige.
  - Collaborates with all active peers who have close topic centroids.
  - Effort prioritizes projects already below 90% of required effort; otherwise best peer fit.

- Mass Producer (`mass_producer`)
  - Take project if the (effort × time window) is realtively low.
  - Collaborates with all active peers within the action mask.
  - Effort goes to the closest-deadline running project that still needs work.

Shared safety: if a running project risks missing its requirement given remaining time, policies skip selecting a new project that step.

## Tips
- JSONL logs can be large; use `jq`, `tail -f`, or sample lines.
- Commit parameter changes alongside their `*_summary.json` for reproducibility.
