import csv
import json
import os

import numpy as np
from env.peer_group_environment import PeerGroupEnvironment
from stats_tracker import SimulationStats


# Convert numpy arrays to lists for JSON serialization (handles nested structures)
def convert_numpy(obj):
    if hasattr(obj, "tolist"):  # numpy arrays
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    return obj


# Initialize environment
env = PeerGroupEnvironment(
    n_agents=8,
    peer_group_size=4,
    n_projects=6,
    max_projects_per_agent=2,
    max_timesteps=100,
    max_rewardless_steps=100,
)

obs, infos = env.reset(seed=123)
stats = SimulationStats()

# Prepare logs directory and files
logs_dir = os.path.join(os.path.dirname(__file__), "log")
os.makedirs(logs_dir, exist_ok=True)
jsonl_path = os.path.join(logs_dir, "stats.jsonl")
csv_path = os.path.join(logs_dir, "stats.csv")

# Create CSV with header if missing
if not os.path.exists(csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats.csv_row().keys()))
        writer.writeheader()

for step in range(100):
    actions = {}
    for agent in env.agents:
        env.action_space(agent).seed(4)
        actions[agent] = env.action_space(agent).sample(mask=env.action_masks[agent])
    obs, rewards, terminations, truncations, infos = env.step(actions)
    stats.update(env, obs, rewards, terminations, truncations)
    print(f"\nStep {step+1}")
    for agent in env.agents:

        obs_converted = convert_numpy(obs[agent])
        print(f"{agent} obs: {json.dumps(obs_converted, indent=2)}")

        # Uncomment and fix the rewards line too
        rewards_converted = convert_numpy(rewards[agent])
        print(f"{agent} reward: {json.dumps(rewards_converted, indent=2)}")
        print(
            f"{agent} terminated: {terminations[agent]}, truncated: {truncations[agent]}"
        )
        # breakpoint()
    # Print a concise stats summary each step
    print(f"Stats: {stats.summary_line()}")

    # Append JSONL row
    with open(jsonl_path, "a") as jf:
        jf.write(json.dumps(stats.to_dict()) + "\n")

# Write final summary JSON
final_summary_path = os.path.join(logs_dir, "final_summary.json")
with open(final_summary_path, "w") as sf:
    json.dump(stats.to_dict(), sf, indent=2)
