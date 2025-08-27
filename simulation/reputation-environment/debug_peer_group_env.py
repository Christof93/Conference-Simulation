import json
import os
import random

import numpy as np
from env.peer_group_environment import PeerGroupEnvironment
from stats_tracker import SimulationStats

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


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
    start_agents=20,
    max_agents=80,
    n_groups=4,
    max_peer_group_size=40,
    n_projects=6,
    max_projects_per_agent=5,
    max_agent_age=500,
    max_rewardless_steps=250,
)

obs, infos = env.reset(seed=SEED)
for i, agent in enumerate(env.possible_agents):
    env.action_space(agent).seed(SEED + i)

stats = SimulationStats()

# Prepare logs directory and files
logs_dir = os.path.join(os.path.dirname(__file__), "log")
os.makedirs(logs_dir, exist_ok=True)
jsonl_path = os.path.join(logs_dir, "stats.jsonl")
projects_jsonl_path = os.path.join(logs_dir, "projects.jsonl")
with open(jsonl_path, "w") as jf:
    jf.write("")
with open(projects_jsonl_path, "w") as jf:
    jf.write("")

for step in range(1000):
    actions = {}
    for agent in env.agents:
        actions[agent] = env.action_space(agent).sample(mask=env.action_masks[agent])
    obs, rewards, terminations, truncations, infos = env.step(actions)
    stats.update(env, obs, rewards, terminations, truncations)
    print(f"\nStep {step+1}")
    for agent in env.agents:
        if agent == "agent_0" and step > 50:
            #     print(env._get_active_projects(0))
            #     print(env.action_masks[agent])
            print(env.agent_active_projects[0])
        obs_converted = convert_numpy(obs[agent])
        # if agent=="agent_0":
        #     print(f"{agent} obs: {json.dumps(obs_converted, indent=2)}")

        # Uncomment and fix the rewards line too
        rewards_converted = convert_numpy(rewards[agent])
        # print(f"{agent} reward: {json.dumps(rewards_converted, indent=2)}")
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

# Write project-level details to JSONL
project_details = stats.get_project_details()
for project in sorted(project_details, key=lambda x: x["start_time"]):
    with open(projects_jsonl_path, "a") as jf:
        jf.write(json.dumps(project) + "\n")

print(
    f"\nSimulation completed. Logged {len(project_details)} finished projects to {projects_jsonl_path}"
)
