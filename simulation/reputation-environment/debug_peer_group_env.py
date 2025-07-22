import numpy as np
from env.peer_group_environment import PeerGroupEnvironment

# Initialize environment
env = PeerGroupEnvironment(
    n_agents=8,
    peer_group_size=4,
    n_projects=6,
    max_projects_per_agent=2,
    max_timesteps=100,
    max_rewardless_steps=100,
)

obs, infos = env.reset()
print("Initial observations:")
for agent, o in obs.items():
    print(f"{agent}: {o}\n")

for step in range(10):
    actions = {}
    for agent in env.agents:
        actions[agent] = env.action_space(agent).sample(mask=env.action_masks[agent])
    obs, rewards, terminations, truncations, infos = env.step(actions)
    print(f"\nStep {step+1}")
    for agent in env.agents:
        print(f"{agent} obs: {obs[agent]}")
        print(f"{agent} reward: {rewards[agent]}")
        print(
            f"{agent} terminated: {terminations[agent]}, truncated: {truncations[agent]}"
        )
