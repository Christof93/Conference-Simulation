import numpy as np
from env.peer_group_environment import PeerGroupEnvironment

# Initialize environment
env = PeerGroupEnvironment(
    n_agents=5,
    peer_group_size=2,
    n_projects=3,
    max_projects_per_agent=2,
    max_timesteps=10,
    max_rewardless_steps=5,
    peer_group_growth_rate=0.5,
)

obs, infos = env.reset()
print("Initial observations:")
for agent, o in obs.items():
    print(f"{agent}: {o}\n")

for step in range(5):
    actions = {}
    for agent in env.agents:
        # Randomly choose a project or none
        choose_project = np.random.randint(0, env.n_projects + 1)
        # Randomly choose which peers to collaborate with
        peer_group = obs[agent]["observation"]["peer_group"]
        collaborate_with = np.random.randint(0, 2, size=peer_group.shape[0])
        # Randomly choose which running projects to put effort into
        put_effort = np.random.randint(0, 2, size=env.max_projects_per_agent)
        actions[agent] = {
            "choose_project": choose_project,
            "collaborate_with": collaborate_with,
            "put_effort": put_effort,
        }
    obs, rewards, terminations, truncations, infos = env.step(actions)
    print(f"\nStep {step+1}")
    for agent in env.agents:
        print(f"{agent} obs: {obs[agent]}")
        print(f"{agent} reward: {rewards[agent]}")
        print(f"{agent} terminated: {terminations[agent]}, truncated: {truncations[agent]}") 