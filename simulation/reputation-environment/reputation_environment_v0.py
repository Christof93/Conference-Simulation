from env.reputation_environment import ReputationEnvironment

env = ReputationEnvironment()
observations, infos = env.reset()
while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample(mask = env.action_masks[agent]) for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)
    breakpoint()
env.render()
print(rewards)
env.close()