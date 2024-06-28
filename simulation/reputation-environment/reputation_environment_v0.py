from env.reputation_environment import ReputationEnvironment

env = ReputationEnvironment(render_mode="ansi")
observations, infos = env.reset()
while env.agents:
    # this is where you would insert your policy
    print("masks: ",env.action_masks)
    actions = {agent: env.action_space(agent).sample(mask = env.action_masks[agent]) for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    env.render()
    breakpoint()
print(rewards)
env.close()