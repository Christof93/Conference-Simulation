import time
from env.custom_environment import CustomActionMaskedEnvironment

env = CustomActionMaskedEnvironment()
observations, infos = env.reset()
print(env.agents)
while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)
env.render()
print(rewards)
env.close()