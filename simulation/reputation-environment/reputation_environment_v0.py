from time import sleep

import numpy as np
from env.reputation_environment import ReputationEnvironment

def simple_policy(agent, environment):
    new_mask = environment.action_masks[agent]
    submittable = new_mask["submit"]["id"][1:]
    # only submit if total effort higher than 40
    worthy_papers = (environment.global_observation["papers"]["total_effort"] > 40) & submittable
    new_mask["submit"]["id"][1:] = worthy_papers.astype(np.int8)
    new_mask["start_with_coauthors"][6:] = 0
    # always contribute if possible
    new_mask["contribute"][0] = 0
    environment.action_masks[agent] = new_mask
    return environment.action_space(agent).sample(mask=new_mask)

def malicious_policy(agent, environment):
    new_mask = environment.action_masks[agent]
    submittable = new_mask["submit"]["id"][1:]
    # submit if total effort higher than 20
    worthy_papers = (environment.global_observation["papers"]["total_effort"] > 20) & submittable
    new_mask["submit"]["id"][1:] = worthy_papers.astype(np.int8)
    new_mask["start_with_coauthors"][12:] = 0
    # always contribute if possible
    new_mask["contribute"][0] = 0
    environment.action_masks[agent] = new_mask
    return environment.action_space(agent).sample(mask=new_mask)

def random_policy(agent, environment):
    env.action_space(agent).sample(mask = environment.action_masks[agent])

if __name__=="__main__":
    env = ReputationEnvironment(n_authors=100, n_conferences=10, render_mode="observation")
    observations, infos = env.reset()
    while env.agents:
        # this is where you would insert your policy
        actions = {agent: simple_policy(agent, env) for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        # breakpoint()
        # sleep(0.5)
        if env.timestep>10:
            break
    env.render()
    print(env.reputations)
    print(env.network)
    env.close()