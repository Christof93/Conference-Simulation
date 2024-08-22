import numpy as np

from env.reputation_environment import ReputationEnvironment
from env.evaluator import EnvironmentRecorder, NetworkEvaluator

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

def diligent_policy(agent, environment):
    new_mask = environment.action_masks[agent]
    submittable = new_mask["submit"]["id"][1:]
    # only submit if total effort higher than 40
    worthy_papers = (environment.global_observation["papers"]["total_effort"] > 50) & submittable
    new_mask["submit"]["id"][1:] = worthy_papers.astype(np.int8)
    new_mask["start_with_coauthors"][6:] = 0
    # always contribute if possible
    new_mask["contribute"][0] = 0
    environment.action_masks[agent] = new_mask
    return environment.action_space(agent).sample(mask=new_mask)

def picky_policy(agent, environment):
    new_mask = environment.action_masks[agent]
    submittable = new_mask["submit"]["id"][1:]
    # only collaborate with high rep agents
    ## TODO look up authors of papers with open positions and and only set to true if certain threshold is surpassed
    # only submit if total effort higher than 40
    worthy_papers = (environment.global_observation["papers"]["total_effort"] > 50) & submittable
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
    new_mask["start_with_coauthors"][8:] = 0
    # always contribute if possible
    new_mask["contribute"][0] = 0
    environment.action_masks[agent] = new_mask
    return environment.action_space(agent).sample(mask=new_mask)

def random_policy(agent, environment):
    env.action_space(agent).sample(mask = environment.action_masks[agent])

if __name__=="__main__":
    env = ReputationEnvironment(n_authors=500, n_conferences=10, render_mode="network", max_rewardless_steps=150, max_agent_steps=5000, max_coauthors=10)
    # env = ReputationEnvironment(n_authors=10, n_conferences=1, render_mode="all", max_rewardless_steps=150, max_agent_steps=5000, max_coauthors=10)
    recorder = EnvironmentRecorder(env)
    observations, infos = env.reset()
    agent_to_strategy = {}
    for agent in env.agents:
        agent_to_strategy[agent] = "honest"
        # if np.random.random() > 0.2:
        #     agent_to_strategy[agent] = "honest"
        # else:
        #     agent_to_strategy[agent] = "malicious"
    recorder.agent_to_strategy = agent_to_strategy

    while len(env.agents) > 0:
        # this is where you would insert your policy
        actions = {}
        for agent in env.agents:
            if agent_to_strategy[agent] == "honest":
                actions[agent] = simple_policy(agent, env)
            elif agent_to_strategy[agent] == "malicious":
                actions[agent] = malicious_policy(agent, env)
            elif agent_to_strategy[agent] == "diligent":
                actions[agent] = diligent_policy(agent, env)
            else:
                actions[agent] = random_policy(agent, env)
        observations, rewards, terminations, truncations, infos = env.step(actions)
        # breakpoint()
        # env.render()
        # sleep(0.5)
        if env.timestep > 1000:
            break
    
    if env.render_mode=="network":
        evaluator = NetworkEvaluator(
            {
                "nodes": list(env.network_nodes.values()), 
                "links": env.network_links,
                "steps": env.timestep,
                "initial_reputation": env.initial_reputation,
                "agent_strategy": agent_to_strategy,
                "remaining_agents": env.agents
            }
        )
        evaluator.report()
    else:
        recorder.report()
    # env.render()
    env.close()