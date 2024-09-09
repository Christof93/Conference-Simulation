import numpy as np

from env.reputation_environment import ReputationEnvironment
from env.evaluator import EnvironmentRecorder, NetworkEvaluator
from gymnasium.spaces.utils import flatten

from custom_policies import *

if __name__=="__main__":
    # env = ReputationEnvironment(n_authors=500, n_conferences=10, render_mode="network", max_rewardless_steps=150, max_agent_steps=5000, max_coauthors=10)
    env = ReputationEnvironment(n_authors=10, n_conferences=1, render_mode="all", max_rewardless_steps=150, max_agent_steps=5000, max_coauthors=10)
    recorder = EnvironmentRecorder(env)
    observations, infos = env.reset()
    agent_to_strategy = {}
    for agent in env.agents:
        agent_to_strategy[agent] = "diligent"
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
                print(flatten(env.action_space(agent), actions[agent]))
                breakpoint()
            elif agent_to_strategy[agent] == "picky":
                actions[agent] = picky_policy(agent, env)
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