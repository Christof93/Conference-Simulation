from env.reputation_environment import ReputationEnvironment
from pettingzoo.test import parallel_api_test

if __name__=="__main__":
    env = ReputationEnvironment(n_authors=100, n_conferences=5, render_mode="network", max_rewardless_steps=150, max_agent_steps=5000)
    parallel_api_test(env, num_cycles=10)
