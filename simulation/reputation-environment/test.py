from pettingzoo.test import parallel_api_test
import env.reputation_environment as rep_env

if __name__=="__main__":
    env = rep_env.ReputationEnvironment(n_authors=10, n_conferences=1, render_mode="all", max_rewardless_steps=150, max_agent_steps=5000, max_coauthors=10)
    env.reset()
    parallel_api_test(env, num_cycles=10)