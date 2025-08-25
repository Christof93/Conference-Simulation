"""
Example script showing how to use the agent policies with the peer group environment.
"""

import json
from pathlib import Path

import numpy as np
from agent_policies import create_mixed_policy_population, get_policy_function
from env.peer_group_environment import PeerGroupEnvironment
from stats_tracker import SimulationStats


def run_simulation_with_policies(
    n_agents: int = 20,
    max_steps: int = 100,
    policy_distribution: dict = None,
    output_file: str = "policy_simulation_results.jsonl",
):
    """
    Run a simulation with different agent policies.

    Args:
        n_agents: Number of agents in the simulation
        max_steps: Maximum number of simulation steps
        policy_distribution: Distribution of policies among agents
        output_file: File to save results
    """

    # Create environment
    env = PeerGroupEnvironment(
        n_agents=n_agents,
        peer_group_size=5,
        n_projects=6,
        max_projects_per_agent=3,
        max_timesteps=max_steps,
        max_rewardless_steps=50,
    )

    # Create agent policy assignments
    agent_policies = create_mixed_policy_population(n_agents, policy_distribution)
    print(
        f"Agent policy distribution: {dict(zip(*np.unique(agent_policies, return_counts=True)))}"
    )

    # Initialize stats tracker
    stats = SimulationStats()

    # Reset environment
    observations, infos = env.reset()

    # Simulation loop
    for step in range(max_steps):
        actions = {}

        # Generate actions for each agent based on their policy
        for agent in env.agents:
            agent_idx = env.agent_to_id[agent]
            policy_name = agent_policies[agent_idx]
            policy_func = get_policy_function(policy_name)

            # Get agent's observation and action mask
            obs = observations[agent]["observation"]
            action_mask = observations[agent]["action_mask"]

            # Generate action using the agent's policy
            action = policy_func(obs, action_mask)
            actions[agent] = action

        # Step the environment
        observations, rewards, terminations, truncations, infos = env.step(actions)

        # Update stats
        stats.update(env, observations, rewards, terminations, truncations)

        # Print progress
        if step % 10 == 0:
            print(f"Step {step}: {stats.summary_line()}")

        # Check if all agents are done
        if all(terminations.values()) or all(truncations.values()):
            print(f"Simulation ended at step {step}")
            break

    # Save results
    results = {
        "final_stats": stats.to_dict(),
        "agent_policies": agent_policies,
        "policy_distribution": policy_distribution
        or {"careerist": 1 / 3, "orthodox_scientist": 1 / 3, "mass_producer": 1 / 3},
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nFinal Results:")
    print(f"Total Steps: {stats.total_steps}")
    print(f"Finished Projects: {stats.finished_projects_count}")
    print(f"Successful Projects: {stats.successful_projects_count}")
    print(
        f"Success Rate: {stats.successful_projects_count / max(stats.finished_projects_count, 1):.3f}"
    )
    print(f"Total Rewards: {stats.total_rewards_distributed:.2f}")

    return results


def compare_policy_performances():
    """Compare the performance of different policy distributions."""

    # Define different policy distributions to test
    policy_configs = {
        "All Careerist": {
            "careerist": 1.0,
            "orthodox_scientist": 0.0,
            "mass_producer": 0.0,
        },
        "All Orthodox": {
            "careerist": 0.0,
            "orthodox_scientist": 1.0,
            "mass_producer": 0.0,
        },
        "All Mass Producer": {
            "careerist": 0.0,
            "orthodox_scientist": 0.0,
            "mass_producer": 1.0,
        },
        "Balanced": {
            "careerist": 1 / 3,
            "orthodox_scientist": 1 / 3,
            "mass_producer": 1 / 3,
        },
        "Careerist Heavy": {
            "careerist": 0.6,
            "orthodox_scientist": 0.2,
            "mass_producer": 0.2,
        },
        "Orthodox Heavy": {
            "careerist": 0.2,
            "orthodox_scientist": 0.6,
            "mass_producer": 0.2,
        },
    }

    results = {}

    for config_name, policy_dist in policy_configs.items():
        print(f"\n{'='*50}")
        print(f"Testing: {config_name}")
        print(f"{'='*50}")

        result = run_simulation_with_policies(
            n_agents=20,
            max_steps=100,
            policy_distribution=policy_dist,
            output_file=f"results_{config_name.lower().replace(' ', '_')}.json",
        )

        results[config_name] = result["final_stats"]

    # Print comparison
    print(f"\n{'='*80}")
    print("POLICY COMPARISON SUMMARY")
    print(f"{'='*80}")

    for config_name, stats in results.items():
        success_rate = stats["successful_projects"] / max(stats["finished_projects"], 1)
        print(
            f"{config_name:20} | Success Rate: {success_rate:.3f} | "
            f"Finished: {stats['finished_projects']:3d} | "
            f"Rewards: {stats['total_rewards_distributed']:6.2f}"
        )


if __name__ == "__main__":
    # Run a single simulation with balanced policies
    print("Running single simulation with balanced policies...")
    run_simulation_with_policies()

    # Compare different policy distributions
    print("\n" + "=" * 80)
    print("Comparing different policy distributions...")
    compare_policy_performances()
