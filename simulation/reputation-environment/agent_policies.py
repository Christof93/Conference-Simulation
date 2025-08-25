"""
Agent Policy Functions for Peer Group Environment

This module contains three different agent policy functions that can be used
instead of random sampling in the peer group environment simulation.

1. careerist: Chooses projects with high potential reward
2. orthodox_scientist: Chooses projects with good fit
3. mass_producer: Chooses projects with low effort and short completion time
"""

from typing import Any, Dict, List, Optional

import numpy as np


def careerist_policy(
    observation: Dict[str, Any],
    action_mask: Dict[str, np.ndarray],
    reward_threshold: float = 0.5,
    **kwargs,
) -> Dict[str, Any]:
    """
    Careerist agent policy: Chooses projects with high potential reward.

    Args:
        observation: Agent's observation containing project opportunities
        action_mask: Available actions mask
        reward_threshold: Minimum reward threshold to consider a project
        **kwargs: Additional arguments

    Returns:
        Dictionary containing the agent's actions
    """
    # Get project opportunities
    project_opportunities = observation.get("project_opportunities", [])

    # Find projects above reward threshold that are allowed by action mask
    valid_projects = []
    choose_project_mask = action_mask.get(
        "choose_project", np.ones(len(project_opportunities) + 1, dtype=np.int8)
    )

    for i, project in enumerate(project_opportunities):
        approx_reward = project.get("approx_reward", 0.0)
        # Check if project meets threshold AND is allowed by action mask
        if approx_reward >= reward_threshold and choose_project_mask[i + 1] == 1:
            valid_projects.append(
                (i + 1, approx_reward)
            )  # +1 because 0 is "no project"

    # Choose project with highest reward
    if valid_projects:
        # Sort by reward (descending) and take the highest
        valid_projects.sort(key=lambda x: x[1], reverse=True)
        chosen_project = valid_projects[0][0]
    else:
        # No projects meet threshold or are allowed, skip
        chosen_project = 0

    # Collaboration: only collaborate with above-average reputation peers
    peer_reputation = observation.get("peer_reputation", [])
    peer_group = observation.get("peer_group", [])

    # Get collaboration mask
    collaborate_mask = action_mask.get(
        "collaborate_with", np.ones(len(peer_group), dtype=np.int8)
    )

    # Collaborate with peers who have above-average reputation AND are allowed by mask
    avg_reputation = np.mean(peer_reputation) if len(peer_reputation) > 0 else 0
    desired_collaboration = (peer_reputation >= avg_reputation).astype(np.int8)
    collaborate_with = desired_collaboration & collaborate_mask

    # Effort: put effort into the project closest to deadline and still under required_effort threshold
    put_effort_mask = action_mask.get(
        "put_effort", np.ones(len(project_opportunities) + 1, dtype=np.int8)
    )
    put_effort = 0  # No valid project found

    # Get running projects for this agent
    running_projects = observation.get("running_projects", {})

    best_project = 0
    if running_projects:
        # Find project closest to deadline that's still under required_effort threshold
        deadline_projects = []
        for proj_id, proj_data in running_projects.items():
            if proj_data.get("time_left", 0) > 0:  # Still has time
                current_effort = proj_data.get("current_effort", 0)
                required_effort = proj_data.get("required_effort", 0)
                if current_effort < required_effort:  # Still under threshold
                    deadline_projects.append(
                        (
                            proj_id,
                            proj_data.get("time_left", 0),
                            current_effort,
                            required_effort,
                        )
                    )

        if deadline_projects:
            # Sort by time left (ascending - closest to deadline first)
            deadline_projects.sort(key=lambda x: x[1])
            best_project = deadline_projects[0][0]

    # Convert project ID to action index if valid
    if best_project in running_projects:
        # Find the index of this project in agent's active projects
        for i, proj_id in enumerate(observation.get("running_projects", {}).keys()):
            if proj_id == best_project:
                if i + 1 < len(put_effort_mask) and put_effort_mask[i + 1] == 1:
                    put_effort = i + 1
                break

    return {
        "choose_project": chosen_project,
        "collaborate_with": collaborate_with,
        "put_effort": put_effort,
    }


def orthodox_scientist_policy(
    observation: Dict[str, Any],
    action_mask: Dict[str, np.ndarray],
    fit_threshold: float = 0.7,
    **kwargs,
) -> Dict[str, Any]:
    """
    Orthodox scientist agent policy: Chooses projects with good fit.

    Args:
        observation: Agent's observation containing project opportunities
        action_mask: Available actions mask
        fit_threshold: Minimum fit threshold to consider a project
        **kwargs: Additional arguments

    Returns:
        Dictionary containing the agent's actions
    """
    # Get project opportunities
    project_opportunities = observation.get("project_opportunities", [])

    # Find projects with good fit that are allowed by action mask
    valid_projects = []
    choose_project_mask = action_mask.get(
        "choose_project", np.ones(len(project_opportunities) + 1, dtype=np.int8)
    )

    for i, project in enumerate(project_opportunities):
        fit = project.get("fit", 0.0)
        # Check if project meets fit threshold AND is allowed by action mask
        if fit >= fit_threshold and choose_project_mask[i + 1] == 1:
            approx_reward = project.get("approx_reward", 0.0)
            valid_projects.append(
                (i + 1, fit, approx_reward)
            )  # +1 because 0 is "no project"

    # Choose project with best fit, then highest reward as tiebreaker
    if valid_projects:
        # Sort by fit (descending), then by reward (descending)
        valid_projects.sort(key=lambda x: (x[1], x[2]), reverse=True)
        chosen_project = valid_projects[0][0]
    else:
        # No projects meet fit threshold, skip
        chosen_project = 0

    # Collaboration: only collaborate with well-fitting peers for the chosen project
    peer_group = observation.get("peer_group", [])
    project_opportunities = observation.get("project_opportunities", [])

    # Get the fit values for the chosen project
    if chosen_project > 0 and chosen_project <= len(project_opportunities):
        chosen_project_data = project_opportunities[chosen_project - 1]
        # Get peer fit values for the chosen project
        peer_fit_values = chosen_project_data.get("peer_fit", [])

        # Collaborate with peers who have good fit (above threshold)
        if len(peer_fit_values) == len(peer_group):
            desired_collaboration = (np.array(peer_fit_values) >= fit_threshold).astype(
                np.int8
            )
        else:
            # Fallback: collaborate with all peers if fit data not available
            desired_collaboration = np.ones(len(peer_group), dtype=np.int8)

        # Apply action mask to collaboration
        collaborate_mask = action_mask.get(
            "collaborate_with", np.ones(len(peer_group), dtype=np.int8)
        )
        collaborate_with = desired_collaboration & collaborate_mask
    else:
        # No project chosen, don't collaborate
        collaborate_with = np.zeros(len(peer_group), dtype=np.int8)

    # Effort: put effort into the best fitting active project until 10% above required_effort threshold
    put_effort_mask = action_mask.get(
        "put_effort", np.ones(len(project_opportunities) + 1, dtype=np.int8)
    )
    put_effort = 0  # No valid project found

    # Get running projects for this agent
    running_projects = observation.get("running_projects", {})

    best_project = 0

    if running_projects:
        # Find project with best fit that's not over the 10% threshold
        best_fit_projects = []
        for proj_id, proj_data in running_projects.items():
            fit = proj_data.get("fit", 0.0)
            current_effort = proj_data.get("current_effort", 0)
            required_effort = proj_data.get("required_effort", 0)
            threshold = required_effort * 1.1  # 10% above required effort

            if current_effort < threshold:  # Still under 10% threshold
                best_fit_projects.append(
                    (proj_id, fit, current_effort, required_effort)
                )

        if best_fit_projects:
            # Sort by fit (descending - best fit first)
            best_fit_projects.sort(key=lambda x: x[1], reverse=True)
            best_project = best_fit_projects[0][0]

    # Convert project ID to action index if valid
    if best_project in running_projects:
        # Find the index of this project in agent's active projects
        for i, proj_id in enumerate(observation.get("running_projects", {}).keys()):
            if proj_id == best_project:
                if i + 1 < len(put_effort_mask) and put_effort_mask[i + 1] == 1:
                    put_effort = i + 1
                break
    return {
        "choose_project": chosen_project,
        "collaborate_with": collaborate_with,
        "put_effort": put_effort,
    }


def mass_producer_policy(
    observation: Dict[str, Any], action_mask: Dict[str, np.ndarray], **kwargs
) -> Dict[str, Any]:
    """
    Mass producer agent policy: Chooses projects with low effort and short completion time.

    Args:
        observation: Agent's observation containing project opportunities
        action_mask: Available actions mask
        **kwargs: Additional arguments

    Returns:
        Dictionary containing the agent's actions
    """
    # Get project opportunities
    project_opportunities = observation.get("project_opportunities", [])

    # Calculate efficiency score for each project (lower effort + shorter time = better)
    project_scores = []
    choose_project_mask = action_mask.get(
        "choose_project", np.ones(len(project_opportunities) + 1, dtype=np.int8)
    )

    for i, project in enumerate(project_opportunities):
        # Check if project is allowed by action mask
        if choose_project_mask[i + 1] == 1:
            required_effort = project.get("required_effort", float("inf"))
            time_window = project.get("time_window", float("inf"))
            approx_reward = project.get("approx_reward", 0.0)

            # Efficiency score: reward / (effort * time) - higher is better
            if required_effort > 0 and time_window > 0:
                efficiency = approx_reward / (required_effort * time_window)
            else:
                efficiency = 0

            project_scores.append(
                (i + 1, efficiency, approx_reward)
            )  # +1 because 0 is "no project"

    # Choose project with highest efficiency, then highest reward as tiebreaker
    if project_scores:
        # Sort by efficiency (descending), then by reward (descending)
        project_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
        chosen_project = project_scores[0][0]
    else:
        # Fallback to first available project
        chosen_project = 1 if len(project_opportunities) > 0 else 0

    # Collaboration: collaborate with everyone (maximize output) but respect action mask
    peer_group = observation.get("peer_group", [])
    collaborate_mask = action_mask.get(
        "collaborate_with", np.ones(len(peer_group), dtype=np.int8)
    )
    collaborate_with = collaborate_mask  # Use mask directly since we want to collaborate with everyone allowed

    # Effort: put effort into the project closest to deadline and still under required_effort threshold
    put_effort_mask = action_mask.get(
        "put_effort", np.ones(len(project_opportunities) + 1, dtype=np.int8)
    )

    put_effort = 0  # No valid project found
    # Get running projects for this agent
    running_projects = observation.get("running_projects", {})

    best_project = 0
    if running_projects:
        # Find project closest to deadline that's still under required_effort threshold
        deadline_projects = []
        for proj_id, proj_data in running_projects.items():
            if proj_data.get("time_left", 0) > 0:  # Still has time
                current_effort = proj_data.get("current_effort", 0)
                required_effort = proj_data.get("required_effort", 0)
                if current_effort < required_effort:  # Still under threshold
                    deadline_projects.append(
                        (
                            proj_id,
                            proj_data.get("time_left", 0),
                            current_effort,
                            required_effort,
                        )
                    )

        if deadline_projects:
            # Sort by time left (ascending - closest to deadline first)
            deadline_projects.sort(key=lambda x: x[1])
            best_project = deadline_projects[0][0]

    # Convert project ID to action index if valid
    if best_project in running_projects:
        # Find the index of this project in agent's active projects
        for i, proj_id in enumerate(observation.get("running_projects", {}).keys()):
            if proj_id == best_project:
                if i + 1 < len(put_effort_mask) and put_effort_mask[i + 1] == 1:
                    put_effort = i + 1
                break

    return {
        "choose_project": chosen_project,
        "collaborate_with": collaborate_with,
        "put_effort": put_effort,
    }


def get_policy_function(policy_name: str):
    """
    Get a policy function by name.

    Args:
        policy_name: Name of the policy ("careerist", "orthodox_scientist", "mass_producer")

    Returns:
        Policy function

    Raises:
        ValueError: If policy name is not recognized
    """
    policies = {
        "careerist": careerist_policy,
        "orthodox_scientist": orthodox_scientist_policy,
        "mass_producer": mass_producer_policy,
    }

    if policy_name not in policies:
        raise ValueError(
            f"Unknown policy: {policy_name}. Available policies: {list(policies.keys())}"
        )

    return policies[policy_name]


def create_mixed_policy_population(
    n_agents: int, policy_distribution: Dict[str, float] = None
) -> List[str]:
    """
    Create a population of agents with different policies.

    Args:
        n_agents: Total number of agents
        policy_distribution: Dictionary mapping policy names to their proportions
                           (e.g., {"careerist": 0.3, "orthodox_scientist": 0.4, "mass_producer": 0.3})

    Returns:
        List of policy names for each agent
    """
    if policy_distribution is None:
        # Default distribution: equal proportions
        policy_distribution = {
            "careerist": 1 / 3,
            "orthodox_scientist": 1 / 3,
            "mass_producer": 1 / 3,
        }

    # Validate distribution
    total_proportion = sum(policy_distribution.values())
    if abs(total_proportion - 1.0) > 1e-6:
        raise ValueError(f"Policy distribution must sum to 1.0, got {total_proportion}")

    # Create agent assignments
    agent_policies = []
    for policy_name, proportion in policy_distribution.items():
        n_policy_agents = int(n_agents * proportion)
        agent_policies.extend([policy_name] * n_policy_agents)

    # Handle rounding errors by adding remaining agents to the first policy
    while len(agent_policies) < n_agents:
        agent_policies.append(list(policy_distribution.keys())[0])

    # Shuffle to randomize agent order
    np.random.shuffle(agent_policies)

    return agent_policies


# Example usage and testing functions
def test_policies():
    """Test the policy functions with sample data."""

    # Sample observation
    sample_observation = {
        "project_opportunities": [
            {"required_effort": 10, "approx_reward": 0.1, "fit": 0.8, "time_window": 5},
            {
                "required_effort": 50,
                "approx_reward": 0.5,
                "fit": 0.6,
                "time_window": 15,
            },
            {
                "required_effort": 100,
                "approx_reward": 1.0,
                "fit": 0.9,
                "time_window": 25,
            },
        ],
        "peer_group": [1, 2, 3],
        "peer_reputation": [0.7, 0.3, 0.8],
    }

    sample_action_mask = {
        "choose_project": np.ones(4, dtype=np.int8),
        "collaborate_with": np.ones(3, dtype=np.int8),
        "put_effort": np.ones(7, dtype=np.int8),
    }

    print("Testing Careerist Policy:")
    careerist_action = careerist_policy(
        sample_observation, sample_action_mask, reward_threshold=0.3
    )
    print(f"  Chosen project: {careerist_action['choose_project']}")
    print(f"  Collaboration: {careerist_action['collaborate_with']}")
    print(f"  Effort: {careerist_action['put_effort']}")

    print("\nTesting Orthodox Scientist Policy:")
    orthodox_action = orthodox_scientist_policy(
        sample_observation, sample_action_mask, fit_threshold=0.7
    )
    print(f"  Chosen project: {orthodox_action['choose_project']}")
    print(f"  Collaboration: {orthodox_action['collaborate_with']}")
    print(f"  Effort: {orthodox_action['put_effort']}")

    print("\nTesting Mass Producer Policy:")
    mass_action = mass_producer_policy(sample_observation, sample_action_mask)
    print(f"  Chosen project: {mass_action['choose_project']}")
    print(f"  Collaboration: {mass_action['collaborate_with']}")
    print(f"  Effort: {mass_action['put_effort']}")


if __name__ == "__main__":
    test_policies()
