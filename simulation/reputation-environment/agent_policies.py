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


def _as_scalar(x: Any) -> float:
    if isinstance(x, (list, tuple, np.ndarray)):
        return float(x[0]) if len(x) > 0 else 0.0
    try:
        return float(x)
    except Exception:
        return 0.0


def _iter_project_opportunities(project_opportunities: Any):
    # project_opportunities: dict like { 'project_0': { 'required_effort': [..], 'prestige': [..], 'novelty':[...], 'time_window':[...]} }
    if isinstance(project_opportunities, dict):
        # preserve insert order for action indexing
        for idx, (proj_key, proj) in enumerate(project_opportunities.items(), start=1):
            yield idx, proj_key, {
                "required_effort": _as_scalar(proj.get("required_effort", 0)),
                "prestige": _as_scalar(proj.get("prestige", 0.0)),
                "novelty": _as_scalar(proj.get("novelty", 0.0)),
                "time_window": _as_scalar(proj.get("time_window", 0)),
            }
    else:
        # Fallback if old list format
        for i, proj in enumerate(project_opportunities or [], start=1):
            yield i, f"project_{i-1}", {
                "required_effort": float(proj.get("required_effort", 0)),
                "prestige": float(proj.get("approx_reward", 0.0)),
                "novelty": float(proj.get("novelty", 0.0)),
                "time_window": float(proj.get("time_window", 0)),
            }


def _iter_running_projects(running_projects: Any):
    # running_projects: dict keyed by project_id, values with arrays for fields
    if not isinstance(running_projects, dict):
        return []
    items = []
    for proj_key, proj in running_projects.items():
        items.append(
            (
                proj_key,
                {
                    "required_effort": _as_scalar(proj.get("required_effort", 0)),
                    "prestige": _as_scalar(proj.get("prestige", 0.0)),
                    "novelty": _as_scalar(proj.get("novelty", 0.0)),
                    "time_left": _as_scalar(proj.get("time_left", 0)),
                    "current_effort": _as_scalar(proj.get("current_effort", 0)),
                    "peer_fit": np.array(proj.get("peer_fit", []), dtype=np.float32),
                },
            )
        )
    return items


def _mask_allowed(mask_arr: Any, idx: int) -> bool:
    if mask_arr is None:
        return True
    try:
        return mask_arr[idx] > 0
    except Exception:
        return False


def careerist_policy(
    observation: Dict[str, Any],
    action_mask: Dict[str, np.ndarray],
    prestige_threshold: float = 0.5,
    **kwargs,
) -> Dict[str, Any]:
    # Unwrap if wrapped as { 'observation': ..., 'action_mask': ... }
    if "project_opportunities" not in observation and "observation" in observation:
        observation = observation["observation"]

    project_opportunities = observation.get("project_opportunities", {})

    current = observation.get("running_projects", {})
    chosen_project = None
    for proj_key, proj in current.items():
        if (
            proj["time_left"] / len(proj["contributors"])
            < proj["required_effort"] - proj["current_effort"]
        ):
            chosen_project = 0
            break
    if chosen_project is None:
        # Choose project: highest prestige above threshold
        valid_projects: List[tuple] = []
        choose_project_mask = action_mask.get(
            "choose_project",
            np.ones(
                len(getattr(project_opportunities, "items", lambda: [])()) + 1,
                dtype=np.int8,
            ),
        )

        for i, proj_key, proj in _iter_project_opportunities(project_opportunities):
            prestige = proj["prestige"]
            if prestige >= prestige_threshold and _mask_allowed(choose_project_mask, i):
                valid_projects.append((i, prestige))
        chosen_project = (
            valid_projects
            and sorted(valid_projects, key=lambda x: x[1], reverse=True)[0][0]
            or 0
        )

    # Collaboration: active peers with above-average reputation
    peer_reputation = np.array(observation.get("peer_reputation", []), dtype=np.float32)
    peer_group_active = np.array(observation.get("peer_group", []), dtype=np.int8)
    collaborate_mask = action_mask.get(
        "collaborate_with", np.ones_like(peer_group_active, dtype=np.int8)
    )
    avg_rep = (
        float(peer_reputation[peer_group_active == 1].mean())
        if peer_group_active.sum() > 0
        else 0.0
    )
    desired = ((peer_reputation >= avg_rep).astype(np.int8)) * (
        peer_group_active > 0
    ).astype(np.int8)
    collaborate_with = (desired > 0) & (collaborate_mask > 0)
    collaborate_with = collaborate_with.astype(np.int8)

    # Effort: project closest to deadline still under required_effort
    put_effort = 0
    put_effort_mask = action_mask.get("put_effort", np.ones(1, dtype=np.int8))
    running_projects = observation.get("running_projects", {})
    candidates = []
    for slot_idx, (proj_key, proj) in enumerate(
        _iter_running_projects(running_projects), start=1
    ):
        time_left = proj["time_left"]
        if time_left <= 0:
            continue
        if proj["current_effort"] < proj["required_effort"]:
            if _mask_allowed(put_effort_mask, slot_idx):
                candidates.append((slot_idx, time_left))
    if candidates:
        candidates.sort(key=lambda x: x[1])
        put_effort = candidates[0][0]

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
    if "project_opportunities" not in observation and "observation" in observation:
        observation = observation["observation"]

    project_opportunities = observation.get("project_opportunities", {})
    current = observation.get("running_projects", {})
    chosen_project = None
    for proj_key, proj in current.items():
        if (
            proj["time_left"] / len(proj["contributors"])
            < proj["required_effort"] - proj["current_effort"]
        ):
            chosen_project = 0
            break
    if chosen_project is None:
        # Choose project: lowest novelty (best fit to existing), tie-breaker by prestige desc
        choose_project_mask = action_mask.get(
            "choose_project",
            np.ones(
                len(getattr(project_opportunities, "items", lambda: [])()) + 1,
                dtype=np.int8,
            ),
        )
        ranked: List[tuple] = []
        for i, proj_key, proj in _iter_project_opportunities(project_opportunities):
            if _mask_allowed(choose_project_mask, i):
                ranked.append((i, proj["novelty"], proj["prestige"]))
        if ranked:
            ranked.sort(key=lambda x: (x[1], -x[2]))  # min novelty, then max prestige
            chosen_project = ranked[0][0]
        else:
            chosen_project = 0

    # Collaboration: if any running project exists, collaborate with peers whose peer_fit >= threshold
    peer_group_active = np.array(observation.get("peer_group", []), dtype=np.int8)
    collaborate_mask = action_mask.get(
        "collaborate_with", np.ones_like(peer_group_active, dtype=np.int8) + 1
    )
    running_projects = observation.get("running_projects", {})
    collaborate_with = np.ones_like(peer_group_active, dtype=np.int8)

    # Effort: best fitting active project under -10% threshold
    put_effort = 0
    put_effort_mask = action_mask.get("put_effort", np.ones(1, dtype=np.int8))
    candidates = []
    for slot_idx, (proj_key, proj) in enumerate(
        _iter_running_projects(running_projects), start=1
    ):
        required = proj["required_effort"]
        threshold = required * 0.9
        if proj["current_effort"] > threshold and _mask_allowed(
            put_effort_mask, slot_idx
        ):
            return {
                "choose_project": chosen_project,
                "collaborate_with": collaborate_with,
                "put_effort": slot_idx,
            }
        elif _mask_allowed(put_effort_mask, slot_idx):
            # use max peer_fit as proxy for fit
            max_fit = (
                float(np.max(proj["peer_fit"])) if len(proj["peer_fit"]) > 0 else 0.0
            )
            candidates.append((slot_idx, max_fit))
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        put_effort = candidates[0][0]

    return {
        "choose_project": chosen_project,
        "collaborate_with": collaborate_with,
        "put_effort": put_effort,
    }


def mass_producer_policy(
    observation: Dict[str, Any], action_mask: Dict[str, np.ndarray], **kwargs
) -> Dict[str, Any]:
    if "project_opportunities" not in observation and "observation" in observation:
        observation = observation["observation"]

    project_opportunities = observation.get("project_opportunities", {})

    # Efficiency: prestige / (effort * time)
    choose_project_mask = action_mask.get(
        "choose_project",
        np.ones(
            len(getattr(project_opportunities, "items", lambda: [])()) + 1,
            dtype=np.int8,
        ),
    )
    scores: List[tuple] = []
    for i, proj_key, proj in _iter_project_opportunities(project_opportunities):
        if not _mask_allowed(choose_project_mask, i):
            continue
        effort = proj["required_effort"]
        time_w = proj["time_window"]
        prestige = proj["prestige"]
        eff = prestige / (effort * time_w) if effort > 0 and time_w > 0 else 0.0
        scores.append((i, eff, prestige))
    if scores:
        scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
        chosen_project = scores[0][0]
    else:
        chosen_project = 0

    # Collaborate with all active peers within mask
    peer_group_active = np.array(observation.get("peer_group", []), dtype=np.int8)
    collaborate_mask = action_mask.get(
        "collaborate_with", np.ones_like(peer_group_active, dtype=np.int8)
    )
    collaborate_with = ((peer_group_active > 0) & (collaborate_mask > 0)).astype(
        np.int8
    )

    # Effort: project closest to deadline under required_effort
    put_effort = 0
    put_effort_mask = action_mask.get("put_effort", np.ones(1, dtype=np.int8))
    running_projects = observation.get("running_projects", {})
    candidates = []
    for slot_idx, (proj_key, proj) in enumerate(
        _iter_running_projects(running_projects), start=1
    ):
        time_left = proj["time_left"]
        if time_left <= 0:
            continue
        if proj["current_effort"] < proj["required_effort"] and _mask_allowed(
            put_effort_mask, slot_idx
        ):
            candidates.append((slot_idx, time_left))
    if candidates:
        candidates.sort(key=lambda x: x[1])
        put_effort = candidates[0][0]

    return {
        "choose_project": chosen_project,
        "collaborate_with": collaborate_with,
        "put_effort": put_effort,
    }


def get_policy_function(policy_name: str):
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
    if policy_distribution is None:
        policy_distribution = {
            "careerist": 1 / 3,
            "orthodox_scientist": 1 / 3,
            "mass_producer": 1 / 3,
        }
    total_proportion = sum(policy_distribution.values())
    if abs(total_proportion - 1.0) > 1e-6:
        raise ValueError(f"Policy distribution must sum to 1.0, got {total_proportion}")
    agent_policies = []
    for policy_name, proportion in policy_distribution.items():
        n_policy_agents = int(n_agents * proportion)
        agent_policies.extend([policy_name] * n_policy_agents)
    while len(agent_policies) < n_agents:
        agent_policies.append(list(policy_distribution.keys())[0])
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
        sample_observation, sample_action_mask, prestige_threshold=0.3
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
