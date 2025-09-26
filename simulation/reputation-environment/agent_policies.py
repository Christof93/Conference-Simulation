"""
Agent Policy Functions for Peer Group Environment

This module contains three different agent policy functions that can be used
instead of random sampling in the peer group environment simulation.

1. careerist: Chooses projects with high potential reward
2. orthodox_scientist: Chooses projects with good fit
3. mass_producer: Chooses projects with low effort and short completion time
"""

from itertools import zip_longest
from typing import Any, Dict, List

import numpy as np


def _as_scalar(x: Any) -> float:
    if isinstance(x, (list, tuple, np.ndarray)):
        return float(x[0]) if len(x) > 0 else 0.0
    try:
        return float(x)
    except Exception:
        return 0.0


def _unwrap_observation(obs: Dict[str, Any]) -> Dict[str, Any]:
    # Unwrap if wrapped as { 'observation': ..., 'action_mask': ... }
    if "project_opportunities" not in obs and "observation" in obs:
        return obs["observation"]
    return obs


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


def _default_choose_project_mask(
    action_mask: Dict[str, Any], project_opportunities: Any
) -> np.ndarray:
    # Binary action space: index 0=None, 1=take the single offered project
    mask = action_mask.get("choose_project")
    if isinstance(mask, np.ndarray) and mask.size == 2:
        return mask
    return np.ones(2, dtype=np.int8)


def _default_collaborate_mask(
    action_mask: Dict[str, Any], peer_group_active: np.ndarray
) -> np.ndarray:
    return action_mask.get(
        "collaborate_with", np.ones_like(peer_group_active, dtype=np.int8)
    )


def _default_put_effort_mask(action_mask: Dict[str, Any]) -> np.ndarray:
    return action_mask.get("put_effort", np.ones(1, dtype=np.int8))


def _get_single_opportunity(project_opportunities: Any):
    # Expect exactly one new opportunity per step
    for _, _, proj in _iter_project_opportunities(project_opportunities):
        return proj
    return None


def _emergency_continue_any(running_projects: Dict[str, Any]) -> bool:
    for proj_key, proj in running_projects.items():
        contributors = len(proj.get("contributors", [])) or 1
        time_left = proj.get("time_left", 0)
        required = proj.get("required_effort", 0)
        current = proj.get("current_effort", 0)
        try:
            if (time_left / contributors) < (required - current):
                return True
        except Exception:
            continue
    return False


def _select_effort_closest_deadline_under_required(
    running_projects: Dict[str, Any], put_effort_mask: np.ndarray
) -> int:
    # choose the running project with smallest time_left that still needs effort
    candidates: List[tuple] = []
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
        return candidates[0][0]
    return 0


def _select_effort_best_fit_or_threshold(
    running_projects: Dict[str, Any],
    put_effort_mask: np.ndarray,
    threshold_ratio: float = 0.9,
) -> int:
    # If any project is above threshold, immediately work on it; else choose by best peer_fit
    candidates: List[tuple] = []
    for slot_idx, (proj_key, proj) in enumerate(
        _iter_running_projects(running_projects), start=1
    ):
        required = proj["required_effort"]
        threshold = required * threshold_ratio
        if proj["current_effort"] > threshold and _mask_allowed(
            put_effort_mask, slot_idx
        ):
            return slot_idx
        if _mask_allowed(put_effort_mask, slot_idx):
            max_fit = (
                float(np.max(proj["peer_fit"])) if len(proj["peer_fit"]) > 0 else 0.0
            )
            candidates.append((slot_idx, max_fit))
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    return 0


def careerist_policy(
    observation: Dict[str, Any],
    action_mask: Dict[str, np.ndarray],
    prestige_threshold: float = 0.5,
    **kwargs,
) -> Dict[str, Any]:
    observation = _unwrap_observation(observation)

    project_opportunities = observation.get("project_opportunities", {})
    current = observation.get("running_projects", {})
    if _emergency_continue_any(current):
        chosen_project = 0
    else:
        choose_project_mask = _default_choose_project_mask(
            action_mask, project_opportunities
        )
        opp = _get_single_opportunity(project_opportunities)
        meets = (
            opp is not None
            and float(opp.get("prestige", 0.0)) >= prestige_threshold
            and _mask_allowed(choose_project_mask, 1)
        )
        chosen_project = 1 if meets else 0

    # Collaboration: active peers with above-average reputation
    peer_reputation = np.array(observation.get("peer_reputation", []), dtype=np.float32)
    peer_group_active = np.array(observation.get("peer_group", []), dtype=np.int8)
    collaborate_mask = _default_collaborate_mask(action_mask, peer_group_active)
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
    put_effort_mask = _default_put_effort_mask(action_mask)
    running_projects = observation.get("running_projects", {})
    put_effort = _select_effort_closest_deadline_under_required(
        running_projects, put_effort_mask
    )

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
    observation = _unwrap_observation(observation)

    project_opportunities = observation.get("project_opportunities", {})
    current = observation.get("running_projects", {})
    if _emergency_continue_any(current):
        chosen_project = 0
    else:
        choose_project_mask = _default_choose_project_mask(
            action_mask, project_opportunities
        )
        opp = _get_single_opportunity(project_opportunities)
        meets = opp is not None and _mask_allowed(choose_project_mask, 1)
        chosen_project = 1 if meets else 0

    # Collaboration: if any running project exists, collaborate with peers whose peer_fit >= threshold
    peer_group_active = np.array(observation.get("peer_group", []), dtype=np.int8)
    collaborate_mask = _default_collaborate_mask(action_mask, peer_group_active)
    running_projects = observation.get("running_projects", {})
    collaborate_with = np.ones_like(peer_group_active, dtype=np.int8)

    # Effort: best fitting active project or above 90% threshold
    put_effort_mask = _default_put_effort_mask(action_mask)
    put_effort = _select_effort_best_fit_or_threshold(
        running_projects, put_effort_mask, threshold_ratio=0.9
    )

    return {
        "choose_project": chosen_project,
        "collaborate_with": collaborate_with,
        "put_effort": put_effort,
    }


def mass_producer_policy(
    observation: Dict[str, Any], action_mask: Dict[str, np.ndarray], **kwargs
) -> Dict[str, Any]:
    observation = _unwrap_observation(observation)

    project_opportunities = observation.get("project_opportunities", {})

    # Efficiency: prestige / (effort * time). With binary space, accept if efficiency > 0 and allowed
    choose_project_mask = _default_choose_project_mask(
        action_mask, project_opportunities
    )
    opp = _get_single_opportunity(project_opportunities)
    if opp is not None and _mask_allowed(choose_project_mask, 1):
        effort = float(opp.get("required_effort", 0.0))
        time_w = float(opp.get("time_window", 0.0))
        prestige = float(opp.get("prestige", 0.0))
        eff = prestige / (effort * time_w) if effort > 0 and time_w > 0 else 0.0
        chosen_project = 1 if eff > 0 else 0
    else:
        chosen_project = 0

    # Collaborate with all active peers within mask
    peer_group_active = np.array(observation.get("peer_group", []), dtype=np.int8)
    collaborate_mask = _default_collaborate_mask(action_mask, peer_group_active)
    collaborate_with = ((peer_group_active > 0) & (collaborate_mask > 0)).astype(
        np.int8
    )

    # Effort: project closest to deadline under required_effort
    put_effort_mask = _default_put_effort_mask(action_mask)
    running_projects = observation.get("running_projects", {})
    put_effort = _select_effort_closest_deadline_under_required(
        running_projects, put_effort_mask
    )

    return {
        "choose_project": chosen_project,
        "collaborate_with": collaborate_with,
        "put_effort": put_effort,
    }


def interleave(lists):
    return [elem for group in zip_longest(*lists) for elem in group if elem is not None]


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


def create_per_group_policy_population(
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
    policy_groups = []
    for policy_name, proportion in policy_distribution.items():
        if proportion > 0:
            n_policy_agents = int(n_agents * proportion)
            policy_groups.append([policy_name] * n_policy_agents)
    while sum([len(group) for group in policy_groups]) < n_agents:
        policy_groups[-1].append(list(policy_distribution.keys())[0])
    return interleave(policy_groups)


if __name__ == "__main__":
    # Keep minimal manual check without noisy prints
    print(create_per_group_policy_population(10))
