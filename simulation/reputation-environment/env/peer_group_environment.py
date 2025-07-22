import functools
from collections import Counter
from copy import copy, deepcopy

import networkx as nx
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary
from pettingzoo import ParallelEnv


class PeerGroupEnvironment(ParallelEnv):
    """Multi-agent environment with peer groups and project opportunities."""

    metadata = {
        "name": "peer_group_environment_v0",
    }

    def __init__(
        self,
        n_agents=20,
        peer_group_size=10,
        n_projects=6,
        max_projects_per_agent=6,
        max_timesteps=1000,
        max_rewardless_steps=24,
        peer_group_growth_rate=0.01,
        validator_skill_mean=0.7,
        validator_skill_std=0.1,
        render_mode=None,
    ):
        self.n_agents = n_agents
        self.peer_group_size = peer_group_size
        self.n_projects = n_projects
        self.max_projects_per_agent = max_projects_per_agent
        self.max_timesteps = max_timesteps
        self.max_rewardless_steps = max_rewardless_steps
        self.peer_group_growth_rate = peer_group_growth_rate
        self.validator_skill_mean = validator_skill_mean
        self.validator_skill_std = validator_skill_std
        self.render_mode = render_mode

        self.possible_agents = [f"agent_{i}" for i in range(self.n_agents)]
        self.agent_to_id = {agent: i for i, agent in enumerate(self.possible_agents)}
        self.agent_peers = []  # List of sets of peer agent ids for each agent
        self.validators = self._init_validators(1)  # Start with 1 validator

        self.timestep = 0
        self.agent_steps = np.zeros(self.n_agents, dtype=np.int32)
        self.rewardless_steps = np.zeros(self.n_agents, dtype=np.int32)
        self.agent_rewards = np.zeros(self.n_agents, dtype=np.float32)
        self.agent_ages = np.zeros(self.n_agents, dtype=np.int32)
        self.agent_completed_projects = np.zeros(self.n_agents, dtype=np.int32)
        self.agent_active_projects = [
            [] for _ in range(self.n_agents)
        ]  # List of project indices
        self.agent_project_effort = [
            {} for _ in range(self.n_agents)
        ]  # {project_idx: effort}
        self.agent_peer_group = [0 for _ in range(self.n_agents)]  # Index of peer group
        self.projects = {}
        self.actions = {}
        self.observations = {}
        self.action_masks = {}
        self.rewards = {}
        self.agents = copy(self.possible_agents)
        self.project_templates = [
            # good fit, low effort, low reward
            {
                "required_effort": 10,
                "approx_reward": 0.1,
                "fit": "high/low",
            },
            # good fit, medium effort, medium reward
            {
                "required_effort": 50,
                "approx_reward": 0.5,
                "fit": "high/low",
            },
            # good fit, high effort, high reward
            {
                "required_effort": 100,
                "approx_reward": 1.0,
                "fit": "high/low",
            },
            # low fit, low effort, low reward
            {
                "required_effort": 10,
                "approx_reward": 0.1,
                "fit": "high/low",
            },
            # low fit, medium effort, medium reward
            {
                "required_effort": 50,
                "approx_reward": 0.5,
                "fit": "high/low",
            },
            # low fit, high effort, high reward
            {
                "required_effort": 100,
                "approx_reward": 1.0,
                "fit": "high/low",
            },
        ]
        self._generate_projects()

    def _init_peer_groups(self):
        self.peer_groups = [set() for _ in range(self.peer_group_size)]
        if self.n_agents % self.peer_group_size != 0:
            raise ValueError(
                f"agents are not evenly distributable into {self.peer_group_size} sized groups."
            )
        n_groups = self.n_agents // self.peer_group_size
        for i in range(self.n_agents):
            self.peer_groups[i % n_groups].add(i)
        # Each agent has a fixed set of peers (not necessarily symmetric)
        self.agent_peer_idx = []  # List of sets of peer agent ids for each agent
        for i in range(self.n_agents):
            # Peers are the all agents in the same peer group
            self.agent_peer_idx.append(i % n_groups)

    def _grow_peer_groups(self):
        # Pick a two random groups.
        self.peer_group_size += 1
        perms = np.random.permutation(list(enumerate(self.peer_groups)))
        if len(perms) % 2 != 0:
            raise ValueError("Peer groups must be even")
        for i in range(0, len(perms), 2):
            group_idx1, group1 = perms[i]
            group_idx2, group2 = perms[i + 1]
            # Pick a random agent from each group which isn't already in the other group.
            try:
                agent_idx1 = np.random.choice(list(group1 - group2))
                agent_idx2 = np.random.choice(list(group2 - group1))

            except ValueError:
                print(
                    f"Warning: Groups {group_idx1} and {group_idx2} couldn't be grown because all members already know each other."
                )
                self._grow_peer_groups()

            # Add agent 1 to agents 2's peer group and vice versa.
            self.peer_groups[group_idx1].add(agent_idx2)
            self.peer_groups[group_idx2].add(agent_idx1)

    def _init_validators(self, n):
        # Each validator has a skill/reputation
        return [
            {
                "skill": np.clip(
                    np.random.normal(
                        self.validator_skill_mean, self.validator_skill_std
                    ),
                    0.1,
                    1.0,
                )
            }
            for _ in range(n)
        ]

    def _generate_projects(self):
        self.open_projects = self.project_templates.copy()
        for i, project in enumerate(self.open_projects):
            project["required_effort"] = project["required_effort"] + np.random.normal(
                0, project["required_effort"] * 0.2
            )
            project["approx_reward"] = project["approx_reward"] + np.random.uniform(
                -0.2, 0.2
            )
            if i % 2 == 0:
                # every second time half the agents are more fit for the project and half are less fit
                project["fit"] = np.array(
                    [
                        (
                            min(0.1, np.random.normal(0.3, 0.15))
                            if i % 2 == 0
                            else max(0.9, np.random.normal(0.7, 0.15))
                        )
                        for _ in range(self.n_agents)
                    ]
                )
            else:
                # the other times the other half are fitter
                project["fit"] = np.array(
                    [
                        (
                            min(0.1, np.random.normal(0.3, 0.15))
                            if i % 2 == 1
                            else max(0.9, np.random.normal(0.7, 0.15))
                        )
                        for _ in range(self.n_agents)
                    ]
                )

            project["validator"] = 0  # Start with validator 0
            project["time_window"] = np.ceil(
                project["required_effort"] * np.random.normal(2, 0.8)
            )
            project["current_effort"] = 0
            project["contributors"] = []
            project["start_time"] = self.timestep
            project["finished"] = False

    def reset(self, seed=None, options=None):
        self.timestep = 0
        self.agent_steps = np.zeros(self.n_agents, dtype=np.int32)
        self.rewardless_steps = np.zeros(self.n_agents, dtype=np.int32)
        self.agent_rewards = np.zeros(self.n_agents, dtype=np.float32)
        self.agent_ages = np.zeros(self.n_agents, dtype=np.int32)
        self.agent_completed_projects = np.zeros(self.n_agents, dtype=np.int32)
        self.agent_active_projects = [[] for _ in range(self.n_agents)]
        self.agent_project_effort = [{} for _ in range(self.n_agents)]
        self._init_peer_groups()
        self.projects = {}
        self.agents = copy(self.possible_agents)
        self.actions = {}
        self.observations = {}
        self.action_masks = {}
        self.rewards = {}
        observations = {}
        for agent in self.agents:
            obs = self._get_observation(agent)
            mask = self._get_action_mask(agent)
            self.observations[agent] = obs
            self.action_masks[agent] = mask
            observations[agent] = {"observation": obs, "action_mask": mask}
        infos = {a: {} for a in self.agents}
        return observations, infos

    def _get_observation(self, agent):
        idx = self.agent_to_id[agent]
        # Peer group: sorted array of peer agent ids
        peer_group = np.array(
            sorted(self.peer_groups[self.agent_peer_idx[idx]]), dtype=np.int32
        )
        peer_reputation = self.agent_rewards[peer_group].astype(np.float32)
        obs = {
            "peer_group": peer_group,
            "peer_reputation": peer_reputation,
            "project_opportunities": self.open_projects,
            "running_projects": self._get_running_projects_obs(idx),
            "age": np.array([self.agent_ages[idx]], dtype=np.int32),
            "accumulated_rewards": np.array(
                [self.agent_rewards[idx]], dtype=np.float32
            ),
        }
        return obs

    def _get_running_projects_obs(self, agent_idx):
        # Projects the agent is currently working on
        running_obs = {}
        for p_idx in self.agent_active_projects[agent_idx]:
            p = self.projects[p_idx]
            running_obs[f"project_{p_idx}"] = {
                "required_effort": np.array([p["required_effort"]], dtype=np.float32),
                "approx_reward": np.array([p["approx_reward"]], dtype=np.float32),
                "fit": p["peer_fit"][np.where(p["contributors"] == agent_idx)[0]],
                "peer_fit": p["peer_fit"],
                "time_left": max(
                    0, p["time_window"] - (self.timestep - p["start_time"])
                ),
                "current_effort": p["current_effort"],
                "self_effort": np.array(
                    [self.agent_project_effort[agent_idx].get(p_idx, 0)], dtype=np.int32
                ),
            }
        return running_obs

    def _get_action_mask(self, agent):
        idx = self.agent_to_id[agent]
        mask = {}
        # Project selection: can only select if under max_projects_per_agent
        can_choose = int(
            len(self.agent_active_projects[idx]) < self.max_projects_per_agent
        )
        mask["choose_project"] = np.zeros(self.n_projects + 1, dtype=np.int8)
        if can_choose:
            mask["choose_project"][:] = 1
        else:
            mask["choose_project"][:] = 0
            mask["choose_project"][0] = 1  # Only 'no project' allowed
        # Peer collaboration: MultiBinary for peer group
        peer_group = sorted(self.peer_groups[self.agent_peer_idx[idx]])
        mask["collaborate_with"] = np.ones(len(peer_group), dtype=np.int8)
        # exclude self collaboration
        mask["collaborate_with"][peer_group.index(idx)] = 0
        # Effort: can only put effort into active projects
        mask["put_effort"] = np.zeros(self.max_projects_per_agent + 1, dtype=np.int8)
        mask["put_effort"][0] = 1  # no effort always possible
        for p_idx in range(len(self.agent_active_projects[idx])):
            mask["put_effort"][p_idx + 1] = 1
        return mask

    def _start_open_project(self, project_idx, contributors):
        project_id = len(self.projects)
        new_running_proj = deepcopy(self.open_projects[project_idx])
        new_running_proj["id"] = f"project_{project_idx}-{self.timestep}"
        new_running_proj["contributors"] = contributors
        new_running_proj["peer_fit"] = new_running_proj["fit"][contributors]
        del new_running_proj["fit"]
        for contributor in contributors:
            self.agent_active_projects[contributor].append(new_running_proj["id"])
            self.agent_project_effort[contributor][new_running_proj["id"]] = 0

        self.projects[new_running_proj["id"]] = new_running_proj
        return project_id

    def _find_project_setting(self, project_idx, peer_group, intents):
        if len(intents) == 0:
            return []
        ## no collaboration
        elif not np.any(intents):
            new_projects = []
            for agent in peer_group:
                running_project_idx = self._start_open_project(project_idx, [agent])
                new_projects.append((running_project_idx, [agent]))
            return new_projects
        else:
            ## Find the biggest overlap of collaborators on the same project as the largest clique in the collaboration graph
            collaborators = max(
                list(nx.find_cliques(nx.from_numpy_array(intents))), key=len
            )
            running_project_idx = self._start_open_project(
                project_idx, peer_group[collaborators]
            )
            ## repeat the process with any remaining agents and possible subgroups
            return [
                (running_project_idx, peer_group[collaborators])
            ] + self._find_project_setting(
                project_idx,
                np.delete(peer_group, collaborators),
                np.delete(intents, collaborators, axis=0),
            )

    def step(self, actions):
        self.actions = actions
        self.timestep += 1
        # Optionally grow peer groups (uncomment if intended)
        if self.timestep % 50 == 0:
            self._grow_peer_groups()

        # Track which open projects are selected to be started this step
        agent_project_choices = {}
        for agent, action in actions.items():
            idx = self.agent_to_id[agent]
            chosen_project = action["choose_project"]

            if (
                chosen_project > 0
                and len(self.agent_active_projects[idx]) < self.max_projects_per_agent
            ):
                open_proj_idx = chosen_project - 1
                if open_proj_idx < len(self.open_projects):
                    agent_project_choices[idx] = open_proj_idx

            # Apply effort to running projects
            if (
                action["put_effort"] > 0
                and len(self.agent_active_projects[idx]) >= action["put_effort"]
            ):
                selected_project = action["put_effort"] - 1
                effort_project_id = self.agent_active_projects[idx][selected_project]
                effort_project = self.projects[effort_project_id]
                effort_amount = effort_project["peer_fit"][
                    np.where(effort_project["contributors"] == idx)[0]
                ]

                self.projects[effort_project_id]["current_effort"] += effort_amount
                self.agent_project_effort[idx][effort_project_id] += effort_amount

        # Collaboration intents (for each agent, with their peers)
        for peer_group in self.peer_groups:
            peer_group = np.array(sorted(peer_group))
            peer_group_choices = [
                agent_project_choices.get(pc_idx, None) for pc_idx in peer_group
            ]
            peer_group_intents = np.array(
                [
                    actions[f"agent_{gm_idx}"]["collaborate_with"]
                    for gm_idx in peer_group
                ]
            )
            for choice, _ in Counter(peer_group_choices).most_common():
                if choice is not None:
                    # get all collaborators which took this choice
                    potential_collaborators = np.where(
                        np.array(peer_group_choices) == choice
                    )[0]
                    collaborator_group = peer_group[potential_collaborators]

                    # find overlaps in collaboration intents of collaborators
                    collaborators_intents = peer_group_intents[
                        np.ix_(potential_collaborators, potential_collaborators)
                    ]
                    # Only keep edges where both i→j and j→i exist
                    collaborators_intents = (
                        collaborators_intents & collaborators_intents.T
                    )
                    running_projects = self._find_project_setting(
                        choice, collaborator_group, collaborators_intents
                    )
                    print(f"new projects: {running_projects}")

        # Check project completion and assign rewards
        self.rewards = {a: 0.0 for a in self.agents}
        for p_idx, p in self.projects.items():
            if (self.timestep - p["start_time"]) >= p["time_window"] and p[
                "finished"
            ] == False:
                validator = self.validators[p["validator"]]
                noise = np.random.normal(1, 1 - validator["skill"])
                quality = (
                    1
                    - (max(0, p["required_effort"] - p["current_effort"]) * noise)
                    / p["required_effort"]
                )
                quality = np.clip(quality, 0, 1)
                if quality > 0.5:
                    reward = p["approx_reward"] + np.random.normal(0, 0.3)
                else:
                    reward = 0
                for idx in p["contributors"]:
                    try:
                        self.agent_active_projects[idx].remove(p["id"])
                    except ValueError:
                        breakpoint()
                    self.agent_rewards[idx] += reward
                    self.agent_completed_projects[idx] += 1
                    self.rewards[f"agent_{idx}"] += reward
                p["finished"] = True

        # Update agent ages, steps, rewardless steps
        for idx, agent in enumerate(self.agents):
            self.agent_ages[idx] += 1
            self.agent_steps[idx] += 1
            if self.rewards[agent] > 0:
                self.rewardless_steps[idx] = 0
            else:
                self.rewardless_steps[idx] += 1

        # Drop and replace agents with too many rewardless steps or max timesteps
        truncations = {
            a: self.rewardless_steps[self.agent_to_id[a]] >= self.max_rewardless_steps
            or self.agent_steps[self.agent_to_id[a]] >= self.max_timesteps
            for a in self.agents
        }
        terminations = {a: False for a in self.agents}
        for agent, trunc in truncations.items():
            if trunc:
                idx = self.agent_to_id[agent]
                self.agent_ages[idx] = 0
                self.agent_completed_projects[idx] = 0
                self.agent_active_projects[idx] = []
                self.agent_project_effort[idx] = {}
                self.agent_rewards[idx] = 0
                self.agent_steps[idx] = 0
                self.rewardless_steps[idx] = 0

        # Prepare next obs/mask
        observations = {}
        for agent in self.agents:
            obs = self._get_observation(agent)
            mask = self._get_action_mask(agent)
            self.observations[agent] = obs
            self.action_masks[agent] = mask
            observations[agent] = {"observation": obs, "action_mask": mask}
        infos = {a: {} for a in self.agents}
        return observations, self.rewards, terminations, truncations, infos

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Dict(
            {
                "peer_group": Box(
                    0, self.n_agents, (self.peer_group_size,), dtype=np.int32
                ),
                "peer_reputation": Box(
                    0, 1e4, (self.peer_group_size,), dtype=np.float32
                ),
                "project_opportunities": Dict(
                    {
                        f"project_{i}": Dict(
                            {
                                "required_effort": Box(0, 20, (1,), dtype=np.int32),
                                "approx_reward": Box(0, 1, (1,), dtype=np.float32),
                                "fit": Box(0, 1, (1,), dtype=np.float32),
                                "peer_fit": Box(
                                    0, 1, (self.peer_group_size), dtype=np.float32
                                ),
                                "time_window": Box(0, 50, (1,), dtype=np.float32),
                            }
                        )
                        for i in range(self.n_projects)
                    }
                ),
                "running_projects": Dict(
                    {
                        f"project_{i}": Dict(
                            {
                                "required_effort": Box(0, 20, (1,), dtype=np.int32),
                                "approx_reward": Box(0, 1, (1,), dtype=np.float32),
                                "fit": Box(0, 1, (1,), dtype=np.float32),
                                "peer_fit": Box(
                                    0, 1, (self.peer_group_size), dtype=np.float32
                                ),
                                "time_left": Box(0, 50, (1,), dtype=np.float32),
                                "current_effort": Box(
                                    0, self.peer_group_size * 50, (1,), dtype=np.float32
                                ),
                            }
                        )
                        for i in range(self.max_projects_per_agent)
                    }
                ),
                "age": Box(0, 1e4, (1,), dtype=np.int32),
                "accumulated_rewards": Box(0, 1e4, (1,), dtype=np.float32),
            }
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # choose_project: Discrete(n_projects+1), collaborate_with: MultiBinary(peer_group_size), put_effort: MultiBinary(n_projects)
        return Dict(
            {
                "choose_project": Discrete(self.n_projects + 1),
                "collaborate_with": MultiBinary(self.peer_group_size),
                "put_effort": Discrete(self.max_projects_per_agent + 1),
            }
        )
