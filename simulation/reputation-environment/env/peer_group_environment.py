from collections import Counter
from copy import copy, deepcopy
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
from gymnasium.spaces import Box
from gymnasium.spaces import Dict as GymDict
from gymnasium.spaces import Discrete, MultiBinary
from pettingzoo import ParallelEnv

from .project import Project


class Novelty_area(NamedTuple):
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def __contains__(self, coordinate: Tuple[float, float]):
        return (
            self.x_min < coordinate[0] < self.x_max
            and self.y_min < coordinate[1] < self.y_max
        )


class PeerGroupEnvironment(ParallelEnv):
    """Multi-agent environment with peer groups and project opportunities."""

    metadata = {
        "name": "peer_group_environment_v0",
    }

    def __init__(
        self,
        start_agents: int = 20,
        max_agents: int = 80,
        max_peer_group_size: int = 60,
        n_groups: int = 4,
        n_projects: int = 6,
        max_projects_per_agent: int = 6,
        max_agent_age: int = 1000,
        max_rewardless_steps: int = 50,
        growth_rate: float = 0.02,
        render_mode: Optional[str] = None,
    ) -> None:
        self.n_agents: int = max_agents
        self.starting_population_size: int = start_agents
        self.n_groups: int = n_groups
        self.max_peer_group_size: int = max_peer_group_size
        self.n_projects: int = n_projects
        self.max_projects_per_agent: int = max_projects_per_agent
        self.max_agent_age: int = max_agent_age
        self.max_rewardless_steps: int = max_rewardless_steps
        self.growth_rate: float = growth_rate
        self.render_mode: Optional[str] = render_mode

        self.possible_agents: List[str] = [f"agent_{i}" for i in range(self.n_agents)]
        self.agent_to_id: Dict[str, int] = {
            agent: i for i, agent in enumerate(self.possible_agents)
        }
        # Cache for action and observation spaces
        self._space_cache: Dict[str, Any] = {}
        self._peer_groups_changed: bool = False
        self.timestep: int = 0
        self.agent_steps = np.zeros(self.n_agents, dtype=np.int32)
        self.rewardless_steps = np.zeros(self.n_agents, dtype=np.int32)
        self.agent_rewards = np.zeros(self.n_agents, dtype=np.float32)
        self.agent_ages = np.zeros(self.n_agents, dtype=np.int32)
        self.agent_completed_projects = [np.zeros(self.n_agents, dtype=np.int32)]
        self.agent_sucessful_projects: List[List[str]] = [
            [] for _ in range(self.n_agents)
        ]
        self.agent_active_projects: List[List[Optional[str]]] = [
            [None for _ in range(self.max_projects_per_agent)]
            for _ in range(self.n_agents)
        ]  # List of project indices
        self.agent_project_effort: List[Dict[str, float]] = [
            {} for _ in range(self.n_agents)
        ]  # {project_idx: effort}
        self.actions: Dict[str, Dict[str, Any]] = {}
        self.observations: Dict[str, Dict[str, Any]] = {}
        self.action_masks: Dict[str, Dict[str, Any]] = {}
        self.rewards: Dict[str, float] = {}
        self.agents: List[str] = []
        self.active_agents = np.zeros(self.n_agents, dtype=np.int8)
        self.terminated_agents = np.zeros(self.n_agents, dtype=np.int8)
        # activate a subset of agents equal to the starting population size
        self.active_agents[: self.starting_population_size] = 1

        # Will be initialized in _init_peer_groups
        self.peer_groups: List[List[int]] = [[] for _ in range(n_groups)]
        self.agent_peer_idx: List[int] = []

        # Will be initialized in _generate_projects
        self.open_projects: List[Dict[str, Any]] = []
        # The chosen projects will be added here
        self.projects: Dict[str, Project] = {}

        self.project_templates: List[Dict[str, Any]] = [
            # low novelty, low effort, low prestige
            {
                "required_effort": 10,
                "prestige": 0.1,
                "novelty": 0.05,
            },
            # low novelty, medium effort, medium prestige
            {
                "required_effort": 50,
                "prestige": 0.5,
                "novelty": 0.05,
            },
            # low novelty, high effort, low prestige
            {
                "required_effort": 100,
                "prestige": 1.0,
                "novelty": 0.05,
            },
            # high novelty, low effort, low prestige
            {
                "required_effort": 10,
                "prestige": 0.1,
                "novelty": 0.5,
            },
            # high novelty, medium effort, medium prestige
            {
                "required_effort": 50,
                "prestige": 0.5,
                "novelty": 0.5,
            },
            # high novelty, high effort, high prestige
            {
                "required_effort": 100,
                "prestige": 1.0,
                "novelty": 0.5,
            },
        ]
        self._generate_projects()

    def _init_peer_groups(self) -> None:
        if self.n_agents < self.n_groups:
            raise ValueError(
                f"{self.n_agents} agents can not be distributed into {self.n_groups}!"
            )
        if self.max_peer_group_size > self.n_agents:
            raise ValueError(f"Peer_group_size can't be bigger than number of agents!")
        # n_groups = self.n_agents // self.peer_group_size
        # self.peer_groups = [[] for _ in range(n_groups)]
        self.peer_group_centroids = np.random.rand(self.n_groups, 2)
        for i in range(self.n_agents):
            self.peer_groups[i % self.n_groups].append(i)
        # Each agent has a fixed set of peers (not necessarily symmetric)
        self.agent_peer_idx = []  # List of sets of peer agent ids for each agent
        for i in range(self.n_agents):
            # Peers are the all agents in the same peer group
            self.agent_peer_idx.append(i % self.n_groups)

    def _connect_peer_groups(self) -> None:
        # Pick two random groups.
        if len(self.peer_groups) % 2 != 0:
            raise ValueError(f"Peer groups must be even, found {len(self.peer_groups)}")

        group_pairs = list(
            np.random.permutation(
                np.array(list(enumerate(self.peer_groups)), dtype=object)
            )
        )
        for i in range(0, len(group_pairs), 2):
            group_idx1, group1 = group_pairs[i]
            group_idx2, group2 = group_pairs[i + 1]
            if (
                len(group1) >= self.max_peer_group_size
                or len(group2) >= self.max_peer_group_size
            ):
                continue
            # Pick a random agent from each group which isn't already in the other group.
            group1 = set(group1)
            group2 = set(group2)
            if len(group1 - group2) == 0 or len(group2 - group1) == 0:
                continue
            try:
                active_only_group1 = self.active_agents[list(group1 - group2)]
                active_only_group2 = self.active_agents[list(group2 - group1)]
                agent_idx1 = np.random.choice(active_only_group1)
                agent_idx2 = np.random.choice(active_only_group2)

            except ValueError:
                print(
                    f"Warning: Groups {group_idx1} and {group_idx2} couldn't be connected because all members already know each other."
                )

            # Add agent 1 to agents 2's peer group and vice versa.
            if agent_idx2 not in self.peer_groups[group_idx1]:
                self.peer_groups[group_idx1].append(agent_idx2)
            if agent_idx1 not in self.peer_groups[group_idx2]:
                self.peer_groups[group_idx2].append(agent_idx1)

    def _activate_agent(self, group_idx: int):
        group = self.peer_groups[group_idx]
        active_in_group = self.active_agents[group]
        for agent_i, active in zip(group, active_in_group):
            if active == 0 and self.terminated_agents[agent_i] == 0:
                self.active_agents[agent_i] = 1
                return agent_i
        return None

    def _generate_projects(self) -> None:
        self.open_projects = self.project_templates.copy()
        for i, project in enumerate(self.open_projects):
            project["required_effort"] = int(
                project["required_effort"]
                + np.random.normal(0, project["required_effort"] * 0.2)
            )
            project["prestige"] = project["prestige"] + np.random.normal(
                0, project["prestige"] * 0.2
            )
            project["novelty"] = project["novelty"] + np.random.normal(
                0, project["novelty"] * 0.2
            )
            project["time_window"] = np.ceil(
                project["required_effort"] * np.random.uniform(0.8, 2)
            )
            project["current_effort"] = 0
            project["contributors"] = []
            project["start_time"] = 0
            project["finished"] = False

    def _get_active_projects(self, agent: int) -> List[str]:
        return [p for p in self.agent_active_projects[agent] if p is not None]

    def _add_active_project(self, agent: int, proj_id: str) -> int:
        for i, slot in enumerate(self.agent_active_projects[agent]):
            if slot is None:
                self.agent_active_projects[agent][i] = proj_id
                return i
        return -1

    def _remove_active_project(self, agent: int, proj_id: str) -> None:
        try:
            idx = self.agent_active_projects[agent].index(proj_id)
            self.agent_active_projects[agent][idx] = None
        except ValueError:
            pass

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        if seed is not None:
            np.random.seed(seed)
        self.timestep = 0
        self.agent_steps = np.zeros(self.n_agents, dtype=np.int32)
        self.rewardless_steps = np.zeros(self.n_agents, dtype=np.int32)
        self.agent_rewards = np.zeros(self.n_agents, dtype=np.float32)
        self.agent_ages = np.zeros(self.n_agents, dtype=np.int32)
        self.agent_completed_projects = np.zeros(self.n_agents, dtype=np.int32)
        self.agent_active_projects = [
            [None for _ in range(self.max_projects_per_agent)]
            for _ in range(self.n_agents)
        ]
        self.agent_project_effort = [{} for _ in range(self.n_agents)]
        self._init_peer_groups()
        self.projects = {}
        self.agents = copy(self.possible_agents)
        self.actions = {}
        self.observations = {}
        self.action_masks = {}
        self.rewards = {}

        # Clear space cache on reset
        self._clear_space_cache()

        observations = {}
        for agent in self.agents:
            obs = self._get_observation(agent)
            mask = self._get_action_mask(agent)
            self.observations[agent] = obs
            self.action_masks[agent] = mask
            observations[agent] = {"observation": obs, "action_mask": mask}
        infos = {a: {} for a in self.agents}
        return observations, infos

    def _clear_space_cache(self) -> None:
        """Clear the cached action and observation spaces."""
        self._space_cache.clear()
        self._peer_groups_changed = False

    def _get_observation(self, agent: str) -> Dict[str, Any]:
        idx = self.agent_to_id[agent]
        # Peer group: array of peer agent ids
        peer_group = np.array(
            self.peer_groups[self.agent_peer_idx[idx]], dtype=np.int32
        )
        peer_reputation = self.agent_rewards[peer_group].astype(np.float32)
        obs = {
            "peer_group": peer_group,
            "peer_reputation": peer_reputation,
            "project_opportunities": self._get_open_projects_obs(idx),
            "running_projects": self._get_running_projects_obs(idx, peer_group),
            "age": np.array([self.agent_ages[idx]], dtype=np.int32),
            "accumulated_rewards": np.array(
                [self.agent_rewards[idx]], dtype=np.float32
            ),
        }
        return obs

    def _get_open_projects_obs(self, agent_idx: int) -> List[Dict[str, Any]]:
        agent_open_projs = deepcopy(self.open_projects)
        return agent_open_projs

    def _get_running_projects_obs(
        self, agent_idx: int, peer_group: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        # Projects the agent is currently working on
        running_obs = {}
        for p_idx in self._get_active_projects(agent_idx):
            p = self.projects[p_idx].as_observation_dict()
            running_obs[p_idx] = {
                "required_effort": p["required_effort"],
                "prestige": p["prestige"],
                "time_left": max(
                    0, p["time_window"] - (self.timestep - p["start_time"])
                ),
                "current_effort": p["current_effort"],
                "contributors": [
                    1 if c in p["contributors"] else 0 for c in peer_group
                ],
                "contributor_effort": [
                    self.agent_project_effort[agent_i].get(p_idx, 0)
                    for agent_i in peer_group
                ],
                "peer_fit": p["peer_fit"],
            }
        return running_obs

    def _get_action_mask(self, agent: str) -> Dict[str, np.ndarray]:
        idx = self.agent_to_id[agent]
        ## this agent dropped out
        if self.active_agents[idx] == 0:
            return {
                "choose_project": np.zeros(self.n_projects + 1, dtype=np.int8),
                "collaborate_with": np.zeros(self.max_peer_group_size, dtype=np.int8),
                "put_effort": np.zeros(self.max_projects_per_agent + 1, dtype=np.int8),
            }
        mask = {}
        # Project selection: can only select if under max_projects_per_agent
        can_choose = int(
            len(self._get_active_projects(idx)) < self.max_projects_per_agent
        )
        mask["choose_project"] = np.zeros(self.n_projects + 1, dtype=np.int8)
        if can_choose:
            mask["choose_project"][:] = 1
        else:
            mask["choose_project"][:] = 0
            mask["choose_project"][0] = 1  # Only 'no project' allowed
        # Peer collaboration: MultiBinary for peer group
        peer_group = self.peer_groups[self.agent_peer_idx[idx]]
        mask["collaborate_with"] = np.zeros(self.max_peer_group_size, dtype=np.int8)
        mask["collaborate_with"][: len(peer_group)] = np.where(
            self.active_agents[peer_group],
            2,  ## if active unmask
            mask["collaborate_with"][: len(peer_group)],  # else keep 0
        )

        # Effort: can only put effort into active projects
        mask["put_effort"] = np.zeros(self.max_projects_per_agent + 1, dtype=np.int8)
        mask["put_effort"][0] = 1  # no effort always possible
        for i, p_idx in enumerate(self.agent_active_projects[idx]):
            if p_idx is not None:
                mask["put_effort"][i + 1] = 1
        return mask

    def _find_project_setting(
        self, project_idx: int, peer_group: np.ndarray, intents: np.ndarray
    ) -> List[Tuple[str, List[int]]]:
        if len(peer_group) == 0:
            return []
        ## no collaboration
        elif not np.any(intents):
            new_projects = []
            for agent in peer_group:
                running_project_idx = self._start_open_project(project_idx, [agent])
                if running_project_idx is not None:
                    new_projects.append((running_project_idx, [agent]))
            return new_projects
        else:
            ## Find the biggest overlap of collaborators on the same project as the largest clique in the collaboration graph
            grouped_collaborators = set()
            running_project_idx = None
            for collaborators in sorted(
                list(nx.find_cliques(nx.from_numpy_array(intents))), key=len
            ):
                already_at_limit = set(
                    [
                        c
                        for c in collaborators
                        if len(self._get_active_projects(c))
                        >= self.max_projects_per_agent
                    ]
                )
                if len(already_at_limit) == 0:
                    running_project_idx = self._start_open_project(
                        project_idx, peer_group[collaborators]
                    )
                    grouped_collaborators |= set(collaborators)
                else:
                    grouped_collaborators |= already_at_limit

            new_project = []

            if running_project_idx is not None:
                new_project = [(running_project_idx, peer_group[collaborators])]

            ## repeat the process with any remaining agents and possible subgroups
            intents[:] = 0
            return new_project + self._find_project_setting(
                project_idx,
                np.delete(peer_group, np.array(list(grouped_collaborators))),
                intents,
            )

    def _start_open_project(
        self, project_idx: int, contributors: List[int]
    ) -> Optional[str]:
        project_id = f"project_{len(self.projects)}-{project_idx}-{self.timestep}"
        suffix = [str(project_idx), str(self.timestep)]
        ## if the project was already added make sure all the contributors are there.
        for contributor in contributors:
            for proj in self._get_active_projects(contributor):
                # the project was already started in the same timestep by this agent
                if proj.split("-")[-2:] == suffix:
                    # ignore
                    if len(contributors) <= len(self.projects[proj].contributors):
                        return None
                    # reconfigure the project
                    else:
                        project_id = proj

        new_running_proj = deepcopy(self.open_projects[project_idx])
        new_running_proj["id"] = project_id
        new_running_proj["start_time"] = self.timestep
        new_running_proj["contributors"] = contributors
        new_running_proj["peer_fit"] = {i: 0 for i in contributors}
        for contributor in contributors:
            self._add_active_project(contributor, new_running_proj["id"])
            self.agent_project_effort[contributor][new_running_proj["id"]] = 0
        proj_object = Project.from_dict(new_running_proj)
        proj_object.kene = self._locate_project_in_plane(proj_object)
        proj_object.peer_fit = [
            self._determine_agent_fit(proj_object, agent_i)
            for agent_i in proj_object.contributors
        ]
        self.projects[new_running_proj["id"]] = proj_object
        return project_id

    def _locate_project_in_plane(self, new_paper: Project) -> np.array:
        # select a random generator paper from all authors in peer group
        all_contributors_papers = []
        for agent_i in new_paper.contributors:
            all_contributors_papers += self.agent_sucessful_projects[agent_i]
        # choose weighted by contributor reputation?
        if len(all_contributors_papers) > 0:
            generator_paper = self.projects[np.random.choice(all_contributors_papers)]
            new_kene = generator_paper.kene
        else:
            peer_group_idx = self.agent_peer_idx[new_paper.contributors[0]]
            new_kene = self.peer_group_centroids[peer_group_idx]

        # select 10-20 papers as citation which are in the area of novelty around the generator
        papers_in_vicinity = []
        vicinity_area = Novelty_area(
            new_kene[0] - new_paper.novelty,
            new_kene[0] + new_paper.novelty,
            new_kene[1] - new_paper.novelty,
            new_kene[1] + new_paper.novelty,
        )
        for paper_id in self.agent_sucessful_projects[agent_i]:
            if (
                self.projects[paper_id].kene in vicinity_area
                and paper_id != generator_paper.project_id
            ):
                papers_in_vicinity.append(paper_id)
        if len(papers_in_vicinity) == 0:
            new_paper.citations = []
            return new_kene + np.random.uniform(0, 0.1, 2)
        n_cited = min(len(papers_in_vicinity), np.random.uniform(10, 20))
        # weighted by n citations?
        cited_papers = np.random.choice(papers_in_vicinity, n_cited)
        m = 0
        for cited_paper in cited_papers:
            cited_paper = self.projects[cited_paper]
            cited_paper.cited_by.append(paper_id)
            cited_position = cited_paper.kene
            m += np.random.uniform(0, 0.1)
            new_kene += (new_kene - cited_position) * (1 - m) / 2
        new_paper.citations = cited_papers
        return new_kene

    def _determine_agent_fit(self, project: Project, agent_i: int) -> float:
        if len(self.agent_sucessful_projects[agent_i]) > 0:
            agent_centroid = np.array(
                [
                    self.projects[p_idx].kene
                    for p_idx in self.agent_sucessful_projects[agent_i]
                ]
            ).mean(axis=0)
        else:
            return 0.5
        return 1 - np.linalg.norm(project.kene - agent_centroid)

    def step(self, actions: Dict[str, Dict[str, Any]]) -> Tuple[
        Dict[str, Dict[str, Any]],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        self.actions = actions
        self.timestep += 1

        # Track which open projects are selected to be started this step
        agent_project_choices: Dict[int] = {}
        for agent, action in actions.items():
            idx = self.agent_to_id[agent]
            chosen_project = action["choose_project"]

            if (
                chosen_project > 0
                and len(self._get_active_projects(idx)) < self.max_projects_per_agent
            ):
                open_proj_idx = chosen_project - 1
                if open_proj_idx < len(self.open_projects):
                    agent_project_choices[idx] = open_proj_idx

            # Apply effort to running projects
            if (
                action["put_effort"] > 0
                and len(self._get_active_projects(idx)) >= action["put_effort"]
            ):
                selected_project = action["put_effort"] - 1
                effort_project_id = self.agent_active_projects[idx][selected_project]
                effort_project = self.projects[effort_project_id]
                contributors_idx = list(effort_project.contributors).index(idx)
                effort_amount = effort_project.peer_fit[contributors_idx]

                self.projects[effort_project_id].add_effort(effort_amount)
                self.agent_project_effort[idx][effort_project_id] += effort_amount

        # Collaboration intents (for each agent, with their peers)
        for peer_group in self.peer_groups:
            peer_group = np.array((peer_group))
            peer_group_choices: List[Optional[int]] = [
                agent_project_choices.get(pc_idx, None) for pc_idx in peer_group
            ]
            peer_group_intents = np.array(
                [
                    actions[f"agent_{gm_idx}"]["collaborate_with"]
                    for gm_idx in peer_group
                ]
            )
            np.fill_diagonal(peer_group_intents, 0)  # no self collaboration

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

                    self._find_project_setting(
                        choice, collaborator_group, collaborators_intents
                    )

        # Check project completion and assign rewards
        self.rewards = {a: 0.0 for a in self.agents}
        for p_idx, p in self.projects.items():
            if p.is_due(self.timestep) and p.finished is False:
                quality = p.calculate_quality(noise_factor=0.5)
                quality = np.clip(quality, 0, 1)

                reward = p.calculate_reward(quality, threshold=0.5, noise_factor=0.15)
                for idx in p.contributors:
                    self._remove_active_project(idx, p_idx)
                    if reward > 0:
                        self.agent_sucessful_projects[idx].append(p.project_id)
                    self.agent_rewards[idx] += reward
                    self.agent_completed_projects[idx] += 1
                    self.rewards[f"agent_{idx}"] += reward
                p.finished = True

        # Update agent ages, steps, rewardless steps
        for idx, agent in enumerate(self.agents):
            self.agent_ages[idx] += 1
            self.agent_steps[idx] += 1
            if self.rewards[agent] > 0:
                self.rewardless_steps[idx] = 0
            else:
                self.rewardless_steps[idx] += 1

        truncations = {a: False for a in self.agents}
        # Drop agents with too many rewardless steps or max timesteps
        terminations = {
            a: (
                self.rewardless_steps[self.agent_to_id[a]] >= self.max_rewardless_steps
                or self.agent_steps[self.agent_to_id[a]] >= self.max_agent_age
            )
            for a in self.agents
        }
        self.terminated_agents = self.terminated_agents | np.fromiter(
            terminations.values(), dtype=bool
        )
        agents_activated_in_step = []
        # connect peer groups
        if self.timestep % 50 == 0:
            self._connect_peer_groups()

        for agent_id, terminated in terminations.items():
            agent_id = self.agent_to_id[agent_id]
            # replace
            if terminated:
                self.active_agents[agent_id] = 0
                group = self.agent_peer_idx[agent_id]
                agents_activated_in_step.append(self._activate_agent(group))
        # grow active agents
        if self.growth_rate < 1:
            if self.timestep % (1 // self.growth_rate) == 0:
                # choice weighted by success?
                group = np.random.choice(range(self.n_groups))
                agents_activated_in_step.append(self._activate_agent(group))
        else:
            each_step = np.floor(self.growth_rate)
            for _ in range(each_step):
                group = np.random.choice(range(self.n_groups))
                agents_activated_in_step.append(self._activate_agent(group))
            if self.timestep % (1 // (self.growth_rate - each_step)) == 0:
                group = np.random.choice(range(self.n_groups))
                agents_activated_in_step.append(self._activate_agent(group))

        # if not all([a is not None for a in agents_activated_in_step]):
        #     print("No more agents to activate!")
        # breakpoint()
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

    def observation_space(self, agent: str) -> GymDict:
        # Check if cache needs to be invalidated
        if self._peer_groups_changed:
            self._clear_space_cache()

        # Check if space is already cached
        cache_key = f"obs_{agent}"
        if cache_key in self._space_cache:
            return self._space_cache[cache_key]

        # Get the actual peer group size for this agent
        idx = self.agent_to_id[agent]
        actual_peer_group_size = len(self.peer_groups[self.agent_peer_idx[idx]])

        space = GymDict(
            {
                "peer_group": Box(
                    0, self.n_agents, (actual_peer_group_size,), dtype=np.int32
                ),
                "peer_reputation": Box(
                    0, 1e4, (actual_peer_group_size,), dtype=np.float32
                ),
                "project_opportunities": GymDict(
                    {
                        f"project_{i}": GymDict(
                            {
                                "required_effort": Box(0, 20, (1,), dtype=np.int32),
                                "prestige": Box(0, 1, (1,), dtype=np.float32),
                                "peer_fit": Box(
                                    0, 1, (actual_peer_group_size), dtype=np.float32
                                ),
                                "time_window": Box(0, 50, (1,), dtype=np.int32),
                            }
                        )
                        for i in range(self.n_projects)
                    }
                ),
                "running_projects": GymDict(
                    {
                        f"project_{i}": GymDict(
                            {
                                "required_effort": Box(0, 20, (1,), dtype=np.int32),
                                "prestige": Box(0, 1, (1,), dtype=np.float32),
                                "peer_fit": Box(
                                    0, 1, (actual_peer_group_size), dtype=np.float32
                                ),
                                "time_left": Box(0, 50, (1,), dtype=np.int32),
                                "current_effort": Box(
                                    0,
                                    actual_peer_group_size * 50,
                                    (1,),
                                    dtype=np.float32,
                                ),
                                "contributors": MultiBinary(actual_peer_group_size),
                                "contributor_effort": Box(
                                    0, 50, (actual_peer_group_size,), dtype=np.float32
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

        # Cache the space
        self._space_cache[cache_key] = space
        return space

    def action_space(self, agent: str) -> GymDict:
        space = GymDict(
            {
                "choose_project": Discrete(self.n_projects + 1),
                "collaborate_with": MultiBinary(self.max_peer_group_size),
                "put_effort": Discrete(self.max_projects_per_agent + 1),
            }
        )
        return space
