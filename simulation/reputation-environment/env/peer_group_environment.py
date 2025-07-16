import functools
import numpy as np
from copy import copy
from gymnasium.spaces import Discrete, MultiBinary, Box, Dict
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
        self.agent_active_projects = [[] for _ in range(self.n_agents)]  # List of project indices
        self.agent_project_effort = [{} for _ in range(self.n_agents)]  # {project_idx: effort}
        self.agent_peer_group = [0 for _ in range(self.n_agents)]  # Index of peer group
        self.projects = []
        self._init_projects()
        self.actions = {}
        self.observations = {}
        self.action_masks = {}
        self.rewards = {}
        self.agents = copy(self.possible_agents)
        self.project_templates = [
            #good fit, low effort, low reward
            {
                "required_effort": 10,
                "approx_reward": 0.1,
                "fit": "high/low",
            },
            #good fit, medium effort, medium reward
            {
                "required_effort": 50,
                "approx_reward": 0.5,
                "fit": "high/low",
            },
            #good fit, high effort, high reward
            {
                "required_effort": 100,
                "approx_reward": 1.0,
                "fit": "high/low",
            },
            #low fit, low effort, low reward
            {
                "required_effort": 10,
                "approx_reward": 0.1,
                "fit": "high/low",
            },
            #low fit, medium effort, medium reward
            {
                "required_effort": 50,
                "approx_reward": 0.5,
                "fit": "high/low",
            },
            #low fit, high effort, high reward
            {
                "required_effort": 100,
                "approx_reward": 1.0,
                "fit": "high/low",
            },
        ]

    def _init_peer_groups(self):
        self.peer_groups = [set() for _ in range(self.peer_group_size)]
        for i in range(self.n_agents):
            self.peer_groups[i%self.peer_group_size].add(i)
        # Each agent has a fixed set of peers (not necessarily symmetric)
        self.agent_peers = []  # List of sets of peer agent ids for each agent
        for i in range(self.n_agents):
            # Peers are the other agents in the same peer group except self
            self.agent_peers.append(self.peer_groups[i%self.peer_group_size] - {i})

    def _grow_peer_groups(self):
        # Pick a two random groups.
        self.peer_group_size += 1
        perms = np.random.permutation(list(enumerate(self.peer_groups)))
        if len(perms) % 2 != 0:
            raise ValueError("Peer groups must be even")
        for i in range(0, len(perms), 2):
            group_idx1, group1 = perms[i]
            group_idx2, group2 = perms[i+1]
            # Pick a random agent from each group which isn't already in the other group.
            try:
                agent_idx1 = np.random.choice(list(group1-group2))
                agent_idx2 = np.random.choice(list(group2-group1))
                
            except ValueError:
                print(f"Warning: Groups {group_idx1} and {group_idx2} couldn't be grown because all members already know each other.")
                self._grow_peer_groups()
            
            # Add agent 1 to agents 2's peer group and vice versa.
            self.peer_groups[group_idx1].add(agent_idx2)
            self.peer_groups[group_idx2].add(agent_idx1)
        
        for i in range(self.n_agents):
            # Peers are the other agents in the same peer group except self
            self.agent_peers.append(self.peer_groups[i%self.peer_group_size] - {i})

    def _init_validators(self, n):
        # Each validator has a skill/reputation
        return [
            {"skill": np.clip(np.random.normal(self.validator_skill_mean, self.validator_skill_std), 0.1, 1.0)}
            for _ in range(n)
        ]
    
    def _generate_projects(self):
        self.open_projects = self.project_templates.copy()
        for i, project in enumerate(self.open_projects):
            project["required_effort"] = project["required_effort"] + np.random.normal(0, project["required_effort"] * 0.2)
            project["approx_reward"] = project["approx_reward"] + np.random.uniform(-0.2, 0.2)
            if i%2==0:
                # every second time half the agents are more fit for the project and half are less fit
                project["fit"] = [
                    min(0.1, np.random.normal(0.3, 0.15)) if i%2==0 
                    else max(0.9, np.random.normal(0.7, 0.15)) 
                    for _ in range(self.n_agents)
                ]
            else:
                # the other times the other half are fitter
                project["fit"] = [
                    min(0.1, np.random.normal(0.3, 0.15)) if i%2==1 
                    else max(0.9, np.random.normal(0.7, 0.15)) 
                    for _ in range(self.n_agents)
                ]

            project["validator"] = 0  # Start with validator 0
            project["time_window"] = np.ceil(project["required_effort"] * np.random.normal(2, 0.8))
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
        self.projects = []
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
        peer_group = np.array(sorted(self.agent_peers[idx]), dtype=np.int32)
        peer_rewards = self.agent_rewards[peer_group].astype(np.float32)
        obs = {
            "peer_group": peer_group,
            "peer_rewards": peer_rewards,
            "project_opportunities": self.open_projects,
            "running_projects": self._get_running_projects_obs(idx),
            "age": np.array([self.agent_ages[idx]], dtype=np.int32),
            "accumulated_rewards": np.array([self.agent_rewards[idx]], dtype=np.float32),
        }
        return obs
    
    def _get_open_projects_obs(self, agent_idx):
        projects = self.open_projects.copy()
        for p in projects:
            fits = p["fit"].copy()
            del p["fit"]
            p["fit"] = fits[agent_idx]
            p["peer_fit"] = [fits[p_i] for p_i in self.agent_peers[agent_idx]]
        return projects
        
    def _get_running_projects_obs(self, agent_idx):
        # Projects the agent is currently working on
        running_obs = {}
        for p_idx in self.agent_active_projects[agent_idx]:
            p = self.projects[p_idx]
            running_obs[f"project_{p_idx}"] = {
                "required_effort": np.array([p["required_effort"]], dtype=np.float32),
                "approx_reward": np.array([p["approx_reward"]], dtype=np.float32),
                "fit": np.array([p["fit"][agent_idx]], dtype=np.float32),
                "peer_fit": np.array([np.mean(p["peer_fit"][agent_idx])], dtype=np.float32),
                "time_left": np.array([max(0, p["time_window"] - (self.timestep - p["start_time"]))], dtype=np.float32),
                "current_effort": np.array([self.agent_project_effort[agent_idx].get(p_idx, 0)], dtype=np.float32),
            }
        return running_obs

    def _get_action_mask(self, agent):
        idx = self.agent_to_id[agent]
        mask = {}
        # Project selection: can only select if under max_projects_per_agent
        can_choose = int(len(self.agent_active_projects[idx]) < self.max_projects_per_agent)
        mask["choose_project"] = np.zeros(self.n_projects + 1, dtype=np.int8)
        if can_choose:
            mask["choose_project"][:] = 1
        else:
            mask["choose_project"][:] = 0
            mask["choose_project"][0] = 1  # Only 'no project' allowed
        # Peer collaboration: MultiBinary for peer group
        group_idx = self.agent_peer_group[idx]
        n_peers = len(self.agent_peers[group_idx])
        mask["collaborate_with"] = np.ones(n_peers, dtype=np.int8)
        # Effort: can only put effort into active projects
        mask["put_effort"] = np.zeros(self.max_projects_per_agent, dtype=np.int8)
        for p_idx in self.agent_active_projects[idx]:
            mask["put_effort"][p_idx] = 1
        return mask

    def step(self, actions):
        self.actions = actions
        self.timestep += 1
        self._grow_peer_groups()
        # Track collaboration intents for each project
        project_collab_intents = {p_idx: set() for p_idx in range(self.n_projects)}
        # First pass: gather project choices and collaboration intents
        for agent, action in actions.items():
            idx = self.agent_to_id[agent]
            # Choose project
            chosen_project = action["choose_project"]
            if chosen_project > 0 and len(self.agent_active_projects[idx]) < self.max_projects_per_agent:
                p_idx = chosen_project - 1
                if not self.projects[p_idx]["finished"] and p_idx not in self.agent_active_projects[idx]:
                    self.agent_active_projects[idx].append(p_idx)
                    self.projects[p_idx]["contributors"].append(idx)
                    self.agent_project_effort[idx][p_idx] = 0
            # Collaboration intent
            peer_group = sorted(self.agent_peers[idx])
            collab_intent = action["collaborate_with"]
            for i, intent in enumerate(collab_intent):
                if intent:
                    peer_idx = peer_group[i]
                    # If peer is also choosing the same project, mark as intent to collaborate
                    if (f"agent_{peer_idx}" in actions and
                        "choose_project" in actions[f"agent_{peer_idx}"] and
                        actions[f"agent_{peer_idx}"]["choose_project"] == chosen_project and
                        chosen_project > 0):
                        project_collab_intents[chosen_project - 1].add(frozenset({idx, peer_idx}))
        # Second pass: ensure collaboration is reflected in contributors
        for p_idx, collab_pairs in project_collab_intents.items():
            for pair in collab_pairs:
                for agent_idx in pair:
                    if p_idx not in self.agent_active_projects[agent_idx]:
                        self.agent_active_projects[agent_idx].append(p_idx)
                        if agent_idx not in self.projects[p_idx]["contributors"]:
                            self.projects[p_idx]["contributors"].append(agent_idx)
                        self.agent_project_effort[agent_idx][p_idx] = self.agent_project_effort[agent_idx].get(p_idx, 0)
        # Put effort
        for agent, action in actions.items():
            idx = self.agent_to_id[agent]
            for p_idx, put in enumerate(action["put_effort"]):
                if put and p_idx in self.agent_active_projects[idx] and not self.projects[p_idx]["finished"]:
                    self.agent_project_effort[idx][p_idx] += 1
                    self.projects[p_idx]["current_effort"] += 1
        # Check project completion and assign rewards
        self.rewards = {a: 0.0 for a in self.agents}
        for p_idx, p in enumerate(self.projects):
            if not p["finished"] and p["current_effort"] >= p["required_effort"]:
                # Validate project
                validator = self.validators[p["validator"]]
                noise = np.random.normal(0, 1 - validator["skill"])
                quality = 1.0 - abs(p["required_effort"] - p["current_effort"] + noise) / max(1, p["required_effort"])
                quality = np.clip(quality, 0, 1)
                for idx in p["contributors"]:
                    reward = p["approx_reward"] * quality
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
        # Drop and replace agents with too many rewardless steps
        truncations = {a: self.rewardless_steps[self.agent_to_id[a]] >= self.max_rewardless_steps or self.agent_steps[self.agent_to_id[a]] >= self.max_timesteps for a in self.agents}
        terminations = {a: False for a in self.agents}
        for agent, trunc in truncations.items():
            if trunc:
                idx = self.agent_to_id[agent]
                # Replace agent
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
        return Dict({
            "peer_group": Box(0, self.n_agents, (self.peer_group_size,), dtype=np.int32),
            "peer_rewards": Box(-1e6, 1e6, (self.peer_group_size,), dtype=np.float32),
            "project_opportunities": Dict({
                f"project_{i}": Dict({
                    "required_effort": Box(0, 1000, (1,), dtype=np.float32),
                    "approx_reward": Box(0, 1, (1,), dtype=np.float32),
                    "fit": Box(0, 1, (1,), dtype=np.float32),
                    "peer_fit": Box(0, 1, (1,), dtype=np.float32),
                    "time_left": Box(0, 1000, (1,), dtype=np.float32),
                }) for i in range(self.n_projects)
            }),
            "running_projects": Dict({
                f"project_{i}": Dict({
                    "required_effort": Box(0, 1000, (1,), dtype=np.float32),
                    "approx_reward": Box(0, 1, (1,), dtype=np.float32),
                    "fit": Box(0, 1, (1,), dtype=np.float32),
                    "peer_fit": Box(0, 1, (1,), dtype=np.float32),
                    "time_left": Box(0, 1000, (1,), dtype=np.float32),
                    "current_effort": Box(0, 1000, (1,), dtype=np.float32),
                }) for i in range(self.max_projects_per_agent)
            }),
            "age": Box(0, 1e4, (1,), dtype=np.int32),
            "accumulated_rewards": Box(0, 1e4, (1,), dtype=np.float32),
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # choose_project: Discrete(n_projects+1), collaborate_with: MultiBinary(peer_group_size), put_effort: MultiBinary(n_projects)
        return Dict({
            "choose_project": Discrete(self.n_projects + 1),
            "collaborate_with": MultiBinary(self.peer_group_size),
            "put_effort": MultiBinary(self.n_projects),
        }) 