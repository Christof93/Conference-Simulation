import functools
import random
from copy import copy

import numpy as np
from gymnasium.spaces import Discrete, Box, Dict, MultiDiscrete

from pettingzoo import ParallelEnv


class ReputationEnvironment(ParallelEnv):
    """The metadata holds environment constants.

    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {
        "name": "reputation_environment_v0",
    }

    def __init__(self, n_authors=2, n_conferences=1):
        """The init method takes in environment arguments.

        Should define the following attributes:
        - escape x and y coordinates
        - guard x and y coordinates
        - prisoner x and y coordinates
        - timestamp
        - possible_agents

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        self.timestep = None
        self.n_authors = n_authors
        self.n_conferences = n_conferences
        self.possible_agents = []
        self.agent_to_id = {}
        self.max_papers = 5
        self.max_authors = self.n_authors
        self.observations = {}
        self.action_masks = {}
        self.rewards = {}
        self.conferences = np.array((self.n_conferences,), dtype=np.int32)
        for i in range(self.n_authors):
            agent = f"author_{i}"
            self.possible_agents.append(agent)
            self.agent_to_id[agent] = i
            
    def _get_action_mask(self, agent, observation=None):
        action_mask = None
        previous_mask = self.action_masks[agent]
        # construct new mask based on observed state of environment
        action_mask = previous_mask
        
        return action_mask
    
    def _observe_after_action(self, agent, action=None):
        observation = self.observations[agent]
        reward = 0
        # make a new observation after applying the changes induced by action
        
        ## check the submission action
        submitted_paper = action["submit"]["id"]
        submit_to = action["submit"]["conference"]

        if submitted_paper > 0:
            observation["papers"]["authors"][submitted_paper:,] = 0
            effort = observation["papers"]["effort"][submitted_paper]
            conference = self.conferences[submit_to]
            reward = self._reward_paper(effort, conference)
        
        ## check the new paper action
        if action["start_with_coauthors"] > 0:
            started = False
            for i, (assigned, wanted, _) in enumerate(observation["papers"]["authors"]):
                ## other author(s) looking for same amount of coauthors
                if assigned < wanted and wanted == action["start_with_coauthors"]:
                    observation["papers"]["authors"][i, 0] += 1  
                    started = True
                    print("assigned to paper: ", observation)
                    break
            if not started:
                for i, (assigned, wanted, _) in enumerate(observation["papers"]["authors"]):
                ## other author(s) looking for same amount of coauthors
                    if wanted == 0:
                        observation["papers"]["authors"][i, 1] = action["start_with_coauthors"]  
                        started = True
                        print("started new paper: ", observation)
                        break
        
        ## contribute action
        observation["papers"]["effort"][action["contribute"]] += 1

        return observation, reward

    def _reward_paper(self, effort, conference):
        potential_reward = self.conferences[conference]/10
        ## accept?
        if sigmoid(effort - potential_reward) + np.random.normal(0, 0.05) > 0.5:
            return potential_reward
        else:
            return 0
    

    def reset(self, seed=None, options=None):
        """Reset set the environment to a starting point.

        It needs to initialize the following attributes:
        - agents
        - timestamp
        - observation
        - infos

        And must set up the environment so that render(), step(), and observe() can be called without issues.
        """
        self.agents = copy(self.possible_agents)
        self.timestep = 0
        self.conferences = np.floor(np.random.normal(50, 25, self.n_conferences))
        self.reputations = np.full((self.n_authors,),20, dtype=np.int32)
        observations = {}
        for agent in self.agents:
            print(self.action_space(agent))
            self.observations[agent] = {
                'spendable_tokens': self.conferences,
                'agent_reputation': self.reputations[self.agent_to_id[agent]],
                'papers': {
                    "effort": np.zeros((self.max_papers,), dtype=np.int32),
                    "authors": np.zeros((self.max_papers * self.n_authors, 3), dtype=np.int32),
                }
            }
            self.action_masks[agent] = {
                "start_with_coauthors": np.ones((self.max_authors + 1,), dtype=np.int8), # start a paper,
                "submit": { # submit a paper,
                    "id": np.zeros((self.n_authors * self.max_papers,), dtype=np.int8),
                    "conference": np.zeros((self.n_conferences + 1,), dtype=np.int8),
                },
                "contribute": np.zeros((self.n_authors * self.max_papers,), dtype=np.int8) # contribute to paper
            }

            observations[agent] = {
                "observation": self.observations[agent], 
                "action_mask": self.action_masks[agent],
            }

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}
        print(observations)
        return observations, infos

    def step(self, actions):
        """Takes in an action for the current agent (specified by agent_selection).

        Needs to update:
        - prisoner x and y coordinates
        - guard x and y coordinates
        - terminations
        - truncations
        - rewards
        - timestamp
        - infos

        And any internal state used by observe() or render()
        """
        # Execute actions

        observations = {}
        # Get dummy infos (not used in this example)
        infos = {}
        # Check termination conditions
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        rewards = {}
        for agent, action in actions.items():
            print(agent)
            print(action)
            # Check truncation conditions (overwrites termination conditions)
            
            # Get observations
            rewards[agent], observation = self._observe_after_action(agent, action)
            # Generate action masks
            action_mask = self._get_action_mask(agent, observation)
            observations[agent] = {
                "observation": observation,
                "action_mask": action_mask,
            }
            infos[agent] = {}
            self.timestep += 1

        return observations, rewards, terminations, truncations, infos

    def render(self):
        """Renders the environment."""

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return Dict({
            'spendable_tokens': Box(
                low=0,
                high=1_000_000,
                shape=(self.n_conferences,),
                dtype=np.int32
            ),
            'agent_reputation': Discrete(1000),  # Example max reputation tokens
            'papers': {
                "effort": Box(
                    low=0,
                    high=12,
                    shape=(self.n_authors * self.max_papers, ),
                    dtype=np.int32
                ),
                "authors": Box(
                    low=0, 
                    high=10, 
                    shape=(self.max_papers * self.n_authors, 3), # assigned authors, wanted authors, finished authors
                    dtype=np.int32
                ),
            }
        })

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Dict({
            "start_with_coauthors": Discrete(self.max_authors + 1), # start a paper,
            "submit": Dict({ # submit a paper,
                "id": Discrete(self.n_authors * self.max_papers),
                "conference": Discrete(self.n_conferences + 1),
            }),
            "contribute": Discrete(self.n_authors * self.max_papers) # contribute to paper
        })
    

def sigmoid(x, alpha=1, x_shift = 10):
    return 1 / (1 + np.exp(-alpha * x + x_shift))