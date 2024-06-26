import functools
import gymnasium
import numpy as np

from enum import Enum
from copy import copy

from gymnasium.spaces import Discrete, Box, Dict, MultiDiscrete
from pettingzoo import ParallelEnv
from pettingzoo.utils import agent_selector


class ReputationEnvironment(ParallelEnv):
    """The metadata holds environment constants.

    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {
        "name": "reputation_environment_v0",
    }

    def __init__(self, n_authors=2, n_conferences=1, reward_scheme="CONVENTIONAL", render_mode=None):
        """The init method defines the following attributes:
        - timestep
        - possible_agents

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        self.render_mode = render_mode
        self.timestep = None
        self.n_authors = n_authors
        self.n_conferences = n_conferences
        self.possible_agents = []
        self.agent_to_id = {}
        self.max_papers = 5
        self.max_authors = self.n_authors
        self.observations = {}
        self.global_observation = {}
        self.actions = {}
        self.action_masks = {}
        self.rewards = {}
        self.conferences = np.array((self.n_conferences,), dtype=np.int32)
        self.paper_to_conference = np.array([-1 for _ in range(self.max_papers * self.n_authors)])
        self.reward_schemes = Enum('reward_scheme', ['CONVENTIONAL', 'TOKENS'])
        self.reward_scheme = self.reward_schemes[reward_scheme]

        for i in range(self.n_authors):
            agent = f"author_{i}"
            self.possible_agents.append(agent)
            self.agent_to_id[agent] = i
    
    def _format_global_observation(self):
        observation = self.global_observation
        spendable_observation = "\n - spendable tokens:\n  - {0}".format(
            '\n  - '.join([
                f'conference_{i}: {s}' 
                for i,s in enumerate(observation['spendable_tokens'])
            ])
        )
        assignments = [
            f'paper_{i}: {assigned}/{wanted} 🧑‍🎓 {effort} 💪 ({("✅" if finished else "⬜️")})' 
            for i, (assigned, wanted, effort, finished) in enumerate(
                zip(
                    observation['papers']['authors']['assigned'],
                    observation['papers']['authors']['wanted'],
                    observation['papers']['total_effort'],
                    observation['papers']['authors']['finished'],
                )
            )
        ]
        assignments_observation = "\n - papers assigned, wanted, effort, finished:\n  - {0}".format('\n  - '.join(assignments))

        formatted_observation = f"\nglobal state: {spendable_observation}{assignments_observation}"
        return formatted_observation

    def _format_observation(self, observation):
        rep_observation = f"\n  - current reputation: {observation['agent_reputation']}"
        efforts = [
            f'spent {effort}/{tot_effort} effort on paper_{i}' 
            for i, (effort, tot_effort) in enumerate(
                zip(
                    observation['papers']['effort'],
                    observation['papers']['total_effort']
                )
            )
            if effort > 0
        ]
        if len(efforts) > 0:
            papers_observation = "\n  - effort spent:\n   - {0}".format('\n   - '.join(efforts))
        else:
            papers_observation = "\n  - no effort spent"

        formatted_observation = f" - observations: {rep_observation}{papers_observation}"
        return formatted_observation

    def _format_action(self, action):
        contribute_action = ""
        start_action = ""
        submit_action = ""
        if action["contribute"] > 0:
            contribute_action = f"\n  - contributes to paper_{action['contribute']-1}."
        
        if action["start_with_coauthors"] == 1:
            start_action = f"\n  - starting paper without coauthors."
        elif action["start_with_coauthors"] > 1:
            start_action = f"\n  - looking for paper with {action['start_with_coauthors'] - 1} coauthor(s)."
        if not action["submit"]["id"] == 0:
            submit_action = f"\n  - submitting paper_{action['submit']['id'] - 1} to conference_{action['submit']['conference']}."
        
        formatted_action = f" - actions: {contribute_action}{start_action}{submit_action}\n"
        return formatted_action 

    def _format_action_mask(self, mask):
        return mask
    
    def _get_action_mask(self, agent, observation=None):
        action_mask = self.action_masks[agent]
        # construct new mask based on observed state of environment

        effort = observation["papers"]["effort"]
        ## not more than max number of papers can be started.
        if sum(effort > 0) >= self.max_papers:
            action_mask["start_with_coauthors"][:] = 0
        else:
            action_mask["start_with_coauthors"][:] = 1
        
        finished = observation["papers"]["authors"]["finished"]
        assigned = observation["papers"]["authors"]["assigned"]
        wanted = observation["papers"]["authors"]["wanted"]
        ## papers which are finished can not be worked on or submitted again
        ## the first action respresents choosing not to act.
        action_mask["contribute"][1:][finished] = 0
        action_mask["submit"]["id"][1:][finished] = 0
        ## papers which are assigned and not finished can be worked on
        action_mask["contribute"][1:] = (effort > 0) & (~ finished)
        ## papers where all wanted are assigned can be submitted
        action_mask["submit"]["id"][1:] = (effort > 0) & (assigned == wanted) & (~ finished)
        return action_mask
    
    def _observe_after_action(self, agent, action=None):
        observation = self.observations[agent]
        # make a new observation after applying the changes induced by action

        ## submission action
        submitted_paper = action["submit"]["id"] - 1
        if submitted_paper >= 0:
            self.global_observation["papers"]["authors"]["finished"][submitted_paper] = True
            self.paper_to_conference[submitted_paper] = action["submit"]["conference"]
        
        ## new paper action (number of coauthors to start a paper)
        if action["start_with_coauthors"] > 0:
            started = False
            author_papers = list(
                zip(
                    self.global_observation["papers"]["authors"]["assigned"],
                    self.global_observation["papers"]["authors"]["wanted"],
                    self.global_observation["papers"]["authors"]["finished"]
                )
            )

            ## first loop: find other people to collaborate with
            for i, (assigned, wanted, _) in enumerate(author_papers):
                ## other author(s) looking for same amount of coauthors
                if assigned < wanted and wanted == action["start_with_coauthors"] and observation["papers"]["effort"][i]==0:
                    self.global_observation["papers"]["authors"]["assigned"][i] += 1
                    observation = self._work_on(observation, i)            
                    started = True
                    break

            ## second loop: new entry in the paper list potentially open for collaborators
            if not started:
                for i, (_, wanted, finished) in enumerate(author_papers):
                    ## start looking for coauthors for new paper
                    if wanted == 0 and not finished:
                        self.global_observation["papers"]["authors"]["wanted"][i] = action["start_with_coauthors"]
                        self.global_observation["papers"]["authors"]["assigned"][i] = 1
                        observation = self._work_on(observation, i)
                        started = True
                        break
        
        ## contribute action
        if action["contribute"]>0:
            paper_number = action["contribute"] - 1
            observation = self._work_on(observation, paper_number)            

        observation["papers"]["total_effort"] = self.global_observation["papers"]["total_effort"]
        observation["papers"]["authors"] = self.global_observation["papers"]["authors"]
        observation["spendable_tokens"] = self.global_observation["spendable_tokens"]
        self.observations[agent] = observation 
        return observation

    def _work_on(self, observation, paper_number):
        observation["papers"]["effort"][paper_number] += 1
        self.global_observation["papers"]["total_effort"][paper_number] += 1
        return observation
    
    def _reward_paper(self, effort, conference):
        if self.reward_scheme is self.reward_schemes.CONVENTIONAL:
            return self._conventional_reward(effort, conference)

    def _restock_conference_rewards(self, index = None):
        if index is None:
            self.conferences = np.abs(np.floor(np.random.normal(100, 25, self.n_conferences)))
        else:
            self.conferences[index] = np.abs(np.floor(np.random.normal(100, 25)))

    def _conventional_reward(self, effort, conference):
        ## how many papers were submitted? How are the tokens split?
        potential_reward = self.conferences[conference] / 10
        ## accept?
        if sigmoid(effort - potential_reward) + np.random.normal(0, 0.05) > 0.5:
            return potential_reward
        else:
            return 0
    
    def _release_finished_paper_slots(self, actions):
        finished = self.global_observation["papers"]["authors"]["finished"]
        self.global_observation["papers"]["authors"]["assigned"][finished] = 0
        self.global_observation["papers"]["authors"]["wanted"][finished] = 0
        self.paper_to_conference[finished] = -1
        self.global_observation["papers"]["total_effort"][finished] = 0
        for agent in actions:
            self.observations[agent]["papers"]["effort"][finished] = 0
        self.global_observation["papers"]["authors"]["finished"][:] = False
        
    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if len(self.agents) > 0:
            for agent in self.agents:
                print(f"\n{agent} (latest reward: {self.rewards[agent]}): ")
                print(self._format_action(self.actions[agent]))
                print(self._format_observation(self.observations[agent]))
                print(self._format_action_mask(self.action_masks[agent]))
        else:
            print("Game over")
        print(self._format_global_observation())

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

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
        self.reputations = np.full((self.n_authors,), 20, dtype=np.int32)
        self._restock_conference_rewards()
        observations = {}
        for agent in self.agents:
            self.observations[agent] = {
                "spendable_tokens": self.conferences,
                "agent_reputation": self.reputations[self.agent_to_id[agent]],
                "papers": {
                    "effort": np.zeros((self.max_papers * self.n_authors,), dtype=np.int32),
                    "total_effort": np.zeros((self.max_papers * self.n_authors,), dtype=np.int32),
                    "authors": {
                        "assigned": np.zeros((self.max_papers * self.n_authors,), dtype=np.int8),
                        "wanted": np.zeros((self.max_papers * self.n_authors,), dtype=np.int8),
                        "finished": np.zeros((self.max_papers * self.n_authors,), dtype=np.int8) > 0, # all False
                    }
                }
            }
            self.action_masks[agent] = {
                "start_with_coauthors": np.ones((self.max_authors + 1,), dtype=np.int8), # start a paper,
                "submit": { # submit a paper, element 0 -> don't submit even if you could
                    "id": np.concatenate((np.ones((1,), dtype=np.int8), np.zeros((self.n_authors * self.max_papers,), dtype=np.int8))),
                    "conference": np.ones((self.n_conferences,), dtype=np.int8),
                },
                "contribute": np.concatenate((np.ones((1,), dtype=np.int8), np.zeros((self.n_authors * self.max_papers,), dtype=np.int8))) # contribute to paper
            }
            self.rewards[agent] = 0
            observations[agent] = {
                "observation": self.observations[agent], 
                "action_mask": self.action_masks[agent],
            }
    
        self.global_observation = {
            "papers": self.observations[agent]["papers"],
            "spendable_tokens": self.observations[agent]["spendable_tokens"]
        }
        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}
        return observations, infos

    def step(self, actions):
        """Takes in actions for the current agents

        Needs to update:
        - terminations
        - truncations
        - rewards
        - timestamp
        - infos

        And any internal state used by observe() or render()
        """
        ## free up finished papers
        self._release_finished_paper_slots(actions)

        # Execute actions
        self.actions = actions
        observations = {}
        # Get dummy infos (not used in this example)
        infos = {}
        # Check termination conditions
        terminations = {a: False for a in self.agents}
        # Check truncation conditions (overwrites termination conditions)
        truncations = {a: False for a in self.agents}
        for agent, action in actions.items():
            # Get observations
            observation = self._observe_after_action(agent, action)
            # Generate action masks
            self.action_masks[agent] = self._get_action_mask(agent, observation)
            observations[agent] = {
                "observation": observation,
                "action_mask": self.action_masks[agent],
            }
            infos[agent] = {}
            
        self.timestep += 1

        finished = self.global_observation["papers"]["authors"]["finished"]
        assigned_to_finished = self.global_observation["papers"]["authors"]["assigned"][finished]

        ## calculate rewards
        rewards = {a: 0 for a in self.agents}
        for agent in self.agents:
            paper_efforts = self.observations[agent]["papers"]["effort"][finished]
            ## collect rewards over all papers given to 
            for paper_i, effort, n_coauthors in zip(np.nonzero(finished)[0], paper_efforts, assigned_to_finished):
                conference = self.paper_to_conference[paper_i]
                reward = self._reward_paper(effort, conference)
                rewards[agent] += reward / (n_coauthors + 1)
            self.rewards[agent] = rewards[agent]
            self.reputations[self.agent_to_id[agent]] += rewards[agent]
        return observations, rewards, terminations, truncations, infos

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
            'papers': Dict({
                "effort": Box(
                    low=0,
                    high=12,
                    shape=(self.n_authors * self.max_papers,),
                    dtype=np.int32
                ),
                "total_effort": Box(
                    low=0,
                    high=12,
                    shape=(self.n_authors * self.max_papers,),
                    dtype=np.int32
                ),
                "authors": 
                    Dict({
                        "assigned": Box(
                            low=0, 
                            high=self.max_authors, 
                            shape=(self.max_papers * self.n_authors,), # assigned authors
                            dtype=np.int8
                        ),
                        "wanted": Box(
                            low=0, 
                            high=self.max_authors, 
                            shape=(self.max_papers * self.n_authors,), # wanted authors
                            dtype=np.int8
                        ),
                        "finished": Box(
                            low=0, 
                            high=self.max_authors, 
                            shape=(self.max_papers * self.n_authors,), # finished authors
                            dtype=bool
                        ),
                    })
            })
        })

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Dict({
            "start_with_coauthors": Discrete(self.max_authors + 1), # start a paper,
            "submit": Dict({ # submit a paper,
                "id": Discrete(self.n_authors * self.max_papers + 1),
                "conference": Discrete(self.n_conferences),
            }),
            "contribute": Discrete(self.n_authors * self.max_papers + 1) # contribute to paper
        })
    

def sigmoid(x, alpha=1, x_shift = 10):
    return 1 / (1 + np.exp(-alpha * x + x_shift))