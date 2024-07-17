import functools
import json
import shortuuid
import numpy as np

from enum import Enum
from copy import copy

import gymnasium
from gymnasium.spaces import Discrete, Box, Dict
from pettingzoo import ParallelEnv


class ReputationEnvironment(ParallelEnv):
    """The metadata holds environment constants.

    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {
        "name": "reputation_environment_v0",
        "render_modes": ["all", "observation", "network"],
    }

    def __init__(
        self,
        n_authors=2,
        n_conferences=1,
        max_concurrent_papers=5,
        reward_scheme="CONVENTIONAL",
        max_coauthors=20,
        max_submissions_per_conference=50,
        render_mode=None,
        max_rewardless_steps=24,
    ):
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
        self.initial_reputation = 0
        self.possible_agents = []
        self.max_concurrent_papers = max_concurrent_papers
        self.n_possible_papers = self.max_concurrent_papers * self.n_authors
        self.max_coauthors = max_coauthors
        self.max_submissions_per_conference = max_submissions_per_conference
        self.observations = {}
        self.global_observation = {}
        self.actions = {}
        self.action_masks = {}
        self.rewards = {}
        self.conferences = np.array((self.n_conferences,), dtype=np.int32)
        self.submission_counter = np.array((self.n_conferences, 2), dtype=np.int64)
        self.paper_to_conference = np.array((self.n_possible_papers,))
        self.agent_to_id = {}
        self.author_to_paper = {}
        self.reward_schemes = Enum("reward_scheme", ["CONVENTIONAL", "TOKENS"])
        self.reward_scheme = self.reward_schemes[reward_scheme]
        self.agent_uuids = []
        self.conference_uuids = []
        self.paper_uuids = [None for _ in range(self.n_possible_papers)]
        self.conference_repetitions = np.array((self.n_conferences,), dtype=np.int32)
        self.network_nodes = {}
        self.network_links = []
        self.on_step = lambda x: None
        self.truncation_threshold = max_rewardless_steps

        for i in range(self.n_authors):
            agent = f"author_{i}"
            self.possible_agents.append(agent)
            self.agent_to_id[agent] = i
            self.author_to_paper[agent] = np.array(
                [False for _ in range(self.n_possible_papers)]
            )

    def _add_network_node(self, name, type, **kwargs):
        node = {
            "id": shortuuid.uuid(),
            "name": name,
            "_type": [type],
            "date": self.timestep,
        }
        node.update(kwargs)
        self.network_nodes[node["id"]] = node
        return node["id"]

    def _add_network_link(self, type, source, target, **kwargs):
        link = {"source": source, "target": target, "_type": [type]}
        link.update(kwargs)
        self.network_links.append(link)

    def _create_paper_connections(self, paper_i, conference, reward):
        ## create paper node
        new_paper_id = self._add_network_node(
            f"Paper_{paper_i}_{self.timestep}",
            "Paper",
            effort_distribution={},
            effort=int(self.global_observation["papers"]["total_effort"][paper_i]),
            reward=float(reward),
            accepted=(1 if reward > 0 else 0),
        )
        self.paper_uuids[paper_i] = new_paper_id

        ## update conference node
        n_submissions = self.submission_counter[conference, 0]
        n_accepted = self.submission_counter[conference, 1]
        self.network_nodes[self.conference_uuids[conference]]["n_submissions"] = int(
            n_submissions
        )
        self.network_nodes[self.conference_uuids[conference]]["accepted"] = int(
            n_accepted
        )

        ## create link
        self._add_network_link(
            new_paper_id,
            self.conference_uuids[conference],
            "_IS_SUBMITTED_TO",
        )
        return new_paper_id

    def _format_global_observation(self):
        observation = self.global_observation
        spendable_observation = "\n - conference reputation:\n  - {0}".format(
            "\n  - ".join(
                [
                    f"conference_{i}: {s}"
                    for i, s in enumerate(observation["spendable_tokens"])
                ]
            )
        )

        assignments = [
            "paper_{:<5}-> {:>2}/{:<2} 🧑‍🎓 {:>3} 💪 ({})".format(
                i, assigned, wanted, effort, ("✅" if finished else "⬜️")
            )
            for i, (assigned, wanted, effort, finished) in enumerate(
                zip(
                    observation["papers"]["authors"]["assigned"],
                    observation["papers"]["authors"]["wanted"],
                    observation["papers"]["total_effort"],
                    observation["papers"]["authors"]["finished"],
                )
            )
            if wanted > 0
        ]

        assignments_observation = (
            "\n - papers assigned, wanted, effort, finished:\n  - {0}".format(
                "\n  - ".join(assignments)
            )
        )

        formatted_observation = (
            f"\nglobal state: {spendable_observation}\n{assignments_observation}"
        )

        return formatted_observation

    def _format_observation(self, observation):
        rep_observation = f"\n  - current reputation: {observation['agent_reputation']}"
        efforts = [
            f"spent {effort}/{tot_effort} effort on paper_{i}"
            for i, (effort, tot_effort) in enumerate(
                zip(
                    observation["papers"]["effort"],
                    observation["papers"]["total_effort"],
                )
            )
            if effort > 0
        ]
        if len(efforts) > 0:
            papers_observation = "\n  - effort spent:\n   - {0}".format(
                "\n   - ".join(efforts)
            )
        else:
            papers_observation = "\n  - no effort spent"

        formatted_observation = (
            f" - observations: {rep_observation}{papers_observation}\n"
        )
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

        formatted_action = (
            f" - actions: {contribute_action}{start_action}{submit_action}\n"
        )

        return formatted_action

    def _format_action_mask(self, mask):
        formatted_mask = " - action mask: "
        formatted_mask += f'\n  - Is allowed to start papers: {np.all(mask["start_with_coauthors"]==1)}'
        formatted_mask += f'\n  - can submit papers {", ". join([str(i)  for i in np.where(mask["submit"]["id"][1:]==1)[0] - 1])}'
        formatted_mask += f'\n  - can contribute to papers {", ". join([str(i)  for i in np.where(mask["contribute"][1:]==1)[0] - 1])}'
        return formatted_mask

    def _get_action_mask(self, agent, observation=None):
        action_mask = self.action_masks[agent]
        # construct new mask based on observed state of environment

        paper_is_assigned = self.author_to_paper[agent]
        ## not more than max number of papers can be started.
        if sum(paper_is_assigned) >= self.max_concurrent_papers:
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
        action_mask["contribute"][1:] = (paper_is_assigned) & (~finished)
        ## papers where all wanted are assigned can be submitted
        action_mask["submit"]["id"][1:] = (
            (paper_is_assigned) & (assigned == wanted) & (~finished)
        )
        return action_mask

    def _observe_after_action(self, agent, action=None):
        observation = self.observations[agent]
        # make a new observation after applying the changes induced by action
        observation["author_reputation"] = self.reputations[self.agent_to_id[agent]]
        ## submission action
        submitted_paper = action["submit"]["id"] - 1
        if submitted_paper >= 0:
            self.global_observation["papers"]["authors"]["finished"][
                submitted_paper
            ] = True
            self.paper_to_conference[submitted_paper] = action["submit"]["conference"]

        ## new paper action (number of coauthors to start a paper)
        if action["start_with_coauthors"] > 0:
            started = False
            author_papers = list(
                zip(
                    self.global_observation["papers"]["authors"]["assigned"],
                    self.global_observation["papers"]["authors"]["wanted"],
                    self.global_observation["papers"]["authors"]["finished"],
                )
            )

            ## first loop: find other people to collaborate with
            for i, (assigned, wanted, _) in enumerate(author_papers):
                ## other author(s) looking for same amount of coauthors
                if (
                    assigned < wanted
                    and wanted == action["start_with_coauthors"]
                    and not self.author_to_paper[agent][i]
                ):
                    self.global_observation["papers"]["authors"]["assigned"][i] += 1
                    self.author_to_paper[agent][i] = True
                    started = True
                    break

            ## second loop: new entry in the paper list potentially open for collaborators
            if not started:
                for i, (_, wanted, finished) in enumerate(author_papers):
                    ## start looking for coauthors for new paper
                    if wanted == 0 and not finished:
                        self.global_observation["papers"]["authors"]["wanted"][i] = (
                            action["start_with_coauthors"]
                        )
                        self.global_observation["papers"]["authors"]["assigned"][i] = 1
                        self.author_to_paper[agent][i] = True
                        started = True
                        break

        ## contribute action
        if action["contribute"] > 0:
            paper_number = action["contribute"] - 1
            observation = self._work_on(observation, paper_number)

        observation["papers"]["total_effort"] = self.global_observation["papers"][
            "total_effort"
        ]
        observation["papers"]["authors"] = self.global_observation["papers"]["authors"]
        observation["spendable_tokens"] = self.global_observation["spendable_tokens"]
        self.observations[agent] = observation
        return observation

    def _work_on(self, observation, paper_number):
        observation["papers"]["effort"][paper_number] += 1
        self.global_observation["papers"]["total_effort"][paper_number] += 1
        return observation

    def _reward_agents(self):
        finished = self.global_observation["papers"]["authors"]["finished"]
        self.rewards = {a: 0 for a in self.agents}
        if not np.any(finished):
            return self.rewards
        finished_effort = self.global_observation["papers"]["total_effort"][finished]
        assigned_to_finished = self.global_observation["papers"]["authors"]["assigned"][
            finished
        ]
        paper_rewards = {}
        for paper_i, effort, n_coauthors in zip(
            np.nonzero(finished)[0], finished_effort, assigned_to_finished
        ):
            conference = self.paper_to_conference[paper_i]
            self.submission_counter[conference, 0] += 1

            if self.reward_scheme is self.reward_schemes.CONVENTIONAL:
                reward = self._conventional_reward(effort, conference)
            else:
                reward = 0

            paper_rewards[paper_i] = reward / (n_coauthors + 1)

            if reward > 0:
                self.submission_counter[conference, 1] += 1

            if self.render_mode == "network":
                new_paper_id = self._create_paper_connections(
                    paper_i, conference, reward
                )

        for agent in self.agents:
            finished_agent_papers = np.nonzero(self.author_to_paper[agent] & finished)[
                0
            ]

            for fp_i in finished_agent_papers:
                author_effort = int(self.observations[agent]["papers"]["effort"][fp_i])
                self.rewards[agent] += paper_rewards[fp_i]

                if self.render_mode == "network":
                    self.network_nodes[self.paper_uuids[fp_i]]["effort_distribution"][
                        agent
                    ] = int(author_effort)
                    self._add_network_link(
                        new_paper_id,
                        self.agent_uuids[self.agent_to_id[agent]],
                        "_HAS_AUTHOR",
                    )

            self.reputations[self.agent_to_id[agent]] += self.rewards[agent]
            if self.rewards[agent] > 0:
                self.rewardless_steps[self.agent_to_id[agent]] = 0

            if self.render_mode == "network":
                self.network_nodes[self.agent_uuids[self.agent_to_id[agent]]][
                    "reputation"
                ] = int(self.reputations[self.agent_to_id[agent]])

        return self.rewards

    def _close_conference(self, conference_nr):
        self.submission_counter[conference_nr, :] = 0
        self._restock_conference_rewards(conference_nr)

    def _restock_conference_rewards(self, index=None):
        if index is None:
            # self.conferences = np.abs(np.floor(np.random.normal(40, 15, self.n_conferences)))
            self.conferences = np.random.choice([200, 400, 800], self.n_conferences)
            self.conference_repetitions = np.zeros(
                (self.n_conferences,), dtype=np.int32
            )
            for i, rating in enumerate(self.conferences):
                conference_id = self._add_network_node(
                    f"conference_{i}_{self.conference_repetitions[i]}",
                    "Conference",
                    year=int(self.conference_repetitions[i]),
                    rank=int(rating),
                    index=i,
                )
                self.conference_uuids.append(conference_id)
        else:
            # self.conferences[index] = np.abs(np.floor(np.random.normal(100, 25)))
            self.conferences[index] = np.random.choice([200, 400, 800])
            self.conference_repetitions[index] += 1
            conference_id = self._add_network_node(
                f"conference_{index}_{self.conference_repetitions[index]}",
                "Conference",
                rank=int(self.conferences[index]),
                year=int(self.conference_repetitions[index]),
                index=index,
            )
            self.conference_uuids[index] = conference_id

    def _conventional_reward(self, effort, conference, arbitrariness=0.3):
        ## how many papers were submitted? How are the tokens split?
        acceptance_threshold = (
            self.conferences[conference] / self.max_submissions_per_conference
        )
        potential_reward = acceptance_threshold
        ## accept?
        paper_rating = sigmoid(
            effort, alpha=0.18, x_shift=acceptance_threshold
        ) + np.random.normal(0, arbitrariness)
        if paper_rating > 0.5:
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
            self.author_to_paper[agent][finished] = False
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
        elif self.render_mode == "all":
            for agent in self.agents:
                print(f"\n{agent} (latest reward: {self.rewards[agent]}): ")
                print(self._format_action(self.actions[agent]))
                print(self._format_observation(self.observations[agent]))
                print(self._format_action_mask(self.action_masks[agent]))
            print(self._format_global_observation())
            print()
        elif self.render_mode == "observation":
            print(self._format_global_observation())
            reputations = "\n - agent reputation, submitted papers, effort:"
            for i, rep in enumerate(self.reputations):
                increase = rep - self.initial_reputation
                reputations += f"\n  - {self.agents[i]}: {rep:>4} ({('+' if increase>=0 else '-')}{increase})"
            print(reputations)
            print()
        elif self.render_mode == "network":
            print(json.dumps(self.network_nodes, indent=2))
        else:
            gymnasium.logger.warn(
                "Render mode not supported. No outputs will be rendered."
            )
            return

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
        self.reputations = np.full(
            (self.n_authors,), self.initial_reputation, dtype=np.int32
        )
        self.rewardless_steps = np.zeros((self.n_authors,), dtype=np.int16)
        self.submission_counter = np.zeros((self.n_conferences, 2), dtype=np.int64)
        self._restock_conference_rewards()
        self.paper_to_conference = np.array([-1 for _ in range(self.n_possible_papers)])
        observations = {}
        for agent in self.agents:
            self.rewards[agent] = 0
            self.author_to_paper[agent] = np.array(
                [False for _ in range(self.n_possible_papers)]
            )
            self.observations[agent] = {
                "spendable_tokens": self.conferences,
                "agent_reputation": self.reputations[self.agent_to_id[agent]],
                "papers": {
                    "effort": np.zeros((self.n_possible_papers,), dtype=np.int32),
                    "total_effort": np.zeros((self.n_possible_papers,), dtype=np.int32),
                    "authors": {
                        "assigned": np.zeros(
                            (self.n_possible_papers,),
                            dtype=np.int8,
                        ),
                        "wanted": np.zeros(
                            (self.n_possible_papers,),
                            dtype=np.int8,
                        ),
                        "finished": np.zeros(
                            (self.n_possible_papers,),
                            dtype=np.int8,
                        )
                        > 0,  # all False
                    },
                },
            }
            self.action_masks[agent] = {
                "start_with_coauthors": np.ones(
                    (self.max_coauthors + 1,), dtype=np.int8
                ),  # start a paper,
                "submit": {  # submit a paper, element 0 -> don't submit even if you could
                    "id": np.concatenate(
                        (
                            np.ones((1,), dtype=np.int8),
                            np.zeros(
                                (self.n_authors * self.max_concurrent_papers,),
                                dtype=np.int8,
                            ),
                        )
                    ),
                    "conference": np.ones((self.n_conferences,), dtype=np.int8),
                },
                "contribute": np.concatenate(
                    (
                        np.ones((1,), dtype=np.int8),
                        np.zeros(
                            (self.n_authors * self.max_concurrent_papers,),
                            dtype=np.int8,
                        ),
                    )
                ),  # contribute to paper
            }
            observations[agent] = {
                "observation": self.observations[agent],
                "action_mask": self.action_masks[agent],
            }
            agent_id = self._add_network_node(
                agent, "Author", truncated=False, reputation=self.initial_reputation
            )
            self.agent_uuids.append(agent_id)

        self.global_observation = {
            "papers": self.observations[agent]["papers"],
            "spendable_tokens": self.observations[agent]["spendable_tokens"],
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
        ## truncate if agent stayed rewardless after certain amount of time
        truncations = {
            a: self.rewardless_steps[self.agent_to_id[a]] >= self.truncation_threshold
            for a in self.agents
        }

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
            if self.render_mode == "network":
                self.network_nodes[self.agent_uuids[self.agent_to_id[agent]]][
                    "truncated"
                ] = truncations[agent]

        ## calculate rewards
        rewards = self._reward_agents()

        ## close conference if enough papers were submitted
        for i in np.nonzero(
            self.submission_counter[:, 1] >= self.max_submissions_per_conference
        )[0]:
            self._close_conference(i)

        self.on_step(self)
        self.agents = [a for a in truncations if not truncations[a]]
        self.timestep += 1
        self.rewardless_steps += 1
        return observations, rewards, terminations, truncations, infos

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return Dict(
            {
                "spendable_tokens": Box(
                    low=0, high=1_000_000, shape=(self.n_conferences,), dtype=np.int32
                ),
                "agent_reputation": Discrete(1000),  # Example max reputation tokens
                "papers": Dict(
                    {
                        "effort": Box(
                            low=0,
                            high=12,
                            shape=(self.n_authors * self.max_concurrent_papers,),
                            dtype=np.int32,
                        ),
                        "total_effort": Box(
                            low=0,
                            high=12,
                            shape=(self.n_authors * self.max_concurrent_papers,),
                            dtype=np.int32,
                        ),
                        "authors": Dict(
                            {
                                "assigned": Box(
                                    low=0,
                                    high=self.max_coauthors,
                                    shape=(self.n_possible_papers,),  # assigned authors
                                    dtype=np.int8,
                                ),
                                "wanted": Box(
                                    low=0,
                                    high=self.max_coauthors,
                                    shape=(self.n_possible_papers,),  # wanted authors
                                    dtype=np.int8,
                                ),
                                "finished": Box(
                                    low=0,
                                    high=self.max_coauthors,
                                    shape=(self.n_possible_papers,),  # finished authors
                                    dtype=bool,
                                ),
                            }
                        ),
                    }
                ),
            }
        )

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Dict(
            {
                "start_with_coauthors": Discrete(
                    self.max_coauthors + 1
                ),  # start a paper,
                "submit": Dict(
                    {  # submit a paper,
                        "id": Discrete(self.n_authors * self.max_concurrent_papers + 1),
                        "conference": Discrete(self.n_conferences),
                    }
                ),
                "contribute": Discrete(
                    self.n_authors * self.max_concurrent_papers + 1
                ),  # contribute to paper
            }
        )


def sigmoid(x, alpha=1, x_shift=10):
    return 1 / (1 + np.exp(-alpha * x + x_shift))
