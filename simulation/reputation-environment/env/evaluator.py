import json
from collections import Counter, defaultdict

import numpy as np


class EnvironmentRecorder:
    def __init__(self, environment):
        self.record_env = environment
        self.actions_taken = {agent: {} for agent in self.record_env.possible_agents}
        self.record_env.on_step = lambda env: self.on_env_step(env)
        self.all_paper_count = {}
        self.nr_steps = 0
        self.author_efforts = {agent: [] for agent in self.record_env.possible_agents}

    def on_env_step(self, env):
        self.nr_steps += 1
        for agent, action in env.actions.items():
            paper_info = self.record_env.observations[agent]["papers"]
            finished = paper_info["authors"]["finished"]
            self.author_efforts[agent] += list(
                paper_info["effort"][finished][
                    np.nonzero(paper_info["effort"][finished])
                ]
            )
            if action["contribute"] > 0:
                self.actions_taken[agent]["contribute"] = (
                    self.actions_taken[agent].get("contribute", 0) + 1
                )

            n_coauthors = int(action["start_with_coauthors"]) - 1
            if n_coauthors >= 0:
                self.actions_taken[agent]["start_with_coauthors"] = self.actions_taken[
                    agent
                ].get("start_with_coauthors", {})

                self.actions_taken[agent]["start_with_coauthors"][n_coauthors] = (
                    self.actions_taken[agent]["start_with_coauthors"].get(
                        n_coauthors, 0
                    )
                    + 1
                )

            if action["submit"]["id"] > 0:
                self.actions_taken[agent]["submit"] = (
                    self.actions_taken[agent].get("submit", 0) + 1
                )
                self.actions_taken[agent]["submit_to"] = self.actions_taken[agent].get(
                    "submit_to", {}
                )
                conf = int(action["submit"]["conference"])
                self.actions_taken[agent]["submit_to"][conf] = (
                    self.actions_taken[agent]["submit_to"].get(conf, 0) + 1
                )

    def get_mean_effort(self, agent=None):
        if agent is None:
            all_efforts = []
            for agent in self.actions_taken:
                all_efforts += self.author_efforts[agent]
            return np.mean(all_efforts)
        return np.mean(self.author_efforts[agent])

    def get_started_paper_count(self, agent=None):
        if agent is None:
            return sum(
                [
                    sum(actions["start_with_coauthors"].values())
                    for actions in self.actions_taken.values()
                ]
            )
        return sum(self.actions_taken[agent]["start_with_coauthors"].values())

    def get_submitted_paper_count(self, agent=None):
        return np.sum(self.record_env.submission_counter[:, 0])

    def get_accepted_paper_count(self, agent=None):
        return np.sum(self.record_env.submission_counter[:, 1])

    def get_avg_coauthors(self, agent=None):
        if agent is None:
            nr_papers = self.get_started_paper_count()
            avg = (
                np.sum(
                    [
                        np.sum(
                            [
                                constellation * val
                                for constellation, val in actions[
                                    "start_with_coauthors"
                                ].items()
                            ]
                        )
                        for actions in self.actions_taken.values()
                    ]
                )
                / nr_papers
            )
        else:
            nr_papers = self.get_started_paper_count(agent)
            avg = (
                np.sum(
                    [
                        constellation * val
                        for constellation, val in self.actions_taken[agent][
                            "start_with_coauthors"
                        ].items()
                    ]
                )
                / nr_papers
            )
        return avg

    def get_percentage_of_strategy(self, strategy=None):
        if strategy is None:
            count_strats = Counter(self.agent_to_strategy.values())
            all_strats = len(self.agent_to_strategy)
            percentage_strats = {
                strat: count / all_strats for strat, count in count_strats.items()
            }
            return percentage_strats

    def report(self):
        # print(self.record_env.network)
        print(
            f"simulation with {self.record_env.n_authors} authors went on for {self.nr_steps} steps."
        )
        print(f"number of papers started: {self.get_started_paper_count()}")
        print(f"number of papers submitted: {self.get_submitted_paper_count()}")
        print(f"number of papers accepted: {self.get_accepted_paper_count()}")
        print(f"average number of coauthors per author: {self.get_avg_coauthors()}")
        print(
            f"percentage of strategies used: {json.dumps(self.get_percentage_of_strategy(), indent=2)}"
        )
        print(
            f"mean reputation increase {np.mean(self.record_env.reputations-self.record_env.initial_reputation)}"
        )
        print(f"mean effort put into submitted papers {self.get_mean_effort()}")
