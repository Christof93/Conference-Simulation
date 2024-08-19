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
        self.agent_to_strategy = {}
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

    def get_median_effort(self, agent=None):
        if agent is None:
            all_efforts = []
            for agent in self.actions_taken:
                all_efforts += self.author_efforts[agent]
            return np.median(all_efforts)
        return np.median(self.author_efforts[agent])

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

    def get_percentage_of_strategy(self, strategy=None, agents=None):
        if agents is None:
            strats = self.agent_to_strategy.values()
        else:
            strats = [self.agent_to_strategy[a] for a in self.agent_to_strategy if a in agents]
        count_strats = Counter(strats)
        all_strats = len(strats)
        percentage_strats = {
            strat: count / all_strats for strat, count in count_strats.items()
        }
        if strategy is None:
            return percentage_strats
        else:
            return percentage_strats[strategy]

    def get_papers_per_conference(self):
        rates = {}
        for node in self.record_env.network_nodes.values():
            if node["_type"][0] == "Conference":
                if "n_submissions" in node:
                    rates[node["name"]] = (
                        node["rank"],
                        node["n_submissions"],
                        node["accepted"],
                    )
                else:
                    rates[node["name"]] = (
                        int(self.record_env.conferences[node["index"]]),
                        int(self.record_env.submission_counter[node["index"], 0]),
                        int(self.record_env.submission_counter[node["index"], 1]),
                    )
        return rates

    def report(self):
        # print(self.record_env.network)
        print(
            f"simulation with {self.record_env.n_authors} authors went on for {self.nr_steps} steps."
        )
        print(f"{len(self.record_env.agents)} agents remain.")
        print(f"number of papers started (including unsubmitted papers): {self.get_started_paper_count()}")
        print(f"number of papers submitted: {self.get_submitted_paper_count()}")
        print(f"number of papers accepted: {self.get_accepted_paper_count()}")
        print(f"average number of coauthors per author (including unsubmitted papers): {self.get_avg_coauthors():.2f}")
        print(f"Agent strategies:")
        print(f"  - before: {', '.join([f'{s}: {c}' for s,c in self.get_percentage_of_strategy().items()])}")
        print(f"  - after: {', '.join([f'{s}: {c:.2f}' for s,c in self.get_percentage_of_strategy(agents=self.record_env.agents).items()])}")
        print(
            f"mean reputation increase: {np.mean(self.record_env.reputations-self.record_env.initial_reputation):.2f}"
        )
        print(f"mean effort put into papers (including unsubmitted papers): {self.get_mean_effort():.2f}")
        print(f"median effort put into papers (including unsubmitted papers): {self.get_median_effort():.2f}")
        print(f"conference submissions, publications and acceptance rates: ")
        for conference, (
            rank,
            submitted,
            accepted,
        ) in self.get_papers_per_conference().items():
            print(
                " - {} (reputation: {}): {:>4}/{:<4} ({})".format(
                    conference,
                    rank,
                    accepted,
                    submitted,
                    (f"{accepted/submitted:.2f}" if submitted > 0 else "-"),
                )
            )


class NetworkEvaluator:
    def __init__(self, network):
        self.record_env = network
        self.nr_steps = network["steps"]
        self.papers = {}
        self.authors = {}
        self.conferences = {}
        self.remaining_agents = network["remaining_agents"]
        for node in self.record_env["nodes"]:
            if node["_type"][0] == "Paper":
                self.papers[node["id"]] = node
            elif node["_type"][0] == "Author":
                self.authors[node["id"]] = node
            elif node["_type"][0] == "Conference":
                self.conferences[node["id"]] = node
        self.agent_to_strategy = network["agent_strategy"]

    def get_mean_effort(self, agent=None):
        if agent is None:
            return np.mean([paper["effort"] for paper in self.papers.values()])
        effort = []
        for paper in self.papers.values():
            if agent in paper["effort_distribution"]:
                effort.append(paper["effort_distribution"][agent])
        return np.mean(effort)

    def get_median_effort(self, agent=None):
        if agent is None:
            return np.median([paper["effort"] for paper in self.papers.values()])
        effort = []
        for paper in self.papers.values():
            if agent in paper["effort_distribution"]:
                effort.append(paper["effort_distribution"][agent])
        return np.median(effort)

    def get_submitted_paper_count(self, agent=None):
        if agent is None:
            return len(self.papers)
        else:
            return len(
                [1 for p in self.papers.values() if agent in p["effort_distribution"]]
            )

    def get_accepted_paper_count(self, agent=None):
        if agent is None:
            return len([1 for p in self.papers.values() if p["accepted"] == 1])
        else:
            return len(
                [
                    1
                    for p in self.papers.values()
                    if agent in p["effort_distribution" and p["accepted"] == 1]
                ]
            )

    def get_avg_coauthors(self, agent=None):
        if agent is None:
            avg = np.mean([len(p["effort_distribution"]) for p in self.papers.values()])
        else:
            avg = np.mean(
                [
                    len(p["effort_distribution"])
                    for p in self.papers.values()
                    if agent in p["effort_distribution"]
                ]
            )

        return avg

    def get_percentage_of_strategy(self, strategy=None, agents=None):
        if agents is None:
            strats = self.agent_to_strategy.values()
        else:
            strats = [self.agent_to_strategy[a] for a in self.agent_to_strategy if a in agents]
        count_strats = Counter(strats)
        all_strats = len(strats)
        percentage_strats = {
            strat: count / all_strats for strat, count in count_strats.items()
        }
        if strategy is None:
            return percentage_strats
        else:
            return percentage_strats[strategy]

    def get_papers_per_conference(self):
        rates = {}
        for conf in self.conferences.values():
            if "n_submissions" in conf:
                rates[conf["name"]] = (
                    conf["rank"],
                    conf["n_submissions"],
                    conf["accepted"],
                )
        return rates

    def report(self):
        # print(json.dumps(self.record_env["nodes"], indent=2))
        print(f"\nAnalysis from network file\n{'-'*20}")
        print(
            f"simulation with {len(self.authors)} authors went on for {self.nr_steps} steps."
        )
        print(f"{len(self.remaining_agents)} agents remain.")
        print(f"number of papers submitted: {self.get_submitted_paper_count()}")
        print(f"number of papers accepted: {self.get_accepted_paper_count()}")
        print(f"average number of coauthors per author: {self.get_avg_coauthors():.2f}")
        print(f"Agent strategies:")
        print(f"  - before: {', '.join([f'{s}: {c}' for s,c in self.get_percentage_of_strategy().items()])}")
        print(f"  - after: {', '.join([f'{s}: {c:.2f}' for s,c in self.get_percentage_of_strategy(agents=self.remaining_agents).items()])}")
        print(
            f"mean reputation increase: {np.mean([a['reputation'] - self.record_env['initial_reputation'] for a in self.authors.values()])}"
        )
        print(f"mean effort put into submitted papers: {self.get_mean_effort():.2f}")
        print(f"median effort put into submitted papers: {self.get_median_effort():.2f}")
        print(f"conference submissions, publications and acceptance rates: ")
        for conference, (
            rank,
            submitted,
            accepted,
        ) in self.get_papers_per_conference().items():
            print(
                f" - {conference} (reputation: {rank}): {accepted:>4}/{submitted:<4} ({accepted/submitted:.2f})"
            )
        # print(self.record_env.network_nodes)
