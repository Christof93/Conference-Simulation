def contributor_reward(contributor_index, beta, epsilon, total_spendable, collaterals):
    collateral = collaterals[contributor_index]
    return (beta * total_spendable) / len(collaterals) + (
        (1 - beta) * total_spendable * (collateral + epsilon)
    ) / (epsilon * len(collaterals) + sum(collaterals))


def distribute_tokens(b, alpha, burned_collateral_sum):
    return b + alpha * burned_collateral_sum


def _distribute_rewards_evenly(p, reward):
    return reward / len(p.contributors)


def _distribute_rewards_multiply(self, p, reward):
    return reward


def _distribute_rewards_by_effort(self, p, reward):
    max_effort = max(
        [self.agent_project_effort[c][p.project_id] for c in p.contributors]
    )
    for idx in p.contributors:
        effort = self.agent_project_effort[idx][p.project_id]
        rel_effort = effort / max_effort if max_effort > 0 else 1 / len(p.contributors)
        return reward * rel_effort


def _distribute_rewards_by_citations(self, p, reward):
    """
    Distribute rewards based on citation patterns found in the completed project.
    Agents who contributed to projects that are cited more frequently get higher rewards.
    """
    # Calculate citation-based weights for each contributor
    citation_weights = []

    for idx in p.contributors:
        # Get all projects this agent has contributed to
        agent_projects = self.agent_successful_projects[idx]

        # Calculate total citations received by this agent's projects
        total_citations = 0
        for project_id in agent_projects:
            if project_id in self.projects:
                project = self.projects[project_id]
                total_citations += len(project.cited_by)

        # Add base weight (to ensure non-zero rewards even for new agents)
        citation_weight = total_citations + 1
        citation_weights.append(citation_weight)

    # Normalize weights so they sum to 1
    total_weight = sum(citation_weights)
    if total_weight > 0:
        normalized_weights = [w / total_weight for w in citation_weights]
    else:
        # Fallback to equal distribution if no citations
        normalized_weights = [1.0 / len(p.contributors)] * len(p.contributors)

    # Distribute rewards based on citation weights
    for i, idx in enumerate(p.contributors):
        self._remove_active_project(idx, p.project_id)
        if reward > 0:
            self.agent_successful_projects[idx].append(p.project_id)

        # Distribute reward proportional to citation weight
        agent_reward = reward * normalized_weights[i]
        self.agent_rewards[idx, self.timestep] += agent_reward
        self.agent_completed_projects[idx] += 1
        self.rewards[f"agent_{idx}"] += agent_reward
