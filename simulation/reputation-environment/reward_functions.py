

def contributor_reward(contributor_index, beta, epsilon, total_spendable, collaterals):
    collateral = collaterals[contributor_index]
    return (beta*total_spendable) / len(collaterals) + ((1-beta) * total_spendable * (collateral + epsilon)) / (epsilon * len(collaterals) + sum(collaterals)) 

def distribute_tokens(b, alpha, burned_collateral_sum):
    return b + alpha * burned_collateral_sum
