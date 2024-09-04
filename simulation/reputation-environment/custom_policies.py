import numpy as np

def simple_policy(agent, environment):
    new_mask = environment.action_masks[agent]
    submittable = new_mask["submit"]["id"][1:]
    # only submit if total effort higher than 40
    worthy_papers = (environment.global_observation["papers"]["total_effort"] > 40) & submittable
    new_mask["submit"]["id"][1:] = worthy_papers.astype(np.int8)
    new_mask["start_with_coauthors"][6:] = 0
    # always contribute if possible
    new_mask["contribute"][0] = 0
    environment.action_masks[agent] = new_mask
    return environment.action_space(agent).sample(mask=new_mask)

def diligent_policy(agent, environment):
    new_mask = environment.action_masks[agent]
    submittable = new_mask["submit"]["id"][1:]
    # only submit if total effort higher than 40
    worthy_papers = (environment.global_observation["papers"]["total_effort"] > 50) & submittable
    new_mask["submit"]["id"][1:] = worthy_papers.astype(np.int8)
    new_mask["start_with_coauthors"][6:] = 0
    # always contribute if possible
    new_mask["contribute"][0] = 0
    environment.action_masks[agent] = new_mask
    return environment.action_space(agent).sample(mask=new_mask)

def picky_policy(agent, environment):
    new_mask = environment.action_masks[agent]
    submittable = new_mask["submit"]["id"][1:]
    # only collaborate with high rep agents
    mean_author_rep_p_paper = np.zeros(new_mask["collaborate"].shape[0]-1)
    assigned = environment.global_observation["papers"]["authors"]["assigned"]
    wanted = environment.global_observation["papers"]["authors"]["wanted"]
    for author, assigned_papers in environment.author_to_paper.items():
        author_i = environment.agent_to_id[author]
        author_rep = environment.observation["author_reputation"][author_i]
        mean_author_rep_p_paper += (assigned_papers*author_rep)/assigned

    ## look up authors of papers with open positions and and only set to true if certain threshold mean reputation is surpassed
    new_mask["collaborate"][:1] = (mean_author_rep_p_paper > 5) & (wanted-assigned>0)
    # only submit if total effort higher than 40
    worthy_papers = (environment.global_observation["papers"]["total_effort"] > 50) & submittable
    new_mask["submit"]["id"][1:] = worthy_papers.astype(np.int8)
    new_mask["submit"]["conference"] = (environment.global_observation["venue_reputation"]==500).astype(np.int8)
    new_mask["start_with_coauthors"][6:] = 0
    # always contribute if possible
    new_mask["contribute"][0] = 0
    environment.action_masks[agent] = new_mask
    return environment.action_space(agent).sample(mask=new_mask)

def malicious_policy(agent, environment):
    new_mask = environment.action_masks[agent]
    submittable = new_mask["submit"]["id"][1:]
    # submit if total effort higher than 20
    worthy_papers = (environment.global_observation["papers"]["total_effort"] > 20) & submittable
    new_mask["submit"]["id"][1:] = worthy_papers.astype(np.int8)
    new_mask["submit"]["conference"] = (environment.global_observation["venue_reputation"]==400).astype(np.int8)
    # always contribute if possible
    new_mask["contribute"][0] = 0
    environment.action_masks[agent] = new_mask
    return environment.action_space(agent).sample(mask=new_mask)

def random_policy(agent, environment):
    environment.action_space(agent).sample(mask = environment.action_masks[agent])