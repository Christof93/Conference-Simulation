import json
from collections import defaultdict

import numpy as np
from neo4j import GraphDatabase
from run_policy_simulation import run_simulation_with_policies
from scipy.stats import wasserstein_distance
# from some_simulator import run_simulation  # your simulator function
from skopt import gp_minimize  # Bayesian optimization
from skopt.space import Categorical, Integer, Real

# --- Neo4j connection setup ---
URI = "bolt://localhost:7690"  # adjust if using neo4j+s:// for Aura
USER = "neo4j"
PASSWORD = "openreview"

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))


def get_collaborators_per_author_or():
    # --- Query for coauthors ---
    # This query collects coauthors for a given author
    COAUTHOR_QUERY = """
    MATCH (a:Author)<-[:_HAS_AUTHOR]-(p:Paper)-[:_HAS_AUTHOR]->(coauthor:Author),
        (p)-[:_IS_SUBMITTED_TO]->(c:Conference)
    WHERE coauthor <> a
    RETURN a.name AS author, collect(DISTINCT coauthor.name) AS coauthors, COLLECT(DISTINCT c.year) AS years, COUNT(p) as p_count
    """
    coauthor_dict = {}
    with driver.session(database="open-review-data") as session:
        results = session.run(COAUTHOR_QUERY)
        for record in results:
            author = record["author"]
            coauthor_dict[author] = record
    return coauthor_dict


def mean_yoy_growth_rate_or():
    QUERY = """
        MATCH (p:Paper)-[:_HAS_AUTHOR]->(a:Author),
            (p)-[:_IS_SUBMITTED_TO]->(c:Conference {name: "ICLR"})
        RETURN c.year AS year, count(DISTINCT a) AS numAuthors
        ORDER BY year ASC
    """
    with driver.session(database="open-review-data") as session:
        results = session.run(QUERY)
        data = [(int(record["year"]), record["numAuthors"]) for record in results]

    # Ensure sorted by year
    data.sort(key=lambda x: x[0])

    # Compute YoY growth rates
    growth_rates = []
    for i in range(1, len(data)):
        prev_year, prev_val = data[i - 1]
        curr_year, curr_val = data[i]
        if prev_val > 0:
            growth = (curr_val - prev_val) / prev_val
            growth_rates.append(growth)

    if growth_rates:
        return np.mean(growth_rates)
    else:
        return None


def get_review_scores_or(
    confs=[
        "ICLR.cc_2018",
        "ICLR.cc_2020",
        "ICLR.cc_2021",
        "ICLR.cc_2022",
        "ICLR.cc_2023",
    ]
):
    score_query = """
        MATCH (p:Paper)-[:_IS_SUBMITTED_TO]->(c:Conference {{id: "{0}"}})
        MATCH (p)-[:_HAS_REVIEW]->(r:Review)
        WITH p, avg(toFloat(r.score)) AS score
        RETURN score
    """
    scores = []
    with driver.session(database="open-review-data") as session:
        for conf in confs:
            results = session.run(score_query.format(conf))
            for record in results:
                if record["score"] is not None:
                    scores.append(record["score"])
    return scores


def get_authors_per_paper_openalex(papers):
    counts = []
    for paper in papers:
        authorships = paper.get("authorships", [])
        counts.append(len(authorships))
    return counts


def get_papers_per_author_openalex(authors):
    totals = []
    for author in authors:
        total_works = sum(
            entry.get("works_count", 0) for entry in author.get("counts_by_year", [])
        )
        totals.append(total_works)
    return totals


def get_author_lifespans_openalex(authors):
    lifespans = []
    for author in authors:
        years = [int(entry.get("year")) for entry in author.get("counts_by_year", [])]
        if years:  # make sure author has data
            lifespan = max(years) - min(years) + 1
        else:
            lifespan = 0
        lifespans.append(lifespan)
    return lifespans


def get_author_lifespans():
    pass


def get_papers_per_author():
    pass


def get_authors_per_paper():
    pass


def build_stats(projects):
    contributor_times = defaultdict(list)
    contributor_papers = defaultdict(int)

    authors_per_paper = []
    quality_scores = []

    # collect info
    for proj in projects:
        start_time = proj.get("start_time")
        if start_time < 100:
            continue
        contributors = proj.get("contributors", [])
        authors_per_paper.append(len(contributors))
        quality_scores.append(proj.get("quality_score"))

        for c in contributors:
            contributor_times[c].append(start_time)
            contributor_papers[c] += 1

    # compute ages
    author_lifespan = []
    papers_per_author = []
    for c, times in contributor_times.items():
        age = max(times) - min(times) if len(times) > 1 else 0
        author_lifespan.append(age)
        papers_per_author.append(contributor_papers[c])

    return {
        "papers_per_author": papers_per_author,
        "authors_per_paper": authors_per_paper,
        "lifespan": author_lifespan,
        "quality": quality_scores,
    }


def save_real_world_data():
    with open("../data/target_corpus_meta_info.json", "r") as f:
        papers = json.load(f)
    papers = np.random.choice(list(papers.values()), 10_000, replace=False)
    with open("../data/openalex_authors_sample.json", "r") as f:
        authors = json.load(f)

    print(f"Sampled {len(papers)} papers")
    print(f"Loaded {len(authors)} authors")
    np.save("author_lifespan.npy", np.array(get_author_lifespans_openalex(authors)))
    np.save("papers_per_author.npy", np.array(get_papers_per_author_openalex(authors)))
    np.save("authors_per_paper.npy", np.array(get_authors_per_paper_openalex(papers)))
    np.save("quality_histogram.npy", np.array(get_review_scores_or()))


def generate_proportions(step=0.1):
    proportions = []
    steps = int(1 / step) + 1
    for i in range(steps):
        for j in range(steps - i):
            k = steps - i - j - 1
            p1, p2, p3 = i * step, j * step, k * step
            proportions.append((round(p1, 5), round(p2, 5), round(p3, 5)))
    return proportions


def main():
    # ---- Step 1: Load real-world histograms ----
    H_real_papers_per_author = np.load("papers_per_author.npy")
    H_real_authors_per_paper = np.load("authors_per_paper.npy")
    H_real_lifespan = np.load("author_lifespan.npy")
    H_real_quality = np.load("quality_histogram.npy")
    # ---- Step 4: Optimize ----
    # Bounds: adjust for your parameter ranges
    np.random.seed(42)
    grid_props = generate_proportions(step=0.2)
    rand_props = np.random.dirichlet([1, 1, 1], size=20).tolist()
    candidates = [tuple(prop) for prop in grid_props + rand_props]
    param_space = [
        Real(0.1, 0.9, name="acceptance_threshold"),
        Real(0.1, 0.9, name="orthodox_novelty_threshold"),
        Real(0.1, 0.9, name="careerist_prestige_threshold"),
        Categorical([True, False], name="policy_aligned_in_group"),  # Boolean
        Categorical(candidates, name="policy_population_proportions"),
    ]

    # ---- Step 2–3: Define loss function ----
    def loss(theta):
        print(theta)
        try:
            sim_run = run_simulation_with_policies(
                n_agents=1_200,
                start_agents=200,
                max_steps=600,
                n_groups=20,
                max_peer_group_size=300,
                policy_distribution={
                    "careerist": theta[4][0],
                    "orthodox_scientist": theta[4][1],
                    "mass_producer": theta[4][2],
                },
                output_file_prefix=None,
                group_policy_homogenous=theta[3],
                acceptance_threshold=theta[0],
                novelty_threshold=theta[1],
                prestige_threshold=theta[2],
            )
        except Exception as e:
            print(e)
            return np.inf

        with open("log/calibration_projects.json", "r") as f:
            run_projects = json.load(f)
        sim_data = build_stats(run_projects)
        # Extract histograms (same bins as real)
        H_sim1 = np.histogram(
            sim_data["papers_per_author"], bins=len(H_real_papers_per_author)
        )[0]
        H_sim2 = np.histogram(
            sim_data["authors_per_paper"], bins=len(H_real_authors_per_paper)
        )[0]
        H_sim3 = np.histogram(sim_data["lifespan"], bins=len(H_real_lifespan))[0]
        H_sim4 = np.histogram(sim_data["quality"], bins=len(H_real_quality))[0]

        # Normalize
        H_sim1 = H_sim1 / H_sim1.sum()
        H_sim2 = H_sim2 / H_sim2.sum()
        H_sim3 = H_sim3 / H_sim3.sum()
        H_sim4 = H_sim4 / H_sim4.sum()

        # Distances
        d1 = wasserstein_distance(H_real_papers_per_author, H_sim1)
        d2 = wasserstein_distance(H_real_authors_per_paper, H_sim2)
        d3 = wasserstein_distance(H_real_lifespan, H_sim3)
        d4 = wasserstein_distance(H_real_quality, H_sim4)
        print(d1, d2, d3, d4)
        return d1 + d2 + d3 + d4  # weighted sum possible

    res = gp_minimize(loss, param_space, n_calls=50, random_state=42)
    print("Best parameters:", res.x)


if __name__ == "__main__":
    main()
