import json
from collections import defaultdict

import numpy as np
from neo4j import GraphDatabase
from run_policy_simulation import run_simulation_with_policies
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample
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


def get_acceptance_rates_or(
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
        WITH p.accepted AS accepted
        RETURN accepted
    """
    scores = []
    with driver.session(database="open-review-data") as session:
        for conf in confs:
            results = session.run(score_query.format(conf))
            for record in results:
                if record["accepted"] is not None:
                    scores.append(record["accepted"])
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
    acceptances = []

    # collect info
    for proj in projects:
        start_time = proj.get("start_time")
        if start_time < 100:
            continue
        contributors = proj.get("contributors", [])
        authors_per_paper.append(len(contributors))
        quality_scores.append(proj.get("quality_score"))
        acceptances.append((1 if proj.get("final_reward") > 0 else 0))
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
        "acceptance": acceptances,
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
    np.save("acceptance.npy", np.array(get_review_scores_or()))
    np.save("quality.npy", np.array(get_acceptance_rates_or()))


def generate_proportions(step=0.1):
    proportions = []
    steps = int(1 / step) + 1
    for i in range(steps):
        for j in range(steps - i):
            k = steps - i - j - 1
            p1, p2, p3 = i * step, j * step, k * step
            proportions.append((round(p1, 5), round(p2, 5), round(p3, 5)))
    return proportions


def sensitivity_analysis(problem):
    # --- Step 2: Sample parameter combinations ---
    param_values = sobol_sample.sample(problem, 64, calc_second_order=False)

    # --- Step 3: Run simulation and collect outputs ---
    def run_model(params):
        acceptance, novelty, prestige, effort, rewardless, group_align = params
        sim_run = run_simulation_with_policies(
            n_agents=400,
            start_agents=100,
            max_steps=400,
            n_groups=10,
            max_peer_group_size=100,
            max_rewardless_steps=rewardless,
            policy_distribution={
                "careerist": 1 / 3,
                "orthodox_scientist": 1 / 3,
                "mass_producer": 1 / 3,
            },
            output_file_prefix=None,
            group_policy_homogenous=group_align,
            acceptance_threshold=acceptance,
            novelty_threshold=novelty,
            prestige_threshold=prestige,
            effort_threshold=effort,
        )

        with open("log/calibration_projects.json", "r") as f:
            run_projects = json.load(f)
        sim_data = build_stats(run_projects)

        # Outputs for sensitivity
        return [
            float(np.mean(sim_data["papers_per_author"])),
            float(np.mean(sim_data["authors_per_paper"])),
            float(np.mean(sim_data["lifespan"])),
            float(np.mean(sim_data["quality"])),
            float(np.mean(sim_data["acceptance"])),
        ]

    Y = []
    for i, p in enumerate(param_values):
        print(f"Sensitivity Analysis run {i+1}/{len(param_values)}")
        outputs = run_model(p)
        if not np.isnan(outputs).any():
            Y.append(outputs)
    Y = np.array(Y)
    # --- Step 4: Sobol sensitivity analysis + Save results ---
    output_names = [
        "papers_per_author",
        "authors_per_paper",
        "lifespan",
        "quality",
        "acceptance",
    ]

    for i, output_name in enumerate(output_names):
        Si = sobol_analyze.analyze(problem, Y[:, i], calc_second_order=False)
        results = {
            "S1": dict(zip(problem["names"], Si["S1"].tolist())),
            "ST": dict(zip(problem["names"], Si["ST"].tolist())),
        }
        out_file = f"sensitivity_{output_name}.json"
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved sensitivity results for {output_name} → {out_file}")


def calibrate(problem, real_data):
    H_real_papers_per_author = real_data["papers_per_author"]
    H_real_authors_per_paper = real_data["authors_per_paper"]
    H_real_lifespan = real_data["lifespan"]
    H_real_quality = real_data["quality"]
    real_acceptance_rate = real_data["acceptance"].mean()

    # Bounds: adjust for your parameter ranges
    # np.random.seed(42)
    # grid_props = generate_proportions(step=0.2)
    # rand_props = np.random.dirichlet([1, 1, 1], size=20).tolist()
    # candidates = [tuple(prop) for prop in rand_props + grid_props]
    # np.random.shuffle(candidates)
    # print(candidates)
    names = problem["names"]
    bounds = problem["bounds"]
    param_space = [
        Real(*bounds[0], name=names[0]),
        Real(*bounds[1], name=names[1]),
        Real(*bounds[2], name=names[2]),
        Integer(*bounds[3], name=names[3]),
        Integer(*bounds[4], name=names[4]),
        Categorical(bounds[5], name=names[5]),  # Boolean
        # Categorical(candidates, name="policy_population_proportions"),
    ]

    # ---- Step 2–3: Define loss function ----
    def loss(theta):
        print(theta)
        try:
            sim_run = run_simulation_with_policies(
                n_agents=1_200,
                start_agents=200,
                max_steps=600,
                # max_steps=10,
                n_groups=20,
                max_peer_group_size=300,
                max_rewardless_steps=theta[names.index("max_rewardless_steps")],
                policy_distribution={
                    "careerist": 1 / 3,  # theta[4][0],
                    "orthodox_scientist": 1 / 3,  # theta[4][1],
                    "mass_producer": 1 / 3,  # theta[4][2],
                },
                output_file_prefix=None,
                group_policy_homogenous=bool(
                    theta[names.index("policy_aligned_in_group")]
                ),
                acceptance_threshold=theta[names.index("acceptance_threshold")],
                novelty_threshold=theta[names.index("orthodox_novelty_threshold")],
                prestige_threshold=theta[names.index("careerist_prestige_threshold")],
                effort_threshold=theta[names.index("mass_producer_effort_threshold")],
            )
        except Exception as e:
            print(e)
            return 1e6

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
        sim_acceptance_rate = np.array(sim_data["acceptance"]).mean()

        # Distances
        d1 = wasserstein_distance(H_real_papers_per_author, H_sim1)
        d2 = wasserstein_distance(H_real_authors_per_paper, H_sim2)
        d3 = wasserstein_distance(H_real_lifespan, H_sim3)
        d4 = wasserstein_distance(H_real_quality, H_sim4)
        d5 = np.abs(real_acceptance_rate - sim_acceptance_rate)
        print(d1, d2, d3, d4, d5)
        return d1 + d2 + d3 + d4 + d5  # weighted sum possible

    res = gp_minimize(loss, param_space, n_calls=50, random_state=42)
    print("Best parameters:", res.x)


def main():
    problem = {
        "num_vars": 6,
        "names": [
            "acceptance_threshold",
            "orthodox_novelty_threshold",
            "careerist_prestige_threshold",
            "mass_producer_effort_threshold",
            "max_rewardless_steps",
            "policy_aligned_in_group",
        ],
        "bounds": [
            [0.2, 0.8],  # Real
            [0.2, 0.8],  # Real
            [0.2, 0.8],  # Real
            [10, 50],  # Integer (approx. continuous for SA)
            [50, 500],  # Integer
            [0, 1],  # Boolean → treat as 0/1
        ],
    }
    # sensitivity_analysis(problem)
    real_data = {
        "papers_per_author": np.histogram(np.load("papers_per_author.npy"), 200)[0],
        "authors_per_paper": np.histogram(np.load("authors_per_paper.npy"), 200)[0],
        "lifespan": np.histogram(np.load("author_lifespan.npy"), 200)[0],
        "quality": np.histogram(np.load("quality_histogram.npy"), 10)[0],
        "acceptance": np.load("acceptance_histogram.npy"),
    }
    # Normalize real data histograms
    real_data["papers_per_author"] = (
        real_data["papers_per_author"] / real_data["papers_per_author"].sum()
    )
    real_data["authors_per_paper"] = (
        real_data["authors_per_paper"] / real_data["authors_per_paper"].sum()
    )
    real_data["lifespan"] = real_data["lifespan"] / real_data["lifespan"].sum()
    real_data["quality"] = real_data["quality"] / real_data["quality"].sum()
    real_data["acceptance"] = real_data["acceptance"].mean()
    
    calibrate(problem, real_data)


if __name__ == "__main__":
    main()
