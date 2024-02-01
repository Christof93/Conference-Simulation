import pickle
import random
random.seed(42)
import json
from itertools import zip_longest
from tqdm import tqdm
from neo4j import GraphDatabase
from collections import defaultdict

from get_data_from_f1000 import get_f1000_data

uri = "neo4j://localhost:7687"

OR_DB_SETUP_SCRIPTS = [
"""//Add normalized score
MATCH (p:Paper)-[:`_IS_SUBMITTED_TO`]->(c:Conference), (p)-[:`_HAS_REVIEW`]->(r:Review) WITH c, MAX(r.score) as max_score_conf 
MATCH (p:Paper)-[:`_IS_SUBMITTED_TO`]->(c:Conference), (p)-[:`_HAS_REVIEW`]->(r:Review)
SET r.score_norm = r.score/toFloat(max_score_conf);
""",
"""//Add max_score to conference
MATCH (p:Paper)-[:`_IS_SUBMITTED_TO`]->(c:Conference), (p)-[:`_HAS_REVIEW`]->(r:Review) WITH c, MAX(r.score) as max_score_conf 
SET c.max_score = max_score_conf;
""",
"""//Add correct avg_score
MATCH (p:Paper)-[:`_IS_SUBMITTED_TO`]->(c:Conference), (p)-[:`_HAS_REVIEW`]->(r:Review) WITH c, MAX(r.score) as max_score_conf 
MATCH (p:Paper)-[:`_IS_SUBMITTED_TO`]->(c:Conference), (p)-[:`_HAS_REVIEW`]->(r:Review) WITH p, COLLECT(r.score) as scores, AVG(r.score) as avg_score
SET p.avg_score = avg_score;
""",
]

F1000_DB_SETUP_SCRIPTS = [
"""// Create an review_about index
CREATE INDEX review_about 
FOR (r:Review)
ON (r.about)
""",
"""// Create an author_orcid index
CREATE INDEX author_orcid 
FOR (n:Author)
ON (n.orcid)
""",
]

def get_subj_obj_label(rel_type):
    if rel_type == "_IS_SUBMITTED_TO":
        return "Paper", "_IS_SUBMITTED_TO", "Conference"
    elif rel_type == "_HAS_AUTHOR":
        return "Paper", "_HAS_AUTHOR", "Author" 
    elif rel_type == "_HAS_REVIEW":
        return "Paper", "_HAS_REVIEW", "Review"
    elif rel_type == "_HAS_REVIEW_AUTHOR":
        return "Review", "_HAS_AUTHOR", "Reviewer"

def run_insert_relation(session, rel_type, params):
    subj_label, rel_type, obj_label = get_subj_obj_label(rel_type)
    insert_query=f'WITH $relations AS batch \
UNWIND batch AS rel \
MATCH (subj:{subj_label} {{id: rel.subj}}) \
MATCH (obj:{obj_label} {{id: rel.obj}}) \
MERGE (subj) -[:{rel_type}]-> (obj) '
    session.execute_write(lambda tx: tx.run(insert_query, relations=params))
    # result.single()

def run_cypher_create(neo4j_driver, db_name, params, label):
    session = neo4j_driver.session(database=db_name)
    try:
        if "relations" in params:
            for rel_type in params["relations"]:
                insert_query = ""
                relations = []
                for i, rel in enumerate(params["relations"][rel_type]):
                    relations.append(rel)
                    if (i+1)%10000==0:
                        run_insert_relation(session, rel_type, relations)
                        relations=[]                       
                run_insert_relation(session, rel_type, relations)
                rel_label = list(range(5))
                rel_label[::2] = get_subj_obj_label(rel_type)
                rel_label[1::2] = ["-", "->"]
                print(f"{i+1} {' '.join(rel_label)} relations successfully inserted to graph store.")

        elif "props" in params:
            insert_query=f"UNWIND $props AS map CREATE (n:{label}) SET n = map RETURN COUNT(n) as cnodes"
            result = session.run(query=insert_query, parameters=params)
            print(result.single()["cnodes"], f"{label} nodes created.")
    finally:
        session.close()

def run_cypher_match(neo4j_driver, db_name, query, params=None):
    session = neo4j_driver.session(database=db_name)
    try:
        result = session.run(query, parameters=params)
        return [line for line in result]
    finally:
        session.close()


def make_author_nodes(authors):
    params = {"props": []}
    ids=set()
    for author in authors.values():
        id = author["id"]
        name = author["name"]
        if name is not None:
            name = name.replace("'",r"\'").replace("\\\\","\\")
        if not name == "Anonymous" and not id in ids:
            params["props"].append({
                "name":name,
                "id":id,
            })
            ids.add(id)
    return params

def make_conf_nodes(confs):
    params = {"props": []}
    for conf in confs:
        for year in confs[conf]:
            params["props"].append({
                "id": f"{conf}_{year}",
                "year": year,
                "name": conf,
            })
    return params

def make_paper_review_nodes_and_relations(confs, papers):
    paper_params = {"props": []}
    review_params = {"props": []}
    relations = defaultdict(list)
    for conf in confs:
        for year in confs[conf]:
            for paper in confs[conf][year]:
                paper_id = paper["reviews"][0]["replyto"]
                conf_id = f"{conf}_{year}"
                # paper _is_submitted to conference
                relations["_IS_SUBMITTED_TO"].append({"subj":paper_id, "obj": conf_id})
                paper_info = papers[paper_id]
                if "pdf" in paper_info.content:
                    paper_pdf = paper_info.content["pdf"]
                else:
                    paper_pdf = None
                if "abstract" in paper_info.content:
                    paper_abstract = paper_info.content["abstract"]
                else:
                    paper_abstract = None
                if "keywords" in paper_info.content:
                    paper_keywords = paper_info.content["keywords"]
                else:
                    paper_keywords = None
                paper_params["props"].append({
                    "id": paper_id,
                    "decision": paper["decision"],
                    "accepted": (1 if (
                                 paper["decision"].lower().startswith("accept") or 
                                 paper["decision"].lower().startswith("conditional accept") or
                                 paper["decision"].lower().startswith("winner") or
                                 paper["decision"].lower().startswith("invite")
                                 )
                                 else 0),
                    "avg_score": paper["avg_score"],
                    "cdate": paper_info.cdate,
                    "abstract": paper_abstract,
                    "keywords": paper_keywords,
                    "title": paper_info.content["title"],
                    "pdf": paper_pdf,
                })
                for author_id in paper_info.content["authorids"]:
                    # author_id _SUBMITS paper_id
                    relations["_HAS_AUTHOR"].append({"subj":paper_id, "obj": author_id})

                for review in paper["reviews"]:
                    review_params["props"].append({
                        "content": json.dumps(review["content"]),
                        "cdate": review["cdate"],
                        "id": review["id"],
                        "score": review["score"],
                        "rating": None,
                    })
                    # paper_id _HAS_REVIEW review_id
                    relations["_HAS_REVIEW"].append({"subj":paper_id, "obj": review["id"]})

    return paper_params, review_params, {"relations":relations}

def make_reviewer_relation(driver, db_name, paper_info, paper_review_tuples):
    reviews_per_paper = defaultdict(list)
    for paper, review in paper_review_tuples:
        reviews_per_paper[paper].append(review)

    author_paper_freqs = run_cypher_match(
        driver,
        db_name,
        "MATCH (paper:Paper)-[r:`_HAS_AUTHOR`]->(a:Author) RETURN a.id, a.name, COUNT(paper) as n ORDER BY n DESC",
    )
    n_papers = sum([int(n) for _, _, n in author_paper_freqs])
    ## extract the probability distribution of authors submitting papers
    authors_with_probs = {authorid: freq/n_papers for authorid, _, freq in author_paper_freqs}
    authors_names = {authorid: authorname for authorid, authorname, _ in author_paper_freqs}

    ## randomly choosing reviewers from authors with probability proportional to number of submitted articles.
    print("randomly assigning reviewers to reviews...")
    reviewers_plus_reviews = []
    for paper in tqdm(reviews_per_paper):
        paper_authors = set(paper_info[paper].content["authorids"])
        paper_reviews = reviews_per_paper[paper]
        # make sure that authors can not be reviewers at the same time..
        possible_reviewers = list(set(authors_with_probs.keys()) - paper_authors)
        # weighting the possible reviewers according to their paper writing probability
        paper_reviewer_probs = [authors_with_probs[r] for r in possible_reviewers]

        paper_reviewers = random.choices(
            population=possible_reviewers,
            weights=paper_reviewer_probs,
            k=len(paper_reviews),
        )
        reviewers_plus_reviews+=list(zip(paper_reviewers, paper_reviews))

    assert(len(reviewers_plus_reviews)==len(paper_review_tuples))
    reviewers = {id for id, _ in reviewers_plus_reviews}
    reviewers = [{"id":id,"name":authors_names[id]} for id in reviewers]
    return {"props":reviewers}, {"relations": {"_HAS_REVIEW_AUTHOR": [{"subj":r, "obj":ra} for ra,r in reviewers_plus_reviews]}}

def set_constraint(driver, db_name, node_label, property):
    driver.execute_query(f"CREATE CONSTRAINT {node_label.lower()}_unique_{property} FOR (n:{node_label}) REQUIRE n.{property} IS UNIQUE", database_=db_name)


def hydrate_open_review_database():
    with open('confs_by_year.pickle', 'rb') as f:
        confs_by_year = pickle.load(f)
    with open('papers.pickle', 'rb') as f:
        papers_by_conf = pickle.load(f)
    num_papers = 0
    papers = {}
    for conf in papers_by_conf:
        for paper_id in papers_by_conf[conf]:
            num_papers+=1
            papers[paper_id]=papers_by_conf[conf][paper_id]
    
    assert num_papers==len(papers)
    authors = defaultdict(dict)
    for paper in papers:
        for author_id, author in zip_longest(papers[paper].content["authorids"], papers[paper].content["authors"]):
            authors[f'{author_id}'] = {"id":author_id, "name": author}

    node_params = {}
    node_params["Author"] = make_author_nodes(authors)
    node_params["Conference"] = make_conf_nodes(confs_by_year)
    node_params["Paper"], node_params["Review"], insert_relations = \
        make_paper_review_nodes_and_relations(confs_by_year, papers)
    paper_plus_reviews = [(r["subj"], r["obj"]) for r in insert_relations["relations"]["_HAS_REVIEW"]]

    with GraphDatabase.driver(uri, auth=("neo4j", "openreview")) as driver:
        for node_label in ["Conference", "Author", "Paper", "Review"]:
            set_constraint(driver, "open-review-data", node_label, "id")
            run_cypher_create(driver, "open-review-data", node_params[node_label], node_label)
        run_cypher_create(driver, "open-review-data", insert_relations, None)

    ## randomly assign authors to reviews and create the reviewer node and relation 
    with GraphDatabase.driver(uri, auth=("neo4j", "openreview")) as driver:
        reviewer_nodes, reviewer_relations = make_reviewer_relation(driver, "open-review-data", papers, paper_plus_reviews)
        set_constraint(driver, "open-review-data", "Reviewer", "id")
        run_cypher_create(driver, "open-review-data", reviewer_nodes, "Reviewer")
        run_cypher_create(driver, "open-review-data", reviewer_relations, None)
        run_setup_scripts(driver, "open-review-data", OR_DB_SETUP_SCRIPTS)

def run_setup_scripts(driver, db_name, cypher_scripts):
    for script in cypher_scripts:
        driver.execute_query(script, database_ = db_name)

def hydrate_f1000_database():
    f1000_data = {}
    for data_type in ["papers", "authors", "reviewers", "reviews"]:
        with open(f"f1000_{data_type}-01-02-2024.pickle", "rb") as fh:
            f1000_data[data_type]=pickle.load(fh)

    insert_relations = {"relations": defaultdict(list)}
    for p in f1000_data["papers"].values():
        p["id"]=p["doi"]
        del p["doi"]
        p["authors_orcid"] = [a["orcid"] if a["orcid"] is not None else "na" for a in p["authors_info"]]
        p["authors_pid"] = [a["pid"] for a in p["authors_info"]]
        del p["authors_info"]
        for author in p["authors_pid"]:
            insert_relations["relations"]["_HAS_AUTHOR"].append({"subj":p["id"], "obj":author})

    for r in f1000_data["reviews"].values():
        r["id"]= r["rid"]
        del r["rid"]
        r["decision"] = r["scores"]["overall"]
        del r["scores"]
        r["reviewer"] = r["reviewer_info"]["reviewer_id"]
        del r["reviewer_info"]
        r["doi"] = r["meta"]["doi"]
        r["sentences"] = [boundary for boundaries in r["meta"]["sentences"]["main"] for boundary in boundaries]
        r["license"] = r["meta"]["license"]
        del r["meta"]
        r["report"]= r["report"]["main"]
        insert_relations["relations"]["_HAS_REVIEW"].append({"subj": r["about"], "obj": r["id"]})
        insert_relations["relations"]["_HAS_REVIEW_AUTHOR"].append({"subj": r["id"], "obj": r["reviewer"]})

    for a in f1000_data["authors"].values():
        a["roles"]=list(a["roles"])
        a["id"]=a["pid"]
        del a["pid"]

    node_params={}
    node_params["Author"] = {"props": [a for a in f1000_data["authors"].values()]}
    node_params["Reviewer"] = {"props":[p|{"id":id} for id, p in f1000_data["reviewers"].items()]}
    node_params["Paper"] = {"props":list(f1000_data["papers"].values())}
    node_params["Review"] = {"props":list(f1000_data["reviews"].values())}
    
    with GraphDatabase.driver(uri, auth=("neo4j", "openreview")) as driver:
        # set_constraint(driver, "f1000-data", "Author", "orcid")
        set_constraint(driver, "f1000-data", "Paper", "doc_id")

        for node_label in ["Author", "Reviewer", "Paper", "Review"]:
            set_constraint(driver, "f1000-data", node_label, "id")
            print(f"trying to insert {len(node_params[node_label]['props'])} {node_label} nodes ...")
            run_cypher_create(driver, "f1000-data", node_params[node_label], node_label)
        run_cypher_create(driver, "f1000-data", insert_relations, None)
        run_setup_scripts(driver, "f1000-data", F1000_DB_SETUP_SCRIPTS)


def delete_nodes(db_name, label=None):
    if label is not None:
        delete_query=f"MATCH (n:{label}) DELETE n"
    else:
        delete_query=f"MATCH (n) DELETE n"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "openreview"))
    try:
        driver.execute_query(delete_query, database_=db_name)
    finally:
        driver.close()

def delete_edges(db_name, label=None):
    if label is not None:
        delete_query=f"MATCH () -[r:{label}]-> () DELETE r"
    else:
        delete_query=f"MATCH () -[r]-> () DELETE r"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "openreview"))
    try:
        driver.execute_query(delete_query, database_=db_name)
    finally:
        driver.close()    
    
def delete_constraints(db_name, constraint_name=None):
    with GraphDatabase.driver(uri, auth=("neo4j", "openreview")) as driver:
        if constraint_name is not None:
            driver.execute_query(f"DROP CONSTRAINT {constraint_name}", database_=db_name)
        else:
            constraints, _, _ = driver.execute_query("SHOW ALL CONSTRAINTS YIELD name", database_=db_name)
            for constraint in constraints:
                driver.execute_query(f"DROP CONSTRAINT {constraint['name']}", database_=db_name)

def delete_indices(db_name, index_name=None):
    with GraphDatabase.driver(uri, auth=("neo4j", "openreview")) as driver:
        if index_name is not None:
            driver.execute_query(f"DROP INDEX {index_name}", database_=db_name)
        else:
            constraints, _, _ = driver.execute_query("SHOW ALL INDEXES YIELD name", database_=db_name)
            for constraint in constraints:
                driver.execute_query(f"DROP INDEX {constraint['name']}", database_=db_name)

def reset_db(db_name):
    delete_edges(db_name)
    delete_nodes(db_name)
    delete_constraints(db_name)
    delete_indices(db_name)

def main():
    reset_db("open-review-data")
    hydrate_open_review_database()
    # reset_db("f1000-data")
    # hydrate_f1000_database()



if __name__=="__main__":
    main()
    
