import os
import json
import re
import pickle

from collections import defaultdict
from dotenv import load_dotenv

import openreview
from datetime import datetime

def get_client(api_version):
    print(api_version)
    # API V2
    if api_version=="2":
        return openreview.api.OpenReviewClient(
            baseurl='https://api2.openreview.net',
            username=os.getenv("OR_USERNAME"),
            password=os.getenv("OR_PASSWORD"),
        )

    # API V1
    if api_version=="1":
        return openreview.Client(
            baseurl='https://api.openreview.net',
            username=os.getenv("OR_USERNAME"),
            password=os.getenv("OR_PASSWORD"),
        )
    else:
        print("Invalid API version!")
        return None

def get_submissions(client, venue, mode, version):
    if version == 1:
        if mode =="single_blind":
            all_subs = client.get_all_notes(invitation=f"{venue}/-/Submission", details="directReplies")
        elif mode=="double_blind":
            all_subs = client.get_all_notes(invitation=f"{venue}/-/Blind_Submission", details="directReplies")
        else:
            all_subs=[]
            print("mode must be single_blind or double_blind")
    elif version == 2:
        all_subs=[]
    else:
        print("version must be 1 or 2.")
        return []
    return all_subs

def get_reviews(paper):
    replies = []
    for reply in paper.details["directReplies"]:
        if reply["invitation"].endswith("Official_Review"):
            replies.append(reply)
    return replies

def main():
    load_dotenv()
    version = str(os.getenv("OR_API_VERSION"))
    client = get_client(version)
    if client is None:
        exit(1)
    venues = client.get_group(id='venues').members
    data = {}
    papers = defaultdict(dict)
    peer_reviews = defaultdict(dict)
    for venue in venues:
        sb_subs = get_submissions(client, venue, "single_blind", version)
        db_subs = get_submissions(client, venue, "double_blind", version)
        subs = sb_subs + db_subs
        for paper in subs:
            papers[venue][paper.id] = paper
            peer_reviews[venue][paper.id] = get_reviews(paper)
    conf_by_year = enrich_peer_review_data(peer_reviews)
    to_json_file(conf_by_year, "open_review_data")
    save_to_disk("confs_by_year.pickle", conf_by_year)
    save_to_disk("papers.pickle", papers)
    save_to_disk("peer_reviews.pickle", peer_reviews)
 
def to_json_file(data_dict, fn):
    with open(f"{fn}-{datetime.now().strftime('%d-%m-%Y')}.json", "w") as outfile:
        json.dump(data_dict, outfile, indent=2)

def get_year(conf_name):
    sub_names = conf_name.split("/")
    for name in sub_names:
        if len(name) == 4 and name[:2] in ["19","20"] and name.isnumeric():
            return name

def get_conference(conf_name):
    conf_name = conf_name.split("/")[0]    
    return conf_name
    
def get_decision(paper):
    for reply in paper.details["directReplies"]:
        if reply["invitation"].endswith("Decision"):
            return reply["content"]["decision"]

def get_meta_review(paper):
    for reply in paper.details["directReplies"]:
        if reply["invitation"].endswith("Meta_Review"):
            return reply

def get_avg_score(reviews):
    return sum([r["score"] for r in reviews if r["score"] is not None])

def get_score(review):
    if review is not None:
        score = None
        for kw in [
            "rating", 
            "final_rating", 
            "final_rating_after_the_rebuttal", 
            "overall_assessment",
            "overall_rating", 
            "review_rating", 
            "score",
            "Overall Score",
            "Overall score",
            "Q6 Overall score",
            "recommendation"]:
            if kw in review["content"]:
                # if kw=="Q6 Overall score":
                    # print(re.match("-?\d+",review["content"][kw]).group(0))
                try:
                    score = re.match("-?\d+",review["content"][kw])
                    if score is not None:
                        break
                except TypeError:
                    continue
        if score is not None:
            return int(score.group(0))


def enrich_peer_review_data(peer_reviews: dict, papers: dict):
    confs_by_year = defaultdict(lambda: defaultdict(list))
    for conf in peer_reviews:
        for paper in peer_reviews[conf]:
            decision = get_decision(papers[conf][paper])
            for pr in peer_reviews[conf][paper]:
                score = get_score(pr)
                pr["score"] = score
            if decision is not None: 
                if len(peer_reviews[conf][paper])>0:
                    confs_by_year[get_conference(conf)][get_year(conf)].append(
                        {
                            "decision": decision,
                            "avg_score": get_avg_score(peer_reviews[conf][paper]),
                            "reviews": peer_reviews[conf][paper]
                        }
                    )
    return confs_by_year
    
def save_to_disk(dict_var, dict_name):
    with open(f"{dict_name}-{datetime.now().strftime('%d-%m-%Y')}", 'wb') as f:
        pickle.dump(dict(dict_var), f, pickle.HIGHEST_PROTOCOL)


if __name__=="__main__":
    main()