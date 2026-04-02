import json
import os
import pickle
import re
from collections import defaultdict
from datetime import datetime

import openreview
from dotenv import load_dotenv


def get_client(api_version):
    print(api_version)
    # API V2
    if api_version == "2":
        return openreview.api.OpenReviewClient(
            baseurl="https://api2.openreview.net",
            username=os.getenv("OR_USERNAME"),
            password=os.getenv("OR_PASSWORD"),
        )

    # API V1
    if api_version == "1":
        return openreview.Client(
            baseurl="https://api.openreview.net",
            username=os.getenv("OR_USERNAME"),
            password=os.getenv("OR_PASSWORD"),
        )
    else:
        print("Invalid API version!")
        return None


def get_submissions(client, venue, version):
    all_subs = []
    if version == "1":
        all_subs = client.get_all_notes(
            invitation=f"{venue}/-/Submission", details="directReplies"
        )
        all_subs += client.get_all_notes(
            invitation=f"{venue}/-/Blind_Submission", details="directReplies"
        )

    elif version == "2":
        try:
            venue_group = client.get_group(venue)
            if venue_group.content is not None:
                submission_name = venue_group.content["submission_id"]["value"]
                all_subs = client.get_all_notes(invitation=f"{submission_name}")

        except openreview.OpenReviewException as e:
            print(e)

    else:
        print("version must be 1 or 2.")

    return all_subs


def get_reviews(
    paper,
    version="1",
    reply_types=[
        "Official_Review",
        "Meta_Review"
        # "Official_Comment",
        # "Decision",
        "Rebuttal",
    ],
):
    replies = []
    if version == "1":
        if paper.details is not None:
            for reply in paper.details["directReplies"]:
                if reply["invitation"].endswith("Official_Review"):
                    replies.append(reply)
    elif version == "2":

        if paper.details is None:
            return []
        for reply in paper.details["replies"]:
            for invitation in reply["invitations"]:
                for reply_type in reply_types:
                    if invitation.endswith(reply_type):
                        replies.append(reply)
    else:
        print("version must be 1 or 2.")

    return replies


def main():
    load_dotenv("credentials.env")
    version = str(os.getenv("OR_API_VERSION"))
    client = get_client(version)
    if client is None:
        exit(1)
    venues = client.get_group(id="venues").members
    print(f"found {len(venues)} venues.")
    data = {}
    papers = defaultdict(dict)
    peer_reviews = defaultdict(dict)
    for i, venue in enumerate(venues):
        print(i, venue)
        subs = get_submissions(client, venue, version)
        for paper in subs:
            papers[venue][paper.id] = paper
            peer_reviews[venue][paper.id] = get_reviews(paper)
    conf_by_year = enrich_peer_review_data(peer_reviews, papers)
    to_json_file(conf_by_year, "open_review_data")
    save_to_disk(conf_by_year, "confs_by_year")
    save_to_disk(papers, "papers")
    save_to_disk(peer_reviews, "peer_reviews")


def to_json_file(data_dict, fn):
    with open(f"{fn}-{datetime.now().strftime('%d-%m-%Y')}.json", "w") as outfile:
        json.dump(data_dict, outfile, indent=2)


def get_year(conf_name):
    sub_names = conf_name.split("/")
    for name in sub_names:
        if len(name) == 4 and name[:2] in ["19", "20"] and name.isnumeric():
            return name


def get_conference(conf_name):
    conf_name = conf_name.split("/")[0]
    return conf_name


def get_decision(paper, version="2"):
    if version == "2":
        if paper.details is None:
            return []
        for reply in paper.details["replies"]:
            for invitation in reply["invitations"]:
                if invitation.endswith("Decision"):
                    return reply["content"]["decision"]["value"]

    elif version == "1":
        for reply in paper.details["directReplies"]:
            if reply["invitation"].endswith("Decision"):
                return reply["content"]["decision"]

    else:
        return None


def get_meta_review(paper, version="2"):
    if version == "2":
        if paper.details is None:
            return None
        for reply in paper.details["replies"]:
            for invitation in reply["invitations"]:
                if invitation.endswith("Meta_Review"):
                    return reply

    elif version == "1":
        for reply in paper.details["directReplies"]:
            if reply["invitation"].endswith("Meta_Review"):
                return reply

    else:
        return None


def get_avg_score(reviews):
    if len(reviews) == 0:
        return 0
    else:
        return sum([r["score"] for r in reviews if r["score"] is not None]) / len(
            reviews
        )


def get_score(review, version="2"):
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
            "recommendation",
        ]:
            if kw in review["content"]:
                # if kw=="Q6 Overall score":
                # print(re.match("-?\d+",review["content"][kw]).group(0))
                try:
                    if version == "1":
                        val = review["content"][kw]["value"]
                        if isinstance(val, int):
                            score = val
                        else:
                            score = re.match("-?\d+", review["content"][kw])
                    elif version == "2":
                        val = review["content"][kw]["value"]
                        if isinstance(val, int):
                            score = val
                        else:
                            score = re.match("-?\d+", review["content"][kw]["value"])
                    else:
                        score = None
                    if score is not None:
                        break
                except TypeError:
                    continue

        if score is not None and isinstance(score, str):
            return int(score.group(0))
        else:
            return score


def simplify_result_dict(result_dict):
    for key in result_dict:
        if key == "value" and len(result_dict) == 1:
            result_dict = result_dict[key]["value"]
        if isinstance(result_dict[key], dict):
            return simplify_result_dict(result_dict["key"])
        else:
            return result_dict


def enrich_peer_review_data(peer_reviews: dict, papers: dict):
    confs_by_year = defaultdict(lambda: defaultdict(list))
    for conf in peer_reviews:
        for paper in peer_reviews[conf]:
            decision = get_decision(papers[conf][paper])
            for i, pr in enumerate(peer_reviews[conf][paper]):
                peer_reviews[conf][paper][i]["score"] = get_score(pr)
            if decision is not None:
                if len(peer_reviews[conf][paper]) > 0:
                    confs_by_year[get_conference(conf)][get_year(conf)].append(
                        {
                            "decision": decision,
                            "avg_score": get_avg_score(peer_reviews[conf][paper]),
                            "reviews": [
                                {
                                    "id": pr["id"],
                                    "reviewer": pr["signatures"][0],
                                    "content": pr["content"],
                                    "score": pr["score"],
                                }
                                for pr in peer_reviews[conf][paper]
                            ],
                            "paper": {
                                "id": papers[conf][paper]["id"],
                                "content": papers[conf][paper]["content"],
                            },
                        }
                    )
    return confs_by_year


def save_to_disk(dict_var, dict_name):
    with open(f"{dict_name}-{datetime.now().strftime('%d-%m-%Y')}.pickle", "wb") as f:
        pickle.dump(dict(dict_var), f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
