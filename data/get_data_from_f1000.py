import json
import glob
from collections import defaultdict
import xml.etree.ElementTree as ET
from datetime import datetime
import hashlib
from get_data_from_or import save_to_disk

def get_reviewer_info(xml_root):
    review_infos = []
    for review in xml_root.iter("sub-article"):
        reviewer_info={}
        if review.attrib["article-type"] == "ref-report":
            affiliation = review.find("./front-stub/contrib-group/aff")[0].tail
            reviewer_info["reviewer_info"] = {}
            reviewer_info["reviewer_info"]["affiliation"] = affiliation
            reviewer_info["reviewer_info"]["name"] = review.find("./front-stub/contrib-group/contrib/name/given-names").text
            reviewer_info["reviewer_info"]["surname"] = review.find("./front-stub/contrib-group/contrib/name/surname").text
            try:
                date = datetime.strptime(
                    review.find("./front-stub/pub-date/day").text + "/" +
                    review.find("./front-stub/pub-date/month").text + "/" +
                    review.find("./front-stub/pub-date/year").text,
                    "%d/%m/%Y"
                )
                date = int(date.timestamp())
            except:
                print(ET.tostring(review.find("./front-stub/pub-date")))
                date = None
            reviewer_info["cdate"]= date
            rel_article = review.find("./front-stub/related-article").attrib["{http://www.w3.org/1999/xlink}href"]
            reviewer_info["about"] = rel_article
            review_infos.append((review.attrib["id"], reviewer_info))
    return review_infos

def get_author_info(xml_root):
    author_infos = []
    contributors = xml_root.find(".//article-meta/contrib-group")
    affs = {aff.attrib["id"]:aff[0].tail for aff in contributors.findall("aff")}
    for contrib_node in contributors.iter("contrib"):
        if contrib_node.attrib["contrib-type"] == "author":
            author_info = {}
            orcid_node = contrib_node.find("uri")
            if orcid_node is not None:
                orcid = orcid_node.text
            else:
                orcid = None
            author_info["orcid"] = orcid

            name_node=contrib_node.find("name")
            if name_node is None:
                try:
                    name_node = contrib_node.find("name-alternatives").find('name[@{http://www.w3.org/XML/1998/namespace}lang="en"]')
                except AttributeError:
                    try:
                        author_info["name"] = contrib_node.find("collab").text
                        author_info["surname"] = None
                        author_info["affiliation"] = None
                        author_info["roles"] = []
                        author_info["is_collective"] = True
                        continue
                    except AttributeError:
                        print(f"no name found: {ET.tostring(contrib_node)}")
                
            author_info["name"] = name_node.find("given-names").text
            author_info["surname"] = name_node.find("surname").text
            aff_id = contrib_node.find("xref[@ref-type='aff']").attrib["rid"]
            author_info["affiliation"] = affs[aff_id]
            author_info["roles"] = [r.text for r in contrib_node.findall("role")]
            author_info["is_collective"] = False
            author_infos.append(author_info)
    return author_infos

def create_author_id(author_dict):
    hash_alg = hashlib.sha256()
    hash_alg.update(author_dict["name"].encode('utf-8'))
    hash_alg.update(" ".encode('utf-8'))
    hash_alg.update(author_dict["surname"].encode('utf-8'))
    hash_alg.update(" ".encode('utf-8'))
    hash_alg.update(author_dict["affiliation"].encode('utf-8'))
    return hash_alg.hexdigest()[:20]

def get_f1000_data():
    papers = defaultdict(dict)
    reviews = defaultdict(dict)
    for filen in glob.glob("nlpeer_v0.1_nopdf/F1000/data/*/v1/*"):
        paper_id = filen.split("/")[3]
        if filen.endswith("/meta.json"):
            with open(filen) as fh:
                paper_meta_obj = json.load(fh)
            papers[paper_id].update(paper_meta_obj)

        elif filen.endswith("/paper.xml"):
            root = ET.parse(filen)
            reviewer_infos= get_reviewer_info(root)
            author_infos = get_author_info(root)
            papers[paper_id]["authors_info"] = author_infos
            for id, info in reviewer_infos:
                reviews[id].update(info)

        elif filen.endswith("/reviews.json"):
            with open(filen) as fh:
                reviews_obj = json.load(fh)

            for review in reviews_obj:
                reviews[review["rid"]].update(review)
    ## retrieve authors
    authors = {}
    for paper in papers.values():
        paper_authors = paper["authors_info"]
        for author in paper_authors:
            id = create_author_id(author)
            author["pid"] = id
            # if author['orcid'] is not None:
            #     # check if author is present without orcid somewhere
            #     prev_roles = set()
            #     if id in authors:
            #         prev_roles = set(authors[id]["roles"])
            #         del authors[id]
            #     # create the author identified by orcid
            #     if not author['orcid'] in authors:
            #         authors[author['orcid']]=author
            #     # update the roles
            #     else:
            #         authors[author['orcid']]["roles"] = set(authors[author['orcid']]["roles"]) | set(author["roles"]) | prev_roles
            # else:
            if not id in authors:
                authors[id] = author
            else:
                authors[id]["roles"] = set(authors[id]["roles"]) | set(author["roles"])
        reviewers = {}

    # retrieve reviewer
    for review in reviews.values():
        id = create_author_id(review["reviewer_info"])
        review["reviewer_info"]["reviewer_id"] = id
        reviewers[id] = {
            "name": review["reviewer_info"]["name"],
            "surname": review["reviewer_info"]["surname"],
            "affiliation": review["reviewer_info"]["affiliation"],
        }
    return papers, authors, reviewers, reviews

if __name__=="__main__":
    papers, authors, reviewers, reviews = get_f1000_data()
    save_to_disk(papers, "f1000_papers")
    save_to_disk(authors, "f1000_authors")
    save_to_disk(reviewers, "f1000_reviewers")
    save_to_disk(reviews, "f1000_reviews")
    