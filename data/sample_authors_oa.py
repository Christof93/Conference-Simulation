import datetime
import json
import time

import requests

BASE_URL = "https://api.openalex.org"
AUTHORS_ENDPOINT = f"{BASE_URL}/authors"
OUTPUT_FILE = "openalex_authors_sample_orcid.json"


def sample_authors(n=1000, per_page=200, seed=42):
    """
    Fetch exactly n authors from OpenAlex deterministically.
    Merge split profiles (same ORCID appearing more than once).
    Then eliminate those with recent publications (last 5 years).
    """
    current_year = datetime.datetime.now().year

    MAX_SAMPLE = 10_000  # OpenAlex hard limit for the sample parameter

    # Fetch authors in chunks of MAX_SAMPLE, each with a unique seed
    fetched_authors = []
    remaining = n
    chunk_index = 0

    while remaining > 0:
        chunk_size = min(remaining, MAX_SAMPLE)
        chunk_seed = seed + chunk_index
        page = 1
        chunk_fetched = []

        print(
            f"  Fetching chunk {chunk_index + 1} ({chunk_size} authors, seed={chunk_seed})..."
        )
        while len(chunk_fetched) < chunk_size:
            params = {
                "sample": chunk_size,
                "page": page,
                "filter": "has_orcid:true",
                "per-page": per_page,
                "seed": chunk_seed,
            }
            r = requests.get(AUTHORS_ENDPOINT, params=params)
            try:
                r.raise_for_status()
            except requests.exceptions.HTTPError:
                print(r.text)
                break
            data = r.json()
            results = data.get("results", [])
            if not results:
                break
            chunk_fetched.extend(results)
            time.sleep(0.1)  # be polite to the API
            page += 1

        fetched_authors.extend(chunk_fetched[:chunk_size])
        remaining -= chunk_size
        chunk_index += 1

    # Merge split profiles by ORCID
    merged_by_orcid = {}
    for a in fetched_authors:
        orcid = a.get("orcid")
        if not orcid:
            continue

        if orcid not in merged_by_orcid:
            # Initialize merged profile
            merged_by_orcid[orcid] = {
                "id": a["id"],
                "orcid": orcid,
                "display_name": a.get("display_name"),
                "works_count": a.get("works_count", 0),
                "counts_by_year": a.get("counts_by_year", []),
                "merged_ids": [a["id"]],
            }
        else:
            # Merge with existing profile
            target = merged_by_orcid[orcid]
            if a["id"] not in target["merged_ids"]:
                target["merged_ids"].append(a["id"])
                target["works_count"] += a.get("works_count", 0)

                # Merge counts_by_year
                existing_years = {y["year"]: y for y in target["counts_by_year"]}
                for yr_info in a.get("counts_by_year", []):
                    y = yr_info["year"]
                    if y in existing_years:
                        existing_years[y]["works_count"] += yr_info.get(
                            "works_count", 0
                        )
                        existing_years[y]["oa_works_count"] += yr_info.get(
                            "oa_works_count", 0
                        )
                        existing_years[y]["cited_by_count"] += yr_info.get(
                            "cited_by_count", 0
                        )
                    else:
                        existing_years[y] = dict(yr_info)

                # Update sorted list
                target["counts_by_year"] = sorted(
                    existing_years.values(), key=lambda x: x["year"], reverse=True
                )

    # Filter out recently active authors from merged groups
    final_authors = []
    for orcid, data in merged_by_orcid.items():
        published_recently = False
        for yr_data in data.get("counts_by_year") or []:
            if (
                yr_data.get("year", 0) >= (current_year - 5)
                and yr_data.get("works_count", 0) > 0
            ):
                published_recently = True
                break

        if not published_recently:
            final_authors.append(data)

    return final_authors


def fetch_publication_info(author_id):
    """
    Get detailed info about one author, including counts_by_year.
    """
    r = requests.get(f"{BASE_URL}/authors/{author_id}?select=id,orcid,counts_by_year")
    r.raise_for_status()
    return r.json()


def main():
    # Step 1: sample authors
    print("Sampling authors from OpenAlex...")
    authors_sample = sample_authors(n=100_000)

    # Step 2: prepare publication info for each merged author
    all_data = []
    for idx, a in enumerate(authors_sample, start=1):
        # We already have the required data without making a second API call
        details = {
            "id": a.get("id"),
            "orcid": a.get("orcid"),
            "display_name": a.get("display_name"),
            "works_count": a.get("works_count"),
            "counts_by_year": a.get("counts_by_year", []),
            "merged_ids": a.get("merged_ids", [a.get("id")]),
        }
        all_data.append(details)

    # Step 3: save to JSON file
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_data, f, indent=2)

    print(f"Saved {len(all_data)} authors with publication info to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
