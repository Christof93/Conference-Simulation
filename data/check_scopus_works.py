import json
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv("credentials.env")


def check_scopus_works(
    input_file="openalex_authors_sample_orcid.json",
    output_file="scopus_validation_results.json",
):
    # Please ensure you set this environment variable before running the script
    scopus_api_key = os.environ.get("SCOPUS_API_KEY")
    scopus_inst_token = os.environ.get(
        "SCOPUS_INST_TOKEN", None
    )  # Optional depending on your network and key permissions

    if not scopus_api_key:
        print("Please set your Scopus API Key as an environment variable.")
        print("Use: export SCOPUS_API_KEY='your_api_key'")
        return

    try:
        with open(input_file, "r") as f:
            authors = json.load(f)
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    print(f"Loaded {len(authors)} authors from {input_file}")

    headers = {"Accept": "application/json", "X-ELS-APIKey": scopus_api_key}

    if scopus_inst_token:
        headers["X-ELS-Insttoken"] = scopus_inst_token

    results = []

    # Process authors sequentially
    for idx, author in enumerate(authors, start=1):
        orcid_url = author.get("orcid")
        if not orcid_url:
            print(
                f"[{idx}/{len(authors)}] Skipping {author.get('id')} - No ORCID found."
            )
            continue

        # The ORCID in OpenAlex typically looks like 'https://orcid.org/0000-0002-XXXX-XXXX'
        orcid_id = orcid_url.split("/")[-1]

        print(
            f"[{idx}/{len(authors)}] Checking Scopus data for ORCID {orcid_id} ({author.get('display_name')})..."
        )

        # We query the Scopus Author Search API
        url = "https://api.elsevier.com/content/search/author"
        params = {"query": f"orcid({orcid_id})", "apiKey": scopus_api_key}

        try:
            r = requests.get(url, headers=headers, params=params)
            r.raise_for_status()
            data = r.json()
            print(data)
            scopus_doc_count = 0
            search_results = data.get("search-results", {})
            entries = search_results.get("entry", [])

            # The API returns entries containing 'document-count'
            # Check for error entry types e.g. [{"error": "Result set was empty"}]
            if entries and "error" not in entries[0]:
                for entry in entries:
                    count_str = entry.get("document-count", "0")
                    # Sum counts across split profiles if Scopus also has them split
                    scopus_doc_count += int(count_str)

            results.append(
                {
                    "openalex_id": author.get("id"),
                    "display_name": author.get("display_name"),
                    "orcid": orcid_id,
                    "openalex_works_count": author.get("works_count"),
                    "scopus_works_count": scopus_doc_count,
                }
            )

        except requests.exceptions.RequestException as e:
            print(f"  Error fetching Scopus data for ORCID {orcid_id}: {e}")
            if e.response is not None and e.response.status_code in (401, 403):
                print(
                    "  Authentication error. Check your API key and institutional access."
                )
                break

        # Add a delay between requests to respect Elsevier's rate limits
        time.sleep(0.5)

    # Save output
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} validation results to {output_file}")


if __name__ == "__main__":
    check_scopus_works("openalex_authors_sample_orcid_100k.json")
