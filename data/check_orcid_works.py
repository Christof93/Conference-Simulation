import json
import os
import time

import requests
from dotenv import load_dotenv

# Load credentials from credentials.env
load_dotenv("credentials.env")


def get_orcid_token():
    """
    Obtain a /read-public access token from ORCID.
    Requires ORCID_CLIENT_ID and ORCID_CLIENT_SECRET in credentials.env.
    """
    client_id = os.environ.get("ORCID_CLIENT_ID")
    client_secret = os.environ.get("ORCID_CLIENT_SECRET")

    if not client_id or not client_secret:
        print("ORCID_CLIENT_ID or ORCID_CLIENT_SECRET not found in environment.")
        return None

    print("Obtaining ORCID access token...")
    url = "https://orcid.org/oauth/token"
    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "client_credentials",
        "scope": "/read-public",
    }
    headers = {"Accept": "application/json"}

    try:
        response = requests.post(url, data=payload, headers=headers)
        response.raise_for_status()
        token_data = response.json()
        return token_data.get("access_token")
    except Exception as e:
        print(f"Error obtaining ORCID token: {e}")
        return None


def check_orcid_works(
    input_file="openalex_authors_sample_orcid_10k.json",
    output_file="orcid_validation_results.json",
):
    """
    Reads authors from input_file and fetches their publication counts from ORCID.
    """
    token = get_orcid_token()
    if not token:
        print("Cannot proceed without an ORCID access token.")
        return

    try:
        with open(input_file, "r") as f:
            authors = json.load(f)
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    print(f"Loaded {len(authors)} authors from {input_file}")

    results = []
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}

    for idx, author in enumerate(authors, start=1):
        orcid_url = author.get("orcid")
        if not orcid_url:
            continue

        orcid_id = orcid_url.split("/")[-1]
        print(
            f"[{idx}/{len(authors)}] Checking ORCID works for {orcid_id} ({author.get('display_name')})..."
        )

        # ORCID Public API endpoint for works summary
        url = f"https://pub.orcid.org/v3.0/{orcid_id}/works"

        try:
            r = requests.get(url, headers=headers)
            r.raise_for_status()
            data = r.json()
            # Collect all publication years
            preferred_works = []
            groups = data.get("group", [])
            if groups:
                for group in groups:
                    summaries = group.get("work-summary", [])
                    if not summaries:
                        continue

                    # ORCID groups similar works. We pick the preferred one
                    # (highest display-index) to avoid double counting.
                    # Usually, the first one in the list is the preferred one if ordered.
                    preferred_work = max(
                        summaries, key=lambda x: int(x.get("display-index", 0))
                    )
                    pub_date = preferred_work.get("publication-date")
                    if pub_date:
                        year_obj = pub_date.get("year")
                        if year_obj:
                            year_val = year_obj.get("value")
                            if year_val:
                                try:
                                    preferred_works.append(
                                        {
                                            "title": preferred_work.get("title")
                                            .get("title")
                                            .get("value"),
                                            "path": preferred_work.get("path"),
                                            "year": int(year_val),
                                        }
                                    )
                                except ValueError:
                                    pass  # Skip invalid years

            results.append(
                {
                    "openalex_id": author.get("id"),
                    "display_name": author.get("display_name"),
                    "orcid": orcid_id,
                    "openalex_works_count": author.get("works_count"),
                    "orcid_works_count": len(preferred_works),
                    "counts_by_year": preferred_works,
                }
            )

        except requests.exceptions.RequestException as e:
            print(f"  Error fetching ORCID data for {orcid_id}: {e}")
            # Optional: handle specific error codes (e.g. 404 if record doesn't exist)

        # Respect rate limits
        time.sleep(0.2)

    # Save output
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} validation results to {output_file}")


if __name__ == "__main__":
    # You can change the input file as needed
    check_orcid_works("openalex_authors_sample_orcid_100k.json")
