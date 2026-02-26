#!/usr/bin/env python3
"""Upload Paper 6 to Zenodo with cross-links to Papers 1-5."""

import requests
import sys
import json

ACCESS_TOKEN = sys.argv[1] if len(sys.argv) > 1 else None
if not ACCESS_TOKEN:
    print("Usage: python zenodo_upload_paper6.py <ACCESS_TOKEN>")
    sys.exit(1)

API = "https://zenodo.org/api"
HEADERS = {"Authorization": f"Bearer {ACCESS_TOKEN}"}

# Paper 5 DOI (the immediate predecessor)
PAPER5_DOI = "10.5281/zenodo.18778768"

# All prior paper DOIs for related identifiers
PRIOR_DOIS = {
    1: "10.5281/zenodo.18778760",
    2: "10.5281/zenodo.18778762",
    3: "10.5281/zenodo.18778764",
    4: "10.5281/zenodo.18778766",
    5: "10.5281/zenodo.18778768",
}

PAPER6 = {
    "file": "docs/arxiv-v6/6-Per-Fact-Graduated-Consolidation.pdf",
    "title": (
        "Per-Fact Graduated Consolidation Resolves the Capacity Ceiling "
        "in Weight-Edited Language Models"
    ),
    "description": (
        "Language models that learn from conversation via direct weight editing "
        "(MEMIT) face a hard capacity ceiling: the 8B Llama model sustains reliable "
        "recall for only ~13 unconstrained edits before cascading interference collapses "
        "performance. Prior attempts to offload knowledge into LoRA adapters failed: "
        "the alignment tax (37% recall degradation on 8B) blocks the transfer pathway, "
        "and per-edit gating produced 0% advancement. We resolve both failures with "
        "per-fact graduated consolidation: each fact independently tracks its consolidation "
        "stage, a graduated dissolution schedule (1.0 -> 0.5 -> 0.1 -> 0.0) progressively "
        "reduces MEMIT influence, and cumulative fusing -- training each cycle on an "
        "already-fused model -- overcomes the alignment tax through incremental prior "
        "erosion. In a capacity sweep on Llama 3.1 8B (4-bit, 2xH100) with {5, 10, 15, 20} "
        "facts across 3 sleep cycles, every condition achieves 100% advancement rate and "
        "1.00 chat recall. MEMIT edits dissolve as designed, making the buffer renewable: "
        "effective lifetime capacity becomes unbounded. This is Paper 6 in the Sleeping LLM "
        "series, superseding the MEMIT-only architecture of Paper 5."
    ),
    "keywords": [
        "language models", "MEMIT", "LoRA", "knowledge editing",
        "memory consolidation", "per-fact gating", "alignment tax",
        "sleep-wake cycle", "continual learning", "lifelong learning"
    ],
    "publication_date": "2026-02-25",
}

CREATOR = {"name": "Baranov, Vladimir", "affiliation": "Independent"}


def create_deposition():
    """Create empty deposition, return (id, bucket_url, prereserved_doi)."""
    r = requests.post(f"{API}/deposit/depositions", json={}, headers=HEADERS)
    r.raise_for_status()
    d = r.json()
    doi = d["metadata"]["prereserve_doi"]["doi"]
    return d["id"], d["links"]["bucket"], doi


def upload_file(bucket_url, filepath):
    """Upload a file to the deposition bucket."""
    filename = filepath.split("/")[-1]
    with open(filepath, "rb") as fp:
        r = requests.put(f"{bucket_url}/{filename}", data=fp, headers=HEADERS)
    r.raise_for_status()
    print(f"  Uploaded {filename} ({r.json()['size']} bytes)")


def set_metadata(dep_id, paper, doi, related_ids=None):
    """Set metadata on a deposition."""
    meta = {
        "metadata": {
            "title": paper["title"],
            "upload_type": "publication",
            "publication_type": "preprint",
            "description": paper["description"],
            "creators": [CREATOR],
            "keywords": paper["keywords"],
            "publication_date": paper["publication_date"],
            "access_right": "open",
            "license": "cc-by-4.0",
            "language": "eng",
            "notes": (
                "Paper 6 in the Sleeping LLM research series on sleep-wake memory "
                "consolidation for lifelong learning in language models. "
                "Supersedes Paper 5 (MEMIT-only) by reintroducing LoRA with "
                "per-fact graduated consolidation."
            ),
        }
    }
    if related_ids:
        meta["metadata"]["related_identifiers"] = related_ids
    r = requests.put(
        f"{API}/deposit/depositions/{dep_id}", json=meta, headers=HEADERS
    )
    r.raise_for_status()
    return r.json()


def publish(dep_id):
    """Publish a deposition (mints the DOI permanently)."""
    r = requests.post(
        f"{API}/deposit/depositions/{dep_id}/actions/publish", headers=HEADERS
    )
    r.raise_for_status()
    return r.json()


def main():
    # Step 1: Create deposition
    print("=" * 60)
    print("CREATING DEPOSITION FOR PAPER 6")
    print("=" * 60)
    dep_id, bucket_url, doi = create_deposition()
    print(f"  Deposition: {dep_id}")
    print(f"  Pre-reserved DOI: {doi}")

    # Step 2: Upload PDF
    print("\n" + "=" * 60)
    print("UPLOADING PDF")
    print("=" * 60)
    upload_file(bucket_url, PAPER6["file"])

    # Step 3: Set metadata with cross-links to all 5 prior papers
    print("\n" + "=" * 60)
    print("SETTING METADATA WITH CROSS-LINKS")
    print("=" * 60)
    related = []
    # Link to Paper 5 as direct predecessor
    related.append({
        "identifier": PAPER5_DOI,
        "relation": "continues",
        "resource_type": "publication-preprint",
    })
    # Link to all prior papers as part of the series
    for paper_num, paper_doi in PRIOR_DOIS.items():
        if paper_doi != PAPER5_DOI:  # already added Paper 5 above
            related.append({
                "identifier": paper_doi,
                "relation": "continues",
                "resource_type": "publication-preprint",
            })

    result = set_metadata(dep_id, PAPER6, doi, related)
    print(f"  Metadata set, DOI: {doi}")

    # Step 4: Publish
    print("\n" + "=" * 60)
    print("PUBLISHING")
    print("=" * 60)
    result = publish(dep_id)
    doi_final = result["doi"]
    doi_url = f"https://doi.org/{doi_final}"
    record_url = result["links"]["record_html"]
    print(f"  PUBLISHED!")
    print(f"  DOI: {doi_final}")
    print(f"  URL: {doi_url}")
    print(f"  Record: {record_url}")

    # Step 5: Update zenodo_results.json
    print("\n" + "=" * 60)
    print("UPDATING RESULTS FILE")
    print("=" * 60)
    try:
        with open("zenodo_results.json", "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        results = []

    results.append({
        "paper": 6,
        "title": PAPER6["title"],
        "doi": doi_final,
        "doi_url": doi_url,
        "record_url": record_url,
    })

    with open("zenodo_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  Results appended to zenodo_results.json")

    # Summary
    print("\n" + "=" * 60)
    print("COMPLETE -- PAPER 6 PUBLISHED")
    print("=" * 60)
    print(f"\n  {PAPER6['title']}")
    print(f"  DOI: {doi_final}")
    print(f"  URL: {doi_url}")
    print(f"  Record: {record_url}")


if __name__ == "__main__":
    main()
