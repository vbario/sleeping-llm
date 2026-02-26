#!/usr/bin/env python3
"""Upload 5 Sleeping LLM papers to Zenodo with cross-links."""

import requests
import sys
import json

ACCESS_TOKEN = sys.argv[1] if len(sys.argv) > 1 else None
if not ACCESS_TOKEN:
    print("Usage: python zenodo_upload.py <ACCESS_TOKEN>")
    sys.exit(1)

API = "https://zenodo.org/api"
HEADERS = {"Authorization": f"Bearer {ACCESS_TOKEN}"}

PAPERS = [
    {
        "file": "docs/arxiv/1-Sleep-Wake-Consolidation.pdf",
        "title": "Sleep-Wake Consolidation for Lifelong Conversational Memory in Local Language Models",
        "description": (
            "We introduce a sleep-wake architecture for lifelong conversational memory in "
            "local language models running on consumer hardware. During wake, the system extracts "
            "facts from conversation and stores them in context. During sleep, it consolidates "
            "these facts into model weights via LoRA fine-tuning using spaced-repetition-inspired "
            "training data. We validate on a 3B parameter model (Llama-3.2-3B-Instruct-4bit) "
            "running on an 8GB MacBook Air M3, demonstrating that sleep cycles produce measurable "
            "memory formation with a narrow viable learning rate window (~1e-4) and a spaced "
            "repetition effect where repeated sleep cycles improve recall. This establishes the "
            "basic feasibility of sleep-wake memory consolidation in local LLMs."
        ),
        "keywords": [
            "language models", "continual learning", "memory consolidation",
            "LoRA", "sleep-wake cycle", "lifelong learning"
        ],
        "publication_date": "2026-02-01",
    },
    {
        "file": "docs/2-Alignment-Tax.pdf",
        "title": "The Alignment Tax on Continual Learning: Inverse Scaling of Memory Consolidation in Language Models",
        "description": (
            "We report a surprising inverse scaling phenomenon in LoRA-based memory consolidation "
            "for language models. At 3B parameters, sleep-wake consolidation achieves 47% factual "
            "recall after training. At 8B, recall drops to 37% with significant confabulation. "
            "At 70B, recall is zero despite successful training (low loss, correct gradient flow). "
            "We identify RLHF alignment as the cause: safety training creates a behavioral prior "
            "that overrides LoRA-injected knowledge at inference time. The effect scales with model "
            "size because larger models receive more extensive alignment training. This 'alignment "
            "tax' on continual learning has implications for any system attempting to inject new "
            "knowledge into aligned language models via parameter-efficient fine-tuning."
        ),
        "keywords": [
            "language models", "alignment tax", "RLHF", "inverse scaling",
            "continual learning", "LoRA", "memory consolidation"
        ],
        "publication_date": "2026-02-08",
    },
    {
        "file": "docs/arxiv-v2/3-Dual-System-Memory-Consolidation.pdf",
        "title": "Dual-System Memory Consolidation for Lifelong Learning in Language Models: Combining Direct Weight Editing with Sleep-Wake Training",
        "description": (
            "We introduce a dual-system memory architecture for language models inspired by "
            "Complementary Learning Systems (CLS) theory. MEMIT (Mass-Editing Memory in "
            "Transformers) serves as fast hippocampal encoding, injecting facts directly into "
            "MLP weights during wake. LoRA fine-tuning serves as slow neocortical consolidation "
            "during sleep. We develop covariance-regularized MEMIT with cross-edit null-space "
            "constraints that prevent new edits from overwriting previous ones, and validate the "
            "dual system across 3B, 8B, and 70B parameter models. Key ablations show: (1) the "
            "dual system outperforms either component alone, (2) null-space constraints achieve "
            "perfect retention across sequential edits, and (3) the Woodbury identity enables "
            "efficient covariance regularization in N x N space rather than d x d."
        ),
        "keywords": [
            "language models", "MEMIT", "knowledge editing", "complementary learning systems",
            "null-space constraints", "dual-system memory", "lifelong learning"
        ],
        "publication_date": "2026-02-12",
    },
    {
        "file": "docs/arxiv-v4/4-Sleeping-LLM.pdf",
        "title": "Sleeping LLM: Two-Phase Memory Consolidation for Lifelong Learning from 3B to 70B Parameters",
        "description": (
            "We present a two-phase sleep architecture for memory consolidation in language "
            "models, with slow-wave sleep (SWS) for individual fact consolidation via per-fact "
            "LoRA training, and REM sleep for knowledge integration via synthetic multi-fact "
            "conversations. We introduce per-fact staged consolidation where each fact independently "
            "advances through stages (0-3) based on individual chat recall testing, replacing "
            "all-or-nothing per-edit gating. Key findings: MEMIT achieves near-zero perplexity "
            "cost for fact injection; REM reduces SWS-induced perplexity damage by 88% at 3B; "
            "per-fact gating achieves 95% consolidation success at 8B; and we discover pathway "
            "separation where MEMIT edits the raw completion pathway while LoRA edits the chat "
            "pathway. We validate across 3B, 8B, and 70B models, demonstrating that the "
            "graduated MEMIT dissolution schedule (scale 1.0 -> 0.5 -> 0.1 -> 0.0) successfully "
            "transfers knowledge from MEMIT to LoRA."
        ),
        "keywords": [
            "language models", "memory consolidation", "sleep-wake cycle",
            "MEMIT", "LoRA", "per-fact gating", "two-phase sleep",
            "continual learning", "lifelong learning"
        ],
        "publication_date": "2026-02-18",
    },
    {
        "file": "docs/arxiv-v5/5-Sleep-Wake-Memory-Convergence.pdf",
        "title": "Sleep-Wake Memory Convergence in Weight-Edited Language Models",
        "description": (
            "We present a sleep-wake architecture that injects facts directly into MLP weights "
            "using MEMIT during wake, then maintains them through sleep cycles of auditing, "
            "constrained refreshing, and pruning. On 8B and 70B models, we identify a sharp wake "
            "capacity threshold: the 8B model sustains 0.92 recall at 13 unconstrained edits, "
            "collapsing to 0.57 at 14 -- a tipping point caused by cascading edit interference. "
            "Sleep maintenance with null-space-constrained refreshes converges to 100% recall even "
            "from severe degradation: 30 facts at 40% recall recover fully within 4 sleep cycles. "
            "The 70B model converges 2x faster and absorbs a second injection wave with zero "
            "degradation, demonstrating that model scale provides more orthogonal weight dimensions "
            "for non-interfering edits. The ratio between wake capacity and sleep capacity defines "
            "optimal sleep frequency -- a 'drowsiness signal' analogous to biological sleep "
            "pressure. We characterize a failure mode: when pruning removes working edits faster "
            "than refresh can replace them, a death spiral drives recall from 97% to 46% over 10 "
            "cycles. Perplexity remains stable throughout convergence (+0.5% for 8B at 14 facts, "
            "0% for 70B), confirming that constrained MEMIT maintenance is a near-free operation. "
            "This paper supersedes our prior work on LoRA-based consolidation, removing LoRA "
            "entirely: MEMIT is now the sole memory mechanism, and sleep performs maintenance "
            "rather than pathway transfer."
        ),
        "keywords": [
            "language models", "MEMIT", "knowledge editing", "memory maintenance",
            "null-space constraints", "sleep-wake cycle", "continual learning",
            "perplexity stability", "lifelong learning"
        ],
        "publication_date": "2026-02-25",
    },
]

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


def set_metadata(dep_id, paper, related_ids=None):
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
                "Part of the Sleeping LLM research series on sleep-wake memory "
                "consolidation for lifelong learning in language models."
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
    # Step 1: Create all depositions and get pre-reserved DOIs
    print("=" * 60)
    print("CREATING DEPOSITIONS")
    print("=" * 60)
    deposits = []
    for i, paper in enumerate(PAPERS):
        dep_id, bucket_url, doi = create_deposition()
        deposits.append({"id": dep_id, "bucket": bucket_url, "doi": doi})
        print(f"  Paper {i+1}: deposition={dep_id}, DOI={doi}")

    # Step 2: Upload files
    print("\n" + "=" * 60)
    print("UPLOADING FILES")
    print("=" * 60)
    for i, paper in enumerate(PAPERS):
        print(f"\nPaper {i+1}: {paper['title'][:60]}...")
        upload_file(deposits[i]["bucket"], paper["file"])

    # Step 3: Set metadata with cross-links
    print("\n" + "=" * 60)
    print("SETTING METADATA WITH CROSS-LINKS")
    print("=" * 60)
    for i, paper in enumerate(PAPERS):
        related = []
        # Link to previous paper
        if i > 0:
            related.append({
                "identifier": deposits[i - 1]["doi"],
                "relation": "continues",
                "resource_type": "publication-preprint",
            })
        # Link to next paper
        if i < len(PAPERS) - 1:
            related.append({
                "identifier": deposits[i + 1]["doi"],
                "relation": "isContinuedBy",
                "resource_type": "publication-preprint",
            })
        result = set_metadata(deposits[i]["id"], paper, related or None)
        print(f"  Paper {i+1}: metadata set, DOI={deposits[i]['doi']}")

    # Step 4: Publish all
    print("\n" + "=" * 60)
    print("PUBLISHING")
    print("=" * 60)
    results = []
    for i, paper in enumerate(PAPERS):
        result = publish(deposits[i]["id"])
        doi_url = f"https://doi.org/{result['doi']}"
        record_url = result["links"]["record_html"]
        results.append({"doi": result["doi"], "doi_url": doi_url, "record_url": record_url})
        print(f"  Paper {i+1}: PUBLISHED")
        print(f"    DOI: {result['doi']}")
        print(f"    URL: {doi_url}")
        print(f"    Record: {record_url}")

    # Summary
    print("\n" + "=" * 60)
    print("COMPLETE -- ALL 5 PAPERS PUBLISHED")
    print("=" * 60)
    for i, paper in enumerate(PAPERS):
        print(f"\n  {i+1}. {paper['title'][:70]}")
        print(f"     DOI: {results[i]['doi']}")
        print(f"     URL: {results[i]['doi_url']}")

    # Save results
    with open("zenodo_results.json", "w") as f:
        json.dump(
            [
                {
                    "paper": i + 1,
                    "title": PAPERS[i]["title"],
                    "doi": results[i]["doi"],
                    "doi_url": results[i]["doi_url"],
                    "record_url": results[i]["record_url"],
                }
                for i in range(len(PAPERS))
            ],
            f,
            indent=2,
        )
    print("\nResults saved to zenodo_results.json")


if __name__ == "__main__":
    main()
