"""Ablation 2: MEMIT Retention Over Conversations.

Tests how MEMIT-injected facts survive:
  1. Filler conversations (unrelated chat turns)
  2. New fact injections (with null-space constraints)
  3. Nap consolidation

Protocol:
  1. Inject 5 facts (batch A)
  2. Chat 20 unrelated turns
  3. Measure recall of batch A
  4. Inject 5 more facts (batch B) — null-space constraints protect A
  5. Measure recall of A and B
  6. Trigger nap
  7. Measure recall of all 10 facts post-nap

Usage:
    python experiments/ablation_retention.py --config experiments/configs/8b_memit.yaml
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.orchestrator import Orchestrator
from src.memory.memit import FactTriple


# ── Batch A: 5 facts ──
BATCH_A = [
    FactTriple("Elena Voronov", "lives in", "Portland"),
    FactTriple("Elena Voronov", "works as", "marine biologist"),
    FactTriple("Marcus Takahashi", "lives in", "Austin"),
    FactTriple("Marcus Takahashi", "works as", "architect"),
    FactTriple("Priya Lindström", "lives in", "Denver"),
]

# ── Batch B: 5 different facts ──
BATCH_B = [
    FactTriple("Tobias Okafor", "lives in", "Seattle"),
    FactTriple("Tobias Okafor", "works as", "chef"),
    FactTriple("Yuki Petrov", "lives in", "Boston"),
    FactTriple("Yuki Petrov", "works as", "photographer"),
    FactTriple("Carlos Navarro", "lives in", "Nashville"),
]

FILLER_MESSAGES = [
    "What's the best way to cook pasta al dente?",
    "Tell me about the history of jazz music.",
    "How do solar panels work?",
    "What are some good hiking trails?",
    "Can you explain how tides are caused by the moon?",
    "What's the difference between a crocodile and an alligator?",
    "Tell me about the Pythagorean theorem.",
    "How does a combustion engine work?",
    "What are the benefits of meditation?",
    "Explain the water cycle briefly.",
    "What is photosynthesis?",
    "Tell me about the Renaissance period.",
    "How do airplanes stay in the air?",
    "What causes earthquakes?",
    "Tell me about the Amazon rainforest.",
    "How does Wi-Fi work?",
    "What are black holes?",
    "Tell me about ancient Egyptian pyramids.",
    "How do vaccines work?",
    "What is the speed of light?",
]


def test_recall(backend, facts, label=""):
    """Test raw completion recall for a list of FactTriples."""
    passed = 0
    results = []
    for fact in facts:
        prompt = fact.to_prompt()
        response = backend.generate(prompt, max_tokens=30, temperature=0.1)
        if response is None:
            response = ""
        found = fact.object.lower() in response.lower()
        if found:
            passed += 1
        results.append({
            "prompt": prompt,
            "expected": fact.object,
            "response": response[:80].strip(),
            "passed": found,
        })
    recall = passed / len(facts) if facts else 0
    if label:
        print(f"  {label}: {recall:.2f} ({passed}/{len(facts)})")
        for r in results:
            status = "PASS" if r["passed"] else "FAIL"
            print(f"    [{status}] \"{r['prompt']}\" → \"{r['response']}\"")
    return recall, results


def clean_artifacts(config):
    """Remove artifacts to ensure clean state."""
    dirs_to_clean = [
        config.paths["current_model"],
        config.paths["checkpoints"],
        config.paths["adapters"],
        config.paths["training"],
        config.paths["conversations"],
    ]
    for dir_path in dirs_to_clean:
        p = Path(dir_path)
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)

    memit_dir = Path(config.paths.get("memit_data", "data/memit"))
    if memit_dir.exists():
        shutil.rmtree(memit_dir)
        memit_dir.mkdir(parents=True, exist_ok=True)

    ledger_path = Path(config.paths.get("memit_ledger", "data/memit/ledger.json"))
    if ledger_path.exists():
        ledger_path.unlink()


def print_section(label):
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="Ablation 2: MEMIT Retention Over Conversations")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    start_time = time.time()

    print("=" * 70)
    print("  ABLATION 2: MEMIT Retention Over Conversations")
    print("=" * 70)

    config = Config(args.config)
    clean_artifacts(config)
    orch = Orchestrator(config, disable_memit=False)

    # Disable auto-triggers
    orch.chat._sleep_callback = None
    orch.chat._nap_callback = None

    results = {
        "model": config.model["path"],
        "memit_lambda": config.get("memit.lambda_reg", 0.1),
        "memit_layers": config.get("memit.target_layers", []),
        "steps": [],
    }

    # ── Step 1: Inject batch A ──
    print_section("Step 1: Inject Batch A (5 facts)")

    t0 = time.time()
    edit_a = orch.memit_engine.inject_facts(BATCH_A)
    inject_a_time = time.time() - t0
    print(f"  Injected {len(BATCH_A)} facts in {inject_a_time:.1f}s")
    print(f"  MEMIT edits: {orch.get_status()['memit_edits']}")

    recall_a_imm, details_a_imm = test_recall(orch.backend, BATCH_A, "Batch A immediate recall")

    results["steps"].append({
        "step": "inject_batch_a",
        "batch_a_recall": recall_a_imm,
        "batch_a_details": details_a_imm,
        "memit_edits": orch.get_status()["memit_edits"],
    })

    # ── Step 2: Chat 20 unrelated turns ──
    print_section("Step 2: Chat 20 Filler Turns")

    for i, msg in enumerate(FILLER_MESSAGES):
        orch.chat.process_input(msg)
        if (i + 1) % 5 == 0:
            print(f"  Completed {i + 1}/{len(FILLER_MESSAGES)} filler turns")

    # ── Step 3: Measure batch A recall after filler ──
    print_section("Step 3: Batch A Recall After Filler")

    recall_a_post_filler, details_a_post = test_recall(orch.backend, BATCH_A, "Batch A after filler")

    results["steps"].append({
        "step": "after_filler",
        "batch_a_recall": recall_a_post_filler,
        "batch_a_details": details_a_post,
        "filler_turns": len(FILLER_MESSAGES),
    })

    # ── Step 4: Inject batch B (with null-space constraints from A) ──
    print_section("Step 4: Inject Batch B (5 more facts, null-space constraints)")

    t0 = time.time()
    edit_b = orch.memit_engine.inject_facts(BATCH_B)
    inject_b_time = time.time() - t0
    print(f"  Injected {len(BATCH_B)} facts in {inject_b_time:.1f}s")
    print(f"  Total MEMIT edits: {orch.get_status()['memit_edits']}")
    print(f"  Total MEMIT facts: {orch.get_status()['memit_facts']}")

    # ── Step 5: Measure recall of both A and B ──
    print_section("Step 5: Recall After Both Batches")

    recall_a_post_b, details_a_post_b = test_recall(orch.backend, BATCH_A, "Batch A after B injection")
    recall_b, details_b = test_recall(orch.backend, BATCH_B, "Batch B recall")

    all_facts = BATCH_A + BATCH_B
    recall_all, details_all = test_recall(orch.backend, all_facts, "Combined (A+B) recall")

    # Null-space effectiveness: how well did A survive B?
    null_space_retention = recall_a_post_b / recall_a_post_filler if recall_a_post_filler > 0 else 0

    results["steps"].append({
        "step": "after_batch_b",
        "batch_a_recall": recall_a_post_b,
        "batch_b_recall": recall_b,
        "combined_recall": recall_all,
        "null_space_retention": null_space_retention,
        "batch_a_details": details_a_post_b,
        "batch_b_details": details_b,
        "memit_edits": orch.get_status()["memit_edits"],
        "memit_facts": orch.get_status()["memit_facts"],
    })

    print(f"\n  Null-space retention (A survival): {null_space_retention:.2f}")

    # ── Step 6: Trigger nap ──
    print_section("Step 6: Trigger Nap")

    pre_nap_edits = orch.get_status()["memit_edits"]
    t0 = time.time()
    try:
        orch._on_nap_trigger("test")
        nap_time = time.time() - t0
        nap_ok = True
        print(f"  Nap completed in {nap_time:.1f}s")
    except Exception as e:
        nap_time = time.time() - t0
        nap_ok = False
        print(f"  Nap failed: {e}")

    post_nap_edits = orch.get_status()["memit_edits"]
    print(f"  MEMIT edits: {pre_nap_edits} → {post_nap_edits}")

    # ── Step 7: Measure post-nap recall ──
    print_section("Step 7: Post-Nap Recall")

    recall_a_post_nap, details_a_nap = test_recall(orch.backend, BATCH_A, "Batch A post-nap")
    recall_b_post_nap, details_b_nap = test_recall(orch.backend, BATCH_B, "Batch B post-nap")
    recall_all_post_nap, details_all_nap = test_recall(orch.backend, all_facts, "Combined post-nap")

    results["steps"].append({
        "step": "post_nap",
        "batch_a_recall": recall_a_post_nap,
        "batch_b_recall": recall_b_post_nap,
        "combined_recall": recall_all_post_nap,
        "nap_ok": nap_ok,
        "nap_time": nap_time,
        "edits_before_nap": pre_nap_edits,
        "edits_after_nap": post_nap_edits,
        "batch_a_details": details_a_nap,
        "batch_b_details": details_b_nap,
    })

    elapsed = time.time() - start_time
    results["total_elapsed_seconds"] = elapsed

    # ── Summary ──
    print_section("RETENTION SUMMARY")

    print(f"  {'Stage':<30} {'Batch A':>10} {'Batch B':>10} {'Combined':>10}")
    print(f"  {'-' * 30} {'-' * 10} {'-' * 10} {'-' * 10}")
    print(f"  {'After A injection':<30} {recall_a_imm:>10.2f} {'---':>10} {'---':>10}")
    print(f"  {'After 20 filler turns':<30} {recall_a_post_filler:>10.2f} {'---':>10} {'---':>10}")
    print(f"  {'After B injection':<30} {recall_a_post_b:>10.2f} {recall_b:>10.2f} {recall_all:>10.2f}")
    print(f"  {'After nap':<30} {recall_a_post_nap:>10.2f} {recall_b_post_nap:>10.2f} {recall_all_post_nap:>10.2f}")
    print()
    print(f"  Null-space retention: {null_space_retention:.2f}")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save
    output_path = Path(args.output) if args.output else Path("experiments/results/ablation_retention.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to {output_path}")


if __name__ == "__main__":
    main()
