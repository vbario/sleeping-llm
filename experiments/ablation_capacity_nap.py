"""Ablation 4: Capacity Ceiling — MEMIT-only vs MEMIT+Nap.

Compares the effective capacity (number of facts at ≥0.7 recall) between:
  1. MEMIT-only: inject facts in batches of 5, never nap
  2. MEMIT+Nap: inject facts in batches of 5, nap after every 10 facts

Continues until recall drops below 0.5 or max_facts is reached.

Usage:
    python experiments/ablation_capacity_nap.py --config experiments/configs/8b_memit.yaml
    python experiments/ablation_capacity_nap.py --config experiments/configs/8b_memit.yaml --max-facts 80
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


# Reuse the fact generator from memit_capacity_test
from experiments.memit_capacity_test import generate_facts, _extract_relation, _extract_object


def facts_to_triples(fact_dicts):
    """Convert fact dicts from generate_facts() to FactTriples."""
    triples = []
    for fact in fact_dicts:
        stmt = fact["statement"]
        # Parse subject
        for rel_phrase in [" lives in ", " works as ", "'s favorite color is ",
                           "'s favorite food is ", " enjoys "]:
            if rel_phrase in stmt or rel_phrase.lower() in stmt.lower():
                subject = stmt.split(rel_phrase)[0].split("'s ")[0] if "'s " in rel_phrase else stmt.split(rel_phrase)[0]
                break
        else:
            subject = stmt.split(" ")[0] + " " + stmt.split(" ")[1]
        triple = FactTriple(
            subject=subject.strip(),
            relation=_extract_relation(stmt),
            object=_extract_object(fact),
        )
        triples.append(triple)
    return triples


def test_all_recall(backend, all_fact_dicts):
    """Test raw completion recall on all injected facts."""
    passed = 0
    for fact in all_fact_dicts:
        prompt = fact.get("raw_prompt", fact["question"])
        response = backend.generate(prompt, max_tokens=30, temperature=0.1)
        if response is None:
            response = ""
        found = any(kw.lower() in response.lower() for kw in fact["expected"])
        if found:
            passed += 1
    recall = passed / len(all_fact_dicts) if all_fact_dicts else 0
    return recall, passed


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


def run_memit_only(config_path, all_facts, batch_size=5, max_facts=60):
    """Condition 1: MEMIT-only, no nap."""
    print(f"\n{'=' * 70}")
    print(f"  CONDITION 1: MEMIT-only (no nap)")
    print(f"{'=' * 70}")

    config = Config(config_path)
    clean_artifacts(config)
    orch = Orchestrator(config, disable_memit=False)
    orch.chat._sleep_callback = None
    orch.chat._nap_callback = None

    checkpoints = []
    injected = []
    fact_idx = 0

    while fact_idx < min(max_facts, len(all_facts)):
        batch_end = min(fact_idx + batch_size, max_facts, len(all_facts))
        batch = all_facts[fact_idx:batch_end]
        triples = facts_to_triples(batch)

        print(f"\n  --- Injecting facts {fact_idx + 1}-{batch_end} ---")
        edit = orch.memit_engine.inject_facts(triples)
        injected.extend(batch)
        fact_idx = batch_end

        # Reset context so recall depends only on weights
        orch.context.reset(keep_summary=False)

        recall, passed = test_all_recall(orch.backend, injected)
        print(f"  Total facts: {len(injected)} | Recall: {recall:.2f} ({passed}/{len(injected)})")

        checkpoints.append({
            "total_facts": len(injected),
            "recall": round(recall, 3),
            "passed": passed,
            "memit_edits": orch.get_status()["memit_edits"],
        })

        if len(injected) >= batch_size * 2 and recall < 0.5:
            print(f"  STOPPING — recall dropped to {recall:.2f}")
            break

    return checkpoints


def run_memit_plus_nap(config_path, all_facts, batch_size=5, nap_every=10, max_facts=60):
    """Condition 2: MEMIT + nap after every nap_every facts."""
    print(f"\n{'=' * 70}")
    print(f"  CONDITION 2: MEMIT + Nap (every {nap_every} facts)")
    print(f"{'=' * 70}")

    config = Config(config_path)
    clean_artifacts(config)
    orch = Orchestrator(config, disable_memit=False)
    orch.chat._sleep_callback = None
    orch.chat._nap_callback = None

    checkpoints = []
    injected = []
    fact_idx = 0
    facts_since_nap = 0
    nap_count = 0

    while fact_idx < min(max_facts, len(all_facts)):
        batch_end = min(fact_idx + batch_size, max_facts, len(all_facts))
        batch = all_facts[fact_idx:batch_end]
        triples = facts_to_triples(batch)

        print(f"\n  --- Injecting facts {fact_idx + 1}-{batch_end} ---")
        edit = orch.memit_engine.inject_facts(triples)
        injected.extend(batch)
        fact_idx = batch_end
        facts_since_nap += len(batch)

        # Check if nap is due
        if facts_since_nap >= nap_every:
            print(f"  Triggering nap (facts since last nap: {facts_since_nap})...")
            t0 = time.time()
            try:
                orch._on_nap_trigger("test")
                nap_time = time.time() - t0
                nap_count += 1
                facts_since_nap = 0
                print(f"  Nap #{nap_count} completed in {nap_time:.1f}s")
                print(f"  Post-nap MEMIT edits: {orch.get_status()['memit_edits']}")
            except Exception as e:
                print(f"  Nap failed: {e}")

        # Reset context for clean recall test
        orch.context.reset(keep_summary=False)

        recall, passed = test_all_recall(orch.backend, injected)
        print(f"  Total facts: {len(injected)} | Recall: {recall:.2f} ({passed}/{len(injected)})")

        checkpoints.append({
            "total_facts": len(injected),
            "recall": round(recall, 3),
            "passed": passed,
            "memit_edits": orch.get_status()["memit_edits"],
            "nap_count": nap_count,
        })

        if len(injected) >= batch_size * 2 and recall < 0.5:
            print(f"  STOPPING — recall dropped to {recall:.2f}")
            break

    return checkpoints


def main():
    parser = argparse.ArgumentParser(description="Ablation 4: Capacity Ceiling — MEMIT vs MEMIT+Nap")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--max-facts", type=int, default=60, help="Max facts to inject (default: 60)")
    parser.add_argument("--batch-size", type=int, default=5, help="Facts per batch (default: 5)")
    parser.add_argument("--nap-every", type=int, default=10, help="Nap after this many new facts (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    start_time = time.time()

    print("=" * 70)
    print("  ABLATION 4: Capacity Ceiling — MEMIT vs MEMIT+Nap")
    print("=" * 70)
    print(f"  Max facts: {args.max_facts}, Batch size: {args.batch_size}, Nap every: {args.nap_every}")

    # Generate facts (same set for both conditions)
    all_facts = generate_facts(args.max_facts, seed=args.seed)

    memit_only_results = run_memit_only(args.config, all_facts, args.batch_size, args.max_facts)
    memit_nap_results = run_memit_plus_nap(args.config, all_facts, args.batch_size, args.nap_every, args.max_facts)

    elapsed = time.time() - start_time

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(f"  CAPACITY COMPARISON SUMMARY")
    print(f"{'=' * 70}")

    # Find capacity at 0.7 threshold
    def capacity_at_threshold(checkpoints, threshold=0.7):
        best = 0
        for cp in checkpoints:
            if cp["recall"] >= threshold:
                best = cp["total_facts"]
        return best

    cap_memit = capacity_at_threshold(memit_only_results)
    cap_nap = capacity_at_threshold(memit_nap_results)

    print(f"\n  MEMIT-only capacity (at 0.7 recall): {cap_memit} facts")
    print(f"  MEMIT+Nap capacity (at 0.7 recall):  {cap_nap} facts")
    if cap_memit > 0:
        print(f"  Improvement: {cap_nap - cap_memit} facts ({(cap_nap/cap_memit - 1)*100:+.0f}%)")

    print(f"\n  {'Facts':>6} {'MEMIT-only':>12} {'MEMIT+Nap':>12}")
    print(f"  {'-' * 6} {'-' * 12} {'-' * 12}")

    max_rows = max(len(memit_only_results), len(memit_nap_results))
    for i in range(max_rows):
        m = memit_only_results[i] if i < len(memit_only_results) else None
        n = memit_nap_results[i] if i < len(memit_nap_results) else None

        facts_m = str(m["total_facts"]) if m else ""
        recall_m = f"{m['recall']:.2f}" if m else ""
        facts_n = str(n["total_facts"]) if n else ""
        recall_n = f"{n['recall']:.2f}" if n else ""

        # Use whichever has facts
        facts_col = facts_m or facts_n
        print(f"  {facts_col:>6} {recall_m:>12} {recall_n:>12}")

    print(f"\n  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save
    results = {
        "config": args.config,
        "max_facts": args.max_facts,
        "batch_size": args.batch_size,
        "nap_every": args.nap_every,
        "memit_only": memit_only_results,
        "memit_plus_nap": memit_nap_results,
        "capacity_memit_only": cap_memit,
        "capacity_memit_plus_nap": cap_nap,
        "total_elapsed_seconds": elapsed,
    }

    output_path = Path(args.output) if args.output else Path("experiments/results/ablation_capacity_nap.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to {output_path}")


if __name__ == "__main__":
    main()
