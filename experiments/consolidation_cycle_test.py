"""Consolidation cycle test — finds optimal batch size and cumulative ceiling.

Tests two things:
  1. BATCH SIZE: How many facts per consolidation? (3, 5, 8, 10)
     Each batch = one FactBuffer → consolidate() → MEMIT inject_facts() call.
     Measures recall of just-injected batch + cumulative recall of all facts.

  2. CUMULATIVE CEILING: How many consolidation cycles before degradation?
     Repeats consolidation cycles (at a fixed batch size) and after each cycle
     measures: recall of all facts so far + PPL drift.
     Finds the point where cumulative recall drops below 0.70.

The combination tells us:
  - How to set max_buffer_size in config
  - When sleep pressure should trigger full sleep
  - What the practical lifetime budget is for wake-phase learning

Usage:
    python experiments/consolidation_cycle_test.py --config config.yaml
    python experiments/consolidation_cycle_test.py --config config.yaml --mode batch
    python experiments/consolidation_cycle_test.py --config config.yaml --mode cumulative --batch-size 5 --max-cycles 10
    python experiments/consolidation_cycle_test.py --config config.yaml --mode both
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

# Re-use synthetic fact generator from capacity test
from experiments.memit_capacity_test import generate_facts, _extract_relation, _extract_object


def _facts_to_triples(facts):
    """Convert generated fact dicts to FactTriple objects."""
    triples = []
    for fact in facts:
        stmt = fact["statement"]
        for rel_phrase in [" lives in ", " works as ", "'s favorite color is ",
                           "'s favorite food is ", " enjoys "]:
            if rel_phrase in stmt or rel_phrase.lower() in stmt.lower():
                subject = (stmt.split(rel_phrase)[0].split("'s ")[0]
                           if "'s " in rel_phrase
                           else stmt.split(rel_phrase)[0])
                break
        else:
            parts = stmt.split(" ")
            subject = parts[0] + " " + parts[1]
        triples.append(FactTriple(
            subject=subject.strip(),
            relation=_extract_relation(stmt),
            object=_extract_object(fact),
        ))
    return triples


def _reset_model(orchestrator, config):
    """Reset model to clean state (base weights, clear ledger)."""
    # Clear data dirs
    for dir_key in ["training", "replay_buffer", "conversations"]:
        d = Path(config.paths.get(dir_key, ""))
        if d.exists():
            shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
    memit_dir = Path(config.paths.get("memit_data", "data/memit"))
    if memit_dir.exists():
        shutil.rmtree(memit_dir)
        memit_dir.mkdir(parents=True, exist_ok=True)

    orchestrator.edit_ledger.clear_all()
    orchestrator.chat.reset_turn_count()
    orchestrator.context.reset(keep_summary=False)
    orchestrator.chat.set_sleep_callback(lambda t: None)
    orchestrator.chat.set_nap_callback(lambda t: None)

    # Reset model weights to base (revert all active MEMIT edits)
    orchestrator.memit_engine.revert_all_active()


def _measure_recall(orchestrator, all_facts):
    """Measure recall of all facts using raw completion prompts.

    Returns:
        (recall_rate, passed, failed, details)
    """
    orchestrator.context.reset(keep_summary=False)
    orchestrator.chat.reset_turn_count()
    orchestrator.chat.set_sleep_callback(lambda t: None)
    orchestrator.chat.set_nap_callback(lambda t: None)

    passed = 0
    failed = 0
    details = []

    for fact in all_facts:
        prompt = fact.get("raw_prompt", fact["question"])
        response = orchestrator.backend.generate(prompt, max_tokens=30)
        if response is None:
            response = ""

        resp_lower = response.lower()
        found = sum(1 for kw in fact["expected"] if kw.lower() in resp_lower)

        if found == len(fact["expected"]):
            passed += 1
        else:
            failed += 1
            details.append({
                "prompt": prompt,
                "expected": fact["expected"],
                "got": response[:80],
            })

    total = len(all_facts)
    recall = passed / total if total else 0
    return recall, passed, failed, details


def _measure_ppl(orchestrator):
    """Measure model perplexity on reference text."""
    try:
        return orchestrator.health_monitor.measure_perplexity()
    except Exception:
        return None


# ─── Test 1: Batch Size ─────────────────────────────────────────────────

def run_batch_size_test(config_path, batch_sizes=None, seed=42):
    """Test different consolidation batch sizes to find the optimal one.

    For each batch size N:
      1. Reset model to base weights
      2. Inject N facts via consolidate()
      3. Measure recall of those N facts
      4. Record recall rate

    This tells us: "What's the best batch size for a single consolidation?"
    """
    if batch_sizes is None:
        batch_sizes = [1, 2, 3, 5, 8, 10, 15]

    max_needed = max(batch_sizes)
    all_facts = generate_facts(max_needed, seed=seed)

    config = Config(config_path)
    model_name = config.model["path"]

    print("=" * 70)
    print("  BATCH SIZE TEST")
    print("=" * 70)
    print(f"  Model:       {model_name}")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  MEMIT layers: {config.get('memit.target_layers', [])}")
    print("=" * 70)
    print()

    print("[INIT] Loading model...")
    orchestrator = Orchestrator(config)

    # Measure baseline PPL
    _reset_model(orchestrator, config)
    baseline_ppl = _measure_ppl(orchestrator)
    print(f"  Baseline PPL: {baseline_ppl}")
    print()

    results = {
        "test": "batch_size",
        "model": model_name,
        "baseline_ppl": baseline_ppl,
        "batch_sizes": [],
    }

    for batch_size in batch_sizes:
        print(f"--- Batch size: {batch_size} ---")
        _reset_model(orchestrator, config)

        batch_facts = all_facts[:batch_size]
        triples = _facts_to_triples(batch_facts)

        # Inject via MEMIT (same as FactBuffer.consolidate does)
        t0 = time.time()
        edit = orchestrator.memit_engine.inject_facts(triples)
        inject_time = time.time() - t0

        if not edit:
            print(f"  FAILED to inject {batch_size} facts")
            results["batch_sizes"].append({
                "batch_size": batch_size,
                "inject_ok": False,
            })
            continue

        # Measure recall
        recall, passed, failed, details = _measure_recall(orchestrator, batch_facts)

        # Measure PPL after injection
        post_ppl = _measure_ppl(orchestrator)
        ppl_drift = ((post_ppl / baseline_ppl) - 1.0) if (baseline_ppl and post_ppl) else None

        entry = {
            "batch_size": batch_size,
            "inject_ok": True,
            "inject_time_s": round(inject_time, 2),
            "recall": round(recall, 3),
            "passed": passed,
            "failed": failed,
            "ppl_after": round(post_ppl, 3) if post_ppl else None,
            "ppl_drift": round(ppl_drift, 4) if ppl_drift is not None else None,
        }
        results["batch_sizes"].append(entry)

        print(f"  Recall: {recall:.2f} ({passed}/{batch_size})")
        print(f"  PPL: {post_ppl:.3f} (drift: {ppl_drift:+.4f})" if ppl_drift is not None else "  PPL: N/A")
        print(f"  Inject time: {inject_time:.2f}s")

        if details:
            for d in details[:2]:
                print(f"    FAIL: {d['prompt']} → expected {d['expected']}, got: {d['got']}")
        print()

    # Summary
    print("=" * 70)
    print("  BATCH SIZE SUMMARY")
    print("=" * 70)
    print(f"  {'Batch':>6} {'Recall':>8} {'PPL Drift':>10} {'Time':>8}")
    print(f"  {'-'*6} {'-'*8} {'-'*10} {'-'*8}")
    for entry in results["batch_sizes"]:
        if entry["inject_ok"]:
            drift_str = f"{entry['ppl_drift']:+.4f}" if entry["ppl_drift"] is not None else "N/A"
            print(f"  {entry['batch_size']:>6} {entry['recall']:>8.2f} {drift_str:>10} {entry['inject_time_s']:>7.1f}s")
        else:
            print(f"  {entry['batch_size']:>6}    FAIL")
    print("=" * 70)

    return results


# ─── Test 2: Cumulative Consolidation Ceiling ────────────────────────────

def run_cumulative_test(config_path, batch_size=5, max_cycles=10,
                        max_total_facts=60, seed=42):
    """Test how many consolidation cycles the model can sustain.

    Simulates the actual wake-phase loop:
      1. Buffer `batch_size` facts
      2. Consolidate (inject_facts)
      3. Measure recall of ALL facts so far + PPL
      4. Repeat until recall < 0.50 or max_cycles reached

    This tells us:
      - How many consolidation cycles before needing sleep
      - Total fact capacity across cycles (cumulative ceiling)
      - PPL drift trajectory
    """
    total_facts_needed = min(batch_size * max_cycles, max_total_facts)
    all_facts = generate_facts(total_facts_needed, seed=seed)

    config = Config(config_path)
    model_name = config.model["path"]

    print("=" * 70)
    print("  CUMULATIVE CONSOLIDATION TEST")
    print("=" * 70)
    print(f"  Model:       {model_name}")
    print(f"  Batch size:  {batch_size} facts/cycle")
    print(f"  Max cycles:  {max_cycles}")
    print(f"  Max facts:   {total_facts_needed}")
    print(f"  MEMIT layers: {config.get('memit.target_layers', [])}")
    print("=" * 70)
    print()

    print("[INIT] Loading model...")
    orchestrator = Orchestrator(config)
    _reset_model(orchestrator, config)

    # Baseline
    baseline_ppl = _measure_ppl(orchestrator)
    print(f"  Baseline PPL: {baseline_ppl}")
    print()

    results = {
        "test": "cumulative",
        "model": model_name,
        "batch_size": batch_size,
        "baseline_ppl": baseline_ppl,
        "cycles": [],
    }

    injected_facts = []
    fact_idx = 0

    for cycle in range(1, max_cycles + 1):
        batch_end = min(fact_idx + batch_size, total_facts_needed)
        if fact_idx >= total_facts_needed:
            print(f"  [Cycle {cycle}] No more facts to inject. Stopping.")
            break

        batch_facts = all_facts[fact_idx:batch_end]
        triples = _facts_to_triples(batch_facts)

        print(f"--- Cycle {cycle}: injecting facts {fact_idx+1}-{batch_end} ---")

        t0 = time.time()
        edit = orchestrator.memit_engine.inject_facts(triples)
        inject_time = time.time() - t0

        if not edit:
            print(f"  FAILED to inject cycle {cycle}")
            results["cycles"].append({
                "cycle": cycle,
                "inject_ok": False,
                "total_facts": len(injected_facts),
            })
            break

        injected_facts.extend(batch_facts)
        fact_idx = batch_end
        total = len(injected_facts)

        # Measure recall of ALL facts
        recall, passed, failed, details = _measure_recall(orchestrator, injected_facts)

        # Measure recall of JUST this batch (batch quality)
        batch_recall, batch_passed, _, _ = _measure_recall(orchestrator, batch_facts)

        # Measure PPL
        post_ppl = _measure_ppl(orchestrator)
        ppl_drift = ((post_ppl / baseline_ppl) - 1.0) if (baseline_ppl and post_ppl) else None

        # Sleep pressure
        status = orchestrator.get_status()
        pressure = status.get("sleep_pressure", 0)

        entry = {
            "cycle": cycle,
            "inject_ok": True,
            "inject_time_s": round(inject_time, 2),
            "batch_size_actual": len(batch_facts),
            "total_facts": total,
            "total_edits": status.get("memit_edits", 0),
            "cumulative_recall": round(recall, 3),
            "cumulative_passed": passed,
            "cumulative_failed": failed,
            "batch_recall": round(batch_recall, 3),
            "batch_passed": batch_passed,
            "ppl_after": round(post_ppl, 3) if post_ppl else None,
            "ppl_drift": round(ppl_drift, 4) if ppl_drift is not None else None,
            "sleep_pressure": round(pressure, 3),
        }
        results["cycles"].append(entry)

        print(f"  Total facts: {total} | Cumulative recall: {recall:.2f} "
              f"({passed}/{total}) | Batch recall: {batch_recall:.2f}")
        drift_str = f"{ppl_drift:+.4f}" if ppl_drift is not None else "N/A"
        print(f"  PPL: {post_ppl:.3f} (drift: {drift_str}) | "
              f"Pressure: {pressure:.3f} | Time: {inject_time:.1f}s")

        if details:
            for d in details[:3]:
                print(f"    FAIL: {d['prompt']} → expected {d['expected']}, got: {d['got']}")
            if len(details) > 3:
                print(f"    ... and {len(details) - 3} more failures")
        print()

        # Stop if cumulative recall is too low
        if total >= batch_size * 2 and recall < 0.50:
            print(f"  STOPPING — cumulative recall dropped to {recall:.2f}")
            break

    # Summary
    print("=" * 70)
    print("  CUMULATIVE CONSOLIDATION SUMMARY")
    print("=" * 70)
    model_short = model_name.split("/")[-1]
    print(f"  Model: {model_short}, batch_size={batch_size}")
    print()
    print(f"  {'Cycle':>6} {'Facts':>6} {'Edits':>6} {'CumRecall':>10} "
          f"{'BatchRecall':>12} {'PPL Drift':>10} {'Pressure':>10}")
    print(f"  {'-'*6} {'-'*6} {'-'*6} {'-'*10} {'-'*12} {'-'*10} {'-'*10}")

    for c in results["cycles"]:
        if c["inject_ok"]:
            drift = f"{c['ppl_drift']:+.4f}" if c["ppl_drift"] is not None else "N/A"
            print(f"  {c['cycle']:>6} {c['total_facts']:>6} {c['total_edits']:>6} "
                  f"{c['cumulative_recall']:>10.2f} {c['batch_recall']:>12.2f} "
                  f"{drift:>10} {c['sleep_pressure']:>10.3f}")
        else:
            print(f"  {c['cycle']:>6} {c['total_facts']:>6}   FAIL")

    # Find degradation points
    peak_recall = 0
    peak_facts = 0
    degrade_70 = None
    degrade_50 = None
    for c in results["cycles"]:
        if not c["inject_ok"]:
            continue
        if c["cumulative_recall"] >= peak_recall:
            peak_recall = c["cumulative_recall"]
            peak_facts = c["total_facts"]
        if degrade_70 is None and c["cumulative_recall"] < 0.70 and c["total_facts"] > batch_size:
            degrade_70 = c["total_facts"]
        if degrade_50 is None and c["cumulative_recall"] < 0.50 and c["total_facts"] > batch_size:
            degrade_50 = c["total_facts"]

    results["peak_recall"] = peak_recall
    results["peak_facts"] = peak_facts
    results["degrade_70_at"] = degrade_70
    results["degrade_50_at"] = degrade_50

    print()
    print(f"  Peak recall: {peak_recall:.2f} at {peak_facts} facts")
    if degrade_70:
        print(f"  Recall < 0.70 at: {degrade_70} facts (CONSOLIDATE HERE → nap/sleep)")
    if degrade_50:
        print(f"  Recall < 0.50 at: {degrade_50} facts (HARD CEILING)")
    else:
        print(f"  Recall stayed >= 0.50 through all {len(results['cycles'])} cycles")

    # Recommendation
    print()
    if degrade_70:
        safe_cycles = (degrade_70 // batch_size) - 1
        print(f"  RECOMMENDATION: max_buffer_size={batch_size}, "
              f"sleep after {safe_cycles} consolidations ({safe_cycles * batch_size} facts)")
    else:
        total_tested = results["cycles"][-1]["total_facts"] if results["cycles"] else 0
        print(f"  RECOMMENDATION: model sustained {total_tested} facts — "
              f"test with more cycles for ceiling")

    print("=" * 70)

    return results


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Consolidation Cycle Test")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML")
    parser.add_argument("--mode", choices=["batch", "cumulative", "both"],
                        default="both",
                        help="Which test to run (default: both)")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Facts per consolidation cycle (cumulative test, default: 5)")
    parser.add_argument("--batch-sizes", type=str, default=None,
                        help="Comma-separated batch sizes for batch test "
                             "(default: 1,2,3,5,8,10,15)")
    parser.add_argument("--max-cycles", type=int, default=10,
                        help="Max consolidation cycles (cumulative test, default: 10)")
    parser.add_argument("--max-total-facts", type=int, default=60,
                        help="Max total facts across all cycles (default: 60)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for fact generation")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    all_results = {}

    if args.mode in ("batch", "both"):
        batch_sizes = (
            [int(x) for x in args.batch_sizes.split(",")]
            if args.batch_sizes
            else None
        )
        all_results["batch"] = run_batch_size_test(
            config_path=args.config,
            batch_sizes=batch_sizes,
            seed=args.seed,
        )
        print()

    if args.mode in ("cumulative", "both"):
        all_results["cumulative"] = run_cumulative_test(
            config_path=args.config,
            batch_size=args.batch_size,
            max_cycles=args.max_cycles,
            max_total_facts=args.max_total_facts,
            seed=args.seed,
        )

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        model_name = Config(args.config).model["path"].split("/")[-1]
        output_path = (Path("experiments/results") /
                       f"consolidation_cycle_{model_name}.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        if isinstance(obj, float) and obj != obj:
            return None
        return obj

    with open(output_path, "w") as f:
        json.dump(clean(all_results), f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
