"""V7 Maintenance Test — Prove sleep detects and fixes degraded MEMIT facts.

Simulates human-like memory interference during wake:
  - Wake injection has NO null-space constraints (new facts can overwrite old ones)
  - Sleep re-injection uses full null-space constraints (consolidation protects all facts)

This mirrors how human memory works:
  - During wake, new learning causes retroactive interference with old memories
  - During sleep, hippocampal replay reconsolidates degraded memories

Uses 8B with 8 layers [12-19]. Without null-space protection, interference causes
degradation after ~10-15 facts, triggering the maintenance path.

Three steps:
  Step 1: Inject facts WITHOUT null-space constraints (simulates wake interference)
  Step 2: Nap — detect degradation
  Step 3: Sleep — fix degradation via revert + re-inject WITH constraints (consolidation)

Usage:
    python experiments/v7_maintenance_test.py --config experiments/configs/8b_v7.yaml
"""

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.orchestrator import Orchestrator
from src.memory.memit import FactTriple


# ── Reference texts for perplexity ──

REFERENCE_TEXTS = [
    (
        "The theory of general relativity, proposed by Albert Einstein in 1915, "
        "describes gravity as the warping of spacetime by mass and energy. "
        "Photosynthesis converts carbon dioxide and water into glucose and oxygen "
        "using energy from sunlight in the chloroplasts of plant cells."
    ),
    (
        "The French Revolution of 1789 overthrew the monarchy and established the "
        "First Republic, fundamentally transforming French society and politics. "
        "DNA stores genetic information in a double helix structure, with base pairs "
        "of adenine-thymine and guanine-cytosine connected by hydrogen bonds."
    ),
    (
        "The Amazon rainforest covers much of South America and contains the greatest "
        "biodiversity of any ecosystem on Earth. Classical music evolved through the "
        "Baroque, Classical, and Romantic periods, with composers like Bach, Mozart, "
        "and Beethoven shaping Western musical tradition."
    ),
]


# ── Helpers (reused from v7_comprehensive_test.py) ──

def measure_perplexity(backend):
    ppls = [backend.compute_perplexity(text) for text in REFERENCE_TEXTS]
    return sum(ppls) / len(ppls)


def test_recall(backend, facts):
    """Test raw completion recall. Returns (fraction, per-fact details)."""
    details = []
    passed = 0
    for fact in facts:
        prompt = fact.to_prompt()
        response = backend.generate(prompt, max_tokens=30, temperature=0.1)
        if response is None:
            response = ""
        hit = fact.object.lower() in response.lower()
        if hit:
            passed += 1
        details.append({
            "subject": fact.subject,
            "relation": fact.relation,
            "expected": fact.object,
            "response": response.strip()[:80],
            "hit": hit,
        })
    fraction = passed / len(facts) if facts else 0
    return fraction, details


def load_fact_pool(path="experiments/data/fact_pool_500.json"):
    pool_path = project_root / path
    with open(pool_path) as f:
        raw = json.load(f)
    return [FactTriple(subject=r["subject"], relation=r["relation"], object=r["object"]) for r in raw]


def clean_artifacts_v7(config):
    for key in ("conversations", "memit_data"):
        d = Path(config.paths[key])
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)


def cleanup_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


def fresh_orchestrator(config):
    cleanup_gpu()
    orch = Orchestrator(config, disable_memit=False)
    orch.chat._sleep_callback = None
    orch.chat._nap_callback = None
    return orch


def destroy_orchestrator(orch):
    if hasattr(orch, 'backend') and hasattr(orch.backend, 'model'):
        del orch.backend.model
    if hasattr(orch, 'backend') and hasattr(orch.backend, 'tokenizer'):
        del orch.backend.tokenizer
    del orch
    cleanup_gpu()


def trigger_sleep(orch):
    orch.sleep_cycle_count += 1
    cycle_id = f"{orch.sleep_cycle_count:04d}"
    result = orch.full_sleep_controller.execute_sleep(
        cycle_id, "full", orch._gather_new_messages,
    )
    refreshed = result.get("facts_refreshed", 0)
    pruned = result.get("facts_pruned", 0)
    orch.health_monitor.record_sleep("full", facts_refreshed=refreshed, facts_pruned=pruned)
    if orch.context.recent_messages:
        orch.context.compact()
    orch.chat.reset_turn_count()
    orch.context.reset(keep_summary=True)
    return result


def trigger_nap(orch):
    orch.sleep_cycle_count += 1
    cycle_id = f"nap_{orch.sleep_cycle_count:04d}"
    return orch.nap_controller.execute_nap(cycle_id)


def teach_facts(orch, facts):
    for fact in facts:
        msg = f"{fact.subject} {fact.relation} {fact.object}."
        orch.chat.process_input(msg)


def ppl_delta_pct(baseline, current):
    if baseline == 0:
        return 0
    return (current - baseline) / baseline


def inject_without_constraints(engine, facts):
    """Inject facts with null-space constraints disabled (wake-mode interference).

    Temporarily hides existing active edits so _compute_keys() won't add
    previous-edit constraint keys. The new edit is still tracked in _active_edits
    after injection, so sleep can later audit and refresh it.
    """
    stashed = list(engine._active_edits)
    engine._active_edits = []
    try:
        result = engine.inject_facts(facts)
    finally:
        # Restore previous edits + any new edit that inject_facts appended
        new_edits = list(engine._active_edits)
        engine._active_edits = stashed + new_edits
    return result


# ── Step 1: Inject until degradation ──

def step_injection(orch, fact_pool, degraded_threshold=0.5, max_facts=30, batch_size=5):
    """Inject facts WITHOUT null-space constraints (wake-mode interference)."""
    print(f"\n{'=' * 70}")
    print(f"  Step 1: Inject until degradation (NO null-space constraints)")
    print(f"  (batch_size={batch_size}, max={max_facts}, threshold={degraded_threshold})")
    print(f"{'=' * 70}")

    baseline_ppl = measure_perplexity(orch.backend)
    print(f"  Baseline PPL: {baseline_ppl:.2f}")

    all_injected = []
    trajectory = []
    batch_num = 0

    while len(all_injected) < max_facts:
        batch_start = len(all_injected)
        batch = fact_pool[batch_start:batch_start + batch_size]
        if not batch:
            break
        batch_num += 1

        print(f"\n  Batch {batch_num}: injecting facts {batch_start+1}-{batch_start+len(batch)} (no constraints)...")
        inject_without_constraints(orch.memit_engine, batch)
        all_injected.extend(batch)

        # Teach so sleep has session data
        teach_facts(orch, batch)

        # Measure recall on ALL injected facts
        recall, details = test_recall(orch.backend, all_injected)
        degraded_facts = [d for d in details if not d["hit"]]
        ppl = measure_perplexity(orch.backend)

        print(f"  After {len(all_injected)} facts: recall={recall:.2f}, "
              f"degraded={len(degraded_facts)}, PPL={ppl:.2f}")

        trajectory.append({
            "batch": batch_num,
            "total_facts": len(all_injected),
            "recall": round(recall, 3),
            "degraded_count": len(degraded_facts),
            "ppl": round(ppl, 3),
        })

        # Stop once we have enough degraded facts
        if len(degraded_facts) >= 3:
            print(f"  Found {len(degraded_facts)} degraded facts — stopping injection.")
            break

        cleanup_gpu()

    final_recall, final_details = test_recall(orch.backend, all_injected)
    degraded = [d for d in final_details if not d["hit"]]

    result = {
        "total_facts": len(all_injected),
        "batches": batch_num,
        "final_recall": round(final_recall, 3),
        "degraded_count": len(degraded),
        "degraded_facts": [
            {"subject": d["subject"], "relation": d["relation"], "expected": d["expected"]}
            for d in degraded
        ],
        "baseline_ppl": round(baseline_ppl, 3),
        "trajectory": trajectory,
    }

    print(f"\n  Step 1 done: {len(all_injected)} facts, recall={final_recall:.2f}, "
          f"{len(degraded)} degraded")
    return result, all_injected


# ── Step 2: Nap → detect degradation ──

def step_nap(orch):
    """Run nap audit and check for degradation detection."""
    print(f"\n{'=' * 70}")
    print(f"  Step 2: Nap — detect degradation")
    print(f"{'=' * 70}")

    nap_result = trigger_nap(orch)
    audited = nap_result.get("audited", 0)
    degraded = nap_result.get("degraded", 0)
    nap_details = nap_result.get("results", [])

    degraded_facts = [r for r in nap_details if not r.get("healthy", True)]

    verdict = "PASS" if degraded > 0 else "FAIL (no degradation detected by nap)"
    print(f"  Nap audited {audited} facts: {degraded} degraded")
    for r in degraded_facts:
        print(f"    DEGRADED: {r['fact']} (recall {r['recall_rate']:.0%})")
    print(f"  Step 2 verdict: {verdict}")

    return {
        "verdict": verdict,
        "audited": audited,
        "degraded": degraded,
        "degraded_facts": degraded_facts,
    }


# ── Step 3: Sleep → fix degradation ──

def step_sleep(orch, all_injected):
    """Run full sleep, measure recall improvement on degraded facts."""
    print(f"\n{'=' * 70}")
    print(f"  Step 3: Sleep — fix degradation")
    print(f"{'=' * 70}")

    # Pre-sleep measurements
    recall_pre, pre_details = test_recall(orch.backend, all_injected)
    ppl_pre = measure_perplexity(orch.backend)
    degraded_before = [d for d in pre_details if not d["hit"]]
    print(f"  Pre-sleep: recall={recall_pre:.2f}, degraded={len(degraded_before)}, PPL={ppl_pre:.2f}")

    # Run sleep
    print(f"  Triggering full sleep...")
    try:
        sleep_result = trigger_sleep(orch)
        facts_refreshed = sleep_result.get("facts_refreshed", 0)
        facts_pruned = sleep_result.get("facts_pruned", 0)
        print(f"  Sleep result: refreshed={facts_refreshed}, pruned={facts_pruned}")
    except Exception as e:
        print(f"  Sleep FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {
            "verdict": "FAIL (sleep crashed)",
            "error": str(e),
            "recall_pre": round(recall_pre, 3),
        }

    # Post-sleep measurements
    recall_post, post_details = test_recall(orch.backend, all_injected)
    ppl_post = measure_perplexity(orch.backend)
    degraded_after = [d for d in post_details if not d["hit"]]
    print(f"  Post-sleep: recall={recall_post:.2f}, degraded={len(degraded_after)}, PPL={ppl_post:.2f}")

    # Check previously-degraded facts specifically
    degraded_subjects = {d["subject"] for d in degraded_before}
    fixed = []
    still_broken = []
    for d in post_details:
        if d["subject"] in degraded_subjects:
            if d["hit"]:
                fixed.append(d["subject"])
            else:
                still_broken.append(d["subject"])

    print(f"  Previously-degraded facts: {len(fixed)} fixed, {len(still_broken)} still broken")

    passed = facts_refreshed > 0 and recall_post >= recall_pre
    verdict = "PASS" if passed else "FAIL"
    print(f"  Step 3 verdict: {verdict}")

    return {
        "verdict": verdict,
        "facts_refreshed": facts_refreshed,
        "facts_pruned": facts_pruned,
        "recall_pre": round(recall_pre, 3),
        "recall_post": round(recall_post, 3),
        "degraded_before": [
            {"subject": d["subject"], "relation": d["relation"], "expected": d["expected"]}
            for d in degraded_before
        ],
        "degraded_after": [
            {"subject": d["subject"], "relation": d["relation"], "expected": d["expected"]}
            for d in degraded_after
        ],
        "fixed": fixed,
        "still_broken": still_broken,
        "ppl_before": round(ppl_pre, 3),
        "ppl_after": round(ppl_post, 3),
    }


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="V7 Maintenance Test")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--output", type=str, default=None, help="Override output JSON path")
    parser.add_argument("--max-facts", type=int, default=30, help="Max facts to inject")
    parser.add_argument("--batch-size", type=int, default=5, help="Facts per injection batch")
    args = parser.parse_args()

    config = Config(args.config)
    model_name = config.model["path"]
    layers = config.get("memit.target_layers", [])
    degraded_threshold = config.get("sleep.maintenance.degraded_threshold", 0.5)

    print("=" * 70)
    print("  V7 MAINTENANCE TEST")
    print("=" * 70)
    print(f"  Model:     {model_name}")
    print(f"  Backend:   {config.model.get('backend', 'mlx')}")
    print(f"  Layers:    {layers} ({len(layers)} layers)")
    print(f"  Threshold: {degraded_threshold}")
    print(f"  Max facts: {args.max_facts}")
    print("=" * 70)

    fact_pool = load_fact_pool()
    print(f"  Loaded {len(fact_pool)} facts from pool")

    total_start = time.time()
    clean_artifacts_v7(config)
    orch = fresh_orchestrator(config)

    results = {
        "config": {
            "model": model_name,
            "backend": config.model.get("backend", "mlx"),
            "layers": layers,
            "num_layers": len(layers),
            "degraded_threshold": degraded_threshold,
            "max_facts": args.max_facts,
            "batch_size": args.batch_size,
        },
    }

    # Step 1: Inject until degradation
    step1_result, all_injected = step_injection(
        orch, fact_pool,
        degraded_threshold=degraded_threshold,
        max_facts=args.max_facts,
        batch_size=args.batch_size,
    )
    results["step_1_injection"] = step1_result

    if step1_result["degraded_count"] == 0:
        print(f"\n  NO DEGRADATION after {step1_result['total_facts']} facts — "
              f"cannot test maintenance. Try more facts or fewer layers.")
        results["step_2_nap"] = {"verdict": "SKIP (no degradation)"}
        results["step_3_sleep"] = {"verdict": "SKIP (no degradation)"}
        results["overall_verdict"] = "SKIP (no degradation observed)"
    else:
        # Step 2: Nap
        step2_result = step_nap(orch)
        results["step_2_nap"] = step2_result

        # Step 3: Sleep
        step3_result = step_sleep(orch, all_injected)
        results["step_3_sleep"] = step3_result

        # Overall
        nap_pass = step2_result["verdict"].startswith("PASS")
        sleep_pass = step3_result["verdict"].startswith("PASS")
        if nap_pass and sleep_pass:
            results["overall_verdict"] = "PASS"
        elif nap_pass:
            results["overall_verdict"] = f"PARTIAL (nap OK, sleep {step3_result['verdict']})"
        else:
            results["overall_verdict"] = f"FAIL (nap: {step2_result['verdict']})"

    total_elapsed = time.time() - total_start
    results["total_elapsed_seconds"] = round(total_elapsed, 1)

    destroy_orchestrator(orch)

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"  OVERALL: {results['overall_verdict']}")
    print(f"  Time: {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min)")
    print(f"{'=' * 70}")

    # Save
    if args.output:
        output_path = Path(args.output)
    else:
        model_short = model_name.split("/")[-1]
        output_path = Path("experiments/results") / f"v7_maintenance_{model_short}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to {output_path}")


if __name__ == "__main__":
    main()
