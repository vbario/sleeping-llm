"""V7 Multi-Cycle Convergence Test — Does iterating sleep cycles converge to stable recall?

Note 112 showed one sleep cycle recovers 5/6 degraded facts but creates 2 new casualties.
This experiment answers: does repeating sleep cycles converge to zero degradation, or is it
whack-a-mole?

Phase A — Initial Wake + Convergence:
  1. Inject facts without constraints (batch_size=1) until degraded >= 3
  2. Run up to N sleep cycles, measuring after each
  3. Converged = degraded == 0 for 2 consecutive cycles

Phase B — Second Wake + Re-Convergence:
  1. From converged state, inject 5 more facts without constraints
  2. Run sleep cycles until convergence again
  3. Proves the system can repeatedly damage → heal → damage → heal

Usage:
    python experiments/v7_convergence_test.py --config experiments/configs/8b_v7.yaml
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


# ── Helpers (same as v7_maintenance_test.py) ──

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


def teach_facts(orch, facts):
    for fact in facts:
        msg = f"{fact.subject} {fact.relation} {fact.object}."
        orch.chat.process_input(msg)


def inject_without_constraints(engine, facts):
    """Inject facts with null-space constraints disabled (wake-mode interference)."""
    stashed = list(engine._active_edits)
    engine._active_edits = []
    try:
        result = engine.inject_facts(facts)
    finally:
        new_edits = list(engine._active_edits)
        engine._active_edits = stashed + new_edits
    return result


# ── Phase A: Initial Wake + Convergence ──

def phase_a(orch, fact_pool, max_inject=30, max_cycles=10, min_degraded=3):
    """Inject facts until degraded >= min_degraded, then run sleep cycles until convergence."""
    print(f"\n{'=' * 70}")
    print(f"  PHASE A: Initial Wake + Convergence")
    print(f"  (max_inject={max_inject}, max_cycles={max_cycles}, min_degraded={min_degraded})")
    print(f"{'=' * 70}")

    # Baseline PPL
    baseline_ppl = measure_perplexity(orch.backend)
    print(f"  Baseline PPL: {baseline_ppl:.2f}")

    # ── Injection (batch_size=1, no constraints) ──
    all_injected = []
    injection_trajectory = []

    for i in range(min(max_inject, len(fact_pool))):
        fact = fact_pool[i]
        print(f"  Injecting fact {i+1}: {fact.subject} {fact.relation} {fact.object}")
        inject_without_constraints(orch.memit_engine, [fact])
        all_injected.append(fact)
        teach_facts(orch, [fact])

        recall, details = test_recall(orch.backend, all_injected)
        degraded_facts = [d for d in details if not d["hit"]]
        ppl = measure_perplexity(orch.backend)

        print(f"    After {len(all_injected)} facts: recall={recall:.2f}, "
              f"degraded={len(degraded_facts)}, PPL={ppl:.2f}")

        injection_trajectory.append({
            "fact_num": i + 1,
            "recall": round(recall, 3),
            "degraded_count": len(degraded_facts),
            "ppl": round(ppl, 3),
        })

        if len(degraded_facts) >= min_degraded:
            print(f"  Found {len(degraded_facts)} degraded facts (>= {min_degraded}) — stopping injection.")
            break

        cleanup_gpu()

    # Final injection measurements
    final_recall, final_details = test_recall(orch.backend, all_injected)
    degraded = [d for d in final_details if not d["hit"]]

    injection_result = {
        "total_facts": len(all_injected),
        "final_recall": round(final_recall, 3),
        "degraded_count": len(degraded),
        "degraded_facts": [d["subject"] for d in degraded],
        "baseline_ppl": round(baseline_ppl, 3),
        "trajectory": injection_trajectory,
    }

    print(f"\n  Injection done: {len(all_injected)} facts, recall={final_recall:.2f}, "
          f"{len(degraded)} degraded")

    if len(degraded) == 0:
        print(f"  No degradation — cannot test convergence.")
        return {
            "injection": injection_result,
            "convergence": {
                "converged": True,
                "cycles_to_converge": 0,
                "trajectory": [],
                "note": "No degradation observed — already converged",
            },
        }, all_injected

    # ── Sleep convergence loop ──
    print(f"\n  Starting sleep convergence loop (max {max_cycles} cycles)...")
    convergence_trajectory = []
    consecutive_zero = 0
    converged = False
    cycles_to_converge = None

    for cycle in range(1, max_cycles + 1):
        print(f"\n  --- Sleep Cycle {cycle} ---")
        cycle_start = time.time()

        try:
            sleep_result = trigger_sleep(orch)
            facts_refreshed = sleep_result.get("facts_refreshed", 0)
            facts_pruned = sleep_result.get("facts_pruned", 0)
            print(f"  Sleep: refreshed={facts_refreshed}, pruned={facts_pruned}")
        except Exception as e:
            print(f"  Sleep FAILED: {e}")
            import traceback
            traceback.print_exc()
            convergence_trajectory.append({
                "cycle": cycle,
                "error": str(e),
            })
            break

        # Measure recall on ALL originally injected facts
        recall, details = test_recall(orch.backend, all_injected)
        degraded_now = [d for d in details if not d["hit"]]
        ppl = measure_perplexity(orch.backend)
        cycle_elapsed = time.time() - cycle_start

        print(f"  After cycle {cycle}: recall={recall:.2f}, degraded={len(degraded_now)}, "
              f"PPL={ppl:.2f}, time={cycle_elapsed:.0f}s")
        for d in degraded_now:
            print(f"    DEGRADED: {d['subject']} (expected: {d['expected']})")

        convergence_trajectory.append({
            "cycle": cycle,
            "recall": round(recall, 3),
            "degraded_count": len(degraded_now),
            "degraded_facts": [d["subject"] for d in degraded_now],
            "facts_refreshed": facts_refreshed,
            "facts_pruned": facts_pruned,
            "ppl": round(ppl, 3),
            "elapsed_seconds": round(cycle_elapsed, 1),
        })

        # Check convergence: degraded == 0 for 2 consecutive cycles
        if len(degraded_now) == 0:
            consecutive_zero += 1
            if consecutive_zero >= 2:
                converged = True
                cycles_to_converge = cycle
                print(f"\n  CONVERGED at cycle {cycle} (2 consecutive zero-degraded)")
                break
        else:
            consecutive_zero = 0

        cleanup_gpu()

    if not converged:
        # Check for stable-but-nonzero convergence
        if len(convergence_trajectory) >= 3:
            last3 = convergence_trajectory[-3:]
            recalls = [c["recall"] for c in last3 if "recall" in c]
            if recalls and max(recalls) - min(recalls) <= 0.02:
                print(f"\n  STABILIZED at recall ~{recalls[-1]:.2f} (not zero-degraded)")

    convergence_result = {
        "converged": converged,
        "cycles_to_converge": cycles_to_converge,
        "total_cycles_run": len(convergence_trajectory),
        "trajectory": convergence_trajectory,
    }

    phase_result = {
        "injection": injection_result,
        "convergence": convergence_result,
    }

    return phase_result, all_injected


# ── Phase B: Second Wake + Re-Convergence ──

def phase_b(orch, fact_pool, all_injected, fact_offset, second_wave=5, max_cycles=10):
    """Inject more facts from converged state, then run sleep cycles again."""
    print(f"\n{'=' * 70}")
    print(f"  PHASE B: Second Wake + Re-Convergence")
    print(f"  (second_wave={second_wave}, max_cycles={max_cycles})")
    print(f"{'=' * 70}")

    # Pre-injection recall on all existing facts
    recall_pre, pre_details = test_recall(orch.backend, all_injected)
    degraded_pre = [d for d in pre_details if not d["hit"]]
    print(f"  Pre-injection: recall={recall_pre:.2f}, degraded={len(degraded_pre)}")

    # Inject new facts without constraints
    new_facts = []
    for i in range(second_wave):
        idx = fact_offset + i
        if idx >= len(fact_pool):
            print(f"  Fact pool exhausted at {idx}")
            break
        fact = fact_pool[idx]
        print(f"  Injecting new fact {i+1}: {fact.subject} {fact.relation} {fact.object}")
        inject_without_constraints(orch.memit_engine, [fact])
        new_facts.append(fact)
        teach_facts(orch, [fact])

    all_injected = all_injected + new_facts

    # Post-injection measurements
    recall_post, post_details = test_recall(orch.backend, all_injected)
    degraded_post = [d for d in post_details if not d["hit"]]
    ppl_post = measure_perplexity(orch.backend)

    print(f"  After {len(new_facts)} new facts: recall={recall_post:.2f}, "
          f"degraded={len(degraded_post)}, PPL={ppl_post:.2f}")
    for d in degraded_post:
        print(f"    DEGRADED: {d['subject']} (expected: {d['expected']})")

    injection_result = {
        "new_facts": len(new_facts),
        "total_facts": len(all_injected),
        "recall_before_injection": round(recall_pre, 3),
        "recall_after_injection": round(recall_post, 3),
        "degraded_count": len(degraded_post),
        "degraded_facts": [d["subject"] for d in degraded_post],
        "ppl": round(ppl_post, 3),
    }

    if len(degraded_post) == 0:
        print(f"  No degradation after second wave — already converged.")
        return {
            "injection": injection_result,
            "convergence": {
                "converged": True,
                "cycles_to_converge": 0,
                "trajectory": [],
                "note": "No degradation after second wave",
            },
        }, all_injected

    # ── Sleep convergence loop (same logic as Phase A) ──
    print(f"\n  Starting Phase B sleep convergence loop (max {max_cycles} cycles)...")
    convergence_trajectory = []
    consecutive_zero = 0
    converged = False
    cycles_to_converge = None

    for cycle in range(1, max_cycles + 1):
        print(f"\n  --- Phase B Sleep Cycle {cycle} ---")
        cycle_start = time.time()

        try:
            sleep_result = trigger_sleep(orch)
            facts_refreshed = sleep_result.get("facts_refreshed", 0)
            facts_pruned = sleep_result.get("facts_pruned", 0)
            print(f"  Sleep: refreshed={facts_refreshed}, pruned={facts_pruned}")
        except Exception as e:
            print(f"  Sleep FAILED: {e}")
            import traceback
            traceback.print_exc()
            convergence_trajectory.append({
                "cycle": cycle,
                "error": str(e),
            })
            break

        # Measure recall on ALL facts (original + new)
        recall, details = test_recall(orch.backend, all_injected)
        degraded_now = [d for d in details if not d["hit"]]
        ppl = measure_perplexity(orch.backend)
        cycle_elapsed = time.time() - cycle_start

        print(f"  After cycle {cycle}: recall={recall:.2f}, degraded={len(degraded_now)}, "
              f"PPL={ppl:.2f}, time={cycle_elapsed:.0f}s")
        for d in degraded_now:
            print(f"    DEGRADED: {d['subject']} (expected: {d['expected']})")

        convergence_trajectory.append({
            "cycle": cycle,
            "recall": round(recall, 3),
            "degraded_count": len(degraded_now),
            "degraded_facts": [d["subject"] for d in degraded_now],
            "facts_refreshed": facts_refreshed,
            "facts_pruned": facts_pruned,
            "ppl": round(ppl, 3),
            "elapsed_seconds": round(cycle_elapsed, 1),
        })

        if len(degraded_now) == 0:
            consecutive_zero += 1
            if consecutive_zero >= 2:
                converged = True
                cycles_to_converge = cycle
                print(f"\n  CONVERGED at cycle {cycle} (2 consecutive zero-degraded)")
                break
        else:
            consecutive_zero = 0

        cleanup_gpu()

    convergence_result = {
        "converged": converged,
        "cycles_to_converge": cycles_to_converge,
        "total_cycles_run": len(convergence_trajectory),
        "trajectory": convergence_trajectory,
    }

    phase_result = {
        "injection": injection_result,
        "convergence": convergence_result,
    }

    return phase_result, all_injected


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="V7 Multi-Cycle Convergence Test")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--max-inject", type=int, default=30, help="Max facts to inject in Phase A")
    parser.add_argument("--max-cycles", type=int, default=10, help="Max sleep cycles per phase")
    parser.add_argument("--second-wave", type=int, default=5, help="Facts to inject in Phase B")
    parser.add_argument("--min-degraded", type=int, default=3, help="Min degraded facts before stopping injection")
    parser.add_argument("--output", type=str, default=None, help="Override output JSON path")
    args = parser.parse_args()

    config = Config(args.config)
    model_name = config.model["path"]
    layers = config.get("memit.target_layers", [])

    print("=" * 70)
    print("  V7 MULTI-CYCLE CONVERGENCE TEST")
    print("=" * 70)
    print(f"  Model:        {model_name}")
    print(f"  Backend:      {config.model.get('backend', 'mlx')}")
    print(f"  Layers:       {layers} ({len(layers)} layers)")
    print(f"  Max inject:   {args.max_inject}")
    print(f"  Max cycles:   {args.max_cycles}")
    print(f"  Min degraded: {args.min_degraded}")
    print(f"  Second wave:  {args.second_wave}")
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
            "max_inject": args.max_inject,
            "max_cycles": args.max_cycles,
            "min_degraded": args.min_degraded,
            "second_wave": args.second_wave,
        },
    }

    # Phase A: Initial wake + convergence
    phase_a_result, all_injected = phase_a(
        orch, fact_pool,
        max_inject=args.max_inject,
        max_cycles=args.max_cycles,
        min_degraded=args.min_degraded,
    )
    results["phase_a"] = phase_a_result

    fact_offset = len(all_injected)

    # Phase B: Second wake + re-convergence
    phase_b_result, all_injected = phase_b(
        orch, fact_pool, all_injected,
        fact_offset=fact_offset,
        second_wave=args.second_wave,
        max_cycles=args.max_cycles,
    )
    results["phase_b"] = phase_b_result

    # Overall verdict
    a_conv = phase_a_result["convergence"]["converged"]
    b_conv = phase_b_result["convergence"]["converged"]
    a_cycles = phase_a_result["convergence"]["cycles_to_converge"]
    b_cycles = phase_b_result["convergence"]["cycles_to_converge"]

    if a_conv and b_conv:
        verdict = f"CONVERGED (A: {a_cycles} cycles, B: {b_cycles} cycles)"
    elif a_conv:
        b_last = phase_b_result["convergence"]["trajectory"]
        b_recall = b_last[-1]["recall"] if b_last else "?"
        verdict = f"PARTIAL (A converged in {a_cycles}, B did not — last recall {b_recall})"
    elif b_conv:
        a_last = phase_a_result["convergence"]["trajectory"]
        a_recall = a_last[-1]["recall"] if a_last else "?"
        verdict = f"PARTIAL (A did not converge — last recall {a_recall}, B converged in {b_cycles})"
    else:
        a_last = phase_a_result["convergence"]["trajectory"]
        b_last = phase_b_result["convergence"]["trajectory"]
        a_recall = a_last[-1]["recall"] if a_last else "?"
        b_recall = b_last[-1]["recall"] if b_last else "?"
        verdict = f"DID NOT CONVERGE (A last recall {a_recall}, B last recall {b_recall})"

    results["overall_verdict"] = verdict

    total_elapsed = time.time() - total_start
    results["total_elapsed_seconds"] = round(total_elapsed, 1)

    destroy_orchestrator(orch)

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"  OVERALL: {verdict}")
    print(f"  Time: {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min)")
    print(f"{'=' * 70}")

    # Save
    if args.output:
        output_path = Path(args.output)
    else:
        model_short = model_name.split("/")[-1]
        output_path = Path("experiments/results") / f"v7_convergence_{model_short}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to {output_path}")


if __name__ == "__main__":
    main()
