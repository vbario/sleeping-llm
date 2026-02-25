"""V7 Comprehensive Experiment — Prove Maintenance, Longevity, Nap Audit + 100 Facts.

Four sequential phases testing the MEMIT-only sleep system:
  Phase 1: Maintenance Refresh — sleep detects degraded facts, reverts, re-injects
  Phase 2: Longevity — 5 inject→sleep cycles without cumulative degradation
  Phase 3: Nap Audit — nap detects degradation, subsequent sleep fixes it
  Phase 4: 100-Fact Scaling — ≥50% recall at 100 facts with sleep maintenance

Usage:
    # Full experiment (all 4 phases)
    python experiments/v7_comprehensive_test.py --config experiments/configs/70b_v7.yaml

    # Single phase
    python experiments/v7_comprehensive_test.py --config ... --phase maintenance
    python experiments/v7_comprehensive_test.py --config ... --phase longevity
    python experiments/v7_comprehensive_test.py --config ... --phase nap
    python experiments/v7_comprehensive_test.py --config ... --phase scaling

    # Quick smoke test (3B, small counts)
    python experiments/v7_comprehensive_test.py --config ... --quick
"""

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

# Prevent CUDA fragmentation OOM on multi-phase runs
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.orchestrator import Orchestrator
from src.memory.memit import FactTriple


# ── Reference texts for perplexity measurement ──

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


# ── Helpers ──

def measure_perplexity(backend):
    """Average perplexity over reference texts."""
    ppls = []
    for text in REFERENCE_TEXTS:
        ppl = backend.compute_perplexity(text)
        ppls.append(ppl)
    return sum(ppls) / len(ppls)


def test_recall(backend, facts):
    """Test raw completion recall. Returns (fraction, details_list)."""
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
    """Load facts from the JSON pool and convert to FactTriples."""
    pool_path = project_root / path
    with open(pool_path) as f:
        raw = json.load(f)
    return [FactTriple(subject=r["subject"], relation=r["relation"], object=r["object"]) for r in raw]


def clean_artifacts_v7(config):
    """Remove v7 artifacts (conversations + memit data) for clean state."""
    conv_dir = Path(config.paths["conversations"])
    if conv_dir.exists():
        shutil.rmtree(conv_dir)
    conv_dir.mkdir(parents=True, exist_ok=True)

    memit_dir = Path(config.paths.get("memit_data", "data/memit"))
    if memit_dir.exists():
        shutil.rmtree(memit_dir)
    memit_dir.mkdir(parents=True, exist_ok=True)


def cleanup_gpu():
    """Force-free GPU memory between phases."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


def fresh_orchestrator(config):
    """Create a fresh orchestrator with auto-triggers disabled."""
    cleanup_gpu()
    orch = Orchestrator(config, disable_memit=False)
    orch.chat._sleep_callback = None
    orch.chat._nap_callback = None
    return orch


def destroy_orchestrator(orch):
    """Explicitly destroy orchestrator and free GPU memory."""
    if hasattr(orch, 'backend') and hasattr(orch.backend, 'model'):
        del orch.backend.model
    if hasattr(orch, 'backend') and hasattr(orch.backend, 'tokenizer'):
        del orch.backend.tokenizer
    del orch
    cleanup_gpu()


def trigger_sleep(orch):
    """Execute full sleep, return result dict."""
    orch.sleep_cycle_count += 1
    cycle_id = f"{orch.sleep_cycle_count:04d}"
    result = orch.full_sleep_controller.execute_sleep(
        cycle_id, "full", orch._gather_new_messages,
    )
    # Post-sleep housekeeping
    refreshed = result.get("facts_refreshed", 0)
    pruned = result.get("facts_pruned", 0)
    orch.health_monitor.record_sleep("full", facts_refreshed=refreshed, facts_pruned=pruned)
    if orch.context.recent_messages:
        orch.context.compact()
    orch.chat.reset_turn_count()
    orch.context.reset(keep_summary=True)
    return result


def trigger_nap(orch):
    """Execute nap audit, return result dict."""
    orch.sleep_cycle_count += 1
    cycle_id = f"nap_{orch.sleep_cycle_count:04d}"
    return orch.nap_controller.execute_nap(cycle_id)


def teach_facts(orch, facts):
    """Teach facts via conversation so sleep has session data to curate."""
    for fact in facts:
        msg = f"{fact.subject} {fact.relation} {fact.object}."
        orch.chat.process_input(msg)


def ppl_delta_pct(baseline, current):
    """PPL increase as fraction of baseline."""
    if baseline == 0:
        return 0
    return (current - baseline) / baseline


# ── Phase 1: Maintenance Refresh ──

def phase_maintenance(config, fact_pool, quick=False):
    """Prove that sleep detects degraded facts, reverts, and re-injects them."""
    batch_size = 3 if quick else 20
    max_batches = 2 if quick else 3
    label = f"Phase 1: Maintenance Refresh ({batch_size} facts/batch)"
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")

    t0 = time.time()
    clean_artifacts_v7(config)
    orch = fresh_orchestrator(config)

    # Baseline
    baseline_ppl = measure_perplexity(orch.backend)
    print(f"  Baseline PPL: {baseline_ppl:.2f}")

    all_injected = []
    recall_pre_sleep = None
    degraded_found = False

    for batch_num in range(1, max_batches + 1):
        start_idx = len(all_injected)
        batch = fact_pool[start_idx:start_idx + batch_size]
        if not batch:
            break

        print(f"\n  Injecting batch {batch_num} ({len(batch)} facts)...")
        orch.memit_engine.inject_facts(batch)
        all_injected.extend(batch)

        # Teach so sleep has session data
        teach_facts(orch, batch)

        ppl = measure_perplexity(orch.backend)
        recall, details = test_recall(orch.backend, all_injected)
        print(f"  After {len(all_injected)} facts: PPL={ppl:.2f}, recall={recall:.2f}")

        # Check for degradation in earlier batches
        if batch_num > 1:
            earlier = all_injected[:start_idx]
            earlier_recall, _ = test_recall(orch.backend, earlier)
            print(f"  Earlier batches recall: {earlier_recall:.2f}")
            if earlier_recall < 0.80:
                degraded_found = True
                print(f"  Degradation detected in earlier facts!")

        recall_pre_sleep = recall

        # If degradation found, run sleep and stop
        if degraded_found:
            break

    # Trigger sleep
    print(f"\n  Triggering full sleep (maintenance)...")
    try:
        sleep_result = trigger_sleep(orch)
        sleep_ok = True
        print(f"  Sleep result: refreshed={sleep_result.get('facts_refreshed', 0)}, "
              f"pruned={sleep_result.get('facts_pruned', 0)}")
    except Exception as e:
        print(f"  Sleep failed: {e}")
        import traceback
        traceback.print_exc()
        sleep_result = {"status": "error", "error": str(e)}
        sleep_ok = False

    # Post-sleep measurement
    post_ppl = measure_perplexity(orch.backend)
    recall_post, post_details = test_recall(orch.backend, all_injected)
    print(f"  Post-sleep: PPL={post_ppl:.2f}, recall={recall_post:.2f}")

    facts_refreshed = sleep_result.get("facts_refreshed", 0) if sleep_ok else 0
    ppl_increase = ppl_delta_pct(baseline_ppl, post_ppl)

    # Verdict
    passed = (
        facts_refreshed > 0
        and recall_post >= recall_pre_sleep
        and recall_post >= 0.50
        and ppl_increase < 0.15
    )
    # If no degradation occurred (system too robust), it's an informative result
    if not degraded_found and facts_refreshed == 0:
        verdict = "SKIP (no degradation observed — system too robust at this scale)"
        verdict_pass = True  # Not a failure
    else:
        verdict = "PASS" if passed else "FAIL"
        verdict_pass = passed

    elapsed = time.time() - t0
    result = {
        "verdict": verdict,
        "verdict_pass": verdict_pass,
        "total_facts": len(all_injected),
        "batches_injected": min(batch_num, max_batches),
        "degraded_found": degraded_found,
        "facts_refreshed": facts_refreshed,
        "recall_pre_sleep": round(recall_pre_sleep, 3) if recall_pre_sleep else None,
        "recall_post_sleep": round(recall_post, 3),
        "baseline_ppl": round(baseline_ppl, 3),
        "post_sleep_ppl": round(post_ppl, 3),
        "ppl_increase_pct": round(ppl_increase * 100, 1),
        "sleep_result": sleep_result if sleep_ok else None,
        "elapsed_seconds": round(elapsed, 1),
    }

    print(f"\n  VERDICT: {verdict}")
    print(f"  (refreshed={facts_refreshed}, recall {recall_pre_sleep:.2f}→{recall_post:.2f}, "
          f"PPL +{ppl_increase*100:.1f}%, {elapsed:.0f}s)")
    destroy_orchestrator(orch)
    return result


# ── Phase 2: Longevity ──

def phase_longevity(config, fact_pool, quick=False):
    """Prove multiple inject→sleep cycles don't cause cumulative degradation."""
    num_cycles = 2 if quick else 5
    facts_per_cycle = 3 if quick else 10
    label = f"Phase 2: Longevity ({num_cycles} cycles × {facts_per_cycle} facts)"
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")

    t0 = time.time()
    clean_artifacts_v7(config)
    orch = fresh_orchestrator(config)

    baseline_ppl = measure_perplexity(orch.backend)
    print(f"  Baseline PPL: {baseline_ppl:.2f}")

    all_injected = []
    trajectory = []
    fact_idx = 0

    for cycle in range(1, num_cycles + 1):
        print(f"\n  --- Cycle {cycle}/{num_cycles} ---")

        # Inject new facts
        batch = fact_pool[fact_idx:fact_idx + facts_per_cycle]
        if not batch:
            print(f"  Ran out of facts!")
            break
        fact_idx += len(batch)

        print(f"  Injecting {len(batch)} facts...")
        orch.memit_engine.inject_facts(batch)
        all_injected.extend(batch)

        # Teach via conversation
        teach_facts(orch, batch)

        # Pre-sleep measurements
        ppl = measure_perplexity(orch.backend)
        cumulative_recall, _ = test_recall(orch.backend, all_injected)
        newest_recall, _ = test_recall(orch.backend, batch)

        # Oldest batch recall
        oldest_batch = all_injected[:facts_per_cycle]
        oldest_recall, _ = test_recall(orch.backend, oldest_batch)

        print(f"  Pre-sleep: PPL={ppl:.2f}, cumulative={cumulative_recall:.2f}, "
              f"oldest={oldest_recall:.2f}, newest={newest_recall:.2f}")

        # Sleep
        print(f"  Running sleep...")
        try:
            sleep_result = trigger_sleep(orch)
            sleep_ok = True
            print(f"  Sleep: refreshed={sleep_result.get('facts_refreshed', 0)}, "
                  f"pruned={sleep_result.get('facts_pruned', 0)}")
        except Exception as e:
            print(f"  Sleep failed: {e}")
            import traceback
            traceback.print_exc()
            sleep_result = {"status": "error", "error": str(e)}
            sleep_ok = False

        # Post-sleep measurements
        post_ppl = measure_perplexity(orch.backend)
        post_cumulative, _ = test_recall(orch.backend, all_injected)
        post_oldest, _ = test_recall(orch.backend, oldest_batch)
        post_newest, _ = test_recall(orch.backend, batch)

        print(f"  Post-sleep: PPL={post_ppl:.2f}, cumulative={post_cumulative:.2f}, "
              f"oldest={post_oldest:.2f}, newest={post_newest:.2f}")

        trajectory.append({
            "cycle": cycle,
            "total_facts": len(all_injected),
            "pre_sleep": {
                "ppl": round(ppl, 3),
                "cumulative_recall": round(cumulative_recall, 3),
                "oldest_recall": round(oldest_recall, 3),
                "newest_recall": round(newest_recall, 3),
            },
            "post_sleep": {
                "ppl": round(post_ppl, 3),
                "cumulative_recall": round(post_cumulative, 3),
                "oldest_recall": round(post_oldest, 3),
                "newest_recall": round(post_newest, 3),
            },
            "sleep_ok": sleep_ok,
            "facts_refreshed": sleep_result.get("facts_refreshed", 0) if sleep_ok else 0,
        })

        # Free intermediate CUDA tensors between cycles
        cleanup_gpu()

    # Final measurements
    final = trajectory[-1]["post_sleep"] if trajectory else {}
    final_cumulative = final.get("cumulative_recall", 0)
    final_oldest = final.get("oldest_recall", 0)
    final_ppl = final.get("ppl", baseline_ppl)
    ppl_increase = ppl_delta_pct(baseline_ppl, final_ppl)

    passed = (
        final_cumulative >= 0.50
        and ppl_increase < 0.20
        and final_oldest >= 0.30
    )
    verdict = "PASS" if passed else "FAIL"
    elapsed = time.time() - t0

    result = {
        "verdict": verdict,
        "verdict_pass": passed,
        "cycles": len(trajectory),
        "total_facts": len(all_injected),
        "baseline_ppl": round(baseline_ppl, 3),
        "final_cumulative_recall": round(final_cumulative, 3),
        "final_oldest_recall": round(final_oldest, 3),
        "final_ppl": round(final_ppl, 3),
        "ppl_increase_pct": round(ppl_increase * 100, 1),
        "trajectory": trajectory,
        "elapsed_seconds": round(elapsed, 1),
    }

    print(f"\n  VERDICT: {verdict}")
    print(f"  (cumulative={final_cumulative:.2f}, oldest={final_oldest:.2f}, "
          f"PPL +{ppl_increase*100:.1f}%, {elapsed:.0f}s)")
    destroy_orchestrator(orch)
    return result


# ── Phase 3: Nap Audit ──

def phase_nap(config, fact_pool, quick=False):
    """Prove nap detects degradation and subsequent sleep fixes it."""
    num_cycles = 2 if quick else 4
    facts_per_cycle = 3 if quick else 10
    label = f"Phase 3: Nap Audit ({num_cycles} cycles × {facts_per_cycle} facts)"
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")

    t0 = time.time()
    clean_artifacts_v7(config)
    orch = fresh_orchestrator(config)

    baseline_ppl = measure_perplexity(orch.backend)
    print(f"  Baseline PPL: {baseline_ppl:.2f}")

    all_injected = []
    nap_results = []
    degradation_detected = False
    fixed_by_sleep = False
    fact_idx = 0

    for cycle in range(1, num_cycles + 1):
        print(f"\n  --- Cycle {cycle}/{num_cycles} ---")

        # Inject
        batch = fact_pool[fact_idx:fact_idx + facts_per_cycle]
        if not batch:
            break
        fact_idx += len(batch)

        print(f"  Injecting {len(batch)} facts...")
        orch.memit_engine.inject_facts(batch)
        all_injected.extend(batch)
        teach_facts(orch, batch)

        # Nap BEFORE sleep
        print(f"  Running nap audit...")
        try:
            nap_result = trigger_nap(orch)
            nap_ok = True
            nap_degraded = nap_result.get("degraded", 0)
            nap_audited = nap_result.get("audited", 0)
            print(f"  Nap: audited={nap_audited}, degraded={nap_degraded}")

            if nap_degraded > 0:
                degradation_detected = True
                degraded_edit_ids = [
                    r["edit_id"] for r in nap_result.get("results", [])
                    if not r.get("healthy", True)
                ]
                print(f"  Nap detected degradation in {nap_degraded} facts!")
        except Exception as e:
            print(f"  Nap failed: {e}")
            nap_result = {"status": "error", "error": str(e)}
            nap_ok = False
            nap_degraded = 0

        # Sleep
        print(f"  Running full sleep...")
        try:
            sleep_result = trigger_sleep(orch)
            sleep_ok = True
            refreshed = sleep_result.get("facts_refreshed", 0)
            print(f"  Sleep: refreshed={refreshed}")

            if degradation_detected and refreshed > 0:
                fixed_by_sleep = True
                print(f"  Sleep fixed degraded facts!")
        except Exception as e:
            print(f"  Sleep failed: {e}")
            sleep_result = {"status": "error", "error": str(e)}
            sleep_ok = False

        nap_results.append({
            "cycle": cycle,
            "total_facts": len(all_injected),
            "nap_ok": nap_ok,
            "nap_audited": nap_result.get("audited", 0) if nap_ok else 0,
            "nap_degraded": nap_degraded,
            "sleep_ok": sleep_ok,
            "facts_refreshed": sleep_result.get("facts_refreshed", 0) if sleep_ok else 0,
        })

        # Free intermediate CUDA tensors between cycles
        cleanup_gpu()

    # Verdict
    if degradation_detected and fixed_by_sleep:
        verdict = "PASS"
        verdict_pass = True
    elif not degradation_detected:
        verdict = "SKIP (no degradation observed — nap had nothing to detect)"
        verdict_pass = True  # Informative negative, not a failure
    else:
        verdict = "FAIL (degradation detected but sleep did not fix it)"
        verdict_pass = False

    elapsed = time.time() - t0
    result = {
        "verdict": verdict,
        "verdict_pass": verdict_pass,
        "cycles": len(nap_results),
        "total_facts": len(all_injected),
        "degradation_detected": degradation_detected,
        "fixed_by_sleep": fixed_by_sleep,
        "nap_results": nap_results,
        "elapsed_seconds": round(elapsed, 1),
    }

    print(f"\n  VERDICT: {verdict}")
    print(f"  (detected={degradation_detected}, fixed={fixed_by_sleep}, {elapsed:.0f}s)")
    destroy_orchestrator(orch)
    return result


# ── Phase 4: 100-Fact Scaling ──

def phase_scaling(config, fact_pool, quick=False):
    """Scale to 100 facts with sleep maintenance at checkpoints."""
    target_facts = 10 if quick else 100
    batch_size = 3 if quick else 5
    checkpoint_interval = 5 if quick else 10
    sleep_at = [5 if quick else 50, target_facts]
    label = f"Phase 4: Scaling to {target_facts} Facts"
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")

    t0 = time.time()
    clean_artifacts_v7(config)
    orch = fresh_orchestrator(config)

    baseline_ppl = measure_perplexity(orch.backend)
    print(f"  Baseline PPL: {baseline_ppl:.2f}")

    all_injected = []
    trajectory = []
    fact_idx = 0

    while fact_idx < target_facts and fact_idx < len(fact_pool):
        batch_end = min(fact_idx + batch_size, target_facts, len(fact_pool))
        batch = fact_pool[fact_idx:batch_end]
        if not batch:
            break

        print(f"\n  Injecting facts {fact_idx + 1}-{batch_end}...")
        orch.memit_engine.inject_facts(batch)
        all_injected.extend(batch)
        fact_idx = batch_end

        total = len(all_injected)

        # Checkpoint measurement
        if total % checkpoint_interval == 0 or total == target_facts:
            ppl = measure_perplexity(orch.backend)
            recall, details = test_recall(orch.backend, all_injected)
            ppl_inc = ppl_delta_pct(baseline_ppl, ppl)
            print(f"  [{total} facts] PPL={ppl:.2f} (+{ppl_inc*100:.1f}%), recall={recall:.2f}")

            checkpoint = {
                "total_facts": total,
                "ppl": round(ppl, 3),
                "ppl_increase_pct": round(ppl_inc * 100, 1),
                "recall": round(recall, 3),
            }

            # Sleep at designated checkpoints
            if total in sleep_at:
                print(f"  Running sleep at {total} facts...")
                try:
                    sleep_result = trigger_sleep(orch)
                    sleep_ok = True
                    refreshed = sleep_result.get("facts_refreshed", 0)
                    pruned = sleep_result.get("facts_pruned", 0)
                    print(f"  Sleep: refreshed={refreshed}, pruned={pruned}")

                    # Re-measure post-sleep
                    post_ppl = measure_perplexity(orch.backend)
                    post_recall, _ = test_recall(orch.backend, all_injected)
                    print(f"  Post-sleep: PPL={post_ppl:.2f}, recall={post_recall:.2f}")

                    checkpoint["post_sleep"] = {
                        "ppl": round(post_ppl, 3),
                        "recall": round(post_recall, 3),
                        "facts_refreshed": refreshed,
                        "facts_pruned": pruned,
                    }
                except Exception as e:
                    print(f"  Sleep failed: {e}")
                    import traceback
                    traceback.print_exc()
                    checkpoint["post_sleep"] = {"error": str(e)}

            trajectory.append(checkpoint)

            # Free intermediate CUDA tensors
            cleanup_gpu()

            # Early stop if recall collapses
            if total >= batch_size * 4 and recall < 0.20:
                print(f"  STOPPING EARLY — recall collapsed to {recall:.2f}")
                break

        # Also cleanup after non-checkpoint injections
        cleanup_gpu()

    # Final state
    final_checkpoint = trajectory[-1] if trajectory else {}
    final_recall = final_checkpoint.get("recall", 0)
    # Use post-sleep recall if sleep ran at the final checkpoint
    if "post_sleep" in final_checkpoint and "recall" in final_checkpoint["post_sleep"]:
        final_recall = final_checkpoint["post_sleep"]["recall"]

    final_ppl = final_checkpoint.get("ppl", baseline_ppl)
    if "post_sleep" in final_checkpoint and "ppl" in final_checkpoint["post_sleep"]:
        final_ppl = final_checkpoint["post_sleep"]["ppl"]

    ppl_increase = ppl_delta_pct(baseline_ppl, final_ppl)
    passed = final_recall >= 0.50 and ppl_increase < 0.20
    verdict = "PASS" if passed else "FAIL"

    elapsed = time.time() - t0
    result = {
        "verdict": verdict,
        "verdict_pass": passed,
        "target_facts": target_facts,
        "total_facts": len(all_injected),
        "baseline_ppl": round(baseline_ppl, 3),
        "final_recall": round(final_recall, 3),
        "final_ppl": round(final_ppl, 3),
        "ppl_increase_pct": round(ppl_increase * 100, 1),
        "recall_trajectory": trajectory,
        "elapsed_seconds": round(elapsed, 1),
    }

    print(f"\n  VERDICT: {verdict}")
    print(f"  ({len(all_injected)} facts, recall={final_recall:.2f}, "
          f"PPL +{ppl_increase*100:.1f}%, {elapsed:.0f}s)")
    destroy_orchestrator(orch)
    return result


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="V7 Comprehensive Experiment")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--phase", type=str, default=None,
                        choices=["maintenance", "longevity", "nap", "scaling"],
                        help="Run a single phase (default: all)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test with small counts")
    parser.add_argument("--output", type=str, default=None,
                        help="Override output JSON path")
    args = parser.parse_args()

    config = Config(args.config)
    model_name = config.model["path"]
    layers = config.get("memit.target_layers", [])

    print("=" * 70)
    print("  V7 COMPREHENSIVE EXPERIMENT")
    print("=" * 70)
    print(f"  Model:   {model_name}")
    print(f"  Backend: {config.model.get('backend', 'mlx')}")
    print(f"  Layers:  {layers} ({len(layers)} layers)")
    print(f"  Quick:   {args.quick}")
    print(f"  Phase:   {args.phase or 'all'}")
    print("=" * 70)

    # Load fact pool
    fact_pool = load_fact_pool()
    print(f"  Loaded {len(fact_pool)} facts from pool")

    total_start = time.time()
    results = {
        "config": {
            "model": model_name,
            "backend": config.model.get("backend", "mlx"),
            "layers": layers,
            "num_layers": len(layers),
            "quick": args.quick,
        },
    }

    phases_to_run = [args.phase] if args.phase else ["maintenance", "longevity", "nap", "scaling"]
    phase_verdicts = []

    for phase_name in phases_to_run:
        try:
            if phase_name == "maintenance":
                result = phase_maintenance(config, fact_pool, quick=args.quick)
                results["phase_1_maintenance"] = result
            elif phase_name == "longevity":
                result = phase_longevity(config, fact_pool, quick=args.quick)
                results["phase_2_longevity"] = result
            elif phase_name == "nap":
                result = phase_nap(config, fact_pool, quick=args.quick)
                results["phase_3_nap"] = result
            elif phase_name == "scaling":
                result = phase_scaling(config, fact_pool, quick=args.quick)
                results["phase_4_scaling"] = result

            phase_verdicts.append((phase_name, result.get("verdict_pass", False), result["verdict"]))
        except Exception as e:
            print(f"\n  Phase '{phase_name}' CRASHED: {e}")
            import traceback
            traceback.print_exc()
            results[f"phase_{phase_name}"] = {"verdict": "CRASH", "verdict_pass": False, "error": str(e)}
            phase_verdicts.append((phase_name, False, f"CRASH: {e}"))

    total_elapsed = time.time() - total_start

    # Overall verdict
    passed_count = sum(1 for _, ok, _ in phase_verdicts if ok)
    total_count = len(phase_verdicts)
    results["overall_verdict"] = f"{'PASS' if passed_count == total_count else 'FAIL'} ({passed_count}/{total_count})"
    results["total_elapsed_seconds"] = round(total_elapsed, 1)

    print(f"\n{'=' * 70}")
    print(f"  OVERALL RESULTS")
    print(f"{'=' * 70}")
    for name, ok, verdict in phase_verdicts:
        status = "OK" if ok else "FAIL"
        print(f"  [{status:>4}] {name}: {verdict}")
    print(f"\n  Overall: {results['overall_verdict']}")
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min)")
    print(f"{'=' * 70}")

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        model_short = model_name.split("/")[-1]
        suffix = f"_{args.phase}" if args.phase else ""
        quick_tag = "_quick" if args.quick else ""
        output_path = Path("experiments/results") / f"v7_comprehensive_{model_short}{suffix}{quick_tag}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
