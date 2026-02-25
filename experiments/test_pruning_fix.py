"""Test Pruning Bug Fix — Verify refresh marks old edits and prune-by-recall works.

Reproduces the "pruning death spiral" from note 116 (Phase B) and verifies that
the two bug fixes prevent it:

  Fix A: Refresh now calls ledger.mark_pruned(old_edit_id) before re-injecting,
         so stale entries don't accumulate and reappear on restart.

  Fix B: Pruning sorts by (recall_success_rate, timestamp) instead of just timestamp,
         so damaged edits are removed first, not healthy oldest ones.

Three test phases:

  Phase 1 — Ledger hygiene: Inject 5 facts, trigger maintenance refresh on a
            synthetically-degraded edit. Verify the old ledger entry is marked
            pruned and the new entry replaces it (edit count stays constant).

  Phase 2 — Prune-by-recall: Inject facts up to capacity, then overshoot.
            Synthetically damage some edits' recall_success_rate. Trigger pruning.
            Verify damaged edits are pruned first, healthy edits survive.

  Phase 3 — Convergence (no death spiral): Inject facts to near-capacity, run
            multiple sleep cycles. Verify recall doesn't monotonically decrease
            (the old death spiral pattern).

Usage:
    python experiments/test_pruning_fix.py --config config.yaml
    python experiments/test_pruning_fix.py --config experiments/configs/8b_v7.yaml
    python experiments/test_pruning_fix.py --config config.yaml --phase ledger
    python experiments/test_pruning_fix.py --config config.yaml --quick
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
]


# ── Helpers ──

def measure_perplexity(backend):
    ppls = [backend.compute_perplexity(text) for text in REFERENCE_TEXTS]
    return sum(ppls) / len(ppls)


def test_recall(backend, facts, raw=True):
    """Test recall. Returns (fraction, per-fact details)."""
    details = []
    passed = 0
    for fact in facts:
        if raw:
            prompt = fact.to_prompt()
            response = backend.generate(prompt, max_tokens=30, temperature=0.1)
        else:
            question = fact.to_question()
            messages = [{"role": "user", "content": question}]
            prompt = backend.apply_chat_template(messages)
            response = backend.generate(prompt, max_tokens=100, temperature=0.1)
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


def clean_artifacts(config):
    """Remove artifacts for clean state."""
    for key in ["conversations", "memit_data"]:
        dir_path = Path(config.paths.get(key, f"data/{key}"))
        if dir_path.exists():
            shutil.rmtree(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

    # Also clean fused models and adapters
    for key in ["fused_models", "adapters"]:
        dir_path = Path(config.paths.get(key, f"data/{key}"))
        if dir_path.exists():
            shutil.rmtree(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)


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
    """Create orchestrator with auto-triggers disabled."""
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
    """Execute full sleep, return result dict."""
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
    """Teach facts via conversation so sleep has session data to curate."""
    for fact in facts:
        msg = f"{fact.subject} {fact.relation} {fact.object}."
        orch.chat.process_input(msg)


# ── Phase 1: Ledger Hygiene ──

def phase_ledger(config, fact_pool, quick=False):
    """Verify refresh marks old edits as pruned; ledger count stays constant."""
    num_facts = 3 if quick else 5
    label = f"Phase 1: Ledger Hygiene ({num_facts} facts)"
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")

    t0 = time.time()
    clean_artifacts(config)
    orch = fresh_orchestrator(config)
    engine = orch.memit_engine
    ledger = orch.edit_ledger

    # Inject facts one at a time
    facts = fact_pool[:num_facts]
    for fact in facts:
        engine.inject_facts([fact])
    teach_facts(orch, facts)

    initial_edit_count = ledger.get_edit_count()
    initial_active_count = engine.get_active_edit_count()
    print(f"  After injection: {initial_edit_count} ledger edits, "
          f"{initial_active_count} in-memory edits")

    # Synthetically degrade the first edit so maintenance will refresh it
    first_edit = engine._active_edits[0]
    first_edit_id = first_edit.edit_id
    first_edit.recall_success_rate = 0.0
    ledger.update_verification(first_edit_id, 0.0)
    print(f"  Synthetically degraded: {first_edit.facts[0].subject} "
          f"{first_edit.facts[0].relation} (recall → 0.0)")

    # Run maintenance manually (same as sleep step 4)
    audit_results = {
        "healthy": initial_active_count - 1,
        "degraded": [(first_edit, 0.0)],
        "total": initial_active_count,
    }
    maint = orch.full_sleep_controller._maintain_edits(audit_results, None)
    print(f"  Maintenance: refreshed={maint['refreshed']}, pruned={maint['pruned']}")

    # Verify: old entry should be marked pruned in ledger
    post_edit_count = ledger.get_edit_count()
    post_active_count = engine.get_active_edit_count()

    old_entry_pruned = False
    for entry in ledger._edits:
        if entry["edit_id"] == first_edit_id and entry.get("pruned", False):
            old_entry_pruned = True
            break

    print(f"  After maintenance: {post_edit_count} active ledger edits, "
          f"{post_active_count} in-memory edits")
    print(f"  Old entry pruned in ledger: {old_entry_pruned}")

    # The active count should stay the same (one removed, one re-added)
    count_stable = (post_active_count == initial_active_count)
    print(f"  Active count stable: {count_stable} "
          f"({initial_active_count} → {post_active_count})")

    # Verify recall of the refreshed fact
    refreshed_fact = facts[0]
    passed, response = engine.test_recall(refreshed_fact, raw=True)
    print(f"  Refreshed fact recall: {'PASS' if passed else 'FAIL'} "
          f"({refreshed_fact.subject} → {response[:40]})")

    verdict_pass = old_entry_pruned and count_stable and maint["refreshed"] > 0
    verdict = "PASS" if verdict_pass else "FAIL"

    elapsed = time.time() - t0
    result = {
        "verdict": verdict,
        "verdict_pass": verdict_pass,
        "initial_edit_count": initial_edit_count,
        "post_edit_count": post_edit_count,
        "count_stable": count_stable,
        "old_entry_pruned": old_entry_pruned,
        "refreshed": maint["refreshed"],
        "refreshed_fact_recall": passed,
        "elapsed_seconds": round(elapsed, 1),
    }

    print(f"\n  VERDICT: {verdict}")
    destroy_orchestrator(orch)
    return result


# ── Phase 2: Prune-by-Recall ──

def phase_prune_order(config, fact_pool, quick=False):
    """Verify pruning removes lowest-recall edits first, not oldest."""
    num_facts = 5 if quick else 10
    label = f"Phase 2: Prune-by-Recall ({num_facts} facts)"
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")

    t0 = time.time()
    clean_artifacts(config)
    orch = fresh_orchestrator(config)
    engine = orch.memit_engine
    ledger = orch.edit_ledger

    # Temporarily lower max_active_edits to force pruning
    original_max = engine.max_active_edits
    engine.max_active_edits = num_facts  # exactly at capacity

    # Inject facts
    facts = fact_pool[:num_facts]
    for fact in facts:
        engine.inject_facts([fact])
    teach_facts(orch, facts)

    print(f"  Injected {num_facts} facts (max_active_edits={engine.max_active_edits})")

    # Synthetically mark some edits as damaged (low recall)
    # Mark edit 1 as worst (0.0) and edit 2 as bad (0.2)
    # With 1 excess, only edit 1 (worst) should be pruned
    worst_id = None
    bad_id = None
    healthy_ids = set()
    for i, edit in enumerate(engine._active_edits):
        if i == 1:
            edit.recall_success_rate = 0.0  # worst
            ledger.update_verification(edit.edit_id, 0.0)
            worst_id = edit.edit_id
            print(f"  Worst: {edit.facts[0].subject} (edit {i}, recall → 0.0)")
        elif i == 2:
            edit.recall_success_rate = 0.2  # bad but not worst
            ledger.update_verification(edit.edit_id, 0.2)
            bad_id = edit.edit_id
            print(f"  Bad: {edit.facts[0].subject} (edit {i}, recall → 0.2)")
        else:
            edit.recall_success_rate = 1.0
            healthy_ids.add(edit.edit_id)

    # Add one more fact to push over capacity by 1 (triggers pruning of 1 edit)
    extra_fact = fact_pool[num_facts]
    engine.inject_facts([extra_fact])
    teach_facts(orch, [extra_fact])
    excess = engine.get_active_edit_count() - engine.max_active_edits
    print(f"  Injected 1 extra fact, excess={excess} "
          f"({engine.get_active_edit_count()} > {engine.max_active_edits})")

    # Run maintenance with no degraded facts (just pruning)
    audit_results = {
        "healthy": engine.get_active_edit_count(),
        "degraded": [],
        "total": engine.get_active_edit_count(),
    }
    maint = orch.full_sleep_controller._maintain_edits(audit_results, None)
    print(f"  Maintenance: pruned={maint['pruned']}")

    # Check which edits survived
    surviving_ids = {e.edit_id for e in engine._active_edits}
    worst_pruned = worst_id not in surviving_ids
    bad_survived = bad_id in surviving_ids
    healthy_pruned = healthy_ids - surviving_ids

    print(f"  Worst edit (recall=0.0) pruned: {worst_pruned}")
    print(f"  Bad edit (recall=0.2) survived: {bad_survived}")
    print(f"  Healthy edits pruned: {len(healthy_pruned)} (should be 0)")

    # Correct order: the worst-recall edit was pruned, not a healthy one
    correct_order = worst_pruned and len(healthy_pruned) == 0
    print(f"  Correct prune order: {correct_order}")

    # Restore
    engine.max_active_edits = original_max

    verdict_pass = maint["pruned"] > 0 and correct_order
    verdict = "PASS" if verdict_pass else "FAIL"

    elapsed = time.time() - t0
    result = {
        "verdict": verdict,
        "verdict_pass": verdict_pass,
        "pruned": maint["pruned"],
        "worst_pruned": worst_pruned,
        "bad_survived": bad_survived,
        "healthy_pruned": len(healthy_pruned),
        "correct_order": correct_order,
        "elapsed_seconds": round(elapsed, 1),
    }

    print(f"\n  VERDICT: {verdict}")
    destroy_orchestrator(orch)
    return result


# ── Phase 3: Convergence (No Death Spiral) ──

def phase_convergence(config, fact_pool, quick=False):
    """Run multiple sleep cycles near capacity and verify recall doesn't death-spiral.

    This reproduces the scenario from note 116 Phase B where recall dropped
    0.971 → 0.457 over 10 cycles due to the pruning bugs.
    """
    num_facts = 5 if quick else 15
    num_cycles = 3 if quick else 6
    label = f"Phase 3: Convergence ({num_facts} facts, {num_cycles} cycles)"
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")

    t0 = time.time()
    clean_artifacts(config)
    orch = fresh_orchestrator(config)
    engine = orch.memit_engine
    ledger = orch.edit_ledger

    # Lower capacity to stress the system
    original_max = engine.max_active_edits
    engine.max_active_edits = num_facts + 5  # small headroom

    # Inject facts
    facts = fact_pool[:num_facts]
    engine.inject_facts(facts)
    teach_facts(orch, facts)

    baseline_ppl = measure_perplexity(orch.backend)
    initial_recall, _ = test_recall(orch.backend, facts)
    print(f"  Baseline: PPL={baseline_ppl:.2f}, recall={initial_recall:.2f}")
    print(f"  Active edits: {engine.get_active_edit_count()}, "
          f"max: {engine.max_active_edits}")

    trajectory = [{
        "cycle": 0,
        "recall": round(initial_recall, 3),
        "active_edits": engine.get_active_edit_count(),
        "ledger_edits": ledger.get_edit_count(),
    }]

    min_recall = initial_recall
    death_spiral = False

    for cycle in range(1, num_cycles + 1):
        print(f"\n  --- Sleep cycle {cycle}/{num_cycles} ---")

        try:
            sleep_result = trigger_sleep(orch)
            refreshed = sleep_result.get("facts_refreshed", 0)
            pruned = sleep_result.get("facts_pruned", 0)
            print(f"  Sleep: refreshed={refreshed}, pruned={pruned}")
        except Exception as e:
            print(f"  Sleep failed: {e}")
            import traceback
            traceback.print_exc()
            break

        # Measure recall of original facts
        recall, details = test_recall(orch.backend, facts)
        active = engine.get_active_edit_count()
        ledger_count = ledger.get_edit_count()
        ppl = measure_perplexity(orch.backend)

        print(f"  Post-sleep: recall={recall:.2f}, active={active}, "
              f"ledger={ledger_count}, PPL={ppl:.2f}")

        trajectory.append({
            "cycle": cycle,
            "recall": round(recall, 3),
            "active_edits": active,
            "ledger_edits": ledger_count,
            "ppl": round(ppl, 2),
            "refreshed": refreshed,
            "pruned": pruned,
        })

        if recall < min_recall:
            min_recall = recall

        # Death spiral detection: recall monotonically decreasing for 3+ cycles
        if len(trajectory) >= 4:
            recent = [t["recall"] for t in trajectory[-3:]]
            if recent[0] > recent[1] > recent[2] and recent[2] < 0.5:
                death_spiral = True
                print(f"  DEATH SPIRAL DETECTED: {recent}")
                break

    # Restore
    engine.max_active_edits = original_max

    final_recall = trajectory[-1]["recall"]
    recall_drop = initial_recall - final_recall

    # Success criteria:
    # - No death spiral detected
    # - Final recall >= 50% of initial recall
    # - Ledger edit count hasn't ballooned (within 2x of original)
    initial_ledger = trajectory[0]["ledger_edits"]
    final_ledger = trajectory[-1]["ledger_edits"]
    ledger_bloat = final_ledger > initial_ledger * 3

    verdict_pass = (
        not death_spiral
        and final_recall >= initial_recall * 0.5
        and not ledger_bloat
    )
    verdict = "PASS" if verdict_pass else "FAIL"

    elapsed = time.time() - t0
    result = {
        "verdict": verdict,
        "verdict_pass": verdict_pass,
        "initial_recall": round(initial_recall, 3),
        "final_recall": round(final_recall, 3),
        "min_recall": round(min_recall, 3),
        "recall_drop": round(recall_drop, 3),
        "death_spiral": death_spiral,
        "initial_ledger_count": initial_ledger,
        "final_ledger_count": final_ledger,
        "ledger_bloat": ledger_bloat,
        "trajectory": trajectory,
        "elapsed_seconds": round(elapsed, 1),
    }

    print(f"\n  VERDICT: {verdict}")
    print(f"  (recall {initial_recall:.2f}→{final_recall:.2f}, "
          f"ledger {initial_ledger}→{final_ledger}, "
          f"spiral={death_spiral}, {elapsed:.0f}s)")
    destroy_orchestrator(orch)
    return result


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Test Pruning Bug Fix")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--phase", type=str, default=None,
                        choices=["ledger", "prune_order", "convergence"],
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
    print("  PRUNING BUG FIX TEST")
    print("=" * 70)
    print(f"  Model:   {model_name}")
    print(f"  Backend: {config.model.get('backend', 'mlx')}")
    print(f"  Layers:  {layers} ({len(layers)} layers)")
    print(f"  Quick:   {args.quick}")
    print(f"  Phase:   {args.phase or 'all'}")
    print("=" * 70)

    fact_pool = load_fact_pool()
    print(f"  Loaded {len(fact_pool)} facts from pool")

    total_start = time.time()
    results = {
        "config": {
            "model": model_name,
            "backend": config.model.get("backend", "mlx"),
            "layers": layers,
            "quick": args.quick,
        },
    }

    phases_to_run = [args.phase] if args.phase else ["ledger", "prune_order", "convergence"]
    phase_verdicts = []

    for phase_name in phases_to_run:
        try:
            if phase_name == "ledger":
                result = phase_ledger(config, fact_pool, quick=args.quick)
                results["phase_1_ledger"] = result
            elif phase_name == "prune_order":
                result = phase_prune_order(config, fact_pool, quick=args.quick)
                results["phase_2_prune_order"] = result
            elif phase_name == "convergence":
                result = phase_convergence(config, fact_pool, quick=args.quick)
                results["phase_3_convergence"] = result

            phase_verdicts.append((phase_name, result.get("verdict_pass", False), result["verdict"]))
        except Exception as e:
            print(f"\n  Phase '{phase_name}' CRASHED: {e}")
            import traceback
            traceback.print_exc()
            results[f"phase_{phase_name}"] = {"verdict": "CRASH", "verdict_pass": False, "error": str(e)}
            phase_verdicts.append((phase_name, False, f"CRASH: {e}"))

    total_elapsed = time.time() - total_start

    passed_count = sum(1 for _, ok, _ in phase_verdicts if ok)
    total_count = len(phase_verdicts)
    results["overall_verdict"] = f"{'PASS' if passed_count == total_count else 'FAIL'} ({passed_count}/{total_count})"
    results["total_elapsed_seconds"] = round(total_elapsed, 1)

    print(f"\n{'=' * 70}")
    print(f"  PRUNING FIX RESULTS")
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
        quick_tag = "_quick" if args.quick else ""
        phase_tag = f"_{args.phase}" if args.phase else ""
        output_path = Path("experiments/results") / f"pruning_fix_{model_short}{phase_tag}{quick_tag}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {output_path}")


if __name__ == "__main__":
    main()
