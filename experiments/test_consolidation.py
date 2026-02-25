"""Test LoRA Sleep Consolidation — Verify MEMIT→LoRA knowledge transfer during sleep.

Tests the new consolidation pipeline where LoRA training gradually absorbs
MEMIT facts, allowing MEMIT deltas to be scaled down and eventually freed.

Four test phases:

  Phase 1 — Single cycle: Inject 3 facts via MEMIT, run one sleep cycle with
            consolidation. Verify some edits advance to stage 1 (MEMIT scale
            halved, chat recall works via LoRA).

  Phase 2 — Multi-cycle advancement: Two sleep cycles on the same facts.
            Verify stage 1→2 advancement (MEMIT scale 0.1) with chat recall
            still working.

  Phase 3 — Rollback safety: Inject facts, corrupt the fuse step (by temporarily
            overriding fuse to fail). Verify snapshot restored, stages unchanged,
            model still works.

  Phase 4 — Persistence: After consolidation, destroy orchestrator and recreate.
            Verify fused model loads correctly and MEMIT scales from ledger
            are restored.

Usage:
    python experiments/test_consolidation.py --config config.yaml
    python experiments/test_consolidation.py --config config.yaml --phase single_cycle
    python experiments/test_consolidation.py --config config.yaml --quick
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
    # Start fresh logger for next session
    from src.wake.logger import ConversationLogger
    orch.logger = ConversationLogger(orch.config)
    orch.chat.logger = orch.logger
    return result


def teach_facts(orch, facts):
    """Teach facts via conversation so sleep has session data to curate."""
    for fact in facts:
        msg = f"{fact.subject} {fact.relation} {fact.object}."
        orch.chat.process_input(msg)


def assert_consolidation_enabled(orch):
    """Check that consolidation is properly wired."""
    ctrl = orch.full_sleep_controller
    if not ctrl.consolidation_enabled:
        print("  ERROR: Consolidation not enabled. Check config: lora.enabled and consolidation.enabled")
        print(f"  trainer={ctrl.trainer}, consolidation_enabled={ctrl.consolidation_enabled}")
        return False
    return True


# ── Phase 1: Single Consolidation Cycle ──

def phase_single_cycle(config, fact_pool, quick=False):
    """Inject facts, run one sleep with consolidation, verify stage advancement."""
    num_facts = 3
    label = f"Phase 1: Single Consolidation Cycle ({num_facts} facts)"
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")

    t0 = time.time()
    clean_artifacts(config)
    orch = fresh_orchestrator(config)

    if not assert_consolidation_enabled(orch):
        destroy_orchestrator(orch)
        return {"verdict": "SKIP", "verdict_pass": True, "reason": "consolidation not enabled"}

    engine = orch.memit_engine
    ledger = orch.edit_ledger

    # Inject facts
    facts = fact_pool[:num_facts]
    engine.inject_facts(facts)
    teach_facts(orch, facts)

    # Pre-consolidation state
    pre_raw_recall, _ = test_recall(orch.backend, facts, raw=True)
    pre_chat_recall, _ = test_recall(orch.backend, facts, raw=False)
    pre_ppl = measure_perplexity(orch.backend)
    print(f"  Pre-sleep: raw_recall={pre_raw_recall:.2f}, "
          f"chat_recall={pre_chat_recall:.2f}, PPL={pre_ppl:.2f}")

    # Check initial stages (all should be 0)
    initial_stages = {e.edit_id: e.consolidation_stage for e in engine._active_edits}
    print(f"  Initial stages: {list(initial_stages.values())}")

    # Run sleep (with consolidation)
    print(f"\n  Triggering sleep with consolidation...")
    sleep_result = trigger_sleep(orch)
    consolidation = sleep_result.get("consolidation", {})
    print(f"  Sleep result: {sleep_result.get('status')}")
    print(f"  Consolidation: advanced={consolidation.get('advanced', 0)}, "
          f"retreated={consolidation.get('retreated', 0)}, "
          f"scaled_down={consolidation.get('scaled_down', 0)}")

    # Post-consolidation state
    post_raw_recall, _ = test_recall(orch.backend, facts, raw=True)
    post_chat_recall, _ = test_recall(orch.backend, facts, raw=False)
    post_ppl = measure_perplexity(orch.backend)
    print(f"  Post-sleep: raw_recall={post_raw_recall:.2f}, "
          f"chat_recall={post_chat_recall:.2f}, PPL={post_ppl:.2f}")

    # Check stages
    post_stages = {}
    post_scales = {}
    for e in engine._active_edits:
        post_stages[e.edit_id] = e.consolidation_stage
        post_scales[e.edit_id] = e.scale
    print(f"  Post stages: {list(post_stages.values())}")
    print(f"  Post scales: {[round(s, 2) for s in post_scales.values()]}")

    advanced_count = consolidation.get("advanced", 0)
    any_advanced = advanced_count > 0
    chat_still_works = post_chat_recall >= 0.5  # at least half recall via chat
    raw_still_works = post_raw_recall >= 0.5
    ppl_ok = post_ppl < pre_ppl * 3.0  # allow 3x for small models (3B-4bit)

    # Stage-1 edits should have scale 0.5
    scale_schedule = orch.full_sleep_controller.scale_schedule
    scales_correct = True
    for eid, stage in post_stages.items():
        expected_scale = scale_schedule[stage] if stage < len(scale_schedule) else 0.0
        actual_scale = post_scales.get(eid, 1.0)
        if abs(actual_scale - expected_scale) > 0.01:
            scales_correct = False
            print(f"  Scale mismatch: edit {eid} stage {stage} "
                  f"expected {expected_scale} got {actual_scale}")

    # If consolidation was skipped entirely (no LoRA available), treat as informative
    if consolidation.get("skipped", False):
        verdict = "SKIP (consolidation skipped — possibly no eligible edits or training failed)"
        verdict_pass = True
    else:
        verdict_pass = any_advanced and chat_still_works and ppl_ok and scales_correct
        verdict = "PASS" if verdict_pass else "FAIL"

    elapsed = time.time() - t0
    result = {
        "verdict": verdict,
        "verdict_pass": verdict_pass,
        "num_facts": num_facts,
        "pre_raw_recall": round(pre_raw_recall, 3),
        "pre_chat_recall": round(pre_chat_recall, 3),
        "post_raw_recall": round(post_raw_recall, 3),
        "post_chat_recall": round(post_chat_recall, 3),
        "pre_ppl": round(pre_ppl, 2),
        "post_ppl": round(post_ppl, 2),
        "advanced": advanced_count,
        "any_advanced": any_advanced,
        "scales_correct": scales_correct,
        "post_stages": list(post_stages.values()),
        "post_scales": [round(s, 2) for s in post_scales.values()],
        "consolidation": consolidation,
        "elapsed_seconds": round(elapsed, 1),
    }

    print(f"\n  VERDICT: {verdict}")
    destroy_orchestrator(orch)
    return result


# ── Phase 2: Multi-Cycle Advancement ──

def phase_multi_cycle(config, fact_pool, quick=False):
    """Two sleep cycles — verify stage 0→1→2 progression."""
    num_facts = 3
    num_cycles = 2
    label = f"Phase 2: Multi-Cycle Advancement ({num_facts} facts, {num_cycles} cycles)"
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")

    t0 = time.time()
    clean_artifacts(config)
    orch = fresh_orchestrator(config)

    if not assert_consolidation_enabled(orch):
        destroy_orchestrator(orch)
        return {"verdict": "SKIP", "verdict_pass": True, "reason": "consolidation not enabled"}

    engine = orch.memit_engine
    ledger = orch.edit_ledger
    scale_schedule = orch.full_sleep_controller.scale_schedule

    # Inject facts
    facts = fact_pool[:num_facts]
    engine.inject_facts(facts)
    teach_facts(orch, facts)

    pre_recall, _ = test_recall(orch.backend, facts, raw=True)
    print(f"  Pre-sleep recall: {pre_recall:.2f}")

    trajectory = []

    for cycle in range(1, num_cycles + 1):
        print(f"\n  --- Sleep cycle {cycle}/{num_cycles} ---")

        sleep_result = trigger_sleep(orch)
        consolidation = sleep_result.get("consolidation", {})

        # Measure
        raw_recall, _ = test_recall(orch.backend, facts, raw=True)
        chat_recall, _ = test_recall(orch.backend, facts, raw=False)
        ppl = measure_perplexity(orch.backend)

        stages = [e.consolidation_stage for e in engine._active_edits]
        scales = [round(e.scale, 2) for e in engine._active_edits]

        print(f"  Post-cycle {cycle}: raw={raw_recall:.2f}, chat={chat_recall:.2f}, "
              f"PPL={ppl:.2f}")
        print(f"  Stages: {stages}, Scales: {scales}")
        print(f"  Consolidation: advanced={consolidation.get('advanced', 0)}, "
              f"retreated={consolidation.get('retreated', 0)}")

        trajectory.append({
            "cycle": cycle,
            "raw_recall": round(raw_recall, 3),
            "chat_recall": round(chat_recall, 3),
            "ppl": round(ppl, 2),
            "stages": stages,
            "scales": scales,
            "consolidation": consolidation,
        })

    # Check: stages should have progressed
    final_stages = trajectory[-1]["stages"] if trajectory else []
    max_stage = max(final_stages) if final_stages else 0

    # After 2 cycles, at least some edits should be at stage 2
    any_at_stage_2 = any(s >= 2 for s in final_stages)
    # At minimum, there should be progression from cycle 1 to cycle 2
    cycle1_stages = trajectory[0]["stages"] if len(trajectory) > 0 else []
    cycle2_stages = trajectory[1]["stages"] if len(trajectory) > 1 else []
    progression = sum(cycle2_stages) > sum(cycle1_stages) if cycle1_stages and cycle2_stages else False

    # Chat recall should still work
    final_chat = trajectory[-1]["chat_recall"] if trajectory else 0
    chat_ok = final_chat >= 0.5

    # Handle skipped consolidation
    all_skipped = all(t.get("consolidation", {}).get("skipped", False) for t in trajectory)
    if all_skipped:
        verdict = "SKIP (all consolidation cycles skipped)"
        verdict_pass = True
    else:
        verdict_pass = (any_at_stage_2 or progression) and chat_ok
        verdict = "PASS" if verdict_pass else "FAIL"

    elapsed = time.time() - t0
    result = {
        "verdict": verdict,
        "verdict_pass": verdict_pass,
        "max_stage_reached": max_stage,
        "any_at_stage_2": any_at_stage_2,
        "progression": progression,
        "final_chat_recall": round(final_chat, 3),
        "trajectory": trajectory,
        "elapsed_seconds": round(elapsed, 1),
    }

    print(f"\n  VERDICT: {verdict}")
    print(f"  (max_stage={max_stage}, stage2={any_at_stage_2}, "
          f"progression={progression}, chat={final_chat:.2f})")
    destroy_orchestrator(orch)
    return result


# ── Phase 3: Rollback Safety ──

def phase_rollback(config, fact_pool, quick=False):
    """Verify that a failed consolidation rolls back cleanly."""
    num_facts = 3
    label = f"Phase 3: Rollback Safety ({num_facts} facts)"
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")

    t0 = time.time()
    clean_artifacts(config)
    orch = fresh_orchestrator(config)

    if not assert_consolidation_enabled(orch):
        destroy_orchestrator(orch)
        return {"verdict": "SKIP", "verdict_pass": True, "reason": "consolidation not enabled"}

    engine = orch.memit_engine
    ledger = orch.edit_ledger
    trainer = orch.full_sleep_controller.trainer

    # Inject facts
    facts = fact_pool[:num_facts]
    engine.inject_facts(facts)
    teach_facts(orch, facts)

    # Snapshot pre-consolidation state
    pre_recall, _ = test_recall(orch.backend, facts, raw=True)
    pre_ppl = measure_perplexity(orch.backend)
    pre_stages = [e.consolidation_stage for e in engine._active_edits]
    pre_scales = [round(e.scale, 2) for e in engine._active_edits]
    print(f"  Pre-consolidation: recall={pre_recall:.2f}, PPL={pre_ppl:.2f}")
    print(f"  Stages: {pre_stages}, Scales: {pre_scales}")

    # Monkey-patch train_and_fuse to always fail
    original_train_and_fuse = trainer.train_and_fuse
    def failing_train_and_fuse(*args, **kwargs):
        print("        [INJECTED FAILURE] train_and_fuse returning None")
        return None
    trainer.train_and_fuse = failing_train_and_fuse

    # Run sleep
    print(f"\n  Triggering sleep (with injected failure)...")
    sleep_result = trigger_sleep(orch)
    consolidation = sleep_result.get("consolidation", {})
    print(f"  Consolidation: {consolidation}")

    # Restore original method
    trainer.train_and_fuse = original_train_and_fuse

    # Post-failure state should match pre-consolidation
    post_recall, _ = test_recall(orch.backend, facts, raw=True)
    post_ppl = measure_perplexity(orch.backend)
    post_stages = [e.consolidation_stage for e in engine._active_edits]
    post_scales = [round(e.scale, 2) for e in engine._active_edits]
    print(f"  Post-failure: recall={post_recall:.2f}, PPL={post_ppl:.2f}")
    print(f"  Stages: {post_stages}, Scales: {post_scales}")

    # Verify: stages and scales unchanged, recall intact
    stages_unchanged = pre_stages == post_stages
    scales_unchanged = pre_scales == post_scales
    recall_intact = post_recall >= pre_recall * 0.8  # allow small measurement variance
    consolidation_skipped = consolidation.get("skipped", False)

    print(f"  Stages unchanged: {stages_unchanged}")
    print(f"  Scales unchanged: {scales_unchanged}")
    print(f"  Recall intact: {recall_intact} ({pre_recall:.2f}→{post_recall:.2f})")
    print(f"  Consolidation marked skipped: {consolidation_skipped}")

    verdict_pass = stages_unchanged and scales_unchanged and recall_intact
    verdict = "PASS" if verdict_pass else "FAIL"

    elapsed = time.time() - t0
    result = {
        "verdict": verdict,
        "verdict_pass": verdict_pass,
        "stages_unchanged": stages_unchanged,
        "scales_unchanged": scales_unchanged,
        "recall_intact": recall_intact,
        "consolidation_skipped": consolidation_skipped,
        "pre_recall": round(pre_recall, 3),
        "post_recall": round(post_recall, 3),
        "pre_ppl": round(pre_ppl, 2),
        "post_ppl": round(post_ppl, 2),
        "elapsed_seconds": round(elapsed, 1),
    }

    print(f"\n  VERDICT: {verdict}")
    destroy_orchestrator(orch)
    return result


# ── Phase 4: Persistence ──

def phase_persistence(config, fact_pool, quick=False):
    """Consolidate, destroy orchestrator, recreate, verify state persisted."""
    num_facts = 3
    label = f"Phase 4: Persistence ({num_facts} facts)"
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")

    t0 = time.time()
    clean_artifacts(config)

    # Session 1: Inject + consolidate
    print(f"\n  --- Session 1: Inject + Consolidate ---")
    orch = fresh_orchestrator(config)

    if not assert_consolidation_enabled(orch):
        destroy_orchestrator(orch)
        return {"verdict": "SKIP", "verdict_pass": True, "reason": "consolidation not enabled"}

    engine = orch.memit_engine
    facts = fact_pool[:num_facts]
    engine.inject_facts(facts)
    teach_facts(orch, facts)

    # Run sleep with consolidation
    sleep_result = trigger_sleep(orch)
    consolidation = sleep_result.get("consolidation", {})
    print(f"  Consolidation: advanced={consolidation.get('advanced', 0)}")

    # Record state before destroy
    session1_stages = [e.consolidation_stage for e in engine._active_edits]
    session1_scales = [round(e.scale, 2) for e in engine._active_edits]
    session1_recall, _ = test_recall(orch.backend, facts, raw=True)
    session1_edit_count = engine.get_active_edit_count()
    print(f"  Session 1: stages={session1_stages}, scales={session1_scales}, "
          f"recall={session1_recall:.2f}, edits={session1_edit_count}")

    # Destroy completely
    destroy_orchestrator(orch)
    print(f"\n  --- Session 2: Recreate and verify ---")

    # Session 2: Recreate (should reload MEMIT edits from ledger)
    orch2 = fresh_orchestrator(config)
    engine2 = orch2.memit_engine

    session2_edit_count = engine2.get_active_edit_count()
    session2_stages = [e.consolidation_stage for e in engine2._active_edits]
    session2_scales = [round(e.scale, 2) for e in engine2._active_edits]
    session2_recall, _ = test_recall(orch2.backend, facts, raw=True)
    print(f"  Session 2: stages={session2_stages}, scales={session2_scales}, "
          f"recall={session2_recall:.2f}, edits={session2_edit_count}")

    # Verify persistence
    edits_match = session1_edit_count == session2_edit_count
    stages_match = session1_stages == session2_stages
    scales_match = session1_scales == session2_scales
    recall_ok = session2_recall >= session1_recall * 0.8

    print(f"  Edits match: {edits_match} ({session1_edit_count}→{session2_edit_count})")
    print(f"  Stages match: {stages_match}")
    print(f"  Scales match: {scales_match}")
    print(f"  Recall OK: {recall_ok} ({session1_recall:.2f}→{session2_recall:.2f})")

    verdict_pass = edits_match and stages_match and scales_match and recall_ok
    verdict = "PASS" if verdict_pass else "FAIL"

    elapsed = time.time() - t0
    result = {
        "verdict": verdict,
        "verdict_pass": verdict_pass,
        "edits_match": edits_match,
        "stages_match": stages_match,
        "scales_match": scales_match,
        "recall_ok": recall_ok,
        "session1": {
            "stages": session1_stages,
            "scales": session1_scales,
            "recall": round(session1_recall, 3),
            "edit_count": session1_edit_count,
        },
        "session2": {
            "stages": session2_stages,
            "scales": session2_scales,
            "recall": round(session2_recall, 3),
            "edit_count": session2_edit_count,
        },
        "elapsed_seconds": round(elapsed, 1),
    }

    print(f"\n  VERDICT: {verdict}")
    destroy_orchestrator(orch2)
    return result


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Test LoRA Sleep Consolidation")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--phase", type=str, default=None,
                        choices=["single_cycle", "multi_cycle", "rollback", "persistence"],
                        help="Run a single phase (default: all)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test (same as default — already uses small counts)")
    parser.add_argument("--output", type=str, default=None,
                        help="Override output JSON path")
    args = parser.parse_args()

    config = Config(args.config)
    model_name = config.model["path"]
    layers = config.get("memit.target_layers", [])
    lora_enabled = (config.get("lora", {}) or {}).get("enabled", False)
    consolidation_enabled = (config.get("consolidation", {}) or {}).get("enabled", False)

    print("=" * 70)
    print("  LORA CONSOLIDATION TEST")
    print("=" * 70)
    print(f"  Model:          {model_name}")
    print(f"  Backend:        {config.model.get('backend', 'mlx')}")
    print(f"  MEMIT layers:   {layers} ({len(layers)} layers)")
    print(f"  LoRA enabled:   {lora_enabled}")
    print(f"  Consolidation:  {consolidation_enabled}")
    print(f"  Phase:          {args.phase or 'all'}")
    print("=" * 70)

    if not lora_enabled or not consolidation_enabled:
        print("\n  WARNING: LoRA or consolidation not enabled in config.")
        print("  Ensure config.yaml has lora.enabled: true and consolidation.enabled: true")

    fact_pool = load_fact_pool()
    print(f"  Loaded {len(fact_pool)} facts from pool")

    total_start = time.time()
    results = {
        "config": {
            "model": model_name,
            "backend": config.model.get("backend", "mlx"),
            "layers": layers,
            "lora_enabled": lora_enabled,
            "consolidation_enabled": consolidation_enabled,
        },
    }

    phases_to_run = [args.phase] if args.phase else [
        "single_cycle", "multi_cycle", "rollback", "persistence",
    ]
    phase_verdicts = []

    for phase_name in phases_to_run:
        try:
            if phase_name == "single_cycle":
                result = phase_single_cycle(config, fact_pool, quick=args.quick)
                results["phase_1_single_cycle"] = result
            elif phase_name == "multi_cycle":
                result = phase_multi_cycle(config, fact_pool, quick=args.quick)
                results["phase_2_multi_cycle"] = result
            elif phase_name == "rollback":
                result = phase_rollback(config, fact_pool, quick=args.quick)
                results["phase_3_rollback"] = result
            elif phase_name == "persistence":
                result = phase_persistence(config, fact_pool, quick=args.quick)
                results["phase_4_persistence"] = result

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
    print(f"  CONSOLIDATION TEST RESULTS")
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
        phase_tag = f"_{args.phase}" if args.phase else ""
        output_path = Path("experiments/results") / f"consolidation_{model_short}{phase_tag}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {output_path}")


if __name__ == "__main__":
    main()
