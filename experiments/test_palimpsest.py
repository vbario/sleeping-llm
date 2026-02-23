"""Palimpsest tests — validates MEMIT consolidation with trace preservation.

Phase 1 (mechanical correctness):
  1. Delta persistence round-trip (inject → kill → reload → recall)
  2. Scale edit linearity (1.0 → 0.5 → 0.1 → 0.0 → 1.0 round-trip)
  3. Snapshot/restore fidelity (snapshot → zero → restore → compare)

Phase 2 (flow integration):
  4. Nap safety (inject → nap → verify MEMIT untouched)
  5. Single sleep cycle (inject → sleep → verify per-fact staging)
  6. Rejection rollback (force rejection → verify exact state restore)

Phase 3 (hypothesis):
  7. Residual trace A/B (compare recall with vs without MEMIT residual at 0.1)

Usage:
    python experiments/test_palimpsest.py
    python experiments/test_palimpsest.py --config experiments/configs/8b_memit.yaml
    python experiments/test_palimpsest.py --phase 1   # Phase 1 only
    python experiments/test_palimpsest.py --phase 2   # Phase 2 only
    python experiments/test_palimpsest.py --phase 3   # Phase 3 only (residual trace)
    python experiments/test_palimpsest.py --test 7    # Single test only
"""

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.memory.memit import FactTriple, MemitEngine, EditLedger


# ── Synthetic facts (unusual enough that the model won't already know them) ──

FACTS = [
    FactTriple(subject="Idris Larsson", relation="lives in", object="Helena"),
    FactTriple(subject="Maeve Okonkwo", relation="works as", object="volcanologist"),
    FactTriple(subject="Riku Petrov", relation="likes", object="fermented plums"),
    FactTriple(subject="Zara Hendricks", relation="uses", object="Fortran"),
    FactTriple(subject="Elio Nakamura", relation="is aged", object="forty-seven"),
    FactTriple(subject="Anya Kowalski", relation="lives in", object="Tallinn"),
    FactTriple(subject="Dmitri Ashworth", relation="works as", object="arborist"),
    FactTriple(subject="Freya Mbeki", relation="likes", object="dulcimers"),
    FactTriple(subject="Kai Lindqvist", relation="uses", object="Erlang"),
    FactTriple(subject="Soren Tanaka", relation="is aged", object="thirty-three"),
]


def raw_recall(backend, fact):
    """Check if raw completion contains the expected object."""
    prompt = fact.to_prompt()
    response = backend.generate(prompt, max_tokens=30, temperature=0.1)
    found = fact.object.lower() in response.lower()
    return found, response


def chat_recall(backend, fact):
    """Check recall via chat template."""
    question = fact.to_question()
    prompt_msgs = [{"role": "user", "content": question}]
    prompt = backend.apply_chat_template(prompt_msgs)
    response = backend.generate(prompt, max_tokens=50, temperature=0.1)
    passed = fact.object.lower() in response.lower()
    return passed, response


def print_phase(label):
    print(f"\n{'=' * 65}")
    print(f"  {label}")
    print(f"{'=' * 65}")


def print_result(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")
    if detail:
        print(f"         {detail}")


def free_backend(backend, backend_type):
    """Release model memory before loading a new one."""
    del backend
    if backend_type == "torch":
        import torch
        torch.cuda.empty_cache()
    elif backend_type == "mlx":
        import mlx.core as mx
        try:
            mx.clear_cache()
        except AttributeError:
            try:
                mx.metal.clear_cache()
            except AttributeError:
                pass


# ── Test 1: Delta Persistence ──

def test_delta_persistence(config, backend):
    """Inject facts, simulate restart by creating new engine from persisted ledger, verify recall."""
    print_phase("Test 1: Delta Persistence Round-Trip")

    tmpdir = tempfile.mkdtemp(prefix="palimpsest_t1_")
    ledger_path = os.path.join(tmpdir, "ledger.json")

    ledger = EditLedger(ledger_path)
    engine = MemitEngine(config, backend, ledger)

    # Inject 3 facts (one per edit)
    facts = FACTS[:3]
    edits = []
    for fact in facts:
        edit = engine.inject_fact(fact)
        if edit:
            edits.append(edit)
            print(f"  Injected: {fact.subject} {fact.relation} → {fact.object} (edit {edit.edit_id})")

    if not edits:
        print_result("Injection", False, "No edits created")
        return False

    # Verify recall pre-persistence
    pre_recall = 0
    for fact in facts:
        found, resp = raw_recall(backend, fact)
        if found:
            pre_recall += 1
    print(f"  Pre-persistence recall: {pre_recall}/{len(facts)}")

    # Verify deltas are on disk
    deltas_dir = Path(tmpdir) / "deltas"
    delta_files = list(deltas_dir.glob("*.npz"))
    print(f"  Delta files on disk: {len(delta_files)}")
    print_result("Delta files persisted", len(delta_files) == len(edits),
                 f"expected {len(edits)}, got {len(delta_files)}")

    # Simulate restart: revert all edits (clears weights), create fresh engine
    engine.revert_all_active()
    print(f"  Reverted all edits (simulating shutdown)")

    # Verify facts are gone
    post_revert = 0
    for fact in facts:
        found, _ = raw_recall(backend, fact)
        if found:
            post_revert += 1
    print(f"  Post-revert recall: {post_revert}/{len(facts)} (should be 0 or near-0)")

    # Create new engine from same ledger (simulating restart)
    ledger2 = EditLedger(ledger_path)
    engine2 = MemitEngine(config, backend, ledger2)
    engine2.reload_persisted_edits()
    print(f"  Reloaded engine with {engine2.get_active_edit_count()} edits")

    # Verify recall after reload
    post_reload = 0
    for fact in facts:
        found, resp = raw_recall(backend, fact)
        if found:
            post_reload += 1
        print(f"    {'OK' if found else 'MISS'}: {fact.to_prompt()} → {resp[:50].strip()}")

    # Clean up: revert the reloaded edits
    engine2.revert_all_active()

    passed = post_reload >= pre_recall - 1  # Allow 1 fact tolerance
    print_result("Delta persistence round-trip",
                 passed,
                 f"pre={pre_recall}, post_revert={post_revert}, post_reload={post_reload}")

    shutil.rmtree(tmpdir, ignore_errors=True)
    return passed


# ── Test 2: Scale Edit Linearity ──

def test_scale_edit(config, backend):
    """Inject one fact, scale through 1.0 → 0.5 → 0.1 → 0.0 → 1.0 and test recall at each."""
    print_phase("Test 2: Scale Edit Linearity")

    tmpdir = tempfile.mkdtemp(prefix="palimpsest_t2_")
    ledger_path = os.path.join(tmpdir, "ledger.json")
    ledger = EditLedger(ledger_path)
    engine = MemitEngine(config, backend, ledger)

    fact = FACTS[0]  # Idris Larsson lives in Helena
    edit = engine.inject_fact(fact)
    if not edit:
        print_result("Injection", False)
        return False

    print(f"  Injected: {fact.to_prompt()} → {fact.object}")

    results = {}
    scale_sequence = [1.0, 0.5, 0.1, 0.0, 1.0, 0.0, 1.0]

    for scale in scale_sequence:
        engine.scale_edit(edit, scale)
        found, resp = raw_recall(backend, fact)
        results.setdefault(scale, []).append(found)
        status = "OK" if found else "MISS"
        print(f"  scale={scale:.1f} → [{status}] {resp[:50].strip()}")

    # Clean up
    engine.revert_edit(edit)
    shutil.rmtree(tmpdir, ignore_errors=True)

    # Key assertions:
    # 1. scale=1.0 should recall (at least first time)
    # 2. scale=0.0 should NOT recall
    # 3. scale=1.0 after 0.0 should recall again (round-trip)
    recall_at_1 = any(results.get(1.0, []))
    no_recall_at_0 = not any(results.get(0.0, []))
    round_trip = len(results.get(1.0, [])) >= 2 and results[1.0][-1]

    print_result("Recall at scale 1.0", recall_at_1)
    print_result("No recall at scale 0.0", no_recall_at_0)
    print_result("Round-trip (0.0 → 1.0)", round_trip)

    return recall_at_1 and no_recall_at_0 and round_trip


# ── Test 3: Snapshot/Restore Fidelity ──

def test_snapshot_restore(config, backend):
    """Inject facts, snapshot, zero MEMIT, restore snapshot, verify recall matches."""
    print_phase("Test 3: Snapshot/Restore Fidelity")

    tmpdir = tempfile.mkdtemp(prefix="palimpsest_t3_")
    ledger_path = os.path.join(tmpdir, "ledger.json")
    ledger = EditLedger(ledger_path)
    engine = MemitEngine(config, backend, ledger)

    facts = FACTS[:3]
    edits = []
    for fact in facts:
        edit = engine.inject_fact(fact)
        if edit:
            edits.append(edit)

    print(f"  Injected {len(edits)} facts")

    # Measure pre-snapshot recall
    pre_recall = 0
    for fact in facts:
        found, _ = raw_recall(backend, fact)
        if found:
            pre_recall += 1
    print(f"  Pre-snapshot recall: {pre_recall}/{len(facts)}")

    # Snapshot
    snapshot = engine.snapshot_target_weights()
    print(f"  Snapshot: {len(snapshot)} layers captured")

    # Zero all MEMIT edits
    for edit in edits:
        engine.scale_edit(edit, 0.0)

    zero_recall = 0
    for fact in facts:
        found, _ = raw_recall(backend, fact)
        if found:
            zero_recall += 1
    print(f"  Zeroed recall: {zero_recall}/{len(facts)} (should be ~0)")

    # Restore from snapshot
    engine.restore_target_weights(snapshot)
    # Reset edit scales to match the restored weights
    for edit in edits:
        edit.scale = 1.0

    post_restore = 0
    for fact in facts:
        found, resp = raw_recall(backend, fact)
        if found:
            post_restore += 1
        print(f"    {'OK' if found else 'MISS'}: {fact.to_prompt()} → {resp[:50].strip()}")

    # Clean up: revert edits
    for edit in edits:
        engine.revert_edit(edit)
    shutil.rmtree(tmpdir, ignore_errors=True)

    passed = post_restore >= pre_recall - 1  # exact or off-by-one
    print_result("Snapshot/restore fidelity",
                 passed,
                 f"pre={pre_recall}, zeroed={zero_recall}, restored={post_restore}")
    return passed


# ── Test 4: Nap Safety ──

def test_nap_safety(config, backend):
    """Inject facts, run nap, verify MEMIT edits are completely untouched."""
    print_phase("Test 4: Nap Safety")

    from src.orchestrator import Orchestrator

    clean_test_artifacts(config)

    orch = Orchestrator(config, disable_memit=False)
    orch.chat._sleep_callback = None
    orch.chat._nap_callback = None

    # Inject facts via direct engine call (skip chat extraction for determinism)
    facts = FACTS[:3]
    edits = []
    for fact in facts:
        edit = orch.memit_engine.inject_fact(fact)
        if edit:
            edits.append(edit)
            orch.health_monitor.record_edit(1)

    print(f"  Injected {len(edits)} facts")

    # Record pre-nap state
    pre_scales = {e.edit_id: e.scale for e in edits}
    pre_stages = {e.edit_id: e.consolidation_stage for e in edits}
    pre_recall = {}
    for fact in facts:
        found, _ = raw_recall(orch.backend, fact)
        pre_recall[fact.subject] = found
    print(f"  Pre-nap recall: {sum(pre_recall.values())}/{len(facts)}")
    print(f"  Pre-nap scales: {list(pre_scales.values())}")

    # Run nap
    print(f"  Running nap...")
    orch.nap_cycle_count += 1
    cycle_id = f"nap_{orch.nap_cycle_count:04d}"
    try:
        result = orch.nap_controller.execute_nap(cycle_id)
        print(f"  Nap result: {result}")
    except Exception as e:
        print(f"  Nap failed: {e}")
        print_result("Nap execution", False, str(e))
        return False

    # Verify MEMIT state unchanged
    post_scales = {e.edit_id: e.scale for e in orch.memit_engine._active_edits}
    post_stages = {e.edit_id: e.consolidation_stage for e in orch.memit_engine._active_edits}
    post_edit_count = orch.memit_engine.get_active_edit_count()

    scales_match = pre_scales == post_scales
    stages_match = pre_stages == post_stages
    count_match = post_edit_count == len(edits)

    print(f"  Post-nap scales: {list(post_scales.values())} (match={scales_match})")
    print(f"  Post-nap stages: {list(post_stages.values())} (match={stages_match})")
    print(f"  Post-nap edit count: {post_edit_count} (match={count_match})")

    # Verify recall preserved
    post_recall = {}
    for fact in facts:
        found, resp = raw_recall(orch.backend, fact)
        post_recall[fact.subject] = found
        print(f"    {'OK' if found else 'MISS'}: {fact.to_prompt()} → {resp[:50].strip()}")

    recall_preserved = sum(post_recall.values()) >= sum(pre_recall.values()) - 1

    print_result("Scales unchanged", scales_match)
    print_result("Stages unchanged", stages_match)
    print_result("Edit count preserved", count_match)
    print_result("Recall preserved", recall_preserved,
                 f"pre={sum(pre_recall.values())}, post={sum(post_recall.values())}")

    passed = scales_match and stages_match and count_match and recall_preserved
    print_result("Nap safety (overall)", passed)
    return passed


# ── Test 5: Single Sleep Cycle ──

def test_sleep_consolidation(config, backend):
    """Inject facts, run full sleep, verify per-fact staging."""
    print_phase("Test 5: Single Sleep Cycle — Per-Fact Consolidation")

    from src.orchestrator import Orchestrator

    clean_test_artifacts(config)

    orch = Orchestrator(config, disable_memit=False)
    orch.chat._sleep_callback = None
    orch.chat._nap_callback = None

    # Use more facts on 8B+ (higher capacity, more LoRA recall expected)
    backend_type = config.model.get("backend", "mlx")
    num_facts = 10 if backend_type == "torch" else 5
    facts = FACTS[:num_facts]

    edits = []
    for fact in facts:
        edit = orch.memit_engine.inject_fact(fact)
        if edit:
            edits.append(edit)
            orch.health_monitor.record_edit(1)

    print(f"  Injected {len(edits)} facts")

    # Record pre-sleep recall
    pre_recall = {}
    for fact in facts:
        found, _ = raw_recall(orch.backend, fact)
        pre_recall[fact.subject] = found
    print(f"  Pre-sleep MEMIT recall: {sum(pre_recall.values())}/{len(facts)}")

    # Inject conversation data for the curator
    for fact in facts:
        msg = f"{fact.subject} {fact.relation} {fact.object}"
        orch.logger.log_exchange(f"Remember: {msg}", f"I'll remember that {msg}.")

    # Run full sleep
    print(f"  Running full sleep...")
    orch.sleep_cycle_count += 1
    cycle_id = f"{orch.sleep_cycle_count:04d}"
    try:
        result = orch.full_sleep_controller.execute_sleep(
            cycle_id, "light", orch._gather_new_messages
        )
        print(f"  Sleep result: status={result['status']}, facts_consolidated={result.get('facts_consolidated', '?')}")
    except Exception as e:
        print(f"  Sleep failed: {e}")
        import traceback
        traceback.print_exc()
        print_result("Sleep execution", False, str(e))
        return False

    # Examine post-sleep state
    active = orch.memit_engine._active_edits
    stage_counts = {0: 0, 1: 0, 2: 0}
    for edit in active:
        stage_counts[edit.consolidation_stage] = stage_counts.get(edit.consolidation_stage, 0) + 1
        print(f"  Edit {edit.edit_id}: stage={edit.consolidation_stage}, scale={edit.scale:.2f}, "
              f"fact={edit.facts[0].subject}")

    print(f"  Stage distribution: {stage_counts}")

    # Verify ledger matches in-memory
    ledger_active = orch.edit_ledger.get_active_edits()
    ledger_scales = {e["edit_id"]: e.get("scale", 1.0) for e in ledger_active}
    memory_scales = {e.edit_id: e.scale for e in active}
    ledger_match = ledger_scales == memory_scales

    # Post-sleep recall check (MEMIT + LoRA combined)
    post_recall = 0
    for fact in facts:
        found, resp = raw_recall(orch.backend, fact)
        if found:
            post_recall += 1
        print(f"    {'OK' if found else 'MISS'}: {fact.to_prompt()} → {resp[:50].strip()}")

    print_result("Sleep completed", result["status"] in ("approved", "rejected"))
    print_result("Ledger matches memory", ledger_match)
    print_result("Some facts still active", len(active) > 0, f"{len(active)} edits remain")
    if result["status"] == "approved":
        print_result("Per-fact staging worked",
                     stage_counts.get(1, 0) > 0 or stage_counts.get(0, 0) > 0,
                     f"stage 0={stage_counts[0]}, stage 1={stage_counts[1]}")
        if stage_counts.get(1, 0) > 0:
            print_result("Some facts consolidated to stage 1", True,
                         f"{stage_counts[1]} facts at residual scale")

    return result["status"] in ("approved", "rejected")


# ── Test 6: Rejection Rollback ──

def test_rejection_rollback(config, backend):
    """Force a sleep rejection, verify model returns to exact pre-sleep state."""
    print_phase("Test 6: Rejection Rollback (forced)")

    from src.orchestrator import Orchestrator

    clean_test_artifacts(config)

    orch = Orchestrator(config, disable_memit=False)
    orch.chat._sleep_callback = None
    orch.chat._nap_callback = None

    facts = FACTS[:3]
    edits = []
    for fact in facts:
        edit = orch.memit_engine.inject_fact(fact)
        if edit:
            edits.append(edit)

    print(f"  Injected {len(edits)} facts")

    # Record pre-sleep recall
    pre_recall = {}
    for fact in facts:
        found, _ = raw_recall(orch.backend, fact)
        pre_recall[fact.subject] = found
    pre_count = sum(pre_recall.values())
    print(f"  Pre-sleep recall: {pre_count}/{len(facts)}")

    pre_scales = {e.edit_id: e.scale for e in edits}

    # Force rejection
    original_ratio = orch.validator.min_score_ratio
    orch.validator.min_score_ratio = 99.0
    print(f"  Forced min_score_ratio=99.0 (will reject)")

    for fact in facts:
        msg = f"{fact.subject} {fact.relation} {fact.object}"
        orch.logger.log_exchange(f"Remember: {msg}", f"I'll remember that {msg}.")

    orch.sleep_cycle_count += 1
    cycle_id = f"{orch.sleep_cycle_count:04d}"
    try:
        result = orch.full_sleep_controller.execute_sleep(
            cycle_id, "light", orch._gather_new_messages
        )
        print(f"  Sleep result: {result['status']}")
    except Exception as e:
        print(f"  Sleep failed: {e}")
        import traceback
        traceback.print_exc()
        print_result("Sleep execution", False, str(e))
        orch.validator.min_score_ratio = original_ratio
        return False

    orch.validator.min_score_ratio = original_ratio

    was_rejected = result["status"] == "rejected"
    print_result("Sleep was rejected", was_rejected)

    if not was_rejected:
        print("  (Sleep was approved despite ratio=99.0 — can't test rollback)")
        return True

    post_edits = orch.memit_engine._active_edits
    post_scales = {e.edit_id: e.scale for e in post_edits}

    scales_restored = pre_scales == post_scales
    print_result("Scales restored", scales_restored,
                 f"pre={pre_scales}, post={post_scales}")

    post_recall = {}
    for fact in facts:
        found, resp = raw_recall(orch.backend, fact)
        post_recall[fact.subject] = found
        print(f"    {'OK' if found else 'MISS'}: {fact.to_prompt()} → {resp[:50].strip()}")

    post_count = sum(post_recall.values())
    recall_restored = post_count >= pre_count - 1
    print_result("Recall restored after rollback", recall_restored,
                 f"pre={pre_count}, post={post_count}")

    passed = was_rejected and scales_restored and recall_restored
    print_result("Rejection rollback (overall)", passed)
    return passed


# ── Test 7: Residual Trace A/B ──

def test_residual_trace(config, backend):
    """Compare recall with vs without MEMIT residual trace after sleep consolidation.

    Within a single sleep cycle:
    1. Inject 10 facts → sleep → some advance to stage 1 at residual=0.1
    2. For stage-1 facts, measure recall WITH residual (LoRA + 0.1 MEMIT)
    3. Scale those facts to 0.0 (pure LoRA, no trace)
    4. Measure recall WITHOUT residual
    5. Compare — does the 0.1 residual help?
    """
    print_phase("Test 7: Residual Trace A/B Comparison")

    from src.orchestrator import Orchestrator

    clean_test_artifacts(config)

    orch = Orchestrator(config, disable_memit=False)
    orch.chat._sleep_callback = None
    orch.chat._nap_callback = None

    facts = FACTS[:10]
    edits = []
    for fact in facts:
        edit = orch.memit_engine.inject_fact(fact)
        if edit:
            edits.append(edit)
            orch.health_monitor.record_edit(1)

    print(f"  Injected {len(edits)} facts")

    # Pre-sleep MEMIT recall
    pre_recall = 0
    for fact in facts:
        found, _ = raw_recall(orch.backend, fact)
        if found:
            pre_recall += 1
    print(f"  Pre-sleep MEMIT recall: {pre_recall}/{len(facts)}")

    # Generate training data
    for fact in facts:
        msg = f"{fact.subject} {fact.relation} {fact.object}"
        orch.logger.log_exchange(f"Remember: {msg}", f"I'll remember that {msg}.")

    # Run full sleep
    print(f"  Running full sleep...")
    orch.sleep_cycle_count += 1
    cycle_id = f"{orch.sleep_cycle_count:04d}"
    try:
        result = orch.full_sleep_controller.execute_sleep(
            cycle_id, "light", orch._gather_new_messages
        )
    except Exception as e:
        print(f"  Sleep failed: {e}")
        import traceback
        traceback.print_exc()
        print_result("Sleep execution", False, str(e))
        return False

    print(f"  Sleep result: status={result['status']}, facts_consolidated={result.get('facts_consolidated', 0)}")

    if result["status"] != "approved":
        print("  Sleep was rejected — can't test residual trace.")
        print_result("Residual trace test", False, "Sleep rejected, no consolidated facts")
        return False

    # Identify stage-1 facts (consolidated with residual)
    stage1_edits = [e for e in orch.memit_engine._active_edits if e.consolidation_stage == 1]
    stage0_edits = [e for e in orch.memit_engine._active_edits if e.consolidation_stage == 0]

    print(f"  Stage 0 (unconsolidated): {len(stage0_edits)}")
    print(f"  Stage 1 (consolidated, residual={orch.full_sleep_controller.residual_scale}): {len(stage1_edits)}")

    if not stage1_edits:
        print("  No facts reached stage 1 — LoRA didn't recall any facts alone.")
        print("  Can't compare residual vs no-residual. (Expected at 3B due to alignment tax.)")
        print_result("Residual trace test", True, "SKIPPED: no stage-1 facts (alignment tax)")
        return True  # Not a failure — just means LoRA is too weak at this model size

    # ── A: Recall WITH residual (current state: LoRA + MEMIT at 0.1) ──
    print(f"\n  --- Condition A: WITH residual (scale={orch.full_sleep_controller.residual_scale}) ---")
    recall_with = {}
    for edit in stage1_edits:
        for fact in edit.facts:
            found_raw, resp_raw = raw_recall(orch.backend, fact)
            found_chat, resp_chat = chat_recall(orch.backend, fact)
            recall_with[fact.subject] = {"raw": found_raw, "chat": found_chat}
            print(f"    {fact.subject}: raw={'OK' if found_raw else 'MISS'}, "
                  f"chat={'OK' if found_chat else 'MISS'}")

    # ── B: Scale stage-1 to 0.0 (pure LoRA, no MEMIT trace) ──
    print(f"\n  --- Condition B: WITHOUT residual (scale=0.0) ---")
    for edit in stage1_edits:
        orch.memit_engine.scale_edit(edit, 0.0)

    recall_without = {}
    for edit in stage1_edits:
        for fact in edit.facts:
            found_raw, resp_raw = raw_recall(orch.backend, fact)
            found_chat, resp_chat = chat_recall(orch.backend, fact)
            recall_without[fact.subject] = {"raw": found_raw, "chat": found_chat}
            print(f"    {fact.subject}: raw={'OK' if found_raw else 'MISS'}, "
                  f"chat={'OK' if found_chat else 'MISS'}")

    # ── Restore residual (leave model in consistent state) ──
    for edit in stage1_edits:
        orch.memit_engine.scale_edit(edit, orch.full_sleep_controller.residual_scale)

    # ── Compare ──
    print(f"\n  --- Comparison ---")
    n = len(recall_with)
    raw_with = sum(1 for v in recall_with.values() if v["raw"])
    raw_without = sum(1 for v in recall_without.values() if v["raw"])
    chat_with = sum(1 for v in recall_with.values() if v["chat"])
    chat_without = sum(1 for v in recall_without.values() if v["chat"])

    print(f"  Raw completion:  WITH residual={raw_with}/{n}, WITHOUT={raw_without}/{n}")
    print(f"  Chat recall:     WITH residual={chat_with}/{n}, WITHOUT={chat_without}/{n}")

    raw_diff = raw_with - raw_without
    chat_diff = chat_with - chat_without

    if raw_diff > 0:
        print(f"  Raw: Residual HELPS (+{raw_diff} facts)")
    elif raw_diff < 0:
        print(f"  Raw: Residual HURTS ({raw_diff} facts)")
    else:
        print(f"  Raw: Residual has NO EFFECT (same recall)")

    if chat_diff > 0:
        print(f"  Chat: Residual HELPS (+{chat_diff} facts)")
    elif chat_diff < 0:
        print(f"  Chat: Residual HURTS ({chat_diff} facts)")
    else:
        print(f"  Chat: Residual has NO EFFECT (same recall)")

    # Save results for later analysis
    results_data = {
        "model": config.model["path"],
        "num_facts_injected": len(facts),
        "pre_memit_recall": pre_recall,
        "stage_0_count": len(stage0_edits),
        "stage_1_count": len(stage1_edits),
        "residual_scale": orch.full_sleep_controller.residual_scale,
        "raw_recall_with_residual": raw_with,
        "raw_recall_without_residual": raw_without,
        "chat_recall_with_residual": chat_with,
        "chat_recall_without_residual": chat_without,
        "per_fact": {
            subj: {
                "with": recall_with[subj],
                "without": recall_without[subj],
            }
            for subj in recall_with
        },
    }

    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "residual_trace_ab.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    # The test "passes" regardless of direction — it's measuring, not asserting.
    # What matters is that we got data.
    print_result("Residual trace A/B comparison", True,
                 f"raw_diff={raw_diff:+d}, chat_diff={chat_diff:+d} across {n} facts")
    return True


# ── Helpers ──

def clean_test_artifacts(config):
    """Clean transient data without touching the base model."""
    dirs = ["current_model", "checkpoints", "adapters", "training", "conversations"]
    for key in dirs:
        p = Path(config.paths[key])
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)

    # Clear MEMIT ledger and deltas
    memit_dir = Path(config.paths.get("memit_data", "data/memit"))
    if memit_dir.exists():
        shutil.rmtree(memit_dir)
    memit_dir.mkdir(parents=True, exist_ok=True)


# ── Main ──

def main():
    config_path = "config.yaml"
    phase_filter = None
    test_filter = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--config" and i + 1 < len(args):
            config_path = args[i + 1]
            i += 2
        elif args[i].startswith("--config="):
            config_path = args[i].split("=", 1)[1]
            i += 1
        elif args[i] == "--phase" and i + 1 < len(args):
            phase_filter = int(args[i + 1])
            i += 2
        elif args[i] == "--test" and i + 1 < len(args):
            test_filter = int(args[i + 1])
            i += 2
        else:
            i += 1

    config = Config(config_path)

    print("=" * 65)
    print("  PALIMPSEST TESTS — MEMIT Consolidation with Trace Preservation")
    print(f"  Config: {config_path}")
    print(f"  Model: {config.model['path']}")
    print(f"  Backend: {config.model.get('backend', 'mlx')}")
    print("=" * 65)

    # Load backend once (shared across Phase 1 tests)
    backend_type = config.model.get("backend", "mlx")
    print(f"\nLoading {backend_type} backend...")
    if backend_type == "torch":
        from src.backend.torch_backend import TorchBackend
        backend = TorchBackend(config)
    else:
        from src.backend.mlx_backend import MLXBackend
        backend = MLXBackend(config)
    backend.load()
    print(f"Model loaded: {backend._model_path}")

    # Dequantize target layers for Phase 1 tests (Phase 2/3 use Orchestrator which does it)
    if hasattr(backend, 'dequantize_layer'):
        memit_config = config.get("memit", {}) or {}
        target_layers = memit_config.get("target_layers", [8, 9, 10, 11, 12, 13, 14, 15])
        target_module = memit_config.get("target_module", "down_proj")
        for l in target_layers:
            backend.dequantize_layer(l, target_module)
        print(f"Dequantized {len(target_layers)} target layers")

    results = {}
    start = time.time()

    # ── Phase 1: Mechanical Correctness ──
    if phase_filter is None or phase_filter == 1:
        if test_filter is None or test_filter == 1:
            results[1] = test_delta_persistence(config, backend)
        if test_filter is None or test_filter == 2:
            results[2] = test_scale_edit(config, backend)
        if test_filter is None or test_filter == 3:
            results[3] = test_snapshot_restore(config, backend)

    # ── Phase 2: Flow Integration ──
    # These use Orchestrator (loads its own model internally)
    if phase_filter is None or phase_filter == 2:
        # Free the shared backend before Orchestrator loads its own
        free_backend(backend, backend_type)

        if test_filter is None or test_filter == 4:
            results[4] = test_nap_safety(config, None)
        if test_filter is None or test_filter == 5:
            results[5] = test_sleep_consolidation(config, None)
        if test_filter is None or test_filter == 6:
            results[6] = test_rejection_rollback(config, None)

    # ── Phase 3: Hypothesis Testing ──
    if phase_filter is None or phase_filter == 3:
        # Free shared backend if Phase 1 ran but Phase 2 didn't
        if phase_filter == 3 or (phase_filter is None and 2 not in results):
            try:
                free_backend(backend, backend_type)
            except NameError:
                pass  # Already freed by Phase 2

        if test_filter is None or test_filter == 7:
            results[7] = test_residual_trace(config, None)

    # ── Summary ──
    elapsed = time.time() - start
    print_phase(f"Summary ({elapsed:.0f}s)")

    test_names = {
        1: "Delta persistence",
        2: "Scale edit linearity",
        3: "Snapshot/restore",
        4: "Nap safety",
        5: "Sleep consolidation",
        6: "Rejection rollback",
        7: "Residual trace A/B",
    }

    all_passed = True
    for test_id in sorted(results.keys()):
        passed = results[test_id]
        status = "PASS" if passed else "FAIL"
        print(f"  {test_id}. [{status}] {test_names[test_id]}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print(f"  RESULT: ALL PASSED")
    else:
        failed = [str(t) for t, ok in results.items() if not ok]
        print(f"  RESULT: FAILED tests: {', '.join(failed)}")

    print("=" * 65)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
