"""Phase 4 palimpsest tests — multi-cycle consolidation and residual sweep.

Test 8: Multi-Cycle Consolidation (Stage 0→1→2)
  - Cycle 1: 10 facts → sleep → stage 0→1
  - Cycle 2: same facts reinforced → stage 1→2

Test 9: Residual Sweep Under Interference
  - For residual_scale in [0.0, 0.1, 0.3, 0.5]:
    - Cycle 1: 5 facts (A) → sleep → consolidate at residual_scale
    - Cycle 2: 5 NEW facts (B) → sleep → LoRA trains on B (interference with A)
    - Measure: does residual protect A recall under B interference?

Usage:
    python3 experiments/test_palimpsest_phase4.py --config experiments/configs/8b_memit.yaml
    python3 experiments/test_palimpsest_phase4.py --config experiments/configs/8b_memit.yaml --test 8
    python3 experiments/test_palimpsest_phase4.py --config experiments/configs/8b_memit.yaml --test 9
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.memory.memit import FactTriple


# ── Synthetic facts ──

FACTS_A = [
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

FACTS_B = [
    FactTriple(subject="Lucia Ferraro", relation="lives in", object="Bruges"),
    FactTriple(subject="Henrik Aziz", relation="works as", object="etymologist"),
    FactTriple(subject="Nadia Ostrowski", relation="likes", object="theremin"),
    FactTriple(subject="Tomasz Eklund", relation="uses", object="Prolog"),
    FactTriple(subject="Cassandra Yuen", relation="is aged", object="fifty-one"),
]


def raw_recall(backend, fact):
    prompt = fact.to_prompt()
    response = backend.generate(prompt, max_tokens=30, temperature=0.1)
    found = fact.object.lower() in response.lower()
    return found, response


def chat_recall(backend, fact):
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


def clean_test_artifacts(config):
    dirs = ["current_model", "checkpoints", "adapters", "training", "conversations"]
    for key in dirs:
        p = Path(config.paths[key])
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)
    memit_dir = Path(config.paths.get("memit_data", "data/memit"))
    if memit_dir.exists():
        shutil.rmtree(memit_dir)
    memit_dir.mkdir(parents=True, exist_ok=True)


def log_facts(orch, facts):
    """Log fact conversations to the current session."""
    for fact in facts:
        msg = f"{fact.subject} {fact.relation} {fact.object}"
        orch.logger.log_exchange(f"Remember: {msg}", f"I'll remember that {msg}.")


def run_sleep_cycle(orch, cycle_num):
    """Run a full sleep cycle and return the result."""
    orch.sleep_cycle_count += 1
    cycle_id = f"{orch.sleep_cycle_count:04d}"
    result = orch.full_sleep_controller.execute_sleep(
        cycle_id, "light", orch._gather_new_messages
    )
    # Post-sleep housekeeping (mimics orchestrator._on_sleep_trigger)
    from src.wake.logger import ConversationLogger
    if orch.context.recent_messages:
        orch.context.compact()
    orch.chat.reset_turn_count()
    orch.context.reset(keep_summary=True)
    orch.logger = ConversationLogger(orch.config)
    orch.chat.logger = orch.logger
    return result


# ── Test 8: Multi-Cycle Consolidation ──

def test_multi_cycle(config):
    """Two sleep cycles: stage 0→1 in cycle 1, stage 1→2 in cycle 2."""
    print_phase("Test 8: Multi-Cycle Consolidation (Stage 0→1→2)")

    from src.orchestrator import Orchestrator

    clean_test_artifacts(config)

    orch = Orchestrator(config, disable_memit=False)
    orch.chat._sleep_callback = None
    orch.chat._nap_callback = None

    facts = FACTS_A[:10]

    # Inject all 10 facts
    edits = []
    for fact in facts:
        edit = orch.memit_engine.inject_fact(fact)
        if edit:
            edits.append(edit)
            orch.health_monitor.record_edit(1)
    print(f"  Injected {len(edits)} facts")

    # Pre-sleep recall
    pre_recall = sum(1 for f in facts if raw_recall(orch.backend, f)[0])
    print(f"  Pre-sleep MEMIT recall: {pre_recall}/{len(facts)}")

    # Log conversations for cycle 1
    log_facts(orch, facts)

    # ── SLEEP CYCLE 1 ──
    print(f"\n  --- Sleep Cycle 1 ---")
    result1 = run_sleep_cycle(orch, 1)
    print(f"  Result: status={result1['status']}, consolidated={result1.get('facts_consolidated', 0)}")

    if result1["status"] != "approved":
        print("  Cycle 1 rejected. Cannot test multi-cycle.")
        print_result("Multi-cycle test", False, "Cycle 1 rejected")
        return False

    # Check stage distribution after cycle 1
    stage_counts_1 = {0: 0, 1: 0, 2: 0}
    stage1_facts = []
    for edit in orch.memit_engine._active_edits:
        stage_counts_1[edit.consolidation_stage] += 1
        if edit.consolidation_stage == 1:
            stage1_facts.append(edit)
    print(f"  After cycle 1: stage 0={stage_counts_1[0]}, stage 1={stage_counts_1[1]}, stage 2={stage_counts_1[2]}")

    if stage_counts_1[1] == 0:
        print("  No facts reached stage 1. Cannot test stage 1→2 transition.")
        print_result("Multi-cycle test", False, "No stage-1 facts after cycle 1")
        return False

    # Log same facts again for cycle 2 training data
    log_facts(orch, facts)

    # ── SLEEP CYCLE 2 ──
    print(f"\n  --- Sleep Cycle 2 ---")
    result2 = run_sleep_cycle(orch, 2)
    print(f"  Result: status={result2['status']}, consolidated={result2.get('facts_consolidated', 0)}")

    # Check stage distribution after cycle 2
    stage_counts_2 = {0: 0, 1: 0, 2: 0}
    for edit in orch.memit_engine._active_edits:
        stage_counts_2[edit.consolidation_stage] += 1
        stage_name = {0: "active", 1: "consolidating", 2: "consolidated"}[edit.consolidation_stage]
        print(f"  Edit {edit.edit_id}: stage={edit.consolidation_stage} ({stage_name}), "
              f"scale={edit.scale:.2f}, fact={edit.facts[0].subject}")

    print(f"  After cycle 2: stage 0={stage_counts_2[0]}, stage 1={stage_counts_2[1]}, stage 2={stage_counts_2[2]}")

    # Post-cycle-2 recall
    post_recall_raw = 0
    post_recall_chat = 0
    for fact in facts:
        raw_ok, raw_resp = raw_recall(orch.backend, fact)
        chat_ok, chat_resp = chat_recall(orch.backend, fact)
        if raw_ok:
            post_recall_raw += 1
        if chat_ok:
            post_recall_chat += 1
        print(f"    {fact.subject}: raw={'OK' if raw_ok else 'MISS'}, chat={'OK' if chat_ok else 'MISS'}")

    print(f"  Post-cycle-2 recall: raw={post_recall_raw}/{len(facts)}, chat={post_recall_chat}/{len(facts)}")

    # Save results
    results = {
        "model": config.model["path"],
        "num_facts": len(facts),
        "cycle_1": {
            "status": result1["status"],
            "facts_consolidated": result1.get("facts_consolidated", 0),
            "stages": stage_counts_1,
        },
        "cycle_2": {
            "status": result2["status"],
            "facts_consolidated": result2.get("facts_consolidated", 0),
            "stages": stage_counts_2,
        },
        "post_recall_raw": post_recall_raw,
        "post_recall_chat": post_recall_chat,
    }
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "multi_cycle.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to experiments/results/multi_cycle.json")

    # Assertions
    has_stage2 = stage_counts_2[2] > 0
    cycle2_approved = result2["status"] == "approved"

    print_result("Cycle 1 approved", result1["status"] == "approved")
    print_result("Stage 1 facts exist after cycle 1", stage_counts_1[1] > 0,
                 f"{stage_counts_1[1]} facts")
    print_result("Cycle 2 approved", cycle2_approved)
    print_result("Stage 2 facts exist after cycle 2", has_stage2,
                 f"{stage_counts_2[2]} facts fully consolidated")

    passed = has_stage2 and cycle2_approved
    print_result("Multi-cycle consolidation (overall)", passed)
    return passed


# ── Test 9: Residual Sweep Under Interference ──

def test_residual_sweep(config):
    """Test whether residual MEMIT trace protects against LoRA interference.

    For each residual_scale:
      1. Inject facts A → sleep cycle 1 → consolidate at residual_scale
      2. Inject facts B (new) → sleep cycle 2 → LoRA trains on B
      3. Measure: does residual protect A recall when LoRA learns B?
    """
    print_phase("Test 9: Residual Sweep Under Interference")

    from src.orchestrator import Orchestrator

    residual_values = [0.0, 0.1, 0.3, 0.5]
    all_results = {}

    for residual_scale in residual_values:
        print(f"\n  {'─' * 55}")
        print(f"  Residual scale = {residual_scale}")
        print(f"  {'─' * 55}")

        clean_test_artifacts(config)

        orch = Orchestrator(config, disable_memit=False)
        orch.chat._sleep_callback = None
        orch.chat._nap_callback = None
        orch.full_sleep_controller.residual_scale = residual_scale

        # ── Phase 1: Inject facts A, sleep cycle 1 ──
        facts_a = FACTS_A[:5]
        edits_a = []
        for fact in facts_a:
            edit = orch.memit_engine.inject_fact(fact)
            if edit:
                edits_a.append(edit)
                orch.health_monitor.record_edit(1)
        print(f"  Injected {len(edits_a)} facts (A)")

        log_facts(orch, facts_a)

        print(f"  Running sleep cycle 1 (consolidate A at residual={residual_scale})...")
        result1 = run_sleep_cycle(orch, 1)
        print(f"  Cycle 1: {result1['status']}, consolidated={result1.get('facts_consolidated', 0)}")

        if result1["status"] != "approved":
            print(f"  Cycle 1 rejected at residual={residual_scale}. Skipping.")
            all_results[residual_scale] = {"status": "cycle1_rejected"}
            continue

        # Record which A facts reached stage 1
        stage1_a_edits = [e for e in orch.memit_engine._active_edits
                          if e.consolidation_stage >= 1]
        stage1_a_facts = []
        for e in stage1_a_edits:
            stage1_a_facts.extend(e.facts)

        n_consolidated = len(stage1_a_edits)
        print(f"  {n_consolidated}/{len(edits_a)} A facts consolidated to stage 1+")

        if n_consolidated == 0:
            print(f"  No A facts consolidated. Skipping residual={residual_scale}.")
            all_results[residual_scale] = {"status": "no_consolidation"}
            continue

        # Measure A recall after cycle 1 (baseline)
        a_recall_after_c1 = {}
        for fact in stage1_a_facts:
            raw_ok, _ = raw_recall(orch.backend, fact)
            chat_ok, _ = chat_recall(orch.backend, fact)
            a_recall_after_c1[fact.subject] = {"raw": raw_ok, "chat": chat_ok}

        # ── Phase 2: Inject facts B, sleep cycle 2 (interference) ──
        facts_b = FACTS_B[:5]
        edits_b = []
        for fact in facts_b:
            edit = orch.memit_engine.inject_fact(fact)
            if edit:
                edits_b.append(edit)
                orch.health_monitor.record_edit(1)
        print(f"  Injected {len(edits_b)} NEW facts (B)")

        log_facts(orch, facts_b)

        print(f"  Running sleep cycle 2 (trains on B, interference with A)...")
        result2 = run_sleep_cycle(orch, 2)
        print(f"  Cycle 2: {result2['status']}, consolidated={result2.get('facts_consolidated', 0)}")

        # ── Phase 3: Measure A recall WITH residual (current state) ──
        print(f"\n  --- Condition: WITH residual={residual_scale} ---")
        a_recall_with = {}
        for fact in stage1_a_facts:
            raw_ok, raw_resp = raw_recall(orch.backend, fact)
            chat_ok, chat_resp = chat_recall(orch.backend, fact)
            a_recall_with[fact.subject] = {"raw": raw_ok, "chat": chat_ok}
            print(f"    {fact.subject}: raw={'OK' if raw_ok else 'MISS'}, "
                  f"chat={'OK' if chat_ok else 'MISS'}")

        # ── Phase 4: Scale A facts to 0.0, measure WITHOUT residual ──
        print(f"\n  --- Condition: WITHOUT residual (scale=0.0) ---")
        for edit in stage1_a_edits:
            orch.memit_engine.scale_edit(edit, 0.0)

        a_recall_without = {}
        for fact in stage1_a_facts:
            raw_ok, raw_resp = raw_recall(orch.backend, fact)
            chat_ok, chat_resp = chat_recall(orch.backend, fact)
            a_recall_without[fact.subject] = {"raw": raw_ok, "chat": chat_ok}
            print(f"    {fact.subject}: raw={'OK' if raw_ok else 'MISS'}, "
                  f"chat={'OK' if chat_ok else 'MISS'}")

        # Restore
        for edit in stage1_a_edits:
            orch.memit_engine.scale_edit(edit, residual_scale)

        # ── Summarize ──
        n = len(stage1_a_facts)
        raw_with = sum(1 for v in a_recall_with.values() if v["raw"])
        raw_without = sum(1 for v in a_recall_without.values() if v["raw"])
        chat_with = sum(1 for v in a_recall_with.values() if v["chat"])
        chat_without = sum(1 for v in a_recall_without.values() if v["chat"])

        raw_c1 = sum(1 for v in a_recall_after_c1.values() if v["raw"])
        chat_c1 = sum(1 for v in a_recall_after_c1.values() if v["chat"])

        print(f"\n  Summary (residual={residual_scale}, {n} consolidated A facts):")
        print(f"    After cycle 1:  raw={raw_c1}/{n}, chat={chat_c1}/{n}")
        print(f"    After cycle 2 WITH residual:    raw={raw_with}/{n}, chat={chat_with}/{n}")
        print(f"    After cycle 2 WITHOUT residual: raw={raw_without}/{n}, chat={chat_without}/{n}")

        raw_diff = raw_with - raw_without
        chat_diff = chat_with - chat_without
        raw_decay = raw_c1 - raw_with  # how much A recall decayed from cycle 1 to cycle 2

        if raw_diff > 0:
            print(f"    Residual HELPS raw (+{raw_diff})")
        elif raw_diff < 0:
            print(f"    Residual HURTS raw ({raw_diff})")
        else:
            print(f"    Residual has NO EFFECT on raw")

        all_results[residual_scale] = {
            "status": "completed",
            "n_consolidated": n_consolidated,
            "n_stage1_facts": n,
            "cycle1": {"raw": raw_c1, "chat": chat_c1},
            "cycle2_with_residual": {"raw": raw_with, "chat": chat_with},
            "cycle2_without_residual": {"raw": raw_without, "chat": chat_without},
            "raw_residual_effect": raw_diff,
            "chat_residual_effect": chat_diff,
            "raw_decay_from_interference": raw_decay,
            "per_fact": {
                subj: {
                    "after_c1": a_recall_after_c1.get(subj, {}),
                    "with_residual": a_recall_with.get(subj, {}),
                    "without_residual": a_recall_without.get(subj, {}),
                } for subj in [f.subject for f in stage1_a_facts]
            },
        }

    # ── Cross-condition comparison ──
    print_phase("Residual Sweep Summary")

    print(f"  {'Scale':<8} {'Consol':<8} {'C1 raw':<8} {'C2+res':<8} {'C2-res':<8} {'Effect':<8} {'Decay':<8}")
    print(f"  {'─' * 56}")
    for scale in residual_values:
        r = all_results.get(scale, {})
        if r.get("status") != "completed":
            print(f"  {scale:<8} {r.get('status', 'error')}")
            continue
        n = r["n_stage1_facts"]
        print(f"  {scale:<8.1f} {r['n_consolidated']:<8} "
              f"{r['cycle1']['raw']}/{n:<5} "
              f"{r['cycle2_with_residual']['raw']}/{n:<5} "
              f"{r['cycle2_without_residual']['raw']}/{n:<5} "
              f"{r['raw_residual_effect']:+d}{'':5} "
              f"{r['raw_decay_from_interference']:+d}")

    # Save results
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    # Convert float keys to strings for JSON
    json_results = {str(k): v for k, v in all_results.items()}
    with open(results_dir / "residual_sweep.json", "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\n  Results saved to experiments/results/residual_sweep.json")

    # The test passes if we got data for at least 3 conditions
    completed = sum(1 for r in all_results.values() if r.get("status") == "completed")
    print_result("Residual sweep completed", completed >= 3,
                 f"{completed}/{len(residual_values)} conditions completed")
    return completed >= 3


# ── Main ──

def main():
    config_path = "config.yaml"
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
        elif args[i] == "--test" and i + 1 < len(args):
            test_filter = int(args[i + 1])
            i += 2
        else:
            i += 1

    config = Config(config_path)

    print("=" * 65)
    print("  PALIMPSEST PHASE 4 — Multi-Cycle & Residual Sweep")
    print(f"  Config: {config_path}")
    print(f"  Model: {config.model['path']}")
    print(f"  Backend: {config.model.get('backend', 'mlx')}")
    print("=" * 65)

    results = {}
    start = time.time()

    if test_filter is None or test_filter == 8:
        results[8] = test_multi_cycle(config)

    if test_filter is None or test_filter == 9:
        results[9] = test_residual_sweep(config)

    elapsed = time.time() - start
    print_phase(f"Summary ({elapsed:.0f}s)")

    test_names = {
        8: "Multi-cycle consolidation",
        9: "Residual sweep under interference",
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
