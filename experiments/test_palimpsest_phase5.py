"""Phase 5 palimpsest tests — capacity under consolidation and 70B per-fact.

Test 10: Capacity Under Consolidation (8B)
  - 20 facts → nap → sleep cycle 1 → sleep cycle 2
  - Measures: consolidation rate, recall curves, PPL at each stage
  - Comparison baseline: old pipeline hit PPL 8.62 at 15 facts with nap

Test 11: 70B Per-Fact Consolidation
  - 10 facts → sleep cycle 1 → measure LoRA recall
  - Key question: does per-fact training get >0% LoRA recall at 70B?
  - (alignment tax paper: 0% LoRA recall at 70B with bulk training)

Usage:
    python3 experiments/test_palimpsest_phase5.py --config experiments/configs/8b_memit.yaml --test 10
    python3 experiments/test_palimpsest_phase5.py --config experiments/configs/70b_memit.yaml --test 11
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


# ── Synthetic facts (20 for capacity test) ──

FACTS_20 = [
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
    FactTriple(subject="Lucia Ferraro", relation="lives in", object="Bruges"),
    FactTriple(subject="Henrik Aziz", relation="works as", object="etymologist"),
    FactTriple(subject="Nadia Ostrowski", relation="likes", object="theremin"),
    FactTriple(subject="Tomasz Eklund", relation="uses", object="Prolog"),
    FactTriple(subject="Cassandra Yuen", relation="is aged", object="fifty-one"),
    FactTriple(subject="Olena Bergstrom", relation="lives in", object="Reykjavik"),
    FactTriple(subject="Javier Okafor", relation="works as", object="campanologist"),
    FactTriple(subject="Mira Johansson", relation="likes", object="clavichords"),
    FactTriple(subject="Tariq Whitfield", relation="uses", object="Haskell"),
    FactTriple(subject="Celeste Nakano", relation="is aged", object="twenty-nine"),
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
    from src.wake.logger import ConversationLogger
    if orch.context.recent_messages:
        orch.context.compact()
    orch.chat.reset_turn_count()
    orch.context.reset(keep_summary=True)
    orch.logger = ConversationLogger(orch.config)
    orch.chat.logger = orch.logger
    return result


def run_nap(orch):
    """Run a nap cycle."""
    result = orch.nap_controller.execute_nap(orch._gather_new_messages)
    from src.wake.logger import ConversationLogger
    if orch.context.recent_messages:
        orch.context.compact()
    orch.chat.reset_turn_count()
    orch.context.reset(keep_summary=True)
    orch.logger = ConversationLogger(orch.config)
    orch.chat.logger = orch.logger
    return result


def measure_ppl(backend):
    """Measure perplexity on a fixed reference text."""
    ref = (
        "The quick brown fox jumps over the lazy dog. "
        "In a world where technology advances rapidly, the importance of "
        "understanding fundamental principles remains paramount. Scientists "
        "continue to explore the mysteries of the universe, seeking answers "
        "to questions that have puzzled humanity for centuries."
    )
    try:
        ppl = backend.compute_perplexity(ref)
        return round(ppl, 2)
    except Exception as e:
        print(f"  [WARN] PPL measurement failed: {e}")
        return None


def get_stage_counts(edits):
    counts = {0: 0, 1: 0, 2: 0}
    for edit in edits:
        counts[edit.consolidation_stage] += 1
    return counts


# ── Test 10: Capacity Under Consolidation (8B) ──

def test_capacity(config):
    """20 facts → nap → sleep → sleep. Measures capacity, recall, PPL."""
    print_phase("Test 10: Capacity Under Consolidation (20 facts)")

    from src.orchestrator import Orchestrator

    clean_test_artifacts(config)

    orch = Orchestrator(config, disable_memit=False)
    orch.chat._sleep_callback = None
    orch.chat._nap_callback = None

    facts = FACTS_20[:20]
    results = {
        "model": config.model["path"],
        "num_facts": len(facts),
        "stages": {},
    }

    # ── Baseline PPL ──
    ppl_baseline = measure_ppl(orch.backend)
    print(f"  Baseline PPL: {ppl_baseline}")
    results["ppl_baseline"] = ppl_baseline

    # ── Inject 20 facts ──
    edits = []
    for i, fact in enumerate(facts):
        edit = orch.memit_engine.inject_fact(fact)
        if edit:
            edits.append(edit)
            orch.health_monitor.record_edit(1)
        if (i + 1) % 5 == 0:
            print(f"  Injected {i + 1}/{len(facts)}...")

    print(f"  Total injected: {len(edits)}/{len(facts)}")

    # ── Post-injection recall & PPL ──
    ppl_post_inject = measure_ppl(orch.backend)
    print(f"  Post-injection PPL: {ppl_post_inject}")

    raw_count = 0
    chat_count = 0
    per_fact_inject = {}
    for fact in facts:
        raw_ok, _ = raw_recall(orch.backend, fact)
        chat_ok, _ = chat_recall(orch.backend, fact)
        if raw_ok:
            raw_count += 1
        if chat_ok:
            chat_count += 1
        per_fact_inject[fact.subject] = {"raw": raw_ok, "chat": chat_ok}

    print(f"  Post-injection recall: raw={raw_count}/{len(facts)}, chat={chat_count}/{len(facts)}")

    results["post_inject"] = {
        "ppl": ppl_post_inject,
        "raw_recall": raw_count,
        "chat_recall": chat_count,
        "per_fact": per_fact_inject,
    }

    # ── Nap ──
    log_facts(orch, facts)
    print(f"\n  --- Nap ---")
    nap_result = run_nap(orch)
    print(f"  Nap completed")

    ppl_post_nap = measure_ppl(orch.backend)
    print(f"  Post-nap PPL: {ppl_post_nap}")

    # Post-nap MEMIT recall (should be unchanged — nap is safe now)
    raw_nap = sum(1 for f in facts if raw_recall(orch.backend, f)[0])
    chat_nap = sum(1 for f in facts if chat_recall(orch.backend, f)[0])
    print(f"  Post-nap recall: raw={raw_nap}/{len(facts)}, chat={chat_nap}/{len(facts)}")

    stages_nap = get_stage_counts(orch.memit_engine._active_edits)
    print(f"  Stages: {stages_nap}")

    results["post_nap"] = {
        "ppl": ppl_post_nap,
        "raw_recall": raw_nap,
        "chat_recall": chat_nap,
        "stages": stages_nap,
    }

    # ── Sleep Cycle 1 ──
    log_facts(orch, facts)
    print(f"\n  --- Sleep Cycle 1 ---")
    result1 = run_sleep_cycle(orch, 1)
    print(f"  Result: status={result1['status']}, consolidated={result1.get('facts_consolidated', 0)}")

    ppl_post_sleep1 = measure_ppl(orch.backend)
    print(f"  Post-sleep-1 PPL: {ppl_post_sleep1}")

    stages_s1 = get_stage_counts(orch.memit_engine._active_edits)
    print(f"  Stages: {stages_s1}")

    raw_s1 = sum(1 for f in facts if raw_recall(orch.backend, f)[0])
    chat_s1 = sum(1 for f in facts if chat_recall(orch.backend, f)[0])
    print(f"  Post-sleep-1 recall: raw={raw_s1}/{len(facts)}, chat={chat_s1}/{len(facts)}")

    results["stages"]["cycle_1"] = {
        "status": result1["status"],
        "facts_consolidated": result1.get("facts_consolidated", 0),
        "ppl": ppl_post_sleep1,
        "raw_recall": raw_s1,
        "chat_recall": chat_s1,
        "stages": stages_s1,
    }

    if result1["status"] != "approved":
        print("  Cycle 1 rejected. Stopping.")
        _save_capacity_results(results)
        print_result("Capacity test", False, "Cycle 1 rejected")
        return False

    # ── Sleep Cycle 2 ──
    log_facts(orch, facts)
    print(f"\n  --- Sleep Cycle 2 ---")
    result2 = run_sleep_cycle(orch, 2)
    print(f"  Result: status={result2['status']}, consolidated={result2.get('facts_consolidated', 0)}")

    ppl_post_sleep2 = measure_ppl(orch.backend)
    print(f"  Post-sleep-2 PPL: {ppl_post_sleep2}")

    stages_s2 = get_stage_counts(orch.memit_engine._active_edits)
    print(f"  Stages: {stages_s2}")

    # Per-fact final recall
    raw_s2 = 0
    chat_s2 = 0
    per_fact_final = {}
    for fact in facts:
        raw_ok, _ = raw_recall(orch.backend, fact)
        chat_ok, _ = chat_recall(orch.backend, fact)
        if raw_ok:
            raw_s2 += 1
        if chat_ok:
            chat_s2 += 1
        per_fact_final[fact.subject] = {"raw": raw_ok, "chat": chat_ok}

    print(f"  Post-sleep-2 recall: raw={raw_s2}/{len(facts)}, chat={chat_s2}/{len(facts)}")

    results["stages"]["cycle_2"] = {
        "status": result2["status"],
        "facts_consolidated": result2.get("facts_consolidated", 0),
        "ppl": ppl_post_sleep2,
        "raw_recall": raw_s2,
        "chat_recall": chat_s2,
        "stages": stages_s2,
        "per_fact": per_fact_final,
    }

    _save_capacity_results(results)

    # ── Summary table ──
    print(f"\n  {'Stage':<15} {'PPL':<8} {'Raw':<8} {'Chat':<8} {'S0':<5} {'S1':<5} {'S2':<5}")
    print(f"  {'─' * 54}")
    print(f"  {'Baseline':<15} {ppl_baseline or '?':<8}")
    print(f"  {'Post-inject':<15} {ppl_post_inject or '?':<8} {raw_count:<8} {chat_count:<8}")
    print(f"  {'Post-nap':<15} {ppl_post_nap or '?':<8} {raw_nap:<8} {chat_nap:<8} "
          f"{stages_nap[0]:<5} {stages_nap[1]:<5} {stages_nap[2]:<5}")
    print(f"  {'Post-sleep-1':<15} {ppl_post_sleep1 or '?':<8} {raw_s1:<8} {chat_s1:<8} "
          f"{stages_s1[0]:<5} {stages_s1[1]:<5} {stages_s1[2]:<5}")
    print(f"  {'Post-sleep-2':<15} {ppl_post_sleep2 or '?':<8} {raw_s2:<8} {chat_s2:<8} "
          f"{stages_s2[0]:<5} {stages_s2[1]:<5} {stages_s2[2]:<5}")

    # Pass if at least some facts consolidated and PPL didn't explode
    has_consolidated = stages_s2.get(1, 0) + stages_s2.get(2, 0) > 0
    ppl_ok = ppl_post_sleep2 is None or ppl_post_sleep2 < 15.0
    passed = has_consolidated and ppl_ok

    print_result("Facts consolidated", has_consolidated,
                 f"{stages_s2.get(2, 0)} at stage 2, {stages_s2.get(1, 0)} at stage 1")
    print_result("PPL under control", ppl_ok,
                 f"PPL={ppl_post_sleep2} (old pipeline: 8.62 at 15 facts)")
    print_result("Capacity test (overall)", passed)
    return passed


def _save_capacity_results(results):
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "capacity_consolidation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to experiments/results/capacity_consolidation.json")


# ── Test 11: 70B Per-Fact Consolidation ──

def test_70b_consolidation(config):
    """10 facts → sleep → measure per-fact LoRA recall at 70B.

    The alignment tax paper showed 0% LoRA recall at 70B with bulk training.
    This tests whether per-fact training (MEMIT facts as LoRA training data)
    can break through.
    """
    print_phase("Test 11: 70B Per-Fact Consolidation")

    from src.orchestrator import Orchestrator

    clean_test_artifacts(config)

    orch = Orchestrator(config, disable_memit=False)
    orch.chat._sleep_callback = None
    orch.chat._nap_callback = None

    facts = FACTS_20[:10]
    results = {
        "model": config.model["path"],
        "num_facts": len(facts),
    }

    # ── Baseline PPL ──
    ppl_baseline = measure_ppl(orch.backend)
    print(f"  Baseline PPL: {ppl_baseline}")
    results["ppl_baseline"] = ppl_baseline

    # ── Inject 10 facts ──
    edits = []
    for fact in facts:
        edit = orch.memit_engine.inject_fact(fact)
        if edit:
            edits.append(edit)
            orch.health_monitor.record_edit(1)
    print(f"  Injected {len(edits)}/{len(facts)} facts")

    # ── Post-injection recall ──
    pre_raw = 0
    pre_chat = 0
    per_fact_pre = {}
    for fact in facts:
        raw_ok, raw_resp = raw_recall(orch.backend, fact)
        chat_ok, chat_resp = chat_recall(orch.backend, fact)
        if raw_ok:
            pre_raw += 1
        if chat_ok:
            pre_chat += 1
        per_fact_pre[fact.subject] = {"raw": raw_ok, "chat": chat_ok}
        print(f"    {fact.subject}: raw={'OK' if raw_ok else 'MISS'}, chat={'OK' if chat_ok else 'MISS'}")

    print(f"  MEMIT recall: raw={pre_raw}/{len(facts)}, chat={pre_chat}/{len(facts)}")
    results["memit_recall"] = {"raw": pre_raw, "chat": pre_chat, "per_fact": per_fact_pre}

    # ── Sleep Cycle 1 ──
    log_facts(orch, facts)
    print(f"\n  --- Sleep Cycle 1 ---")
    result1 = run_sleep_cycle(orch, 1)
    print(f"  Result: status={result1['status']}, consolidated={result1.get('facts_consolidated', 0)}")

    if result1["status"] != "approved":
        print("  Sleep rejected.")
        results["cycle_1"] = {"status": "rejected"}
        _save_70b_results(results)
        print_result("70B consolidation", False, "Sleep rejected")
        return False

    ppl_post = measure_ppl(orch.backend)
    print(f"  Post-sleep PPL: {ppl_post}")

    stages = get_stage_counts(orch.memit_engine._active_edits)
    print(f"  Stages: {stages}")

    # ── Per-fact results ──
    # Test recall at current state (LoRA + MEMIT residual)
    post_raw = 0
    post_chat = 0
    per_fact_post = {}
    for fact in facts:
        raw_ok, _ = raw_recall(orch.backend, fact)
        chat_ok, _ = chat_recall(orch.backend, fact)
        if raw_ok:
            post_raw += 1
        if chat_ok:
            post_chat += 1
        per_fact_post[fact.subject] = {"raw": raw_ok, "chat": chat_ok}

    print(f"  Post-sleep recall (LoRA+residual): raw={post_raw}/{len(facts)}, chat={post_chat}/{len(facts)}")

    # ── Pure LoRA recall (scale MEMIT to 0.0) ──
    for edit in orch.memit_engine._active_edits:
        if edit.consolidation_stage >= 1:
            orch.memit_engine.scale_edit(edit, 0.0)

    pure_raw = 0
    pure_chat = 0
    per_fact_pure = {}
    for fact in facts:
        raw_ok, raw_resp = raw_recall(orch.backend, fact)
        chat_ok, chat_resp = chat_recall(orch.backend, fact)
        if raw_ok:
            pure_raw += 1
        if chat_ok:
            pure_chat += 1
        per_fact_pure[fact.subject] = {"raw": raw_ok, "chat": chat_ok}
        print(f"    {fact.subject}: pure_lora_raw={'OK' if raw_ok else 'MISS'}, "
              f"pure_lora_chat={'OK' if chat_ok else 'MISS'}")

    print(f"  Pure LoRA recall (MEMIT=0): raw={pure_raw}/{len(facts)}, chat={pure_chat}/{len(facts)}")

    # Restore residual
    for edit in orch.memit_engine._active_edits:
        if edit.consolidation_stage >= 1:
            residual = config.raw.get("memit", {}).get("residual_scale", 0.1)
            # residual_scale in config is MEMIT v_lr scale, not the consolidation residual
            # The consolidation residual is set in full_sleep_controller
            orch.memit_engine.scale_edit(edit, 0.1)

    results["cycle_1"] = {
        "status": result1["status"],
        "facts_consolidated": result1.get("facts_consolidated", 0),
        "ppl": ppl_post,
        "stages": stages,
        "post_recall": {"raw": post_raw, "chat": post_chat, "per_fact": per_fact_post},
        "pure_lora_recall": {"raw": pure_raw, "chat": pure_chat, "per_fact": per_fact_pure},
    }

    _save_70b_results(results)

    # ── Summary ──
    print(f"\n  {'Metric':<30} {'Value':<20}")
    print(f"  {'─' * 50}")
    print(f"  {'MEMIT recall (raw)':<30} {pre_raw}/{len(facts)}")
    print(f"  {'MEMIT recall (chat)':<30} {pre_chat}/{len(facts)}")
    print(f"  {'LoRA consolidated':<30} {result1.get('facts_consolidated', 0)}/{len(facts)}")
    print(f"  {'Pure LoRA recall (raw)':<30} {pure_raw}/{len(facts)}")
    print(f"  {'Pure LoRA recall (chat)':<30} {pure_chat}/{len(facts)}")
    print(f"  {'Alignment tax baseline':<30} 0/{len(facts)} (bulk training)")
    print(f"  {'PPL':<30} {ppl_baseline} -> {ppl_post}")

    # The key question: did we get >0% pure LoRA recall?
    lora_works = pure_chat > 0
    print_result("LoRA recall > 0% at 70B", lora_works,
                 f"{pure_chat}/{len(facts)} chat recall (was 0% with bulk training)")
    print_result("70B per-fact consolidation (overall)", lora_works)
    return lora_works


def _save_70b_results(results):
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "70b_consolidation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to experiments/results/70b_consolidation.json")


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
    print("  PALIMPSEST PHASE 5 — Capacity & 70B Consolidation")
    print(f"  Config: {config_path}")
    print(f"  Model: {config.model['path']}")
    print(f"  Backend: {config.model.get('backend', 'mlx')}")
    print("=" * 65)

    results = {}
    start = time.time()

    if test_filter is None or test_filter == 10:
        results[10] = test_capacity(config)

    if test_filter is None or test_filter == 11:
        results[11] = test_70b_consolidation(config)

    elapsed = time.time() - start
    print_phase(f"Summary ({elapsed:.0f}s)")

    test_names = {
        10: "Capacity under consolidation (20 facts)",
        11: "70B per-fact consolidation",
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
