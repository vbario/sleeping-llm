"""Consolidation Capacity Sweep — How many facts can MEMIT→LoRA absorb?

Direct rematch of note 72's ablation where nap consolidation *degraded*
recall (0.80→0.60 at 10 facts). Tests whether per-fact graduated dissolution
solves the scaling problem.

Sweep matrix:
  fact_counts × cycles

For each (fact_count, max_cycles) condition:
  1. Clean state, inject N facts via MEMIT
  2. Run up to max_cycles sleep cycles with consolidation
  3. After each cycle, measure:
     - Per-fact stage distribution (how many at stage 0/1/2/3)
     - Raw recall (MEMIT pathway)
     - Chat recall (LoRA pathway)
     - PPL drift
     - Edit scale (min of fact_stages → scale_schedule)
  4. Stop early if all facts reach stage 3

Usage:
    python experiments/sweep_consolidation_capacity.py --config experiments/configs/8b_consolidation.yaml
    python experiments/sweep_consolidation_capacity.py --config experiments/configs/8b_consolidation.yaml --facts 5,10
    python experiments/sweep_consolidation_capacity.py --config experiments/configs/8b_consolidation.yaml --cycles 5
"""

import argparse
import gc
import json
import os
import shutil
import sys
import time
from collections import Counter
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
    """Test recall. Returns (fraction, per-fact hit list)."""
    hits = []
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
        hits.append(fact.object.lower() in response.lower())
    fraction = sum(hits) / len(facts) if facts else 0
    return fraction, hits


def load_fact_pool(path="experiments/data/fact_pool_500.json"):
    pool_path = project_root / path
    with open(pool_path) as f:
        raw = json.load(f)
    return [FactTriple(subject=r["subject"], relation=r["relation"], object=r["object"]) for r in raw]


def clean_artifacts(config):
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
    cleanup_gpu()
    orch = Orchestrator(config, disable_memit=False)
    orch.chat._sleep_callback = None
    orch.chat._nap_callback = None
    # Disable fact extractor during sleep curation to prevent re-injection
    # of facts that were already directly injected via MEMIT.
    orch.full_sleep_controller.fact_extractor = None
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
    from src.wake.logger import ConversationLogger
    orch.logger = ConversationLogger(orch.config)
    orch.chat.logger = orch.logger
    return result


def teach_facts(orch, facts):
    for fact in facts:
        msg = f"{fact.subject} {fact.relation} {fact.object}."
        orch.chat.process_input(msg)


def get_fact_stages(engine, original_facts=None):
    """Collect per-fact stages across active edits.

    If original_facts is provided, only return stages for those facts
    (matched by subject+relation). Otherwise returns all stages.
    """
    if original_facts is None:
        stages = []
        for edit in engine._active_edits:
            stages.extend(edit.fact_stages)
        return stages

    stages = []
    for fact in original_facts:
        stage = 0
        for edit in engine._active_edits:
            for ei, ef in enumerate(edit.facts):
                if ef.subject == fact.subject and ef.relation == fact.relation:
                    if ei < len(edit.fact_stages):
                        stage = edit.fact_stages[ei]
                    break
        stages.append(stage)
    return stages


def stage_distribution(stages):
    """Count facts at each stage."""
    counts = Counter(stages)
    return {f"stage_{i}": counts.get(i, 0) for i in range(4)}


# ── Single condition ──

def run_condition(config, fact_pool, num_facts, max_cycles):
    """Run one (fact_count, cycles) condition. Returns trajectory."""
    label = f"{num_facts} facts × {max_cycles} cycles"
    print(f"\n{'=' * 70}")
    print(f"  CONDITION: {label}")
    print(f"{'=' * 70}")

    t0 = time.time()
    clean_artifacts(config)
    orch = fresh_orchestrator(config)

    if not orch.full_sleep_controller.consolidation_enabled:
        print("  SKIP: consolidation not enabled")
        destroy_orchestrator(orch)
        return {"verdict": "SKIP", "reason": "consolidation not enabled"}

    engine = orch.memit_engine

    # Inject facts
    facts = fact_pool[:num_facts]
    engine.inject_facts(facts)
    teach_facts(orch, facts)

    # Baseline measurements
    baseline_raw, raw_hits = test_recall(orch.backend, facts, raw=True)
    baseline_chat, chat_hits = test_recall(orch.backend, facts, raw=False)
    baseline_ppl = measure_perplexity(orch.backend)
    baseline_stages = get_fact_stages(engine, facts)

    print(f"  Baseline: raw={baseline_raw:.2f}, chat={baseline_chat:.2f}, "
          f"PPL={baseline_ppl:.2f}")
    print(f"  Stages: {stage_distribution(baseline_stages)}")

    trajectory = []
    trajectory.append({
        "cycle": 0,
        "raw_recall": round(baseline_raw, 3),
        "chat_recall": round(baseline_chat, 3),
        "ppl": round(baseline_ppl, 2),
        "stages": stage_distribution(baseline_stages),
        "fact_stages": baseline_stages,
        "scales": [round(e.scale, 2) for e in engine._active_edits],
        "consolidation": {},
        "elapsed": 0,
    })

    for cycle in range(1, max_cycles + 1):
        cycle_t0 = time.time()
        print(f"\n  --- Cycle {cycle}/{max_cycles} ---")

        sleep_result = trigger_sleep(orch)
        consolidation = sleep_result.get("consolidation", {})

        raw_recall, raw_hits = test_recall(orch.backend, facts, raw=True)
        chat_recall, chat_hits = test_recall(orch.backend, facts, raw=False)
        ppl = measure_perplexity(orch.backend)
        fact_stgs = get_fact_stages(engine, facts)
        dist = stage_distribution(fact_stgs)
        scales = [round(e.scale, 2) for e in engine._active_edits]
        cycle_elapsed = time.time() - cycle_t0

        print(f"  Cycle {cycle}: raw={raw_recall:.2f}, chat={chat_recall:.2f}, "
              f"PPL={ppl:.2f}")
        print(f"  Stages: {dist}, Scales: {scales}")
        print(f"  Consolidation: advanced={consolidation.get('advanced', 0)}, "
              f"retreated={consolidation.get('retreated', 0)}, "
              f"scaled_down={consolidation.get('scaled_down', 0)}")

        # Per-fact detail for chat recall (only original facts)
        # Build a lookup: fact key → (edit, fact_idx_in_edit)
        chat_by_stage = {}
        for fi, fact in enumerate(facts):
            # Find this fact's stage in the engine
            stage = 0
            for edit in engine._active_edits:
                for ei, ef in enumerate(edit.facts):
                    if ef.subject == fact.subject and ef.relation == fact.relation:
                        if ei < len(edit.fact_stages):
                            stage = edit.fact_stages[ei]
                        break
            if stage not in chat_by_stage:
                chat_by_stage[stage] = {"total": 0, "passed": 0}
            chat_by_stage[stage]["total"] += 1
            if fi < len(chat_hits) and chat_hits[fi]:
                chat_by_stage[stage]["passed"] += 1
        for s in sorted(chat_by_stage):
            info = chat_by_stage[s]
            rate = info["passed"] / info["total"] if info["total"] else 0
            print(f"    Stage {s}: chat {info['passed']}/{info['total']} ({rate:.0%})")

        trajectory.append({
            "cycle": cycle,
            "raw_recall": round(raw_recall, 3),
            "chat_recall": round(chat_recall, 3),
            "ppl": round(ppl, 2),
            "stages": dist,
            "fact_stages": fact_stgs,
            "scales": scales,
            "consolidation": consolidation,
            "chat_by_stage": {str(k): v for k, v in chat_by_stage.items()},
            "elapsed": round(cycle_elapsed, 1),
        })

        # Early stop: all facts at stage 3
        if dist.get("stage_3", 0) == num_facts:
            print(f"  All {num_facts} facts fully consolidated — stopping early")
            break

    total_elapsed = time.time() - t0

    # Summary
    final = trajectory[-1]
    final_stages = final["stages"]
    advancement_rate = 1.0 - (final_stages.get("stage_0", 0) / num_facts) if num_facts else 0
    fully_consolidated = final_stages.get("stage_3", 0)
    ppl_drift = (final["ppl"] - baseline_ppl) / baseline_ppl if baseline_ppl else 0

    result = {
        "num_facts": num_facts,
        "max_cycles": max_cycles,
        "cycles_run": len(trajectory) - 1,
        "baseline_raw_recall": round(baseline_raw, 3),
        "baseline_chat_recall": round(baseline_chat, 3),
        "baseline_ppl": round(baseline_ppl, 2),
        "final_raw_recall": final["raw_recall"],
        "final_chat_recall": final["chat_recall"],
        "final_ppl": final["ppl"],
        "ppl_drift_pct": round(ppl_drift * 100, 1),
        "final_stages": final_stages,
        "advancement_rate": round(advancement_rate, 3),
        "fully_consolidated": fully_consolidated,
        "trajectory": trajectory,
        "elapsed_seconds": round(total_elapsed, 1),
    }

    print(f"\n  SUMMARY: {num_facts} facts, {len(trajectory)-1} cycles")
    print(f"  Chat recall: {baseline_chat:.2f} → {final['chat_recall']:.2f}")
    print(f"  PPL drift: {ppl_drift:+.1%}")
    print(f"  Advancement: {advancement_rate:.0%} of facts moved past stage 0")
    print(f"  Fully consolidated (stage 3): {fully_consolidated}/{num_facts}")
    print(f"  Time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")

    destroy_orchestrator(orch)
    return result


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Consolidation Capacity Sweep")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--facts", type=str, default="5,10,15,20",
                        help="Comma-separated fact counts (default: 5,10,15,20)")
    parser.add_argument("--cycles", type=int, default=3,
                        help="Max sleep cycles per condition (default: 3)")
    parser.add_argument("--output", type=str, default=None,
                        help="Override output JSON path")
    args = parser.parse_args()

    fact_counts = [int(x) for x in args.facts.split(",")]
    max_cycles = args.cycles

    config = Config(args.config)
    model_name = config.model["path"]
    backend_name = config.model.get("backend", "mlx")
    memit_layers = config.get("memit.target_layers", [])

    print("=" * 70)
    print("  CONSOLIDATION CAPACITY SWEEP")
    print("=" * 70)
    print(f"  Model:       {model_name}")
    print(f"  Backend:     {backend_name}")
    print(f"  MEMIT layers: {memit_layers} ({len(memit_layers)} layers)")
    print(f"  Conditions:  {fact_counts} facts × {max_cycles} cycles each")
    print("=" * 70)

    fact_pool = load_fact_pool()
    print(f"  Loaded {len(fact_pool)} facts from pool")

    max_needed = max(fact_counts)
    if max_needed > len(fact_pool):
        print(f"  WARNING: need {max_needed} facts but pool has {len(fact_pool)}")
        fact_counts = [n for n in fact_counts if n <= len(fact_pool)]

    total_start = time.time()
    results = {
        "config": {
            "model": model_name,
            "backend": backend_name,
            "memit_layers": memit_layers,
            "fact_counts": fact_counts,
            "max_cycles": max_cycles,
        },
        "conditions": {},
    }

    for num_facts in fact_counts:
        try:
            cond_result = run_condition(config, fact_pool, num_facts, max_cycles)
            results["conditions"][str(num_facts)] = cond_result
        except Exception as e:
            print(f"\n  CRASHED at {num_facts} facts: {e}")
            import traceback
            traceback.print_exc()
            results["conditions"][str(num_facts)] = {
                "verdict": "CRASH", "error": str(e),
            }

    total_elapsed = time.time() - total_start

    # ── Summary table ──
    print(f"\n{'=' * 70}")
    print(f"  SWEEP RESULTS")
    print(f"{'=' * 70}")
    print(f"  {'Facts':>5} │ {'Cycles':>6} │ {'Chat':>6} │ {'PPL Δ':>7} │ {'Adv%':>5} │ {'S3':>4} │ {'Time':>6}")
    print(f"  {'─'*5}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*7}─┼─{'─'*5}─┼─{'─'*4}─┼─{'─'*6}")

    for num_facts in fact_counts:
        key = str(num_facts)
        c = results["conditions"].get(key, {})
        if "verdict" in c and c["verdict"] in ("SKIP", "CRASH"):
            print(f"  {num_facts:>5} │ {'—':>6} │ {'—':>6} │ {'—':>7} │ {'—':>5} │ {'—':>4} │ {c.get('verdict')}")
            continue
        cycles = c.get("cycles_run", 0)
        chat = c.get("final_chat_recall", 0)
        ppl_d = c.get("ppl_drift_pct", 0)
        adv = c.get("advancement_rate", 0)
        s3 = c.get("fully_consolidated", 0)
        elapsed = c.get("elapsed_seconds", 0)
        print(f"  {num_facts:>5} │ {cycles:>6} │ {chat:>6.2f} │ {ppl_d:>+6.1f}% │ {adv:>4.0%} │ {s3:>4} │ {elapsed:>5.0f}s")

    print(f"\n  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"{'=' * 70}")

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        model_short = model_name.split("/")[-1]
        output_path = Path("experiments/results") / f"sweep_consolidation_{model_short}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {output_path}")


if __name__ == "__main__":
    main()
