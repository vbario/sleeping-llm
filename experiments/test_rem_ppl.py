"""MEMIT Sleep Experiment — inject facts, run MEMIT maintenance sleep, measure health.

Tests the MEMIT-only sleep pipeline (no LoRA). Replaces the old SWS-only vs SWS+REM
A/B comparison with a single-condition test of the new 6-step maintenance pipeline.

Measurement points:
  0. Baseline (clean model)        -> PPL
  1. After 10 MEMIT facts          -> PPL, recall
  2. After 20 MEMIT facts          -> PPL, recall
  3. Post sleep (maintenance)      -> PPL, recall, sleep result dict

Usage:
    # Full 20-fact test
    python experiments/test_rem_ppl.py --config config.yaml

    # Fewer facts
    python experiments/test_rem_ppl.py --config config.yaml --num-facts 10
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


# ── Facts: 2 batches of 10 (5 people x city + job each) ──

BATCH_1 = [
    FactTriple("Elena Voronov", "lives in", "Portland"),
    FactTriple("Elena Voronov", "works as", "marine biologist"),
    FactTriple("Marcus Takahashi", "lives in", "Austin"),
    FactTriple("Marcus Takahashi", "works as", "architect"),
    FactTriple("Priya Lindström", "lives in", "Denver"),
    FactTriple("Priya Lindström", "works as", "violinist"),
    FactTriple("Tobias Okafor", "lives in", "Seattle"),
    FactTriple("Tobias Okafor", "works as", "chef"),
    FactTriple("Yuki Petrov", "lives in", "Boston"),
    FactTriple("Yuki Petrov", "works as", "photographer"),
]

BATCH_2 = [
    FactTriple("Carlos Navarro", "lives in", "Nashville"),
    FactTriple("Carlos Navarro", "works as", "guitarist"),
    FactTriple("Amara Chen", "lives in", "Chicago"),
    FactTriple("Amara Chen", "works as", "neurosurgeon"),
    FactTriple("Felix Johansson", "lives in", "Miami"),
    FactTriple("Felix Johansson", "works as", "pilot"),
    FactTriple("Leila Dubois", "lives in", "San Diego"),
    FactTriple("Leila Dubois", "works as", "ceramicist"),
    FactTriple("Ravi Kowalski", "lives in", "Philadelphia"),
    FactTriple("Ravi Kowalski", "works as", "librarian"),
]

# Teaching messages — one per person, combining both facts
TEACHING_MESSAGES = [
    "Elena Voronov lives in Portland and works as a marine biologist.",
    "Marcus Takahashi is an architect who lives in Austin.",
    "Priya Lindström lives in Denver and works as a violinist.",
    "Tobias Okafor is a chef from Seattle.",
    "Yuki Petrov is a photographer based in Boston.",
    "Carlos Navarro lives in Nashville and works as a guitarist.",
    "Amara Chen is a neurosurgeon who lives in Chicago.",
    "Felix Johansson lives in Miami and works as a pilot.",
    "Leila Dubois is a ceramicist based in San Diego.",
    "Ravi Kowalski lives in Philadelphia and works as a librarian.",
]

# Reference texts for perplexity measurement
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
    """Measure perplexity averaged over multiple reference texts."""
    ppls = []
    for text in REFERENCE_TEXTS:
        ppl = backend.compute_perplexity(text)
        ppls.append(ppl)
    return sum(ppls) / len(ppls)


def test_recall(backend, facts):
    """Raw completion per fact. Returns (fraction, per_fact_details)."""
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


def clean_artifacts(config):
    """Remove artifacts to ensure clean state."""
    # Clean conversation logs
    conv_dir = Path(config.paths["conversations"])
    if conv_dir.exists():
        shutil.rmtree(conv_dir)
    conv_dir.mkdir(parents=True, exist_ok=True)

    # Clean MEMIT data
    memit_dir = Path(config.paths.get("memit_data", "data/memit"))
    if memit_dir.exists():
        shutil.rmtree(memit_dir)
    memit_dir.mkdir(parents=True, exist_ok=True)


# ── Core: run the experiment ──

def run_experiment(config_path, num_facts):
    """Run the full MEMIT sleep experiment.

    Args:
        config_path: Path to config YAML
        num_facts: Total number of facts to inject (split into 2 equal batches)

    Returns:
        dict with experiment results and trajectory
    """
    print(f"\n{'=' * 70}")
    print(f"  MEMIT SLEEP EXPERIMENT ({num_facts} facts)")
    print(f"{'=' * 70}")

    start_time = time.time()

    # 1. Load fresh config
    config = Config(config_path)

    # 2. Clean state + fresh orchestrator
    clean_artifacts(config)
    orch = Orchestrator(config, disable_memit=False)

    # 3. Disable auto-triggers
    orch.chat._sleep_callback = None
    orch.chat._nap_callback = None

    # Select fact batches based on num_facts
    half = num_facts // 2
    batch_1 = BATCH_1[:half]
    batch_2 = BATCH_2[:num_facts - half]
    all_facts = batch_1 + batch_2

    trajectory = []
    step = 0

    # 4. Baseline PPL (step 0)
    ppl = measure_perplexity(orch.backend)
    print(f"\n  [{step}] Baseline perplexity: {ppl:.2f}")
    trajectory.append({
        "step": step,
        "event": "baseline",
        "perplexity": round(ppl, 3),
        "total_facts": 0,
        "recall": None,
    })
    step += 1

    # 5. Inject batch 1, measure PPL + recall (step 1)
    print(f"\n  Injecting batch 1 ({len(batch_1)} facts)...")
    orch.memit_engine.inject_facts(batch_1)
    ppl = measure_perplexity(orch.backend)
    recall, recall_details_1 = test_recall(orch.backend, batch_1)
    print(f"  [{step}] After {len(batch_1)} facts: PPL={ppl:.2f}, recall={recall:.2f}")
    trajectory.append({
        "step": step,
        "event": f"memit_inject_{len(batch_1)}",
        "perplexity": round(ppl, 3),
        "total_facts": len(batch_1),
        "recall": round(recall, 3),
        "recall_details": recall_details_1,
    })
    step += 1

    # 6. Inject batch 2, measure PPL + recall (step 2)
    print(f"\n  Injecting batch 2 ({len(batch_2)} facts)...")
    orch.memit_engine.inject_facts(batch_2)
    ppl = measure_perplexity(orch.backend)
    recall, recall_details_2 = test_recall(orch.backend, all_facts)
    print(f"  [{step}] After {len(all_facts)} facts: PPL={ppl:.2f}, recall={recall:.2f}")
    trajectory.append({
        "step": step,
        "event": f"memit_inject_{len(all_facts)}",
        "perplexity": round(ppl, 3),
        "total_facts": len(all_facts),
        "recall": round(recall, 3),
        "recall_details": recall_details_2,
    })
    step += 1

    # 7. Teach via conversation (so sleep has session data to curate)
    print(f"\n  Teaching via conversation...")
    num_people = num_facts // 2  # 2 facts per person
    for msg in TEACHING_MESSAGES[:num_people]:
        orch.chat.process_input(msg)
    print(f"  Taught {min(num_people, len(TEACHING_MESSAGES))} messages")

    # 8. Trigger sleep — MEMIT maintenance pipeline
    print(f"\n  Triggering sleep (MEMIT maintenance)...")
    t0 = time.time()
    sleep_result = None
    sleep_ok = False
    try:
        orch.sleep_cycle_count += 1
        cycle_id = f"{orch.sleep_cycle_count:04d}"
        sleep_result = orch.full_sleep_controller.execute_sleep(
            cycle_id, "full", orch._gather_new_messages,
        )
        sleep_ok = True
        sleep_time = time.time() - t0
        print(f"  Sleep completed in {sleep_time:.1f}s")
        print(f"  Sleep result: {sleep_result}")
    except Exception as e:
        sleep_time = time.time() - t0
        print(f"  Sleep failed: {e}")
        import traceback
        traceback.print_exc()
        sleep_result = {"status": "error", "error": str(e)}

    # Post-sleep housekeeping
    if sleep_ok:
        refreshed = sleep_result.get("facts_refreshed", 0) if sleep_result else 0
        pruned = sleep_result.get("facts_pruned", 0) if sleep_result else 0
        orch.health_monitor.record_sleep("full",
                                          facts_refreshed=refreshed,
                                          facts_pruned=pruned)
        if orch.context.recent_messages:
            orch.context.compact()
        orch.chat.reset_turn_count()
        orch.context.reset(keep_summary=True)

    # 9. Post-sleep PPL + recall (step 3)
    try:
        ppl = measure_perplexity(orch.backend)
        recall, recall_details_post = test_recall(orch.backend, all_facts)
    except Exception as measure_err:
        print(f"  Post-sleep measurement failed: {measure_err}")
        ppl = float("nan")
        recall = float("nan")
        recall_details_post = []
    print(f"  [{step}] Post-sleep: PPL={ppl:.2f}, recall={recall:.2f}")
    trajectory.append({
        "step": step,
        "event": "post_sleep",
        "perplexity": round(ppl, 3),
        "total_facts": len(all_facts),
        "recall": round(recall, 3),
        "recall_details": recall_details_post,
        "sleep_ok": sleep_ok,
        "sleep_time": round(sleep_time, 1),
    })

    elapsed = time.time() - start_time

    # 10. Build result
    return {
        "num_facts": num_facts,
        "trajectory": trajectory,
        "baseline_ppl": trajectory[0]["perplexity"],
        "post_sleep_ppl": trajectory[-1]["perplexity"],
        "post_sleep_recall": trajectory[-1]["recall"],
        "sleep_result": sleep_result,
        "elapsed_seconds": round(elapsed, 1),
    }


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="MEMIT Sleep Experiment")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--num-facts", type=int, default=20, help="Total facts to inject (default: 20)")
    parser.add_argument("--output", type=str, default=None, help="Override output JSON path")
    # Legacy compat — ignored, single condition now
    parser.add_argument("--mode", type=str, default="both", help="(ignored, kept for compat)")
    args = parser.parse_args()

    print("=" * 70)
    print("  MEMIT SLEEP EXPERIMENT")
    print(f"  Facts: {args.num_facts}")
    print("=" * 70)

    start_time = time.time()
    result = run_experiment(args.config, num_facts=args.num_facts)

    # Print trajectory summary
    traj = result["trajectory"]
    print(f"\n  Trajectory:")
    print(f"  {'Step':>4} {'Event':<24} {'PPL':>8} {'Facts':>6} {'Recall':>8}")
    print(f"  {'-' * 4} {'-' * 24} {'-' * 8} {'-' * 6} {'-' * 8}")
    for t in traj:
        recall_str = f"{t['recall']:.3f}" if t['recall'] is not None else "---"
        print(f"  {t['step']:>4} {t['event']:<24} {t['perplexity']:>8.3f} "
              f"{t['total_facts']:>6} {recall_str:>8}")

    # Sleep result summary
    sr = result.get("sleep_result", {})
    if sr:
        print(f"\n  Sleep Result:")
        print(f"    Status:     {sr.get('status', 'n/a')}")
        print(f"    Audited:    {sr.get('audited', 'n/a')}")
        print(f"    Refreshed:  {sr.get('facts_refreshed', 'n/a')}")
        print(f"    Pruned:     {sr.get('facts_pruned', 'n/a')}")
        ppl_before = sr.get('ppl_before')
        ppl_after = sr.get('ppl_after')
        if ppl_before is not None:
            print(f"    PPL:        {ppl_before} → {ppl_after}")

    total_elapsed = time.time() - start_time
    print(f"\n  Total time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")

    # Verdict
    baseline = result["baseline_ppl"]
    post = result["post_sleep_ppl"]
    delta = post - baseline
    recall = result["post_sleep_recall"]
    print(f"\n  VERDICT:")
    print(f"    PPL:    {baseline:.2f} → {post:.2f} ({delta:+.3f}, {delta/baseline*100:+.1f}%)")
    print(f"    Recall: {recall:.2f}")
    ppl_ok = "PASS" if delta / baseline < 0.15 else "FAIL"
    recall_ok = "PASS" if recall >= 0.5 else "FAIL"
    print(f"    PPL health:   {ppl_ok}")
    print(f"    Recall check: {recall_ok}")

    # Save results
    output_path = Path(args.output) if args.output else Path("experiments/results/test_rem_ppl.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_result = dict(result)
    save_result["trajectory"] = [
        {k: v for k, v in t.items() if k != "recall_details"}
        for t in result["trajectory"]
    ]
    save_result["total_elapsed_seconds"] = round(total_elapsed, 1)

    with open(output_path, "w") as f:
        json.dump(save_result, f, indent=2, default=str)
    print(f"  Results saved to {output_path}")


if __name__ == "__main__":
    main()
