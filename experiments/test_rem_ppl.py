"""REM PPL Experiment — A/B comparison of SWS-only vs SWS+REM sleep.

Hypothesis: REM integration phase trains on multi-fact synthetic conversations,
which should reduce PPL scaling compared to SWS-only (isolated Q&A pairs that
double PPL at 20 facts: 6.5->13.3).

Design:
  Condition A (SWS-only):  Deep sleep with REM disabled
  Condition B (SWS+REM):   Deep sleep with REM enabled

Both conditions: same 20 facts, same teaching messages, same base model,
same config (only rem.enabled differs).

Measurement points per condition:
  0. Baseline (clean model)        -> PPL
  1. After 10 MEMIT facts          -> PPL, recall
  2. After 20 MEMIT facts          -> PPL, recall
  3. Post deep sleep               -> PPL, recall, sleep result dict

Usage:
    # Full A/B comparison
    python experiments/test_rem_ppl.py --config config.yaml

    # Single condition
    python experiments/test_rem_ppl.py --config config.yaml --mode sws
    python experiments/test_rem_ppl.py --config config.yaml --mode rem

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

# Reference texts for perplexity measurement (identical to ablation_perplexity.py)
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
    dirs_to_clean = [
        config.paths["current_model"],
        config.paths["checkpoints"],
        config.paths["adapters"],
        config.paths["training"],
        config.paths["conversations"],
    ]
    for dir_path in dirs_to_clean:
        p = Path(dir_path)
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)

    memit_dir = Path(config.paths.get("memit_data", "data/memit"))
    if memit_dir.exists():
        shutil.rmtree(memit_dir)
        memit_dir.mkdir(parents=True, exist_ok=True)

    ledger_path = Path(config.paths.get("memit_ledger", "data/memit/ledger.json"))
    if ledger_path.exists():
        ledger_path.unlink()


# ── Core: run one condition ──

def run_condition(config_path, rem_enabled, num_facts):
    """Run one experimental condition end-to-end.

    Args:
        config_path: Path to config YAML
        rem_enabled: Whether to enable REM phase
        num_facts: Total number of facts to inject (split into 2 equal batches)

    Returns:
        dict with condition results and trajectory
    """
    condition_name = "SWS+REM" if rem_enabled else "SWS-only"
    print(f"\n{'=' * 70}")
    print(f"  CONDITION: {condition_name} (rem.enabled={rem_enabled})")
    print(f"{'=' * 70}")

    start_time = time.time()

    # 1. Load fresh config and override rem.enabled
    config = Config(config_path)
    rem_defaults = {
        "enabled": rem_enabled,
        "learning_rate": 5e-5,
        "epochs": 1,
        "num_integrations": 10,
        "temperature": 0.8,
        "max_ppl_increase": 0.10,
    }
    existing = config._data.get("rem", {})
    if existing is None:
        existing = {}
    for k, v in rem_defaults.items():
        existing.setdefault(k, v)
    existing["enabled"] = rem_enabled
    config._data["rem"] = existing

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

    # 7. Teach via conversation (so sleep has data)
    print(f"\n  Teaching via conversation...")
    # Scale teaching messages to num_facts (one per person)
    num_people = num_facts // 2  # 2 facts per person
    for msg in TEACHING_MESSAGES[:num_people]:
        orch.chat.process_input(msg)
    print(f"  Taught {min(num_people, len(TEACHING_MESSAGES))} messages")

    # 8. Force deep sleep — call execute_sleep directly to capture result dict
    print(f"\n  Triggering deep sleep (rem.enabled={rem_enabled})...")
    t0 = time.time()
    sleep_result = None
    sleep_ok = False
    try:
        orch.sleep_cycle_count += 1
        cycle_id = f"{orch.sleep_cycle_count:04d}"
        sleep_result = orch.full_sleep_controller.execute_sleep(
            cycle_id, "deep", orch._gather_new_messages,
        )
        sleep_ok = True
        sleep_time = time.time() - t0
        print(f"  Sleep completed in {sleep_time:.1f}s")
        print(f"  Sleep result: {sleep_result}")
    except Exception as e:
        sleep_time = time.time() - t0
        print(f"  Sleep failed: {e}")
        sleep_result = {"status": "error", "error": str(e)}

    # Replicate _on_sleep_trigger housekeeping
    if sleep_ok:
        orch.light_sleep_count = 0
        facts_consolidated = sleep_result.get("facts_consolidated", 0) if sleep_result else 0
        orch.health_monitor.record_sleep("full", facts_consolidated=facts_consolidated)
        if orch.context.recent_messages:
            orch.context.compact()
        orch.chat.reset_turn_count()
        orch.context.reset(keep_summary=True)
    else:
        # Sleep failed (e.g. REM rollback error) — model may be broken.
        # Reload base model so we can still measure post-sleep metrics.
        print("  Reloading base model after sleep failure...")
        try:
            orch.backend.reload(config.model["path"])
            # Re-apply MEMIT edits from in-memory state
            if orch.memit_engine.enabled and hasattr(orch.backend, "dequantize_layer"):
                orch.memit_engine._dequantize_target_layers()
            for edit in list(orch.memit_engine._active_edits):
                scaled_deltas = {}
                for layer_idx, delta in edit.layer_deltas.items():
                    scaled_deltas[layer_idx] = orch.memit_engine._scale_tensor(delta, edit.scale)
                for layer_idx, scaled_delta in scaled_deltas.items():
                    orch.memit_engine._apply_delta(layer_idx, scaled_delta)
            print("  Base model reloaded with MEMIT edits re-applied.")
        except Exception as reload_err:
            print(f"  Base model reload also failed: {reload_err}")

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
        "condition": condition_name,
        "rem_enabled": rem_enabled,
        "num_facts": num_facts,
        "trajectory": trajectory,
        "baseline_ppl": trajectory[0]["perplexity"],
        "post_sleep_ppl": trajectory[-1]["perplexity"],
        "post_sleep_recall": trajectory[-1]["recall"],
        "sleep_result": sleep_result,
        "elapsed_seconds": round(elapsed, 1),
    }


# ── Main ──

def print_comparison(result_a, result_b):
    """Print side-by-side comparison table and verdict."""
    print(f"\n{'=' * 70}")
    print(f"  COMPARISON: SWS-only vs SWS+REM")
    print(f"{'=' * 70}")

    baseline_a = result_a["baseline_ppl"]
    baseline_b = result_b["baseline_ppl"]
    post_a = result_a["post_sleep_ppl"]
    post_b = result_b["post_sleep_ppl"]
    delta_a = post_a - baseline_a
    delta_b = post_b - baseline_b
    recall_a = result_a["post_sleep_recall"]
    recall_b = result_b["post_sleep_recall"]

    print(f"\n  {'Metric':<32} {'SWS-only':>12} {'SWS+REM':>12} {'Delta':>12}")
    print(f"  {'-' * 32} {'-' * 12} {'-' * 12} {'-' * 12}")
    print(f"  {'Baseline PPL':<32} {baseline_a:>12.3f} {baseline_b:>12.3f} {baseline_b - baseline_a:>+12.3f}")
    print(f"  {'Post-sleep PPL':<32} {post_a:>12.3f} {post_b:>12.3f} {post_b - post_a:>+12.3f}")
    print(f"  {'PPL delta from baseline':<32} {delta_a:>+12.3f} {delta_b:>+12.3f} {delta_b - delta_a:>+12.3f}")
    print(f"  {'Post-sleep recall':<32} {recall_a:>12.3f} {recall_b:>12.3f} {recall_b - recall_a:>+12.3f}")

    # REM phase details (from treatment condition)
    rem = result_b.get("sleep_result", {}).get("rem")
    if rem:
        print(f"\n  REM Phase Details:")
        print(f"    Status:       {rem.get('status', 'n/a')}")
        print(f"    Integrations: {rem.get('integrations', 'n/a')}")
        sws_ppl = rem.get("sws_ppl")
        rem_ppl = rem.get("rem_ppl")
        if sws_ppl is not None:
            print(f"    SWS PPL:      {sws_ppl:.3f}")
        if rem_ppl is not None:
            print(f"    REM PPL:      {rem_ppl:.3f}")
        recall_rate = rem.get("recall_rate")
        if recall_rate is not None:
            print(f"    Recall rate:  {recall_rate:.2f}")

    # Verdict
    ppl_improved = post_b < post_a
    recall_maintained = recall_b >= recall_a - 0.10  # allow 10% drop

    print(f"\n  VERDICT:")
    ppl_yn = "YES" if ppl_improved else "NO"
    recall_yn = "YES" if recall_maintained else "NO"
    print(f"    PPL improved:      {ppl_yn} ({post_a:.2f} -> {post_b:.2f})")
    print(f"    Recall maintained: {recall_yn} ({recall_a:.2f} -> {recall_b:.2f})")
    print()


def main():
    parser = argparse.ArgumentParser(description="REM PPL Experiment: A/B comparison")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--mode", type=str, default="both", choices=["sws", "rem", "both"],
                        help="Which conditions to run (default: both)")
    parser.add_argument("--num-facts", type=int, default=20, help="Total facts to inject (default: 20)")
    parser.add_argument("--output", type=str, default=None, help="Override output JSON path")
    args = parser.parse_args()

    print("=" * 70)
    print("  REM PPL EXPERIMENT")
    print(f"  Mode: {args.mode} | Facts: {args.num_facts}")
    print("=" * 70)

    results = {}
    start_time = time.time()

    # Condition A: SWS-only
    if args.mode in ("sws", "both"):
        results["sws_only"] = run_condition(args.config, rem_enabled=False, num_facts=args.num_facts)

    # Condition B: SWS+REM
    if args.mode in ("rem", "both"):
        results["sws_rem"] = run_condition(args.config, rem_enabled=True, num_facts=args.num_facts)

    # Comparison (if both)
    if args.mode == "both" and "sws_only" in results and "sws_rem" in results:
        print_comparison(results["sws_only"], results["sws_rem"])

    # Per-condition summary
    for key, result in results.items():
        traj = result["trajectory"]
        print(f"\n  {result['condition']} Trajectory:")
        print(f"  {'Step':>4} {'Event':<24} {'PPL':>8} {'Facts':>6} {'Recall':>8}")
        print(f"  {'-' * 4} {'-' * 24} {'-' * 8} {'-' * 6} {'-' * 8}")
        for t in traj:
            recall_str = f"{t['recall']:.3f}" if t['recall'] is not None else "---"
            print(f"  {t['step']:>4} {t['event']:<24} {t['perplexity']:>8.3f} "
                  f"{t['total_facts']:>6} {recall_str:>8}")

    total_elapsed = time.time() - start_time
    print(f"\n  Total time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")

    # Save results
    output_path = Path(args.output) if args.output else Path("experiments/results/test_rem_ppl.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Strip recall_details from trajectory for cleaner JSON (keep top-level only)
    save_results = {}
    for key, result in results.items():
        save_result = dict(result)
        save_result["trajectory"] = [
            {k: v for k, v in t.items() if k != "recall_details"}
            for t in result["trajectory"]
        ]
        save_results[key] = save_result
    save_results["total_elapsed_seconds"] = round(total_elapsed, 1)

    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"  Results saved to {output_path}")


if __name__ == "__main__":
    main()
