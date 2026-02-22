"""MEMIT benchmark — tests MEMIT injection, nap consolidation, and full sleep.

Tests the MEMIT + nap/sleep pipeline in 4 phases:
  Phase A: Inject facts via conversation → MEMIT auto-injects → test recall IMMEDIATELY
  Phase B: Run a nap → test recall (MEMIT consolidated to LoRA)
  Phase C: Run full sleep → test recall (full pipeline)
  Phase D: Distractors → test retention

Compares against LoRA-only baseline (MEMIT disabled).

Usage:
    python experiments/memit_benchmark.py --config experiments/configs/3b_memit.yaml
    python experiments/memit_benchmark.py --config experiments/configs/70b_memit.yaml --no-retention
    python experiments/memit_benchmark.py --config experiments/configs/3b_memit.yaml --memit-only
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is on the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.orchestrator import Orchestrator
from experiments.scaling_benchmark import score_response, load_facts


def test_recall_quiet(orchestrator, facts):
    """Test recall of individual facts. Returns list of scored results."""
    results = []
    for fact_group in facts["facts"]:
        for q_data in fact_group["recall_questions"]:
            question = q_data["question"]
            response = orchestrator.process_message(question)
            score = score_response(response, q_data)
            result = {"question": question, "response": response, **score}
            results.append(result)
    return results


def test_generalization_quiet(orchestrator, facts):
    """Test generalization questions. Returns list of scored results."""
    results = []
    for q_data in facts.get("generalization_questions", []):
        question = q_data["question"]
        response = orchestrator.process_message(question)
        score = score_response(response, q_data)
        result = {"question": question, "response": response, **score}
        results.append(result)
    return results


def avg(values):
    return sum(values) / len(values) if values else 0.0


def summarize_results(results, label=""):
    """Compute and print summary for a set of results."""
    recall = avg([r["recall"] for r in results])
    precision = avg([r["precision"] for r in results])
    perfect = sum(1 for r in results if r["recall"] == 1.0 and r["precision"] == 1.0)
    total = len(results)

    print(f"  {label}Recall: {recall:.2f}  Precision: {precision:.2f}  Perfect: {perfect}/{total}")

    for r in results:
        status = "PASS" if r["recall"] == 1.0 and r["precision"] == 1.0 else "PART" if r["recall"] > 0 else "FAIL"
        resp_short = r["response"][:80].replace("\n", " ")
        print(f"    [{status}] {r['question']}")
        print(f"           {resp_short}...")
        if r["expected_missing"]:
            print(f"           Missing: {r['expected_missing']}")
        if r["forbidden_found"]:
            print(f"           Forbidden: {r['forbidden_found']}")

    return {
        "avg_recall": recall,
        "avg_precision": precision,
        "perfect_count": perfect,
        "total_questions": total,
        "details": results,
    }


def run_memit_benchmark(config_path, facts_path, sleep_cycles=2,
                        test_retention_flag=True, memit_only=False):
    """Run the full MEMIT benchmark.

    Args:
        config_path: Path to config YAML (must include memit/health/nap sections)
        facts_path: Path to test facts JSON
        sleep_cycles: Number of full sleep cycles in Phase C
        test_retention_flag: Whether to run Phase D
        memit_only: If True, skip nap and full sleep phases (test MEMIT injection only)

    Returns:
        Full results dict
    """
    start_time = time.time()

    config = Config(config_path)
    facts = load_facts(facts_path)

    memit_enabled = config.get("memit.enabled", True)
    model_name = config.model["path"]

    print("=" * 70)
    print("  MEMIT BENCHMARK")
    print("=" * 70)
    print(f"  Model:   {model_name}")
    print(f"  Backend: {config.model.get('backend', 'mlx')}")
    print(f"  MEMIT:   {'ENABLED' if memit_enabled else 'DISABLED (LoRA baseline)'}")
    print(f"  LoRA:    rank={config.lora['rank']}, LR={config.lora['light_learning_rate']}")
    if memit_enabled:
        memit_cfg = config.get("memit", {})
        print(f"  MEMIT layers: {memit_cfg.get('target_layers', 'default')}")
        print(f"  MEMIT lambda: {memit_cfg.get('lambda_reg', 0.5)}")
    print(f"  Phases:  A(wake)" + (" B(nap) C(sleep)" if not memit_only else "") +
          (" D(retention)" if test_retention_flag and not memit_only else ""))
    print("=" * 70)
    print()

    # --- Initialize ---
    print("[INIT] Loading model and factory reset...")
    orchestrator = Orchestrator(config)
    orchestrator.factory_reset()
    print()

    # Disable auto-sleep/nap during testing — we control the timing
    orchestrator.chat.set_sleep_callback(lambda t: None)
    orchestrator.chat.set_nap_callback(lambda t: None)

    results = {
        "config": {
            "model": model_name,
            "backend": config.model.get("backend", "mlx"),
            "memit_enabled": memit_enabled,
            "memit_layers": config.get("memit.target_layers", []),
            "memit_lambda": config.get("memit.lambda_reg", 0.5),
            "lora_rank": config.lora["rank"],
            "lora_alpha": config.lora["alpha"],
            "learning_rate": config.lora["light_learning_rate"],
            "epochs": config.lora["light_epochs"],
        },
    }

    # ============================================================
    # PHASE A: Inject facts, test IMMEDIATE recall (MEMIT path)
    # ============================================================
    print("=" * 70)
    print("  PHASE A: Wake-phase injection + immediate recall")
    print("=" * 70)

    # Inject all facts via conversation
    print("\n  Injecting facts...")
    msg_count = 0
    for fact_group in facts["facts"]:
        for statement in fact_group["statements"]:
            print(f"    > {statement[:70]}...")
            response = orchestrator.process_message(statement)
            msg_count += 1

    print(f"\n  Injected {msg_count} messages")

    # Report MEMIT state
    status = orchestrator.get_status()
    print(f"  MEMIT edits: {status.get('memit_edits', 0)}")
    print(f"  MEMIT facts: {status.get('memit_facts', 0)}")
    print(f"  Sleep pressure: {status.get('sleep_pressure', 0)}")

    # Test recall IMMEDIATELY — no sleep yet
    print(f"\n  Testing immediate recall ({msg_count} facts injected, 0 sleep cycles)...")
    phase_a_recall = test_recall_quiet(orchestrator, facts)
    results["phase_a_recall"] = summarize_results(phase_a_recall, "Phase A Recall: ")

    print(f"\n  Testing immediate generalization...")
    phase_a_gen = test_generalization_quiet(orchestrator, facts)
    results["phase_a_generalization"] = summarize_results(phase_a_gen, "Phase A Gen:    ")

    if memit_only:
        elapsed = time.time() - start_time
        results["elapsed_seconds"] = elapsed
        _print_final_summary(results, elapsed)
        return results

    # ============================================================
    # PHASE B: Nap — consolidate MEMIT facts into LoRA
    # ============================================================
    print("\n" + "=" * 70)
    print("  PHASE B: Nap consolidation")
    print("=" * 70)

    print("\n  Running nap cycle...")
    nap_gen = orchestrator.trigger_nap_web()
    for event in nap_gen:
        if event.get("status") == "done":
            detail = event.get("detail", "")
            print(f"    Step {event.get('step', '?')}: {event.get('label', '?')} — {detail}")

    # Reset context for clean testing
    orchestrator.context.reset(keep_summary=True)
    orchestrator.chat.reset_turn_count()
    orchestrator.chat.set_sleep_callback(lambda t: None)
    orchestrator.chat.set_nap_callback(lambda t: None)

    # Report post-nap state
    status = orchestrator.get_status()
    print(f"\n  Post-nap MEMIT edits: {status.get('memit_edits', 0)}")
    print(f"  Post-nap MEMIT facts: {status.get('memit_facts', 0)}")

    print(f"\n  Testing post-nap recall...")
    phase_b_recall = test_recall_quiet(orchestrator, facts)
    results["phase_b_recall"] = summarize_results(phase_b_recall, "Phase B Recall: ")

    print(f"\n  Testing post-nap generalization...")
    phase_b_gen = test_generalization_quiet(orchestrator, facts)
    results["phase_b_generalization"] = summarize_results(phase_b_gen, "Phase B Gen:    ")

    # ============================================================
    # PHASE C: Full sleep — deep consolidation
    # ============================================================
    print("\n" + "=" * 70)
    print(f"  PHASE C: Full sleep ({sleep_cycles} cycle(s))")
    print("=" * 70)

    for i in range(sleep_cycles):
        print(f"\n  Sleep cycle {i+1}/{sleep_cycles}...")
        sleep_gen = orchestrator.trigger_sleep_web()
        for event in sleep_gen:
            if event.get("status") == "done":
                detail = event.get("detail", "")
                print(f"    Step {event.get('step', '?')}: {event.get('label', '?')} — {detail}")

    # Reset for testing
    orchestrator.context.reset(keep_summary=True)
    orchestrator.chat.reset_turn_count()
    orchestrator.chat.set_sleep_callback(lambda t: None)
    orchestrator.chat.set_nap_callback(lambda t: None)

    print(f"\n  Testing post-sleep recall...")
    phase_c_recall = test_recall_quiet(orchestrator, facts)
    results["phase_c_recall"] = summarize_results(phase_c_recall, "Phase C Recall: ")

    print(f"\n  Testing post-sleep generalization...")
    phase_c_gen = test_generalization_quiet(orchestrator, facts)
    results["phase_c_generalization"] = summarize_results(phase_c_gen, "Phase C Gen:    ")

    # ============================================================
    # PHASE D: Retention after distraction
    # ============================================================
    if test_retention_flag:
        print("\n" + "=" * 70)
        print("  PHASE D: Retention after distraction")
        print("=" * 70)

        # Run distractor conversation
        distractors = facts.get("distractor_conversation", [])
        print(f"\n  Running {len(distractors)} distractor messages...")
        for msg in distractors:
            orchestrator.process_message(msg)

        # Sleep on the distractor data
        print("  Running distractor sleep cycle...")
        sleep_gen = orchestrator.trigger_sleep_web()
        for event in sleep_gen:
            if event.get("status") == "done":
                detail = event.get("detail", "")
                print(f"    Step {event.get('step', '?')}: {event.get('label', '?')} — {detail}")

        orchestrator.context.reset(keep_summary=True)
        orchestrator.chat.reset_turn_count()
        orchestrator.chat.set_sleep_callback(lambda t: None)
        orchestrator.chat.set_nap_callback(lambda t: None)

        print(f"\n  Testing retention recall...")
        phase_d_recall = test_recall_quiet(orchestrator, facts)
        results["phase_d_recall"] = summarize_results(phase_d_recall, "Phase D Recall: ")
    else:
        print("\n  [Skipping Phase D — retention test]")

    elapsed = time.time() - start_time
    results["elapsed_seconds"] = elapsed

    _print_final_summary(results, elapsed)
    return results


def _print_final_summary(results, elapsed):
    """Print the final comparison table."""
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)

    cfg = results["config"]
    model_short = cfg["model"].split("/")[-1]
    memit_str = "MEMIT ON" if cfg["memit_enabled"] else "MEMIT OFF"
    print(f"  Model: {model_short} | {memit_str} | LR={cfg['learning_rate']}")
    print()

    headers = f"  {'Phase':<25} {'Recall':>8} {'Precision':>10} {'Perfect':>10}"
    print(headers)
    print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*10}")

    for key, label in [
        ("phase_a_recall", "A: Wake (immediate)"),
        ("phase_b_recall", "B: Post-nap"),
        ("phase_c_recall", "C: Post-sleep"),
        ("phase_d_recall", "D: Retention"),
    ]:
        if key in results:
            r = results[key]
            print(f"  {label:<25} {r['avg_recall']:>8.2f} {r['avg_precision']:>10.2f} "
                  f"{r['perfect_count']:>4}/{r['total_questions']:<5}")

    for key, label in [
        ("phase_a_generalization", "A: Wake gen"),
        ("phase_b_generalization", "B: Post-nap gen"),
        ("phase_c_generalization", "C: Post-sleep gen"),
    ]:
        if key in results:
            r = results[key]
            print(f"  {label:<25} {r['avg_recall']:>8.2f} {r['avg_precision']:>10.2f} "
                  f"{r['perfect_count']:>4}/{r['total_questions']:<5}")

    print(f"\n  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 70)


def run_comparison(config_memit_path, config_baseline_path, facts_path,
                   sleep_cycles=2, no_retention=False):
    """Run MEMIT and baseline side-by-side, print comparison."""
    print("\n" + "#" * 70)
    print("  RUNNING MEMIT VARIANT")
    print("#" * 70 + "\n")

    memit_results = run_memit_benchmark(
        config_memit_path, facts_path,
        sleep_cycles=sleep_cycles,
        test_retention_flag=not no_retention,
    )

    print("\n" + "#" * 70)
    print("  RUNNING BASELINE (NO MEMIT)")
    print("#" * 70 + "\n")

    baseline_results = run_memit_benchmark(
        config_baseline_path, facts_path,
        sleep_cycles=sleep_cycles,
        test_retention_flag=not no_retention,
    )

    # Print side-by-side comparison
    print("\n" + "=" * 70)
    print("  MEMIT vs BASELINE COMPARISON")
    print("=" * 70)

    print(f"\n  {'Phase':<25} {'MEMIT':>10} {'Baseline':>10} {'Delta':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")

    for key, label in [
        ("phase_a_recall", "A: Wake recall"),
        ("phase_b_recall", "B: Post-nap recall"),
        ("phase_c_recall", "C: Post-sleep recall"),
        ("phase_d_recall", "D: Retention"),
    ]:
        m = memit_results.get(key, {}).get("avg_recall", None)
        b = baseline_results.get(key, {}).get("avg_recall", None)
        if m is not None and b is not None:
            delta = m - b
            sign = "+" if delta > 0 else ""
            print(f"  {label:<25} {m:>10.2f} {b:>10.2f} {sign}{delta:>9.2f}")
        elif m is not None:
            print(f"  {label:<25} {m:>10.2f} {'N/A':>10}")

    print("=" * 70)

    return {"memit": memit_results, "baseline": baseline_results}


def main():
    parser = argparse.ArgumentParser(description="MEMIT Benchmark for Sleeping LLM")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML (with MEMIT settings)")
    parser.add_argument("--baseline-config", type=str, default=None,
                        help="Path to baseline config (MEMIT disabled). Runs comparison if provided.")
    parser.add_argument("--facts", type=str, default="experiments/facts/test_facts.json",
                        help="Path to test facts JSON")
    parser.add_argument("--sleep-cycles", type=int, default=2,
                        help="Number of full sleep cycles in Phase C")
    parser.add_argument("--no-retention", action="store_true",
                        help="Skip Phase D (retention test)")
    parser.add_argument("--memit-only", action="store_true",
                        help="Only test Phase A (wake injection), skip nap/sleep")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    if args.baseline_config:
        results = run_comparison(
            args.config, args.baseline_config, args.facts,
            sleep_cycles=args.sleep_cycles,
            no_retention=args.no_retention,
        )
    else:
        results = run_memit_benchmark(
            config_path=args.config,
            facts_path=args.facts,
            sleep_cycles=args.sleep_cycles,
            test_retention_flag=not args.no_retention,
            memit_only=args.memit_only,
        )

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        model_name = results.get("config", {}).get("model", "unknown").split("/")[-1]
        memit_tag = "memit" if results.get("config", {}).get("memit_enabled", True) else "baseline"
        output_path = Path("experiments/results") / f"{model_name}_{memit_tag}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Strip non-serializable details from results before saving
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        if isinstance(obj, float):
            if obj != obj:  # NaN
                return None
            return obj
        return obj

    with open(output_path, "w") as f:
        json.dump(clean_for_json(results), f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
