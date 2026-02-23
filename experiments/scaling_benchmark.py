"""Scaling benchmark — automated memory retention test.

Injects facts into the model via conversation, runs sleep cycles,
then tests recall, precision, generalization, and retention.

Usage:
    python experiments/scaling_benchmark.py --config experiments/configs/8b_lr1e4.yaml
    python experiments/scaling_benchmark.py --config config.yaml --facts experiments/facts/test_facts.json
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


def load_facts(facts_path):
    """Load test facts from JSON file."""
    with open(facts_path) as f:
        return json.load(f)


def inject_facts(orchestrator, facts):
    """Inject facts into the model via conversation.

    Sends each fact statement as a user message and consumes the response.
    Returns the number of messages sent.
    """
    msg_count = 0
    for fact_group in facts["facts"]:
        for statement in fact_group["statements"]:
            print(f"  Injecting: {statement[:60]}...")
            response = orchestrator.process_message(statement)
            msg_count += 1
    return msg_count


def run_sleep_cycles(orchestrator, num_cycles):
    """Run N sleep cycles, consuming the streaming generator."""
    for i in range(num_cycles):
        print(f"  Sleep cycle {i+1}/{num_cycles}...")
        gen = orchestrator.trigger_sleep_web()
        last_event = None
        for event in gen:
            last_event = event
            if event.get("status") == "done":
                detail = event.get("detail", "")
                print(f"    Step {event['step']}: {event['label']} — {detail}")
        print()


def score_response(response, question_data):
    """Score a model response against expected/forbidden keywords.

    Returns:
        dict with recall, precision, expected_found, forbidden_found
    """
    response_lower = response.lower()

    expected = question_data["expected"]
    forbidden = question_data.get("forbidden", [])

    expected_found = [kw for kw in expected if kw.lower() in response_lower]
    forbidden_found = [kw for kw in forbidden if kw.lower() in response_lower]

    recall = len(expected_found) / len(expected) if expected else 1.0
    precision = 1.0 if not forbidden_found else max(0.0, 1.0 - 0.5 * len(forbidden_found))

    return {
        "recall": recall,
        "precision": precision,
        "expected_found": expected_found,
        "expected_missing": [kw for kw in expected if kw.lower() not in response_lower],
        "forbidden_found": forbidden_found,
    }


def test_recall(orchestrator, facts):
    """Test recall of individual facts.

    Returns list of scored results.
    """
    results = []
    for fact_group in facts["facts"]:
        for q_data in fact_group["recall_questions"]:
            question = q_data["question"]
            response = orchestrator.process_message(question)

            score = score_response(response, q_data)
            result = {
                "question": question,
                "response": response,
                **score,
            }
            results.append(result)

            status = "PASS" if score["recall"] == 1.0 and score["precision"] == 1.0 else "PARTIAL" if score["recall"] > 0 else "FAIL"
            print(f"  [{status}] {question}")
            print(f"         Response: {response[:100]}...")
            if score["expected_missing"]:
                print(f"         Missing: {score['expected_missing']}")
            if score["forbidden_found"]:
                print(f"         Forbidden: {score['forbidden_found']}")

    return results


def test_generalization(orchestrator, facts):
    """Test generalization questions that require combining facts.

    Returns list of scored results.
    """
    results = []
    for q_data in facts.get("generalization_questions", []):
        question = q_data["question"]
        response = orchestrator.process_message(question)

        score = score_response(response, q_data)
        result = {
            "question": question,
            "response": response,
            **score,
        }
        results.append(result)

        status = "PASS" if score["recall"] == 1.0 else "PARTIAL" if score["recall"] > 0 else "FAIL"
        print(f"  [{status}] {question}")
        print(f"         Response: {response[:100]}...")

    return results


def test_retention(orchestrator, facts, distractor_turns=5, sleep_cycles=2):
    """Test retention after distractor conversation and additional sleep.

    Injects unrelated conversation, sleeps, then retests recall.
    """
    # Run distractor conversation (auto-sleep may trigger here — that's fine)
    distractors = facts.get("distractor_conversation", [])
    for msg in distractors[:distractor_turns]:
        orchestrator.process_message(msg)

    # Sleep on the distractor data
    run_sleep_cycles(orchestrator, sleep_cycles)

    # Disable auto-sleep before retesting recall
    orchestrator.chat.set_sleep_callback(lambda t: None)
    orchestrator.chat.reset_turn_count()

    # Retest recall
    print("  Retesting recall after distraction...")
    return test_recall(orchestrator, facts)


def compute_summary(recall_results, generalization_results, retention_results=None):
    """Compute aggregate scores."""
    def avg(values):
        return sum(values) / len(values) if values else 0.0

    summary = {
        "recall": {
            "avg_recall": avg([r["recall"] for r in recall_results]),
            "avg_precision": avg([r["precision"] for r in recall_results]),
            "perfect_count": sum(1 for r in recall_results if r["recall"] == 1.0 and r["precision"] == 1.0),
            "total_questions": len(recall_results),
        },
        "generalization": {
            "avg_recall": avg([r["recall"] for r in generalization_results]),
            "avg_precision": avg([r["precision"] for r in generalization_results]),
            "total_questions": len(generalization_results),
        },
    }

    if retention_results:
        summary["retention"] = {
            "avg_recall": avg([r["recall"] for r in retention_results]),
            "avg_precision": avg([r["precision"] for r in retention_results]),
            "recall_drop": summary["recall"]["avg_recall"] - avg([r["recall"] for r in retention_results]),
        }

    return summary


def run_benchmark(config_path, facts_path, sleep_cycles=2, test_retention_flag=True):
    """Run a complete benchmark."""
    start_time = time.time()

    # Load config and facts
    config = Config(config_path)
    facts = load_facts(facts_path)

    print("=" * 60)
    print(f"  SCALING BENCHMARK")
    print(f"  Model: {config.model['path']}")
    print(f"  Backend: {config.model.get('backend', 'mlx')}")
    print(f"  LoRA: rank={config.lora['rank']}, alpha={config.lora['alpha']}, layers={config.lora['layers']}")
    print(f"  LR: {config.lora['light_learning_rate']}, Epochs: {config.lora['light_epochs']}")
    print(f"  Sleep cycles: {sleep_cycles}")
    print("=" * 60)
    print()

    # Initialize
    print("[1/6] Initializing orchestrator...")
    orchestrator = Orchestrator(config)

    # Factory reset to start clean
    print("[2/6] Factory reset...")
    orchestrator.factory_reset()
    print()

    # Inject facts
    print("[3/6] Injecting facts...")
    msg_count = inject_facts(orchestrator, facts)
    print(f"  Injected {msg_count} messages\n")

    # Sleep
    print(f"[4/6] Running {sleep_cycles} sleep cycle(s)...")
    run_sleep_cycles(orchestrator, sleep_cycles)

    # Restart context — preserve summary (mirrors real system behavior after sleep)
    orchestrator.context.reset(keep_summary=True)
    orchestrator.chat.reset_turn_count()

    # Disable auto-sleep during testing — test questions must NOT trigger sleep
    # (otherwise the model trains on wrong answers, contaminating the experiment)
    orchestrator.chat.set_sleep_callback(lambda t: None)

    # Test recall
    print("[5/6] Testing recall...")
    recall_results = test_recall(orchestrator, facts)
    print()

    # Test generalization
    print("  Testing generalization...")
    gen_results = test_generalization(orchestrator, facts)
    print()

    # Test retention (optional)
    retention_results = None
    if test_retention_flag:
        print("[6/6] Testing retention after distraction...")
        # Re-enable sleep for the distraction + sleep portion
        orchestrator.chat.set_sleep_callback(orchestrator._on_sleep_trigger)
        retention_results = test_retention(orchestrator, facts)
        print()
    else:
        print("[6/6] Skipping retention test\n")

    # Compute summary
    summary = compute_summary(recall_results, gen_results, retention_results)
    elapsed = time.time() - start_time

    # Build full results
    results = {
        "config": {
            "model": config.model["path"],
            "backend": config.model.get("backend", "mlx"),
            "lora_rank": config.lora["rank"],
            "lora_alpha": config.lora["alpha"],
            "lora_layers": config.lora["layers"],
            "learning_rate": config.lora["light_learning_rate"],
            "epochs": config.lora["light_epochs"],
            "sleep_cycles": sleep_cycles,
        },
        "summary": summary,
        "recall_details": recall_results,
        "generalization_details": gen_results,
        "retention_details": retention_results,
        "elapsed_seconds": elapsed,
    }

    # Print summary
    print("=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Recall:         {summary['recall']['avg_recall']:.2f} ({summary['recall']['perfect_count']}/{summary['recall']['total_questions']} perfect)")
    print(f"  Precision:      {summary['recall']['avg_precision']:.2f}")
    print(f"  Generalization: {summary['generalization']['avg_recall']:.2f}")
    if retention_results:
        print(f"  Retention:      {summary['retention']['avg_recall']:.2f} (drop: {summary['retention']['recall_drop']:.2f})")
    print(f"  Time:           {elapsed:.1f}s")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description="Sleeping LLM Scaling Benchmark")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    parser.add_argument("--facts", type=str, default="experiments/facts/test_facts.json", help="Path to test facts JSON")
    parser.add_argument("--sleep-cycles", type=int, default=2, help="Number of sleep cycles")
    parser.add_argument("--no-retention", action="store_true", help="Skip retention test")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    results = run_benchmark(
        config_path=args.config,
        facts_path=args.facts,
        sleep_cycles=args.sleep_cycles,
        test_retention_flag=not args.no_retention,
    )

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")
    else:
        # Auto-generate output path
        model_name = results["config"]["model"].split("/")[-1]
        lr = results["config"]["learning_rate"]
        output_path = Path("experiments/results") / f"{model_name}_lr{lr}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
