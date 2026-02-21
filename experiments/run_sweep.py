"""Sweep runner — runs all experiment configs and aggregates results.

Usage:
    python experiments/run_sweep.py --configs experiments/configs/ --output experiments/results/
    python experiments/run_sweep.py --configs experiments/configs/ --filter "8b_*"
"""

import argparse
import csv
import glob
import json
import sys
import time
from pathlib import Path

# Ensure project root is on the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.scaling_benchmark import run_benchmark


def run_sweep(configs_dir, output_dir, facts_path, sleep_cycles, filter_pattern=None, no_retention=False):
    """Run all configs in directory, aggregate results."""
    configs_dir = Path(configs_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find config files
    pattern = filter_pattern or "*.yaml"
    config_files = sorted(configs_dir.glob(pattern))

    if not config_files:
        print(f"No config files found matching {configs_dir / pattern}")
        return

    print(f"Found {len(config_files)} config(s) to run:")
    for f in config_files:
        print(f"  - {f.name}")
    print()

    all_results = []
    sweep_start = time.time()

    for i, config_path in enumerate(config_files):
        run_id = config_path.stem
        print(f"\n{'#' * 60}")
        print(f"  RUN {i+1}/{len(config_files)}: {run_id}")
        print(f"{'#' * 60}\n")

        try:
            results = run_benchmark(
                config_path=str(config_path),
                facts_path=facts_path,
                sleep_cycles=sleep_cycles,
                test_retention_flag=not no_retention,
            )
            results["run_id"] = run_id

            # Save individual results
            run_output = output_dir / f"{run_id}.json"
            with open(run_output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n  Saved: {run_output}")

            all_results.append(results)

        except Exception as e:
            print(f"\n  ERROR: {run_id} failed: {e}")
            all_results.append({
                "run_id": run_id,
                "error": str(e),
                "config": {"model": "?"},
                "summary": None,
            })

    sweep_elapsed = time.time() - sweep_start

    # Generate summary CSV
    csv_path = output_dir / "summary.csv"
    write_summary_csv(all_results, csv_path)

    # Print comparison table
    print_comparison_table(all_results)

    print(f"\nTotal sweep time: {sweep_elapsed:.0f}s ({sweep_elapsed/60:.1f} min)")
    print(f"Summary CSV: {csv_path}")


def write_summary_csv(all_results, csv_path):
    """Write a summary CSV with one row per run."""
    fieldnames = [
        "run_id", "model", "rank", "lr", "epochs",
        "recall", "precision", "perfect",
        "generalization", "retention", "retention_drop",
        "time_s", "error",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in all_results:
            if result.get("error"):
                writer.writerow({
                    "run_id": result["run_id"],
                    "error": result["error"],
                })
                continue

            summary = result["summary"]
            config = result["config"]
            row = {
                "run_id": result["run_id"],
                "model": config["model"].split("/")[-1],
                "rank": config["lora_rank"],
                "lr": config["learning_rate"],
                "epochs": config["epochs"],
                "recall": f"{summary['recall']['avg_recall']:.3f}",
                "precision": f"{summary['recall']['avg_precision']:.3f}",
                "perfect": f"{summary['recall']['perfect_count']}/{summary['recall']['total_questions']}",
                "generalization": f"{summary['generalization']['avg_recall']:.3f}",
                "retention": f"{summary['retention']['avg_recall']:.3f}" if summary.get("retention") else "-",
                "retention_drop": f"{summary['retention']['recall_drop']:.3f}" if summary.get("retention") else "-",
                "time_s": f"{result['elapsed_seconds']:.0f}",
                "error": "",
            }
            writer.writerow(row)


def print_comparison_table(all_results):
    """Print a formatted comparison table to stdout."""
    print(f"\n{'=' * 90}")
    print(f"  COMPARISON TABLE")
    print(f"{'=' * 90}")
    print(f"  {'Run ID':<16} {'Model':<28} {'Recall':>7} {'Prec':>7} {'Gen':>7} {'Retain':>7} {'Time':>6}")
    print(f"  {'-'*16} {'-'*28} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*6}")

    for result in all_results:
        if result.get("error"):
            print(f"  {result['run_id']:<16} {'ERROR':<28} {'-':>7} {'-':>7} {'-':>7} {'-':>7} {'-':>6}")
            continue

        summary = result["summary"]
        config = result["config"]
        model_short = config["model"].split("/")[-1][:27]
        retention = f"{summary['retention']['avg_recall']:.2f}" if summary.get("retention") else "-"

        print(f"  {result['run_id']:<16} {model_short:<28} "
              f"{summary['recall']['avg_recall']:>7.2f} "
              f"{summary['recall']['avg_precision']:>7.2f} "
              f"{summary['generalization']['avg_recall']:>7.2f} "
              f"{retention:>7} "
              f"{result['elapsed_seconds']:>5.0f}s")

    print(f"{'=' * 90}")


def main():
    parser = argparse.ArgumentParser(description="Sleeping LLM Scaling Sweep")
    parser.add_argument("--configs", type=str, default="experiments/configs/", help="Directory containing config YAMLs")
    parser.add_argument("--output", type=str, default="experiments/results/", help="Directory for results")
    parser.add_argument("--facts", type=str, default="experiments/facts/test_facts.json", help="Path to test facts JSON")
    parser.add_argument("--sleep-cycles", type=int, default=2, help="Sleep cycles per run")
    parser.add_argument("--filter", type=str, default=None, help="Glob pattern to filter configs (e.g. '8b_*')")
    parser.add_argument("--no-retention", action="store_true", help="Skip retention tests (faster)")
    args = parser.parse_args()

    run_sweep(
        configs_dir=args.configs,
        output_dir=args.output,
        facts_path=args.facts,
        sleep_cycles=args.sleep_cycles,
        filter_pattern=args.filter,
        no_retention=args.no_retention,
    )


if __name__ == "__main__":
    main()
