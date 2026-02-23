"""MEMIT sweep runner — runs MEMIT vs baseline comparison across model sizes.

Runs all *_memit.yaml configs paired with their *_nomemit.yaml baselines,
then prints a unified comparison table.

Usage:
    # Run all model sizes (3B, 8B, 70B):
    python experiments/run_memit_sweep.py

    # Run only 3B:
    python experiments/run_memit_sweep.py --filter 3b

    # MEMIT-only mode (skip nap/sleep, just test wake injection):
    python experiments/run_memit_sweep.py --memit-only

    # Quick test (skip retention phase):
    python experiments/run_memit_sweep.py --no-retention --sleep-cycles 1
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is on the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.memit_benchmark import run_memit_benchmark


def discover_pairs(configs_dir, filter_prefix=None):
    """Find matching *_memit.yaml / *_nomemit.yaml pairs.

    Returns list of (size_label, memit_path, baseline_path) tuples.
    """
    configs_dir = Path(configs_dir)
    pairs = []

    memit_configs = sorted(configs_dir.glob("*_memit.yaml"))
    for memit_path in memit_configs:
        # Extract size prefix: "3b_memit.yaml" -> "3b"
        stem = memit_path.stem  # "3b_memit"
        size = stem.replace("_memit", "")  # "3b"

        if filter_prefix and size != filter_prefix:
            continue

        baseline_path = configs_dir / f"{size}_nomemit.yaml"
        if baseline_path.exists():
            pairs.append((size, memit_path, baseline_path))
        else:
            print(f"  Warning: no baseline found for {memit_path.name} (expected {baseline_path.name})")
            pairs.append((size, memit_path, None))

    return pairs


def run_sweep(configs_dir, output_dir, facts_path, sleep_cycles,
              filter_prefix=None, no_retention=False, memit_only=False):
    """Run MEMIT vs baseline across all model sizes."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_pairs(configs_dir, filter_prefix)
    if not pairs:
        print("No MEMIT config pairs found.")
        return

    print(f"Found {len(pairs)} model size(s) to benchmark:")
    for size, memit_path, baseline_path in pairs:
        baseline_str = baseline_path.name if baseline_path else "NONE"
        print(f"  {size.upper()}: {memit_path.name} vs {baseline_str}")
    print()

    all_results = {}
    sweep_start = time.time()

    for i, (size, memit_path, baseline_path) in enumerate(pairs):
        print(f"\n{'#' * 70}")
        print(f"  MODEL {i+1}/{len(pairs)}: {size.upper()}")
        print(f"{'#' * 70}")

        # Run MEMIT variant
        print(f"\n  --- MEMIT variant ({memit_path.name}) ---")
        try:
            memit_results = run_memit_benchmark(
                config_path=str(memit_path),
                facts_path=facts_path,
                sleep_cycles=sleep_cycles,
                test_retention_flag=not no_retention,
                memit_only=memit_only,
            )

            # Save individual results
            out_file = output_dir / f"{size}_memit.json"
            _save_results(memit_results, out_file)

        except Exception as e:
            print(f"\n  ERROR: MEMIT variant failed: {e}")
            memit_results = {"error": str(e)}

        # Run baseline variant
        baseline_results = None
        if baseline_path and not memit_only:
            print(f"\n  --- Baseline variant ({baseline_path.name}) ---")
            try:
                baseline_results = run_memit_benchmark(
                    config_path=str(baseline_path),
                    facts_path=facts_path,
                    sleep_cycles=sleep_cycles,
                    test_retention_flag=not no_retention,
                    memit_only=False,
                )

                out_file = output_dir / f"{size}_nomemit.json"
                _save_results(baseline_results, out_file)

            except Exception as e:
                print(f"\n  ERROR: Baseline variant failed: {e}")
                baseline_results = {"error": str(e)}

        all_results[size] = {
            "memit": memit_results,
            "baseline": baseline_results,
        }

    sweep_elapsed = time.time() - sweep_start

    # Print unified comparison
    _print_sweep_comparison(all_results, memit_only)

    # Save aggregate results
    aggregate_path = output_dir / "memit_sweep_results.json"
    _save_results(all_results, aggregate_path)

    print(f"\nTotal sweep time: {sweep_elapsed:.0f}s ({sweep_elapsed/60:.1f} min)")
    print(f"Results saved to: {output_dir}")


def _save_results(results, path):
    """Save results dict to JSON, handling non-serializable values."""
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        if isinstance(obj, float):
            if obj != obj:  # NaN
                return None
            return obj
        return obj

    with open(path, "w") as f:
        json.dump(clean(results), f, indent=2)
    print(f"  Saved: {path}")


def _get_recall(results, phase_key):
    """Extract avg_recall from a phase, or None."""
    if not results or results.get("error"):
        return None
    phase = results.get(phase_key, {})
    return phase.get("avg_recall")


def _print_sweep_comparison(all_results, memit_only=False):
    """Print unified comparison table across all model sizes."""
    print(f"\n{'=' * 90}")
    print(f"  MEMIT SWEEP — UNIFIED COMPARISON")
    print(f"{'=' * 90}")

    if memit_only:
        phases = [("phase_a_recall", "Wake (A)")]
    else:
        phases = [
            ("phase_a_recall", "Wake (A)"),
            ("phase_b_recall", "Nap (B)"),
            ("phase_c_recall", "Sleep (C)"),
            ("phase_d_recall", "Retain (D)"),
        ]

    # Header
    phase_headers = "".join(f" {label:>12}" for _, label in phases)
    print(f"\n  {'Model':<12} {'Variant':<10}{phase_headers}")
    print(f"  {'-'*12} {'-'*10}" + " ".join(f"{'-'*12}" for _ in phases))

    for size in sorted(all_results.keys(), key=lambda s: _model_sort_key(s)):
        data = all_results[size]

        # MEMIT row
        memit = data.get("memit")
        vals = []
        for phase_key, _ in phases:
            r = _get_recall(memit, phase_key)
            vals.append(f"{r:.2f}" if r is not None else "-")
        vals_str = "".join(f" {v:>12}" for v in vals)
        print(f"  {size.upper():<12} {'MEMIT':<10}{vals_str}")

        # Baseline row
        baseline = data.get("baseline")
        if baseline and not baseline.get("error"):
            vals = []
            for phase_key, _ in phases:
                r = _get_recall(baseline, phase_key)
                vals.append(f"{r:.2f}" if r is not None else "-")
            vals_str = "".join(f" {v:>12}" for v in vals)
            print(f"  {'':<12} {'baseline':<10}{vals_str}")

            # Delta row
            deltas = []
            for phase_key, _ in phases:
                m = _get_recall(memit, phase_key)
                b = _get_recall(baseline, phase_key)
                if m is not None and b is not None:
                    d = m - b
                    sign = "+" if d > 0 else ""
                    deltas.append(f"{sign}{d:.2f}")
                else:
                    deltas.append("-")
            deltas_str = "".join(f" {d:>12}" for d in deltas)
            print(f"  {'':<12} {'delta':<10}{deltas_str}")

        print()

    print(f"{'=' * 90}")


def _model_sort_key(size_str):
    """Sort model sizes numerically: 3b < 8b < 70b."""
    num = size_str.lower().replace("b", "")
    try:
        return int(num)
    except ValueError:
        return 999


def main():
    parser = argparse.ArgumentParser(description="MEMIT Sweep — compare MEMIT vs baseline across model sizes")
    parser.add_argument("--configs", type=str, default="experiments/configs/",
                        help="Directory containing config YAMLs")
    parser.add_argument("--output", type=str, default="experiments/results/",
                        help="Directory for results")
    parser.add_argument("--facts", type=str, default="experiments/facts/test_facts.json",
                        help="Path to test facts JSON")
    parser.add_argument("--sleep-cycles", type=int, default=2,
                        help="Number of full sleep cycles in Phase C")
    parser.add_argument("--filter", type=str, default=None,
                        help="Model size prefix to filter (e.g. '3b', '8b', '70b')")
    parser.add_argument("--no-retention", action="store_true",
                        help="Skip Phase D (retention test)")
    parser.add_argument("--memit-only", action="store_true",
                        help="Only test Phase A (MEMIT wake injection, no nap/sleep)")
    args = parser.parse_args()

    run_sweep(
        configs_dir=args.configs,
        output_dir=args.output,
        facts_path=args.facts,
        sleep_cycles=args.sleep_cycles,
        filter_prefix=args.filter,
        no_retention=args.no_retention,
        memit_only=args.memit_only,
    )


if __name__ == "__main__":
    main()
