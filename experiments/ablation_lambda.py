"""Ablation 3: Lambda Regularization Sweep.

Injects the same 10 facts with different lambda_reg values to find the
optimal trade-off between recall and model coherence (perplexity).

Lambda values: 0.01, 0.05, 0.1, 0.5, 1.0

For each lambda:
  - Inject 10 facts
  - Measure recall (% correct raw completions)
  - Measure perplexity on reference text
  - Measure mean delta norm across edited layers

Usage:
    python experiments/ablation_lambda.py --config experiments/configs/8b_memit.yaml
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


LAMBDA_VALUES = [0.01, 0.05, 0.1, 0.5, 1.0]

TEST_FACTS = [
    FactTriple("Elena Voronov", "lives in", "Portland"),
    FactTriple("Elena Voronov", "works as", "marine biologist"),
    FactTriple("Marcus Takahashi", "lives in", "Austin"),
    FactTriple("Marcus Takahashi", "works as", "architect"),
    FactTriple("Priya Lindström", "lives in", "Denver"),
    FactTriple("Tobias Okafor", "lives in", "Seattle"),
    FactTriple("Tobias Okafor", "works as", "chef"),
    FactTriple("Yuki Petrov", "lives in", "Boston"),
    FactTriple("Yuki Petrov", "works as", "photographer"),
    FactTriple("Carlos Navarro", "lives in", "Nashville"),
]

REFERENCE_TEXT = (
    "The theory of general relativity, proposed by Albert Einstein in 1915, "
    "describes gravity as the warping of spacetime by mass and energy. "
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen "
    "using energy from sunlight in the chloroplasts of plant cells. "
    "The French Revolution of 1789 overthrew the monarchy and established the "
    "First Republic, fundamentally transforming French society and politics. "
    "DNA stores genetic information in a double helix structure, with base pairs "
    "of adenine-thymine and guanine-cytosine connected by hydrogen bonds."
)


def test_recall(backend, facts):
    """Test raw completion recall."""
    passed = 0
    results = []
    for fact in facts:
        prompt = fact.to_prompt()
        response = backend.generate(prompt, max_tokens=30, temperature=0.1)
        if response is None:
            response = ""
        found = fact.object.lower() in response.lower()
        if found:
            passed += 1
        results.append({
            "prompt": prompt,
            "expected": fact.object,
            "response": response[:80].strip(),
            "passed": found,
        })
    recall = passed / len(facts) if facts else 0
    return recall, results


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


def compute_delta_norms(memit_engine):
    """Compute mean L2 norm of weight deltas across active edits."""
    norms = []
    for edit in memit_engine.ledger.get_active_edits():
        # Each edit stores layer_deltas as a dict
        for layer_idx, delta in edit.get("layer_deltas", {}).items():
            try:
                import torch
                if isinstance(delta, torch.Tensor):
                    norms.append(delta.float().norm().item())
            except (ImportError, AttributeError):
                pass
    return sum(norms) / len(norms) if norms else 0.0


def run_lambda_trial(config_path, lambda_val):
    """Run a single trial with a specific lambda value."""
    print(f"\n{'─' * 60}")
    print(f"  Lambda = {lambda_val}")
    print(f"{'─' * 60}")

    config = Config(config_path)

    # Override lambda_reg in config
    if "memit" not in config._data:
        config._data["memit"] = {}
    config._data["memit"]["lambda_reg"] = lambda_val

    clean_artifacts(config)
    orch = Orchestrator(config, disable_memit=False)
    orch.chat._sleep_callback = None
    orch.chat._nap_callback = None

    # Baseline perplexity
    baseline_ppl = orch.backend.compute_perplexity(REFERENCE_TEXT)
    print(f"  Baseline perplexity: {baseline_ppl:.2f}")

    # Inject all 10 facts
    t0 = time.time()
    edit = orch.memit_engine.inject_facts(TEST_FACTS)
    inject_time = time.time() - t0
    print(f"  Injected {len(TEST_FACTS)} facts in {inject_time:.1f}s")

    # Measure recall
    recall, details = test_recall(orch.backend, TEST_FACTS)
    passed = sum(1 for d in details if d["passed"])
    print(f"  Recall: {recall:.2f} ({passed}/{len(TEST_FACTS)})")

    for d in details:
        s = "PASS" if d["passed"] else "FAIL"
        print(f"    [{s}] \"{d['prompt']}\" → \"{d['response']}\"")

    # Post-injection perplexity
    post_ppl = orch.backend.compute_perplexity(REFERENCE_TEXT)
    ppl_change = post_ppl - baseline_ppl
    print(f"  Post-injection perplexity: {post_ppl:.2f} (delta: {ppl_change:+.2f})")

    # Delta norms (from edit ledger)
    mean_delta_norm = 0.0
    if edit and hasattr(edit, 'layer_deltas') and edit.layer_deltas:
        norms = []
        for layer_idx, delta in edit.layer_deltas.items():
            try:
                import torch
                if isinstance(delta, torch.Tensor):
                    norms.append(delta.float().norm().item())
                else:
                    # MLX array
                    import mlx.core as mx
                    norms.append(mx.sqrt(mx.sum(delta * delta)).item())
            except Exception:
                pass
        mean_delta_norm = sum(norms) / len(norms) if norms else 0.0
    print(f"  Mean delta norm: {mean_delta_norm:.4f}")

    return {
        "lambda": lambda_val,
        "recall": recall,
        "passed": passed,
        "total": len(TEST_FACTS),
        "baseline_perplexity": baseline_ppl,
        "post_perplexity": post_ppl,
        "perplexity_delta": ppl_change,
        "mean_delta_norm": mean_delta_norm,
        "inject_time": inject_time,
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser(description="Ablation 3: Lambda Regularization Sweep")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--lambdas", type=str, default=None,
                        help="Comma-separated lambda values (default: 0.01,0.05,0.1,0.5,1.0)")
    args = parser.parse_args()

    lambda_values = LAMBDA_VALUES
    if args.lambdas:
        lambda_values = [float(x) for x in args.lambdas.split(",")]

    start_time = time.time()

    print("=" * 70)
    print("  ABLATION 3: Lambda Regularization Sweep")
    print("=" * 70)
    print(f"  Lambda values: {lambda_values}")
    print(f"  Facts: {len(TEST_FACTS)}")

    all_results = {
        "lambda_values": lambda_values,
        "num_facts": len(TEST_FACTS),
        "trials": [],
    }

    for lam in lambda_values:
        trial = run_lambda_trial(args.config, lam)
        all_results["trials"].append(trial)

    elapsed = time.time() - start_time
    all_results["total_elapsed_seconds"] = elapsed

    # ── Summary table ──
    print(f"\n{'=' * 70}")
    print(f"  LAMBDA SWEEP SUMMARY")
    print(f"{'=' * 70}")

    print(f"  {'Lambda':>8} {'Recall':>8} {'Pass':>6} {'PPL Base':>10} {'PPL Post':>10} {'PPL Δ':>8} {'Delta Norm':>12}")
    print(f"  {'-' * 8} {'-' * 8} {'-' * 6} {'-' * 10} {'-' * 10} {'-' * 8} {'-' * 12}")

    for t in all_results["trials"]:
        print(f"  {t['lambda']:>8.3f} {t['recall']:>8.2f} {t['passed']:>4}/{t['total']:<1} "
              f"{t['baseline_perplexity']:>10.2f} {t['post_perplexity']:>10.2f} "
              f"{t['perplexity_delta']:>+8.2f} {t['mean_delta_norm']:>12.4f}")

    # Find optimal lambda
    best = max(all_results["trials"], key=lambda t: t["recall"])
    print(f"\n  Best lambda: {best['lambda']} (recall={best['recall']:.2f}, PPL delta={best['perplexity_delta']:+.2f})")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save
    output_path = Path(args.output) if args.output else Path("experiments/results/ablation_lambda.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Results saved to {output_path}")


if __name__ == "__main__":
    main()
