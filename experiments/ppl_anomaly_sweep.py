"""PPL anomaly sweep — systematic investigation of non-monotonic PPL scaling.

Problem: 3B +1.6%, 8B +14.1%, 70B +0.3% PPL increase after sleep.
This script runs a systematic sweep to characterize and explain the anomaly.

Experimental phases:
  1. Reproduce: Confirm anomaly across model sizes
  2. Rank sweep: LoRA rank 8/16/32/64 (8B only)
  3. Layer sweep: Different target layer ranges (8B only)
  4. LR sweep: Learning rates 2e-5 to 2e-4 (8B only)
  5. Quantization: BF16 vs NF4 (8B only, requires GPU)

Each condition: inject 10 facts → sleep → measure PPL delta.

Usage:
    python experiments/ppl_anomaly_sweep.py --phase 1              # Reproduce
    python experiments/ppl_anomaly_sweep.py --phase 2              # Rank sweep (8B)
    python experiments/ppl_anomaly_sweep.py --phase all            # Full sweep
    python experiments/ppl_anomaly_sweep.py --phase 1 --repeats 3  # Repeat for noise check
"""

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.memory.memit import FactTriple
from src.orchestrator import Orchestrator

RESULTS_DIR = Path("results")

# 10 standard test facts for all conditions
STANDARD_FACTS = [
    FactTriple("Maren", "lives in", "Oslo"),
    FactTriple("Soren", "works as", "architect"),
    FactTriple("Elina", "lives in", "Helsinki"),
    FactTriple("Kiran", "works as", "botanist"),
    FactTriple("Thalia", "lives in", "Athens"),
    FactTriple("Lucien", "works as", "violinist"),
    FactTriple("Amara", "lives in", "Nairobi"),
    FactTriple("Callum", "works as", "geologist"),
    FactTriple("Yuki", "lives in", "Kyoto"),
    FactTriple("Rohan", "works as", "pilot"),
]


def get_phase_configs():
    """Return sweep configurations for each phase."""
    return {
        1: {
            "name": "Reproduce across model sizes",
            "conditions": [
                {"label": "3B", "config": "experiments/configs/3b_baseline.yaml"},
                {"label": "8B", "config": "experiments/configs/8b_lr1e4.yaml"},
                {"label": "70B", "config": "experiments/configs/70b_baseline.yaml"},
            ],
        },
        2: {
            "name": "LoRA rank sweep (8B)",
            "conditions": [
                {"label": "rank_8", "config": "experiments/configs/8b_lr1e4.yaml",
                 "overrides": {"lora": {"rank": 8, "alpha": 16}}},
                {"label": "rank_16", "config": "experiments/configs/8b_lr1e4.yaml",
                 "overrides": {"lora": {"rank": 16, "alpha": 32}}},
                {"label": "rank_32", "config": "experiments/configs/8b_lr1e4.yaml",
                 "overrides": {"lora": {"rank": 32, "alpha": 64}}},
                {"label": "rank_64", "config": "experiments/configs/8b_lr1e4.yaml",
                 "overrides": {"lora": {"rank": 64, "alpha": 128}}},
            ],
        },
        3: {
            "name": "Layer sweep (8B)",
            "conditions": [
                {"label": "layers_8_15", "config": "experiments/configs/8b_lr1e4.yaml",
                 "overrides": {"memit": {"target_layers": [8, 9, 10, 11, 12, 13, 14, 15]}}},
                {"label": "layers_12_19", "config": "experiments/configs/8b_lr1e4.yaml",
                 "overrides": {"memit": {"target_layers": [12, 13, 14, 15, 16, 17, 18, 19]}}},
                {"label": "layers_16_23", "config": "experiments/configs/8b_lr1e4.yaml",
                 "overrides": {"memit": {"target_layers": [16, 17, 18, 19, 20, 21, 22, 23]}}},
                {"label": "layers_12_15", "config": "experiments/configs/8b_lr1e4.yaml",
                 "overrides": {"memit": {"target_layers": [12, 13, 14, 15]}}},
            ],
        },
        4: {
            "name": "Learning rate sweep (8B)",
            "conditions": [
                {"label": "lr_2e5", "config": "experiments/configs/8b_lr1e4.yaml",
                 "overrides": {"lora": {"light_learning_rate": 2e-5}}},
                {"label": "lr_5e5", "config": "experiments/configs/8b_lr1e4.yaml",
                 "overrides": {"lora": {"light_learning_rate": 5e-5}}},
                {"label": "lr_1e4", "config": "experiments/configs/8b_lr1e4.yaml",
                 "overrides": {"lora": {"light_learning_rate": 1e-4}}},
                {"label": "lr_2e4", "config": "experiments/configs/8b_lr1e4.yaml",
                 "overrides": {"lora": {"light_learning_rate": 2e-4}}},
            ],
        },
        5: {
            "name": "Quantization (8B BF16 vs NF4)",
            "conditions": [
                {"label": "8B_nf4", "config": "experiments/configs/8b_lr1e4.yaml"},
                # BF16 requires a separate config with quantization disabled
                {"label": "8B_bf16", "config": "experiments/configs/8b_lr1e4.yaml",
                 "overrides": {"model": {"quantization": "none"}}},
            ],
        },
    }


def apply_overrides(config, overrides):
    """Apply nested dict overrides to a Config object."""
    for section, values in overrides.items():
        if hasattr(config, '_data') and section in config._data:
            if isinstance(values, dict):
                config._data[section].update(values)
            else:
                config._data[section] = values


def clean_for_condition(config):
    """Clean artifacts between conditions."""
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

    ledger_path = Path(config.paths.get("memit_ledger", "data/memit/ledger.json"))
    if ledger_path.exists():
        ledger_path.unlink()


def measure_ppl(orch) -> float:
    """Measure perplexity on identity reference text."""
    identity_dir = Path(orch.config.paths["core_identity"])
    identity_file = identity_dir / "identity.jsonl"
    if not identity_file.exists():
        return -1.0
    texts = []
    with open(identity_file) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    texts.append(item.get("text", ""))
                except json.JSONDecodeError:
                    continue
    ref_text = " ".join(texts)[:2000]
    if not ref_text:
        return -1.0
    try:
        return round(orch.backend.compute_perplexity(ref_text), 4)
    except Exception:
        return -1.0


def run_condition(config_path, overrides=None, label=""):
    """Run a single experimental condition: inject facts → sleep → measure PPL.

    Returns:
        dict with ppl_pre, ppl_post, ppl_delta_pct, recall, wall_time
    """
    config = Config(config_path)
    if overrides:
        apply_overrides(config, overrides)

    print(f"\n  Condition: {label}")
    print(f"  Config: {config_path}")

    clean_for_condition(config)

    t0 = time.time()
    orch = Orchestrator(config, disable_memit=False)
    orch.chat._sleep_callback = None
    orch.chat._nap_callback = None

    # Measure baseline PPL
    ppl_pre = measure_ppl(orch)
    print(f"    Baseline PPL: {ppl_pre:.4f}")

    # Inject 10 facts
    injected = 0
    for fact in STANDARD_FACTS:
        edit = orch.memit_engine.inject_fact(fact)
        if edit:
            injected += 1
    print(f"    Injected {injected}/{len(STANDARD_FACTS)} facts")

    # Brief chat for curation
    for fact in STANDARD_FACTS[:3]:
        try:
            orch.chat.process_input(f"Tell me about {fact.subject}")
        except Exception:
            pass

    # Trigger sleep
    try:
        orch._on_sleep_trigger("ppl_sweep")
        sleep_ok = True
    except Exception as e:
        print(f"    Sleep failed: {e}")
        sleep_ok = False

    # Measure post-sleep PPL
    ppl_post = measure_ppl(orch)

    # Measure raw recall
    recalled = 0
    for fact in STANDARD_FACTS:
        prompt = fact.to_prompt()
        response = orch.backend.generate(prompt, max_tokens=20, temperature=0.1)
        if response and fact.object.lower() in response.lower():
            recalled += 1

    wall_time = time.time() - t0

    ppl_delta = -1.0
    if ppl_pre > 0 and ppl_post > 0:
        ppl_delta = round((ppl_post - ppl_pre) / ppl_pre * 100, 2)

    result = {
        "label": label,
        "ppl_pre": ppl_pre,
        "ppl_post": ppl_post,
        "ppl_delta_pct": ppl_delta,
        "recall": f"{recalled}/{len(STANDARD_FACTS)}",
        "recall_rate": round(recalled / len(STANDARD_FACTS), 2),
        "sleep_ok": sleep_ok,
        "wall_time_seconds": round(wall_time, 1),
    }

    print(f"    PPL: {ppl_pre:.4f} → {ppl_post:.4f} ({ppl_delta:+.2f}%)")
    print(f"    Recall: {recalled}/{len(STANDARD_FACTS)}")
    print(f"    Time: {wall_time:.1f}s")

    # Cleanup
    del orch

    return result


def run_phase(phase_num, repeats=1):
    """Run all conditions in a phase, optionally with repeats."""
    phases = get_phase_configs()
    if phase_num not in phases:
        print(f"Unknown phase: {phase_num}")
        return []

    phase = phases[phase_num]
    print(f"\n{'=' * 60}")
    print(f"  Phase {phase_num}: {phase['name']}")
    print(f"  Conditions: {len(phase['conditions'])}, Repeats: {repeats}")
    print(f"{'=' * 60}")

    all_results = []
    for condition in phase["conditions"]:
        for rep in range(repeats):
            label = condition["label"]
            if repeats > 1:
                label = f"{label}_rep{rep + 1}"
            result = run_condition(
                config_path=condition["config"],
                overrides=condition.get("overrides"),
                label=label,
            )
            result["phase"] = phase_num
            result["condition"] = condition["label"]
            result["repeat"] = rep + 1
            all_results.append(result)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="PPL anomaly sweep")
    parser.add_argument("--phase", type=str, default="1", help="Phase number (1-5) or 'all'")
    parser.add_argument("--repeats", type=int, default=1, help="Repeats per condition")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = RESULTS_DIR / f"ppl_sweep_{timestamp}.json"

    all_results = []

    if args.phase == "all":
        for phase_num in range(1, 6):
            results = run_phase(phase_num, args.repeats)
            all_results.extend(results)
    else:
        phase_num = int(args.phase)
        all_results = run_phase(phase_num, args.repeats)

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print(f"\n{'=' * 60}")
    print(f"  Summary")
    print(f"{'=' * 60}")
    print(f"  {'Label':<20} {'PPL Pre':>10} {'PPL Post':>10} {'Delta':>10} {'Recall':>10}")
    print(f"  {'-' * 20} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")
    for r in all_results:
        delta_str = f"{r['ppl_delta_pct']:+.2f}%" if r['ppl_delta_pct'] != -1 else "N/A"
        print(f"  {r['label']:<20} {r['ppl_pre']:>10.4f} {r['ppl_post']:>10.4f} {delta_str:>10} {r['recall']:>10}")

    print(f"\n  Results saved: {output_path}")


if __name__ == "__main__":
    main()
