"""Ablation 5: Perplexity Through Lifecycle.

Tracks model coherence (perplexity) throughout the full lifecycle:
  1. Baseline
  2. After each of 5 MEMIT injections
  3. After nap
  4. After each of 5 more MEMIT injections
  5. After full sleep

Outputs a trajectory suitable for plotting: step vs perplexity, annotated with events.

Usage:
    python experiments/ablation_perplexity.py --config experiments/configs/8b_memit.yaml
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


# ── Facts: 2 batches of 5 ──
BATCH_1 = [
    FactTriple("Elena Voronov", "lives in", "Portland"),
    FactTriple("Elena Voronov", "works as", "marine biologist"),
    FactTriple("Marcus Takahashi", "lives in", "Austin"),
    FactTriple("Marcus Takahashi", "works as", "architect"),
    FactTriple("Priya Lindström", "lives in", "Denver"),
]

BATCH_2 = [
    FactTriple("Tobias Okafor", "lives in", "Seattle"),
    FactTriple("Tobias Okafor", "works as", "chef"),
    FactTriple("Yuki Petrov", "lives in", "Boston"),
    FactTriple("Yuki Petrov", "works as", "photographer"),
    FactTriple("Carlos Navarro", "lives in", "Nashville"),
]

# Teaching messages (for sleep to have conversation data)
TEACHING_MESSAGES = [
    "Elena Voronov lives in Portland and works as a marine biologist.",
    "Marcus Takahashi is an architect who lives in Austin.",
    "Priya Lindström lives in Denver.",
    "Tobias Okafor is a chef from Seattle.",
    "Yuki Petrov is a photographer based in Boston.",
    "Carlos Navarro lives in Nashville.",
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


def measure_perplexity(backend):
    """Measure perplexity averaged over multiple reference texts."""
    ppls = []
    for text in REFERENCE_TEXTS:
        ppl = backend.compute_perplexity(text)
        ppls.append(ppl)
    return sum(ppls) / len(ppls)


def test_recall(backend, facts):
    """Quick recall check, return fraction correct."""
    passed = 0
    for fact in facts:
        prompt = fact.to_prompt()
        response = backend.generate(prompt, max_tokens=30, temperature=0.1)
        if response is None:
            response = ""
        if fact.object.lower() in response.lower():
            passed += 1
    return passed / len(facts) if facts else 0


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


def main():
    parser = argparse.ArgumentParser(description="Ablation 5: Perplexity Through Lifecycle")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    start_time = time.time()

    print("=" * 70)
    print("  ABLATION 5: Perplexity Through Lifecycle")
    print("=" * 70)

    config = Config(args.config)
    clean_artifacts(config)
    orch = Orchestrator(config, disable_memit=False)
    orch.chat._sleep_callback = None
    orch.chat._nap_callback = None

    trajectory = []  # List of {step, event, perplexity, recall, ...}
    step = 0

    # ── Baseline ──
    ppl = measure_perplexity(orch.backend)
    print(f"\n  [{step}] Baseline perplexity: {ppl:.2f}")
    trajectory.append({
        "step": step,
        "event": "baseline",
        "perplexity": round(ppl, 3),
        "total_facts": 0,
        "memit_edits": 0,
        "recall": None,
    })
    step += 1

    # ── Batch 1: inject facts one at a time, measure after each ──
    print(f"\n  Injecting Batch 1 (facts 1-5)...")
    for i, fact in enumerate(BATCH_1):
        edit = orch.memit_engine.inject_facts([fact])
        ppl = measure_perplexity(orch.backend)
        all_facts_so_far = BATCH_1[:i + 1]
        recall = test_recall(orch.backend, all_facts_so_far)
        status = orch.get_status()

        print(f"  [{step}] After fact {i + 1}: PPL={ppl:.2f}, recall={recall:.2f}")
        trajectory.append({
            "step": step,
            "event": f"memit_inject_{i + 1}",
            "perplexity": round(ppl, 3),
            "total_facts": i + 1,
            "memit_edits": status["memit_edits"],
            "recall": round(recall, 3),
            "fact": fact.to_prompt(),
        })
        step += 1

    # ── Nap ──
    print(f"\n  Triggering nap...")
    pre_nap_edits = orch.get_status()["memit_edits"]
    t0 = time.time()
    try:
        orch._on_nap_trigger("test")
        nap_time = time.time() - t0
        nap_ok = True
        print(f"  Nap completed in {nap_time:.1f}s")
    except Exception as e:
        nap_time = time.time() - t0
        nap_ok = False
        print(f"  Nap failed: {e}")

    ppl = measure_perplexity(orch.backend)
    recall = test_recall(orch.backend, BATCH_1)
    post_nap_edits = orch.get_status()["memit_edits"]
    print(f"  [{step}] After nap: PPL={ppl:.2f}, recall={recall:.2f}, edits: {pre_nap_edits}→{post_nap_edits}")

    trajectory.append({
        "step": step,
        "event": "nap",
        "perplexity": round(ppl, 3),
        "total_facts": len(BATCH_1),
        "memit_edits": post_nap_edits,
        "recall": round(recall, 3),
        "nap_ok": nap_ok,
        "nap_time": round(nap_time, 1),
    })
    step += 1

    # ── Batch 2: inject facts one at a time ──
    print(f"\n  Injecting Batch 2 (facts 6-10)...")
    for i, fact in enumerate(BATCH_2):
        edit = orch.memit_engine.inject_facts([fact])
        ppl = measure_perplexity(orch.backend)
        all_facts_so_far = BATCH_1 + BATCH_2[:i + 1]
        recall = test_recall(orch.backend, all_facts_so_far)
        status = orch.get_status()

        fact_num = len(BATCH_1) + i + 1
        print(f"  [{step}] After fact {fact_num}: PPL={ppl:.2f}, recall={recall:.2f}")
        trajectory.append({
            "step": step,
            "event": f"memit_inject_{fact_num}",
            "perplexity": round(ppl, 3),
            "total_facts": fact_num,
            "memit_edits": status["memit_edits"],
            "recall": round(recall, 3),
            "fact": fact.to_prompt(),
        })
        step += 1

    # ── Teach via conversation (so full sleep has data) ──
    print(f"\n  Teaching via conversation for sleep data...")
    for msg in TEACHING_MESSAGES:
        orch.chat.process_input(msg)

    # ── Full sleep ──
    print(f"\n  Triggering full sleep...")
    pre_sleep_edits = orch.get_status()["memit_edits"]
    t0 = time.time()
    try:
        orch._on_sleep_trigger("test")
        sleep_time = time.time() - t0
        sleep_ok = True
        print(f"  Sleep completed in {sleep_time:.1f}s")
    except Exception as e:
        sleep_time = time.time() - t0
        sleep_ok = False
        print(f"  Sleep failed: {e}")

    ppl = measure_perplexity(orch.backend)
    all_facts = BATCH_1 + BATCH_2
    recall = test_recall(orch.backend, all_facts)
    post_sleep_edits = orch.get_status()["memit_edits"]
    print(f"  [{step}] After sleep: PPL={ppl:.2f}, recall={recall:.2f}, edits: {pre_sleep_edits}→{post_sleep_edits}")

    trajectory.append({
        "step": step,
        "event": "full_sleep",
        "perplexity": round(ppl, 3),
        "total_facts": len(all_facts),
        "memit_edits": post_sleep_edits,
        "recall": round(recall, 3),
        "sleep_ok": sleep_ok,
        "sleep_time": round(sleep_time, 1),
    })

    elapsed = time.time() - start_time

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(f"  PERPLEXITY TRAJECTORY SUMMARY")
    print(f"{'=' * 70}")

    print(f"\n  {'Step':>4} {'Event':<20} {'PPL':>8} {'Facts':>6} {'Recall':>8} {'Edits':>6}")
    print(f"  {'-' * 4} {'-' * 20} {'-' * 8} {'-' * 6} {'-' * 8} {'-' * 6}")

    for t in trajectory:
        recall_str = f"{t['recall']:.2f}" if t['recall'] is not None else "---"
        print(f"  {t['step']:>4} {t['event']:<20} {t['perplexity']:>8.2f} "
              f"{t['total_facts']:>6} {recall_str:>8} {t['memit_edits']:>6}")

    baseline_ppl = trajectory[0]["perplexity"]
    final_ppl = trajectory[-1]["perplexity"]
    print(f"\n  Perplexity: {baseline_ppl:.2f} → {final_ppl:.2f} (delta: {final_ppl - baseline_ppl:+.2f})")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save
    results = {
        "model": config.model["path"],
        "trajectory": trajectory,
        "baseline_perplexity": baseline_ppl,
        "final_perplexity": final_ppl,
        "perplexity_delta": round(final_ppl - baseline_ppl, 3),
        "total_elapsed_seconds": elapsed,
    }

    output_path = Path(args.output) if args.output else Path("experiments/results/ablation_perplexity.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to {output_path}")


if __name__ == "__main__":
    main()
