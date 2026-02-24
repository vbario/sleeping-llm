"""Multi-session longevity experiment — does the system degrade over many cycles?

Systematically exercises the full lifecycle over N cycles, measuring:
  - Perplexity (PPL) before/after each sleep
  - Cumulative recall (ALL facts injected so far)
  - Per-batch recall (oldest batch vs newest batch)
  - Replay buffer statistics
  - Consolidation success rate

Protocol per cycle:
  1. Inject K facts via MEMIT (from pre-generated fact pool)
  2. Chat briefly (generates conversation data for curation)
  3. Trigger sleep
  4. Measure all metrics
  5. Log trajectory point

Usage:
    python experiments/longevity_test.py                              # default: 50 cycles, 10 facts/cycle
    python experiments/longevity_test.py --cycles 3 --facts-per-cycle 3  # smoke test
    python experiments/longevity_test.py --config experiments/configs/longevity_3b.yaml
    python experiments/longevity_test.py --resume results/longevity_20260223_120000.json
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


DEFAULT_CONFIG = "experiments/configs/longevity_3b.yaml"
DEFAULT_FACT_POOL = "experiments/data/fact_pool_500.json"
RESULTS_DIR = Path("results")
CHECKPOINT_INTERVAL = 5  # Save trajectory every N cycles


def load_fact_pool(path: str) -> list:
    """Load pre-generated fact pool from JSON."""
    with open(path) as f:
        return json.load(f)


def inject_facts_batch(orch, facts_batch: list) -> int:
    """Inject a batch of facts via MEMIT and return count injected."""
    from src.memory.memit import FactTriple
    injected = 0
    for fact in facts_batch:
        triple = FactTriple(
            subject=fact["subject"],
            relation=fact["relation"],
            object=fact["object"],
        )
        edit = orch.memit_engine.inject_fact(triple)
        if edit:
            injected += 1
    return injected


def chat_about_facts(orch, facts_batch: list):
    """Chat briefly about injected facts to generate curation data."""
    for fact in facts_batch[:3]:  # Only first 3 to keep it short
        msg = f"Tell me about {fact['subject']} and {fact['object']}"
        try:
            orch.chat.process_input(msg)
        except Exception:
            pass  # Non-critical


def measure_recall(orch, all_facts: list, batch_size: int) -> dict:
    """Measure cumulative and per-batch recall.

    Returns:
        dict with cumulative_recall, oldest_batch_recall, newest_batch_recall,
        and per_batch_recalls list.
    """
    if not all_facts:
        return {"cumulative_recall": 1.0, "oldest_batch_recall": 1.0,
                "newest_batch_recall": 1.0, "per_batch_recalls": []}

    total_recalled = 0
    per_batch_recalls = []

    # Split into batches
    batches = []
    for i in range(0, len(all_facts), batch_size):
        batches.append(all_facts[i:i + batch_size])

    for batch_idx, batch in enumerate(batches):
        batch_recalled = 0
        for fact in batch:
            prompt = f"{fact['subject']} {fact['relation']}"
            response = orch.backend.generate(prompt, max_tokens=20, temperature=0.1)
            if response and fact["object"].lower() in response.lower():
                batch_recalled += 1
                total_recalled += 1
        batch_recall = batch_recalled / len(batch) if batch else 0
        per_batch_recalls.append(round(batch_recall, 3))

    cumulative = total_recalled / len(all_facts) if all_facts else 1.0

    return {
        "cumulative_recall": round(cumulative, 3),
        "oldest_batch_recall": per_batch_recalls[0] if per_batch_recalls else 1.0,
        "newest_batch_recall": per_batch_recalls[-1] if per_batch_recalls else 1.0,
        "per_batch_recalls": per_batch_recalls,
    }


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


def save_trajectory(trajectory: list, output_path: Path):
    """Save trajectory to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(trajectory, f, indent=2)


def clean_artifacts(config):
    """Remove artifacts from previous runs."""
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


def main():
    parser = argparse.ArgumentParser(description="Longevity experiment")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Config YAML path")
    parser.add_argument("--fact-pool", default=DEFAULT_FACT_POOL, help="Fact pool JSON path")
    parser.add_argument("--cycles", type=int, default=50, help="Number of sleep cycles")
    parser.add_argument("--facts-per-cycle", type=int, default=10, help="Facts injected per cycle")
    parser.add_argument("--clean", action="store_true", help="Clean artifacts before starting")
    parser.add_argument("--resume", type=str, default=None, help="Resume from trajectory JSON")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    config = Config(args.config)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = RESULTS_DIR / f"longevity_{timestamp}.json"

    # Load fact pool
    fact_pool = load_fact_pool(args.fact_pool)
    total_facts_needed = args.cycles * args.facts_per_cycle
    if len(fact_pool) < total_facts_needed:
        print(f"WARNING: fact pool has {len(fact_pool)} facts, need {total_facts_needed}. Will wrap around.")

    # Resume or clean start
    trajectory = []
    start_cycle = 0
    all_injected_facts = []

    if args.resume:
        with open(args.resume) as f:
            trajectory = json.load(f)
        start_cycle = len(trajectory)
        # Reconstruct injected facts
        for point in trajectory:
            cycle_idx = point["cycle"]
            batch_start = cycle_idx * args.facts_per_cycle
            batch_end = batch_start + args.facts_per_cycle
            batch = [fact_pool[i % len(fact_pool)] for i in range(batch_start, batch_end)]
            all_injected_facts.extend(batch)
        print(f"Resuming from cycle {start_cycle}, {len(all_injected_facts)} facts already injected")
    elif args.clean:
        print("Cleaning artifacts...")
        clean_artifacts(config)

    # Boot
    print(f"Loading model...")
    orch = Orchestrator(config, disable_memit=False)

    # Disable auto-triggers
    orch.chat._sleep_callback = None
    orch.chat._nap_callback = None

    print(f"\n{'=' * 60}")
    print(f"  Longevity Experiment: {args.cycles} cycles x {args.facts_per_cycle} facts")
    print(f"  Config: {args.config}")
    print(f"  Output: {output_path}")
    print(f"{'=' * 60}\n")

    for cycle in range(start_cycle, args.cycles):
        cycle_start = time.time()
        print(f"\n--- Cycle {cycle + 1}/{args.cycles} ---")

        # 1. Select fact batch
        batch_start = cycle * args.facts_per_cycle
        batch_end = batch_start + args.facts_per_cycle
        facts_batch = [fact_pool[i % len(fact_pool)] for i in range(batch_start, batch_end)]

        # 2. Measure pre-sleep PPL
        ppl_pre = measure_ppl(orch)

        # 3. Inject facts via MEMIT
        injected = inject_facts_batch(orch, facts_batch)
        all_injected_facts.extend(facts_batch[:injected])
        print(f"  Injected {injected}/{len(facts_batch)} facts (total: {len(all_injected_facts)})")

        # 4. Chat briefly for curation data
        chat_about_facts(orch, facts_batch)

        # 5. Trigger sleep
        sleep_start = time.time()
        try:
            orch._on_sleep_trigger("longevity_test")
            sleep_ok = True
        except Exception as e:
            print(f"  Sleep failed: {e}")
            sleep_ok = False
        sleep_time = time.time() - sleep_start

        # 6. Measure post-sleep PPL
        ppl_post = measure_ppl(orch)

        # 7. Measure recall
        recall = measure_recall(orch, all_injected_facts, args.facts_per_cycle)

        # 8. Get system stats
        status = orch.get_status()
        replay_stats = status.get("replay_buffer", {})

        # 9. Log trajectory point
        point = {
            "cycle": cycle,
            "ppl_pre": ppl_pre,
            "ppl_post": ppl_post,
            "cumulative_recall": recall["cumulative_recall"],
            "oldest_batch_recall": recall["oldest_batch_recall"],
            "newest_batch_recall": recall["newest_batch_recall"],
            "per_batch_recalls": recall["per_batch_recalls"],
            "sleep_ok": sleep_ok,
            "replay_buffer_size": replay_stats.get("count", 0),
            "memit_edits": status.get("memit_edits", 0),
            "memit_facts": status.get("memit_facts", 0),
            "facts_injected_this_cycle": injected,
            "total_facts_injected": len(all_injected_facts),
            "wall_time_seconds": round(time.time() - cycle_start, 1),
            "sleep_time_seconds": round(sleep_time, 1),
        }
        trajectory.append(point)

        # Print summary
        ppl_delta = ""
        if ppl_pre > 0 and ppl_post > 0:
            ppl_change = (ppl_post - ppl_pre) / ppl_pre * 100
            ppl_delta = f" ({ppl_change:+.1f}%)"
        print(f"  PPL: {ppl_pre:.2f} → {ppl_post:.2f}{ppl_delta}")
        print(f"  Recall: cumulative={recall['cumulative_recall']:.2f}, "
              f"oldest={recall['oldest_batch_recall']:.2f}, "
              f"newest={recall['newest_batch_recall']:.2f}")

        # Forgetting detection
        if recall["oldest_batch_recall"] < 0.5 and recall["newest_batch_recall"] > 0.7:
            print(f"  WARNING: Possible catastrophic forgetting detected!")

        # Checkpoint
        if (cycle + 1) % CHECKPOINT_INTERVAL == 0 or cycle == args.cycles - 1:
            save_trajectory(trajectory, output_path)
            print(f"  Checkpoint saved: {output_path}")

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"  Longevity Experiment Complete")
    print(f"{'=' * 60}")
    print(f"  Cycles: {len(trajectory)}")
    print(f"  Total facts: {len(all_injected_facts)}")

    if trajectory:
        final = trajectory[-1]
        first = trajectory[0]
        print(f"  PPL trajectory: {first.get('ppl_pre', '?')} → {final.get('ppl_post', '?')}")
        print(f"  Final cumulative recall: {final['cumulative_recall']:.2f}")
        print(f"  Final oldest batch recall: {final['oldest_batch_recall']:.2f}")

    print(f"  Results: {output_path}")


if __name__ == "__main__":
    main()
