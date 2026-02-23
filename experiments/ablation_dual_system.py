"""Ablation 1: MEMIT-only vs LoRA-only vs MEMIT+LoRA (full dual system).

Three conditions on the same 5 facts:
  1. MEMIT-only  — inject via MEMIT, no sleep. Measure recall immediately & after filler.
  2. LoRA-only   — no MEMIT. Teach via conversation → sleep. Reload model. Measure recall.
  3. MEMIT+LoRA  — inject via MEMIT → nap → full sleep. Reload model. Measure recall.

Outputs JSON with per-condition metrics:
  - immediate_recall, post_restart_recall, perplexity, time_to_first_recall

Usage:
    python experiments/ablation_dual_system.py --config experiments/configs/8b_memit.yaml
    python experiments/ablation_dual_system.py --config experiments/configs/8b_memit.yaml --output results.json
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


# ── Test facts (identical across all 3 conditions) ──

TEST_FACTS = [
    FactTriple("Elena Voronov", "lives in", "Portland"),
    FactTriple("Elena Voronov", "works as", "marine biologist"),
    FactTriple("Marcus Takahashi", "lives in", "Austin"),
    FactTriple("Marcus Takahashi", "works as", "architect"),
    FactTriple("Priya Lindström", "lives in", "Denver"),
]

# Conversation messages that teach the same facts (for LoRA-only condition)
TEACHING_MESSAGES = [
    "Elena Voronov lives in Portland and works as a marine biologist.",
    "Marcus Takahashi is an architect who lives in Austin.",
    "Priya Lindström lives in Denver.",
]

FILLER_MESSAGES = [
    "What's the best way to cook pasta al dente?",
    "Tell me about the history of jazz music.",
    "How do solar panels work?",
    "What are some good hiking trails?",
    "Can you explain how tides are caused by the moon?",
    "What's the difference between a crocodile and an alligator?",
    "Tell me about the Pythagorean theorem.",
    "How does a combustion engine work?",
    "What are the benefits of meditation?",
    "Explain the water cycle briefly.",
    "What is photosynthesis?",
    "Tell me about the Renaissance period.",
    "How do airplanes stay in the air?",
    "What causes earthquakes?",
    "Tell me about the Amazon rainforest.",
    "How does Wi-Fi work?",
    "What are black holes?",
    "Tell me about ancient Egyptian pyramids.",
    "How do vaccines work?",
    "What is the speed of light?",
]

REFERENCE_TEXT = (
    "The theory of general relativity, proposed by Albert Einstein in 1915, "
    "describes gravity as the warping of spacetime by mass and energy. "
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen "
    "using energy from sunlight in the chloroplasts of plant cells. "
    "The French Revolution of 1789 overthrew the monarchy and established the "
    "First Republic, fundamentally transforming French society and politics."
)


def test_recall(backend, facts):
    """Test raw completion recall for a list of FactTriples."""
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


def print_section(label):
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")


def run_condition_memit_only(config_path):
    """Condition 1: MEMIT-only — inject facts, no sleep."""
    print_section("CONDITION 1: MEMIT-only")

    config = Config(config_path)
    clean_artifacts(config)
    orch = Orchestrator(config, disable_memit=False)

    # Disable auto-triggers
    orch.chat._sleep_callback = None
    orch.chat._nap_callback = None

    # Measure baseline perplexity
    baseline_ppl = orch.backend.compute_perplexity(REFERENCE_TEXT)
    print(f"  Baseline perplexity: {baseline_ppl:.2f}")

    # Inject facts via MEMIT
    t0 = time.time()
    edit = orch.memit_engine.inject_facts(TEST_FACTS)
    inject_time = time.time() - t0
    print(f"  Injected {len(TEST_FACTS)} facts in {inject_time:.1f}s")

    # Immediate recall
    recall_imm, results_imm = test_recall(orch.backend, TEST_FACTS)
    print(f"  Immediate recall: {recall_imm:.2f} ({sum(r['passed'] for r in results_imm)}/{len(TEST_FACTS)})")

    # Post-injection perplexity
    post_inject_ppl = orch.backend.compute_perplexity(REFERENCE_TEXT)
    print(f"  Post-injection perplexity: {post_inject_ppl:.2f}")

    # Chat 20 unrelated turns (filler)
    print(f"  Chatting {len(FILLER_MESSAGES)} filler turns...")
    for msg in FILLER_MESSAGES:
        orch.chat.process_input(msg)

    # Post-filler recall
    recall_post, results_post = test_recall(orch.backend, TEST_FACTS)
    print(f"  Post-filler recall: {recall_post:.2f} ({sum(r['passed'] for r in results_post)}/{len(TEST_FACTS)})")

    post_filler_ppl = orch.backend.compute_perplexity(REFERENCE_TEXT)
    print(f"  Post-filler perplexity: {post_filler_ppl:.2f}")

    return {
        "condition": "memit_only",
        "immediate_recall": recall_imm,
        "post_filler_recall": recall_post,
        "post_restart_recall": None,  # No restart for MEMIT-only
        "baseline_perplexity": baseline_ppl,
        "post_inject_perplexity": post_inject_ppl,
        "post_filler_perplexity": post_filler_ppl,
        "time_to_first_recall": inject_time,
        "inject_time": inject_time,
        "details_immediate": results_imm,
        "details_post_filler": results_post,
    }


def run_condition_lora_only(config_path):
    """Condition 2: LoRA-only — teach via conversation, sleep, reload."""
    print_section("CONDITION 2: LoRA-only (no MEMIT)")

    config = Config(config_path)
    clean_artifacts(config)
    orch = Orchestrator(config, disable_memit=True)  # MEMIT disabled

    orch.chat._sleep_callback = None
    orch.chat._nap_callback = None

    # Baseline perplexity
    baseline_ppl = orch.backend.compute_perplexity(REFERENCE_TEXT)
    print(f"  Baseline perplexity: {baseline_ppl:.2f}")

    # Teach facts via conversation
    t0 = time.time()
    for msg in TEACHING_MESSAGES:
        print(f"  Teaching: \"{msg[:60]}...\"")
        orch.chat.process_input(msg)

    # No immediate raw recall (facts are only in context, not weights)
    recall_pre, _ = test_recall(orch.backend, TEST_FACTS)
    print(f"  Pre-sleep raw recall: {recall_pre:.2f} (expected ~0 without MEMIT)")

    # Trigger full sleep
    print(f"  Triggering full sleep...")
    sleep_t0 = time.time()
    orch._on_sleep_trigger("test")
    sleep_time = time.time() - sleep_t0
    total_time = time.time() - t0
    print(f"  Sleep completed in {sleep_time:.1f}s")

    # Post-sleep perplexity (same model in memory)
    post_sleep_ppl = orch.backend.compute_perplexity(REFERENCE_TEXT)

    # Post-sleep recall (in current session — context may help)
    recall_post, results_post = test_recall(orch.backend, TEST_FACTS)
    print(f"  Post-sleep recall: {recall_post:.2f}")

    # Simulate restart: reload model from models/current, clear context
    print(f"  Simulating restart (clearing context, no model reload)...")
    orch.context.reset(keep_summary=False)

    recall_restart, results_restart = test_recall(orch.backend, TEST_FACTS)
    print(f"  Post-restart recall: {recall_restart:.2f}")

    post_restart_ppl = orch.backend.compute_perplexity(REFERENCE_TEXT)
    print(f"  Post-restart perplexity: {post_restart_ppl:.2f}")

    return {
        "condition": "lora_only",
        "immediate_recall": recall_pre,
        "post_sleep_recall": recall_post,
        "post_restart_recall": recall_restart,
        "baseline_perplexity": baseline_ppl,
        "post_sleep_perplexity": post_sleep_ppl,
        "post_restart_perplexity": post_restart_ppl,
        "time_to_first_recall": total_time,
        "sleep_time": sleep_time,
        "details_post_sleep": results_post,
        "details_post_restart": results_restart,
    }


def run_condition_dual_system(config_path):
    """Condition 3: MEMIT+LoRA — inject via MEMIT, nap, then full sleep."""
    print_section("CONDITION 3: MEMIT+LoRA (full dual system)")

    config = Config(config_path)
    clean_artifacts(config)
    orch = Orchestrator(config, disable_memit=False)

    orch.chat._sleep_callback = None
    orch.chat._nap_callback = None

    # Baseline perplexity
    baseline_ppl = orch.backend.compute_perplexity(REFERENCE_TEXT)
    print(f"  Baseline perplexity: {baseline_ppl:.2f}")

    # Inject facts via MEMIT
    t0_inject = time.time()
    edit = orch.memit_engine.inject_facts(TEST_FACTS)
    inject_time = time.time() - t0_inject
    print(f"  Injected {len(TEST_FACTS)} facts in {inject_time:.1f}s")

    # Immediate recall (MEMIT provides this)
    recall_imm, results_imm = test_recall(orch.backend, TEST_FACTS)
    print(f"  Immediate recall (MEMIT): {recall_imm:.2f}")

    post_inject_ppl = orch.backend.compute_perplexity(REFERENCE_TEXT)

    # Trigger nap (consolidate MEMIT → LoRA)
    print(f"  Triggering nap...")
    nap_t0 = time.time()
    orch._on_nap_trigger("test")
    nap_time = time.time() - nap_t0
    print(f"  Nap completed in {nap_time:.1f}s")

    post_nap_edits = orch.get_status()["memit_edits"]
    print(f"  Post-nap MEMIT edits: {post_nap_edits}")

    recall_post_nap, results_nap = test_recall(orch.backend, TEST_FACTS)
    print(f"  Post-nap recall: {recall_post_nap:.2f}")

    post_nap_ppl = orch.backend.compute_perplexity(REFERENCE_TEXT)

    # Chat to generate conversation data for full sleep
    for msg in TEACHING_MESSAGES:
        orch.chat.process_input(msg)

    # Trigger full sleep
    print(f"  Triggering full sleep...")
    sleep_t0 = time.time()
    orch._on_sleep_trigger("test")
    sleep_time = time.time() - sleep_t0
    print(f"  Sleep completed in {sleep_time:.1f}s")

    post_sleep_ppl = orch.backend.compute_perplexity(REFERENCE_TEXT)

    # Post-sleep recall
    recall_post_sleep, results_sleep = test_recall(orch.backend, TEST_FACTS)
    print(f"  Post-sleep recall: {recall_post_sleep:.2f}")

    # Simulate restart
    print(f"  Simulating restart...")
    orch.context.reset(keep_summary=False)

    recall_restart, results_restart = test_recall(orch.backend, TEST_FACTS)
    print(f"  Post-restart recall: {recall_restart:.2f}")

    post_restart_ppl = orch.backend.compute_perplexity(REFERENCE_TEXT)

    return {
        "condition": "memit_plus_lora",
        "immediate_recall": recall_imm,
        "post_nap_recall": recall_post_nap,
        "post_sleep_recall": recall_post_sleep,
        "post_restart_recall": recall_restart,
        "baseline_perplexity": baseline_ppl,
        "post_inject_perplexity": post_inject_ppl,
        "post_nap_perplexity": post_nap_ppl,
        "post_sleep_perplexity": post_sleep_ppl,
        "post_restart_perplexity": post_restart_ppl,
        "time_to_first_recall": inject_time,
        "inject_time": inject_time,
        "nap_time": nap_time,
        "sleep_time": sleep_time,
        "details_immediate": results_imm,
        "details_post_nap": results_nap,
        "details_post_sleep": results_sleep,
        "details_post_restart": results_restart,
    }


def main():
    parser = argparse.ArgumentParser(description="Ablation 1: Dual System Comparison")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    start_time = time.time()

    print("=" * 70)
    print("  ABLATION 1: MEMIT-only vs LoRA-only vs MEMIT+LoRA")
    print("=" * 70)

    all_results = {}

    # Run all three conditions
    all_results["memit_only"] = run_condition_memit_only(args.config)
    all_results["lora_only"] = run_condition_lora_only(args.config)
    all_results["memit_plus_lora"] = run_condition_dual_system(args.config)

    elapsed = time.time() - start_time
    all_results["total_elapsed_seconds"] = elapsed

    # ── Summary table ──
    print_section("COMPARISON SUMMARY")

    def _get(d, key, default="N/A"):
        v = d.get(key)
        return f"{v:.2f}" if isinstance(v, (int, float)) else str(default)

    header = f"  {'Metric':<30} {'MEMIT-only':>12} {'LoRA-only':>12} {'MEMIT+LoRA':>12}"
    print(header)
    print(f"  {'-' * 30} {'-' * 12} {'-' * 12} {'-' * 12}")

    m = all_results["memit_only"]
    l = all_results["lora_only"]
    d = all_results["memit_plus_lora"]

    rows = [
        ("Immediate recall", "immediate_recall", "immediate_recall", "immediate_recall"),
        ("Post-restart recall", "post_filler_recall", "post_restart_recall", "post_restart_recall"),
        ("Baseline perplexity", "baseline_perplexity", "baseline_perplexity", "baseline_perplexity"),
        ("Final perplexity", "post_filler_perplexity", "post_restart_perplexity", "post_restart_perplexity"),
        ("Time to first recall (s)", "time_to_first_recall", "time_to_first_recall", "time_to_first_recall"),
    ]
    for label, mk, lk, dk in rows:
        print(f"  {label:<30} {_get(m, mk):>12} {_get(l, lk):>12} {_get(d, dk):>12}")

    print(f"\n  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save results
    output_path = Path(args.output) if args.output else Path("experiments/results/ablation_dual_system.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Results saved to {output_path}")


if __name__ == "__main__":
    main()
