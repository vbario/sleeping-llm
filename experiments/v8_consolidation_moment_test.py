"""V8 Consolidation Moment Experiment — Verify surprise-gated buffered memory.

Six sequential phases testing the consolidation-moment system:
  Phase 1: Buffer Mechanics — facts buffer in RAM, not in MEMIT, until consolidation
  Phase 2: Surprise Gating — high-surprise messages trigger automatic consolidation
  Phase 3: Batch vs Per-Turn — compare recall quality of batch vs legacy injection
  Phase 4: Pre-Sleep Flush — buffer is flushed before sleep begins
  Phase 5: Volatility — buffered facts lost on crash, consolidated facts survive
  Phase 6: Buffer Overflow — overflow triggers forced consolidation

Usage:
    # Full experiment (all 6 phases)
    python experiments/v8_consolidation_moment_test.py --config config.yaml

    # Single phase
    python experiments/v8_consolidation_moment_test.py --config config.yaml --phase buffer_mechanics
    python experiments/v8_consolidation_moment_test.py --config config.yaml --phase surprise_gating
    python experiments/v8_consolidation_moment_test.py --config config.yaml --phase batch_vs_perturn
    python experiments/v8_consolidation_moment_test.py --config config.yaml --phase pre_sleep_flush
    python experiments/v8_consolidation_moment_test.py --config config.yaml --phase volatility
    python experiments/v8_consolidation_moment_test.py --config config.yaml --phase buffer_overflow

    # Quick smoke test (3B, small counts)
    python experiments/v8_consolidation_moment_test.py --config config.yaml --quick
"""

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

# Prevent CUDA fragmentation OOM on multi-phase runs
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.orchestrator import Orchestrator
from src.memory.memit import FactTriple


# ── Reference texts for perplexity measurement ──

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

# Surprise-triggering messages (contain correction/emphasis markers)
SURPRISE_MESSAGES = [
    "Actually, remember that my real name is Viktor",
    "No, I was wrong earlier. I actually live in Berlin now.",
    "This is important — don't forget I'm allergic to shellfish",
]


# ── Helpers ──

def measure_perplexity(backend):
    """Average perplexity over reference texts."""
    ppls = []
    for text in REFERENCE_TEXTS:
        ppl = backend.compute_perplexity(text)
        ppls.append(ppl)
    return sum(ppls) / len(ppls)


def test_recall(backend, facts):
    """Test raw completion recall. Returns (fraction, details_list)."""
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


def load_fact_pool(path="experiments/data/fact_pool_500.json"):
    """Load facts from the JSON pool and convert to FactTriples."""
    pool_path = project_root / path
    with open(pool_path) as f:
        raw = json.load(f)
    return [FactTriple(subject=r["subject"], relation=r["relation"], object=r["object"]) for r in raw]


def clean_artifacts(config):
    """Remove artifacts (conversations + memit data) for clean state."""
    conv_dir = Path(config.paths["conversations"])
    if conv_dir.exists():
        shutil.rmtree(conv_dir)
    conv_dir.mkdir(parents=True, exist_ok=True)

    memit_dir = Path(config.paths.get("memit_data", "data/memit"))
    if memit_dir.exists():
        shutil.rmtree(memit_dir)
    memit_dir.mkdir(parents=True, exist_ok=True)


def cleanup_gpu():
    """Force-free GPU memory between phases."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


def fresh_orchestrator(config, consolidation_enabled=True):
    """Create a fresh orchestrator with auto-triggers disabled.

    Args:
        config: Config object
        consolidation_enabled: If False, override consolidation_moment.enabled
    """
    cleanup_gpu()

    # Override consolidation_moment.enabled before constructing orchestrator
    if not consolidation_enabled:
        config._data["consolidation_moment"]["enabled"] = False

    orch = Orchestrator(config, disable_memit=False)
    orch.chat._sleep_callback = None
    orch.chat._nap_callback = None

    # Restore config for next orchestrator (don't leave it mutated)
    if not consolidation_enabled:
        config._data["consolidation_moment"]["enabled"] = True

    return orch


def destroy_orchestrator(orch):
    """Explicitly destroy orchestrator and free GPU memory."""
    if hasattr(orch, 'backend') and hasattr(orch.backend, 'model'):
        del orch.backend.model
    if hasattr(orch, 'backend') and hasattr(orch.backend, 'tokenizer'):
        del orch.backend.tokenizer
    del orch
    cleanup_gpu()


def trigger_sleep(orch):
    """Execute full sleep (with pre-sleep buffer flush), return result dict."""
    # Flush fact buffer before sleep (mirrors _on_sleep_trigger)
    if orch.fact_buffer and not orch.fact_buffer.is_empty:
        print(f"  Pre-sleep consolidation: {orch.fact_buffer.size} buffered fact(s)")
        orch.fact_buffer.consolidate(reason="pre_sleep")

    orch.sleep_cycle_count += 1
    cycle_id = f"{orch.sleep_cycle_count:04d}"
    result = orch.full_sleep_controller.execute_sleep(
        cycle_id, "full", orch._gather_new_messages,
    )
    # Post-sleep housekeeping
    refreshed = result.get("facts_refreshed", 0)
    pruned = result.get("facts_pruned", 0)
    orch.health_monitor.record_sleep("full", facts_refreshed=refreshed, facts_pruned=pruned)
    if orch.context.recent_messages:
        orch.context.compact()
    orch.chat.reset_turn_count()
    orch.context.reset(keep_summary=True)
    return result


def teach_facts_buffered(orch, facts):
    """Send facts as chat messages. If consolidation is enabled, they'll buffer.
    If disabled, they'll inject per-turn (legacy)."""
    for fact in facts:
        msg = f"{fact.subject} {fact.relation} {fact.object}."
        orch.chat.process_input(msg)


def ppl_delta_pct(baseline, current):
    """PPL increase as fraction of baseline."""
    if baseline == 0:
        return 0
    return (current - baseline) / baseline


# ── Phase 1: Buffer Mechanics ──

def phase_buffer_mechanics(config, fact_pool, quick=False):
    """Verify facts enter the buffer and don't touch model weights until consolidation.

    Adds facts directly to the buffer (bypasses extraction/surprise pipeline)
    to test the buffer→consolidate→MEMIT mechanism in isolation.
    """
    num_facts = 3
    label = f"Phase 1: Buffer Mechanics ({num_facts} facts)"
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")

    t0 = time.time()
    clean_artifacts(config)
    orch = fresh_orchestrator(config, consolidation_enabled=True)

    # Verify consolidation-moment is wired up
    assert orch.fact_buffer is not None, "fact_buffer not initialized — consolidation_moment not enabled?"
    print(f"  Fact buffer initialized (max_size={orch.fact_buffer.max_buffer_size})")

    # Baseline
    baseline_ppl = measure_perplexity(orch.backend)
    edits_before = orch.memit_engine.get_active_edit_count()
    print(f"  Baseline PPL: {baseline_ppl:.2f}, MEMIT edits: {edits_before}")

    # Add facts directly to buffer (bypasses extraction + surprise)
    facts = fact_pool[:num_facts]
    print(f"\n  Adding {num_facts} facts directly to buffer...")
    for fact in facts:
        orch.fact_buffer.add(fact, turn=0, surprise=0.0)
        print(f"    Added: '{fact.subject} {fact.relation} {fact.object}' "
              f"→ buffer={orch.fact_buffer.size}")

    buffer_after_add = orch.fact_buffer.size
    edits_after_add = orch.memit_engine.get_active_edit_count()
    print(f"\n  After adding: buffer={buffer_after_add}, edits={edits_after_add}")

    # Assert: facts should be in the buffer, NOT in MEMIT
    facts_buffered = buffer_after_add == num_facts
    no_premature_edits = edits_after_add == edits_before
    print(f"  Facts buffered (== {num_facts}): {facts_buffered}")
    print(f"  No premature MEMIT edits: {no_premature_edits}")

    # Manual consolidation
    print(f"\n  Triggering manual consolidation...")
    edit = orch.fact_buffer.consolidate(reason="manual")
    buffer_after_consolidation = orch.fact_buffer.size
    edits_after_consolidation = orch.memit_engine.get_active_edit_count()
    print(f"  After consolidation: buffer={buffer_after_consolidation}, "
          f"edits={edits_after_consolidation}")

    buffer_empty = buffer_after_consolidation == 0
    edits_created = edits_after_consolidation > edits_before
    print(f"  Buffer empty: {buffer_empty}")
    print(f"  MEMIT edits created: {edits_created}")

    # Test recall on consolidated facts
    recall, details = test_recall(orch.backend, facts)
    print(f"  Recall on consolidated facts: {recall:.2f} ({len(facts)} facts)")

    # Verdict: buffer mechanics work if facts buffered, then consolidated into MEMIT
    passed = facts_buffered and no_premature_edits and buffer_empty and edits_created
    verdict = "PASS" if passed else "FAIL"

    elapsed = time.time() - t0
    result = {
        "verdict": verdict,
        "verdict_pass": passed,
        "num_facts": num_facts,
        "buffer_after_add": buffer_after_add,
        "edits_after_add": edits_after_add,
        "buffer_after_consolidation": buffer_after_consolidation,
        "edits_after_consolidation": edits_after_consolidation,
        "facts_buffered": facts_buffered,
        "no_premature_edits": no_premature_edits,
        "buffer_empty": buffer_empty,
        "edits_created": edits_created,
        "recall": round(recall, 3),
        "recall_details": details,
        "baseline_ppl": round(baseline_ppl, 3),
        "elapsed_seconds": round(elapsed, 1),
    }

    print(f"\n  VERDICT: {verdict}")
    print(f"  (buffered={buffer_after_add}, edits={edits_after_consolidation}, "
          f"recall={recall:.2f}, {elapsed:.0f}s)")
    destroy_orchestrator(orch)
    return result


# ── Phase 2: Surprise Gating ──

def phase_surprise_gating(config, fact_pool, quick=False):
    """Verify high-surprise messages trigger automatic consolidation.

    The default threshold (0.6) is too low — any new fact gives novelty=1.0,
    which alone produces surprise = 0.5/0.8 = 0.625 > 0.6. We raise the
    threshold to 0.9 so that:
      - Mundane messages (novelty only): 0.625 < 0.9 → buffer
      - Surprise messages (novelty + markers): ~0.96 > 0.9 → consolidate
    """
    label = "Phase 2: Surprise Gating"
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")

    t0 = time.time()
    clean_artifacts(config)
    orch = fresh_orchestrator(config, consolidation_enabled=True)
    assert orch.fact_buffer is not None

    # Raise threshold so novelty alone doesn't trigger consolidation
    original_threshold = orch.surprise_estimator.consolidation_threshold
    orch.surprise_estimator.consolidation_threshold = 0.9
    print(f"  Surprise threshold: {original_threshold} → 0.9 (raised for test)")

    edits_before = orch.memit_engine.get_active_edit_count()
    print(f"  Starting edits: {edits_before}")

    # Send 2 mundane facts (novelty-only surprise ~0.625, below 0.9 threshold)
    mundane = [
        "My favorite color is blue",
        "I have a cat named Luna",
    ]
    print(f"\n  Sending {len(mundane)} mundane messages...")
    for msg in mundane:
        orch.chat.process_input(msg)
        print(f"    Sent: '{msg}' → buffer={orch.fact_buffer.size}, "
              f"edits={orch.memit_engine.get_active_edit_count()}")

    buffer_after_mundane = orch.fact_buffer.size
    edits_after_mundane = orch.memit_engine.get_active_edit_count()
    print(f"\n  After mundane: buffer={buffer_after_mundane}, edits={edits_after_mundane}")

    mundane_buffered = buffer_after_mundane > 0
    mundane_no_edits = edits_after_mundane == edits_before
    print(f"  Mundane facts buffered: {mundane_buffered}")
    print(f"  No premature edits: {mundane_no_edits}")

    # Send surprise message (markers "actually" + "remember" → marker=0.9, pushes above 0.9)
    surprise_msg = SURPRISE_MESSAGES[0]  # "Actually, remember that my real name is Viktor"
    print(f"\n  Sending surprise message: '{surprise_msg}'")
    orch.chat.process_input(surprise_msg)

    buffer_after_surprise = orch.fact_buffer.size
    edits_after_surprise = orch.memit_engine.get_active_edit_count()
    consolidation_count = orch.fact_buffer._consolidation_count
    print(f"  After surprise: buffer={buffer_after_surprise}, "
          f"edits={edits_after_surprise}, consolidations={consolidation_count}")

    surprise_triggered = consolidation_count > 0
    edits_exist = edits_after_surprise > edits_before
    print(f"  Surprise triggered consolidation: {surprise_triggered}")
    print(f"  MEMIT edits created: {edits_exist}")

    # Verdict: mundane must buffer AND surprise must trigger consolidation
    passed = mundane_buffered and mundane_no_edits and surprise_triggered and edits_exist
    verdict = "PASS" if passed else "FAIL"

    elapsed = time.time() - t0
    result = {
        "verdict": verdict,
        "verdict_pass": passed,
        "threshold_used": 0.9,
        "buffer_after_mundane": buffer_after_mundane,
        "edits_after_mundane": edits_after_mundane,
        "buffer_after_surprise": buffer_after_surprise,
        "edits_after_surprise": edits_after_surprise,
        "consolidation_count": consolidation_count,
        "mundane_buffered": mundane_buffered,
        "mundane_no_edits": mundane_no_edits,
        "surprise_triggered": surprise_triggered,
        "edits_exist": edits_exist,
        "elapsed_seconds": round(elapsed, 1),
    }

    print(f"\n  VERDICT: {verdict}")
    print(f"  (mundane_buf={buffer_after_mundane}, surprise_triggered={surprise_triggered}, "
          f"edits={edits_after_surprise}, {elapsed:.0f}s)")
    destroy_orchestrator(orch)
    return result


# ── Phase 3: Batch vs Per-Turn Recall Quality ──

def phase_batch_vs_perturn(config, fact_pool, quick=False):
    """Compare recall quality of batch injection vs legacy per-fact injection.

    Injects directly to MEMIT (bypasses chat extraction) for a clean comparison:
      Run A: inject_facts([batch]) — single batch call (what consolidate() does)
      Run B: inject_fact(f) × N — one call per fact (legacy per-turn behavior)
    """
    num_facts = 3 if quick else 5
    label = f"Phase 3: Batch vs Per-Turn ({num_facts} facts)"
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")

    t0 = time.time()
    facts = fact_pool[:num_facts]

    # ── Run A: Batch injection (single call) ──
    print(f"\n  --- Run A: Batch Injection ---")
    clean_artifacts(config)
    orch_a = fresh_orchestrator(config, consolidation_enabled=True)

    baseline_ppl_a = measure_perplexity(orch_a.backend)
    print(f"  Baseline PPL: {baseline_ppl_a:.2f}")

    print(f"  Injecting {num_facts} facts as single batch...")
    orch_a.memit_engine.inject_facts(facts)
    edits_a = orch_a.memit_engine.get_active_edit_count()
    print(f"  After batch inject: edits={edits_a}")

    recall_a, details_a = test_recall(orch_a.backend, facts)
    ppl_a = measure_perplexity(orch_a.backend)
    ppl_inc_a = ppl_delta_pct(baseline_ppl_a, ppl_a)
    print(f"  Run A: recall={recall_a:.2f}, PPL={ppl_a:.2f} (+{ppl_inc_a*100:.1f}%)")
    destroy_orchestrator(orch_a)

    # ── Run B: Per-fact injection (one call per fact) ──
    print(f"\n  --- Run B: Per-Fact Injection ---")
    clean_artifacts(config)
    orch_b = fresh_orchestrator(config, consolidation_enabled=True)

    baseline_ppl_b = measure_perplexity(orch_b.backend)
    print(f"  Baseline PPL: {baseline_ppl_b:.2f}")

    print(f"  Injecting {num_facts} facts one at a time...")
    for fact in facts:
        orch_b.memit_engine.inject_fact(fact)
    edits_b = orch_b.memit_engine.get_active_edit_count()
    print(f"  After per-fact inject: edits={edits_b}")

    recall_b, details_b = test_recall(orch_b.backend, facts)
    ppl_b = measure_perplexity(orch_b.backend)
    ppl_inc_b = ppl_delta_pct(baseline_ppl_b, ppl_b)
    print(f"  Run B: recall={recall_b:.2f}, PPL={ppl_b:.2f} (+{ppl_inc_b*100:.1f}%)")
    destroy_orchestrator(orch_b)

    # ── Compare ──
    batch_at_least_as_good = recall_a >= recall_b
    passed = batch_at_least_as_good
    verdict = "PASS" if passed else "FAIL"

    elapsed = time.time() - t0
    result = {
        "verdict": verdict,
        "verdict_pass": passed,
        "num_facts": num_facts,
        "run_a_batch": {
            "recall": round(recall_a, 3),
            "ppl": round(ppl_a, 3),
            "ppl_increase_pct": round(ppl_inc_a * 100, 1),
            "edits": edits_a,
            "details": details_a,
        },
        "run_b_perturn": {
            "recall": round(recall_b, 3),
            "ppl": round(ppl_b, 3),
            "ppl_increase_pct": round(ppl_inc_b * 100, 1),
            "edits": edits_b,
            "details": details_b,
        },
        "batch_at_least_as_good": batch_at_least_as_good,
        "elapsed_seconds": round(elapsed, 1),
    }

    print(f"\n  VERDICT: {verdict}")
    print(f"  (batch={recall_a:.2f} vs perturn={recall_b:.2f}, "
          f"PPL batch +{ppl_inc_a*100:.1f}% vs perturn +{ppl_inc_b*100:.1f}%, {elapsed:.0f}s)")
    return result


# ── Phase 4: Pre-Sleep Flush ──

def phase_pre_sleep_flush(config, fact_pool, quick=False):
    """Verify the buffer is flushed before sleep begins.

    Adds facts directly to the buffer to guarantee they're present,
    then triggers sleep and verifies the pre-sleep flush consolidated them.
    """
    num_facts = 3
    label = f"Phase 4: Pre-Sleep Flush ({num_facts} facts)"
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")

    t0 = time.time()
    clean_artifacts(config)
    orch = fresh_orchestrator(config, consolidation_enabled=True)
    assert orch.fact_buffer is not None

    facts = fact_pool[:num_facts]

    # Disable surprise auto-consolidation so we control when flush happens
    orch.surprise_estimator.consolidation_threshold = 10.0  # effectively disabled

    # Teach via chat so sleep has session data to curate
    print(f"  Teaching facts via chat for sleep session data...")
    teach_facts_buffered(orch, facts)

    # Extractor may or may not have extracted — add facts directly to ensure buffer is populated
    print(f"  Adding {num_facts} facts directly to buffer...")
    for fact in facts:
        orch.fact_buffer.add(fact, turn=0, surprise=0.0)

    buffer_before_sleep = orch.fact_buffer.size
    edits_before_sleep = orch.memit_engine.get_active_edit_count()
    print(f"  Before sleep: buffer={buffer_before_sleep}, edits={edits_before_sleep}")

    buffered_before = buffer_before_sleep > 0
    print(f"  Facts buffered: {buffered_before}")

    # Trigger full sleep (includes pre-sleep flush)
    print(f"\n  Triggering full sleep...")
    try:
        sleep_result = trigger_sleep(orch)
        sleep_ok = True
        print(f"  Sleep completed: refreshed={sleep_result.get('facts_refreshed', 0)}, "
              f"pruned={sleep_result.get('facts_pruned', 0)}")
    except Exception as e:
        print(f"  Sleep failed: {e}")
        import traceback
        traceback.print_exc()
        sleep_result = {"status": "error", "error": str(e)}
        sleep_ok = False

    buffer_after_sleep = orch.fact_buffer.size
    edits_after_sleep = orch.memit_engine.get_active_edit_count()
    print(f"  After sleep: buffer={buffer_after_sleep}, edits={edits_after_sleep}")

    buffer_flushed = buffer_after_sleep == 0
    edits_created = edits_after_sleep > edits_before_sleep
    print(f"  Buffer flushed: {buffer_flushed}")
    print(f"  MEMIT edits created: {edits_created}")

    # Test recall
    recall, details = test_recall(orch.backend, facts)
    print(f"  Recall: {recall:.2f}")

    passed = buffered_before and buffer_flushed and edits_created and sleep_ok
    verdict = "PASS" if passed else "FAIL"

    elapsed = time.time() - t0
    result = {
        "verdict": verdict,
        "verdict_pass": passed,
        "num_facts": num_facts,
        "buffer_before_sleep": buffer_before_sleep,
        "buffer_after_sleep": buffer_after_sleep,
        "edits_before_sleep": edits_before_sleep,
        "edits_after_sleep": edits_after_sleep,
        "buffered_before": buffered_before,
        "buffer_flushed": buffer_flushed,
        "edits_created": edits_created,
        "sleep_ok": sleep_ok,
        "recall": round(recall, 3),
        "recall_details": details,
        "elapsed_seconds": round(elapsed, 1),
    }

    print(f"\n  VERDICT: {verdict}")
    print(f"  (buffer {buffer_before_sleep}→{buffer_after_sleep}, "
          f"edits {edits_before_sleep}→{edits_after_sleep}, "
          f"recall={recall:.2f}, {elapsed:.0f}s)")
    destroy_orchestrator(orch)
    return result


# ── Phase 5: Volatility (Retrograde Amnesia) ──

def phase_volatility(config, fact_pool, quick=False):
    """Verify buffered facts are lost on crash but consolidated facts survive.

    Step 1: Add facts to buffer, consolidate them → in MEMIT
    Step 2: Add more facts to buffer (don't consolidate) → volatile only
    Step 3: Destroy orchestrator (simulate crash)
    Step 4: Recreate → consolidated facts survive, buffered facts are gone
    """
    label = "Phase 5: Volatility (Retrograde Amnesia)"
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")

    t0 = time.time()
    clean_artifacts(config)

    # Step 1: Create orchestrator, buffer facts, consolidate them
    print(f"\n  Step 1: Buffer facts and consolidate into MEMIT...")
    orch = fresh_orchestrator(config, consolidation_enabled=True)
    assert orch.fact_buffer is not None

    consolidated_facts = fact_pool[:2]
    for fact in consolidated_facts:
        orch.fact_buffer.add(fact, turn=0, surprise=0.0)
    print(f"  Buffered {len(consolidated_facts)} facts: buffer={orch.fact_buffer.size}")

    # Consolidate (inject into MEMIT)
    orch.fact_buffer.consolidate(reason="manual")
    edits_consolidated = orch.memit_engine.get_active_edit_count()
    print(f"  After consolidation: buffer={orch.fact_buffer.size}, edits={edits_consolidated}")

    # Verify consolidated facts are in MEMIT
    recall_consolidated_pre, _ = test_recall(orch.backend, consolidated_facts)
    print(f"  Consolidated facts recall: {recall_consolidated_pre:.2f}")

    # Step 2: Add 2 more facts to buffer (do NOT consolidate)
    print(f"\n  Step 2: Buffer 2 more facts (volatile only, no consolidation)...")
    buffered_facts = fact_pool[2:4]
    for fact in buffered_facts:
        orch.fact_buffer.add(fact, turn=1, surprise=0.0)

    buffer_before_crash = orch.fact_buffer.size
    edits_before_crash = orch.memit_engine.get_active_edit_count()
    print(f"  Before crash: buffer={buffer_before_crash}, edits={edits_before_crash}")

    has_buffered = buffer_before_crash > 0
    edits_unchanged = edits_before_crash == edits_consolidated
    print(f"  Has buffered facts: {has_buffered}")
    print(f"  No new MEMIT edits (buffered only): {edits_unchanged}")

    # Step 3: Simulate crash
    print(f"\n  Step 3: Simulating crash (destroying orchestrator)...")
    destroy_orchestrator(orch)

    # Step 4: Create new orchestrator (reloads persisted edits)
    print(f"  Step 4: Creating new orchestrator (reload persisted edits)...")
    orch2 = fresh_orchestrator(config, consolidation_enabled=True)
    assert orch2.fact_buffer is not None

    buffer_after_crash = orch2.fact_buffer.size
    edits_after_crash = orch2.memit_engine.get_active_edit_count()
    print(f"  After restart: buffer={buffer_after_crash}, edits={edits_after_crash}")

    # Assert: buffer is empty (volatile — lost on crash)
    buffer_lost = buffer_after_crash == 0
    # Assert: consolidated edits survive
    edits_survived = edits_after_crash >= edits_consolidated
    print(f"  Buffer lost (volatility): {buffer_lost}")
    print(f"  Consolidated edits survived: {edits_survived}")

    # Test recall: consolidated facts should work
    recall_consolidated_post, details_consolidated = test_recall(orch2.backend, consolidated_facts)
    recall_buffered_post, details_buffered = test_recall(orch2.backend, buffered_facts)
    print(f"  Consolidated facts recall: {recall_consolidated_post:.2f}")
    print(f"  Buffered facts recall: {recall_buffered_post:.2f}")

    consolidated_survived = recall_consolidated_post > 0

    passed = has_buffered and buffer_lost and edits_survived and consolidated_survived
    verdict = "PASS" if passed else "FAIL"

    elapsed = time.time() - t0
    result = {
        "verdict": verdict,
        "verdict_pass": passed,
        "consolidated_facts": len(consolidated_facts),
        "buffered_facts": len(buffered_facts),
        "edits_consolidated": edits_consolidated,
        "buffer_before_crash": buffer_before_crash,
        "edits_before_crash": edits_before_crash,
        "buffer_after_crash": buffer_after_crash,
        "edits_after_crash": edits_after_crash,
        "has_buffered": has_buffered,
        "buffer_lost": buffer_lost,
        "edits_survived": edits_survived,
        "recall_consolidated_pre": round(recall_consolidated_pre, 3),
        "recall_consolidated_post": round(recall_consolidated_post, 3),
        "recall_buffered_post": round(recall_buffered_post, 3),
        "consolidated_survived": consolidated_survived,
        "elapsed_seconds": round(elapsed, 1),
    }

    print(f"\n  VERDICT: {verdict}")
    print(f"  (consolidated recall={recall_consolidated_post:.2f}, "
          f"buffer_lost={buffer_lost}, edits_survived={edits_survived}, {elapsed:.0f}s)")
    destroy_orchestrator(orch2)
    return result


# ── Phase 6: Buffer Overflow ──

def phase_buffer_overflow(config, fact_pool, quick=False):
    """Verify overflow policy triggers forced consolidation."""
    max_buffer = 5
    inject_count = max_buffer + 2  # 7 facts to trigger overflow
    label = f"Phase 6: Buffer Overflow (max={max_buffer}, inject={inject_count})"
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")

    t0 = time.time()
    clean_artifacts(config)

    # Override max_buffer_size for this phase
    original_max = config._data["consolidation_moment"].get("max_buffer_size", 20)
    config._data["consolidation_moment"]["max_buffer_size"] = max_buffer

    orch = fresh_orchestrator(config, consolidation_enabled=True)
    assert orch.fact_buffer is not None

    # Restore config
    config._data["consolidation_moment"]["max_buffer_size"] = original_max

    print(f"  Fact buffer max_size: {orch.fact_buffer.max_buffer_size}")
    assert orch.fact_buffer.max_buffer_size == max_buffer, \
        f"Expected max_buffer_size={max_buffer}, got {orch.fact_buffer.max_buffer_size}"

    edits_before = orch.memit_engine.get_active_edit_count()
    consolidations_before = orch.fact_buffer._consolidation_count
    print(f"  Starting: edits={edits_before}, consolidations={consolidations_before}")

    # Inject facts directly into buffer (bypass chat dedup for determinism)
    facts = fact_pool[:inject_count]
    print(f"\n  Injecting {inject_count} facts directly into buffer...")
    for i, fact in enumerate(facts):
        orch.fact_buffer.add(fact, turn=i, surprise=0.0)
        print(f"    Added fact {i+1}: buffer={orch.fact_buffer.size}, "
              f"consolidations={orch.fact_buffer._consolidation_count}")

    buffer_after = orch.fact_buffer.size
    consolidations_after = orch.fact_buffer._consolidation_count
    edits_after = orch.memit_engine.get_active_edit_count()
    total_consolidated = orch.fact_buffer._total_facts_consolidated
    print(f"\n  After injection: buffer={buffer_after}, edits={edits_after}, "
          f"consolidations={consolidations_after}, total_consolidated={total_consolidated}")

    # Assert: overflow triggered at least one consolidation
    overflow_triggered = consolidations_after > consolidations_before
    # Assert: buffer is not larger than max
    buffer_within_limit = buffer_after <= max_buffer
    print(f"  Overflow triggered consolidation: {overflow_triggered}")
    print(f"  Buffer within limit (<= {max_buffer}): {buffer_within_limit}")

    passed = overflow_triggered and buffer_within_limit
    verdict = "PASS" if passed else "FAIL"

    elapsed = time.time() - t0
    result = {
        "verdict": verdict,
        "verdict_pass": passed,
        "max_buffer_size": max_buffer,
        "inject_count": inject_count,
        "buffer_after": buffer_after,
        "edits_before": edits_before,
        "edits_after": edits_after,
        "consolidations_before": consolidations_before,
        "consolidations_after": consolidations_after,
        "total_facts_consolidated": total_consolidated,
        "overflow_triggered": overflow_triggered,
        "buffer_within_limit": buffer_within_limit,
        "elapsed_seconds": round(elapsed, 1),
    }

    print(f"\n  VERDICT: {verdict}")
    print(f"  (consolidations={consolidations_after}, buffer={buffer_after}/{max_buffer}, "
          f"edits={edits_after}, {elapsed:.0f}s)")
    destroy_orchestrator(orch)
    return result


# ── Main ──

PHASE_MAP = {
    "buffer_mechanics": ("phase_1_buffer_mechanics", phase_buffer_mechanics),
    "surprise_gating": ("phase_2_surprise_gating", phase_surprise_gating),
    "batch_vs_perturn": ("phase_3_batch_vs_perturn", phase_batch_vs_perturn),
    "pre_sleep_flush": ("phase_4_pre_sleep_flush", phase_pre_sleep_flush),
    "volatility": ("phase_5_volatility", phase_volatility),
    "buffer_overflow": ("phase_6_buffer_overflow", phase_buffer_overflow),
}


def main():
    parser = argparse.ArgumentParser(description="V8 Consolidation Moment Experiment")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--phase", type=str, default=None,
                        choices=list(PHASE_MAP.keys()),
                        help="Run a single phase (default: all)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test with small counts")
    parser.add_argument("--output", type=str, default=None,
                        help="Override output JSON path")
    args = parser.parse_args()

    config = Config(args.config)
    model_name = config.model["path"]
    layers = config.get("memit.target_layers", [])
    cm_config = config.get("consolidation_moment", {}) or {}

    print("=" * 70)
    print("  V8 CONSOLIDATION MOMENT EXPERIMENT")
    print("=" * 70)
    print(f"  Model:       {model_name}")
    print(f"  Backend:     {config.model.get('backend', 'mlx')}")
    print(f"  Layers:      {layers} ({len(layers)} layers)")
    print(f"  CM enabled:  {cm_config.get('enabled', False)}")
    print(f"  Buffer max:  {cm_config.get('max_buffer_size', 20)}")
    print(f"  Surprise th: {(cm_config.get('surprise', {}) or {}).get('threshold', 0.6)}")
    print(f"  Quick:       {args.quick}")
    print(f"  Phase:       {args.phase or 'all'}")
    print("=" * 70)

    # Load fact pool
    fact_pool = load_fact_pool()
    print(f"  Loaded {len(fact_pool)} facts from pool")

    total_start = time.time()
    results = {
        "config": {
            "model": model_name,
            "backend": config.model.get("backend", "mlx"),
            "layers": layers,
            "num_layers": len(layers),
            "consolidation_moment": cm_config,
            "quick": args.quick,
        },
    }

    phases_to_run = [args.phase] if args.phase else list(PHASE_MAP.keys())
    phase_verdicts = []

    for phase_name in phases_to_run:
        result_key, phase_fn = PHASE_MAP[phase_name]
        try:
            result = phase_fn(config, fact_pool, quick=args.quick)
            results[result_key] = result
            phase_verdicts.append((phase_name, result.get("verdict_pass", False), result["verdict"]))
        except Exception as e:
            print(f"\n  Phase '{phase_name}' CRASHED: {e}")
            import traceback
            traceback.print_exc()
            results[result_key] = {"verdict": "CRASH", "verdict_pass": False, "error": str(e)}
            phase_verdicts.append((phase_name, False, f"CRASH: {e}"))

    total_elapsed = time.time() - total_start

    # Overall verdict
    passed_count = sum(1 for _, ok, _ in phase_verdicts if ok)
    total_count = len(phase_verdicts)
    results["overall_verdict"] = f"{'PASS' if passed_count == total_count else 'FAIL'} ({passed_count}/{total_count})"
    results["total_elapsed_seconds"] = round(total_elapsed, 1)

    print(f"\n{'=' * 70}")
    print(f"  OVERALL RESULTS")
    print(f"{'=' * 70}")
    for name, ok, verdict in phase_verdicts:
        status = "OK" if ok else "FAIL"
        print(f"  [{status:>4}] {name}: {verdict}")
    print(f"\n  Overall: {results['overall_verdict']}")
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min)")
    print(f"{'=' * 70}")

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        model_short = model_name.split("/")[-1]
        suffix = f"_{args.phase}" if args.phase else ""
        quick_tag = "_quick" if args.quick else ""
        output_path = Path("experiments/results") / f"v8_consolidation_moment_{model_short}{suffix}{quick_tag}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
