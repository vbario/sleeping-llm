"""End-to-end lifecycle test — wake → MEMIT → nap → sleep.

Exercises the full system lifecycle programmatically:
  A. Boot & baseline
  B. Chat with 3 facts (MEMIT injects)
  C. Verify MEMIT recall via raw completion
  D. Trigger nap (LoRA consolidation)
  E. Post-nap chat with 2 more facts
  F. Trigger full sleep
  G. Summary

Usage:
    python experiments/test_lifecycle.py
    python experiments/test_lifecycle.py --config path/to/config.yaml
"""

import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.orchestrator import Orchestrator


# ── Test facts: user messages with extractable personal info ──

PHASE_B_MESSAGES = [
    "My name is Viktor and I live in Portland",
    "I work as a marine biologist",
    "My favorite color is teal",
]

PHASE_E_MESSAGES = [
    "My dog's name is Biscuit",
    "I use Python and Rust",
]

# Expected (subject, relation_fragment, object) for raw completion checks
PHASE_B_EXPECTED = [
    ("Viktor", "lives in", "Portland"),
    ("The user", "works as", "marine biologist"),
    ("The user", "'s favorite color is", "teal"),
]

PHASE_E_EXPECTED = [
    ("The user's dog", "is named", "Biscuit"),
    ("The user", "uses", "Python"),
]


def raw_recall_check(backend, subject, relation, expected_obj):
    """Check raw completion recall (MEMIT pathway)."""
    prompt = f"{subject} {relation}"
    response = backend.generate(prompt, max_tokens=20, temperature=0.1)
    found = expected_obj.lower() in response.lower()
    return found, prompt, response


def print_phase(label):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "config.yaml"
    for arg in sys.argv[1:]:
        if arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]
        elif arg == "--config" and sys.argv.index(arg) + 1 < len(sys.argv):
            config_path = sys.argv[sys.argv.index(arg) + 1]

    config = Config(config_path)

    results = {}  # phase -> (passed, detail)

    # ── Phase A: Boot & Baseline ──
    print_phase("Phase A: Boot & Baseline")

    # Disable auto sleep/nap triggers so we control them manually
    original_sleep_callback = None
    original_nap_callback = None

    t0 = time.time()
    orch = Orchestrator(config, disable_memit=False)
    boot_time = time.time() - t0
    print(f"  Boot time: {boot_time:.1f}s")

    # Prevent auto-triggers during chat phases
    original_sleep_callback = orch.chat._sleep_callback
    original_nap_callback = orch.chat._nap_callback
    orch.chat._sleep_callback = None
    orch.chat._nap_callback = None

    status = orch.get_status()
    print(f"  Model: {status['model']}")
    print(f"  MEMIT enabled: {status['memit_enabled']}")
    print(f"  MEMIT edits: {status['memit_edits']}")
    print(f"  Sleep pressure: {status['sleep_pressure']}")
    print(f"  Trigger mode: {config.sleep.get('trigger_mode', 'turns')}")

    phase_a_ok = status["memit_enabled"] and status["memit_edits"] == 0
    results["A"] = (phase_a_ok, f"MEMIT={status['memit_enabled']}, edits={status['memit_edits']}")
    print(f"  {'PASS' if phase_a_ok else 'FAIL'}: system initialized correctly")

    # ── Phase B: Chat with Facts ──
    print_phase("Phase B: Chat with Facts (3 exchanges)")

    prev_edits = 0
    for i, msg in enumerate(PHASE_B_MESSAGES):
        print(f"\n  [{i+1}/{len(PHASE_B_MESSAGES)}] User: \"{msg}\"")
        t0 = time.time()
        response = orch.chat.process_input(msg)
        elapsed = time.time() - t0
        print(f"  Assistant: {response[:120] if response else '(None)'}...")
        print(f"  ({elapsed:.1f}s)")

        status = orch.get_status()
        new_edits = status["memit_edits"]
        new_facts = status["memit_facts"]
        pressure = status["sleep_pressure"]
        print(f"  Status: edits={new_edits}, facts={new_facts}, pressure={pressure}")

    status = orch.get_status()
    phase_b_ok = status["memit_facts"] > 0
    results["B"] = (phase_b_ok, f"facts={status['memit_facts']}, pressure={status['sleep_pressure']}")
    print(f"\n  {'PASS' if phase_b_ok else 'FAIL'}: MEMIT injected {status['memit_facts']} facts")

    # ── Phase C: Verify MEMIT Recall ──
    print_phase("Phase C: Verify MEMIT Recall (raw completion)")

    recall_passed = 0
    recall_total = len(PHASE_B_EXPECTED)

    for subject, relation, obj in PHASE_B_EXPECTED:
        found, prompt, response = raw_recall_check(orch.backend, subject, relation, obj)
        status_str = "PASS" if found else "FAIL"
        if found:
            recall_passed += 1
        print(f"  [{status_str}] '{prompt}' → '{response[:60].strip()}'  (expected: '{obj}')")

    phase_c_ok = recall_passed > 0  # At least some facts recalled
    results["C"] = (phase_c_ok, f"{recall_passed}/{recall_total} raw recall")
    print(f"\n  {'PASS' if phase_c_ok else 'FAIL'}: {recall_passed}/{recall_total} facts recalled via raw completion")

    # ── Phase D: Trigger Nap ──
    print_phase("Phase D: Trigger Nap")

    pre_nap_edits = orch.get_status()["memit_edits"]
    print(f"  Pre-nap MEMIT edits: {pre_nap_edits}")

    t0 = time.time()
    try:
        orch._on_nap_trigger("test")
        nap_time = time.time() - t0
        nap_ok = True
        nap_detail = f"completed in {nap_time:.1f}s"
    except Exception as e:
        nap_time = time.time() - t0
        nap_ok = False
        nap_detail = f"failed: {e}"

    post_nap_status = orch.get_status()
    post_nap_edits = post_nap_status["memit_edits"]
    print(f"  Nap {nap_detail}")
    print(f"  Post-nap MEMIT edits: {post_nap_edits} (was {pre_nap_edits})")

    # Check if edits were reverted (they should be on successful nap)
    edits_reverted = post_nap_edits < pre_nap_edits
    print(f"  Edits reverted: {edits_reverted}")

    # Test raw recall after nap (facts should be in LoRA now)
    post_nap_recall = 0
    for subject, relation, obj in PHASE_B_EXPECTED:
        found, prompt, response = raw_recall_check(orch.backend, subject, relation, obj)
        if found:
            post_nap_recall += 1
        status_str = "PASS" if found else "FAIL"
        print(f"  [{status_str}] Post-nap: '{prompt}' → '{response[:60].strip()}'")

    phase_d_ok = nap_ok
    results["D"] = (phase_d_ok, f"nap={nap_detail}, post_recall={post_nap_recall}/{recall_total}")
    print(f"\n  {'PASS' if phase_d_ok else 'FAIL'}: {nap_detail}")

    # ── Phase E: Post-Nap Chat ──
    print_phase("Phase E: Post-Nap Chat (2 more facts)")

    for i, msg in enumerate(PHASE_E_MESSAGES):
        print(f"\n  [{i+1}/{len(PHASE_E_MESSAGES)}] User: \"{msg}\"")
        t0 = time.time()
        response = orch.chat.process_input(msg)
        elapsed = time.time() - t0
        print(f"  Assistant: {response[:120] if response else '(None)'}...")
        print(f"  ({elapsed:.1f}s)")

        status = orch.get_status()
        print(f"  Status: edits={status['memit_edits']}, facts={status['memit_facts']}, pressure={status['sleep_pressure']}")

    # Check new facts via raw recall
    new_recall_passed = 0
    for subject, relation, obj in PHASE_E_EXPECTED:
        found, prompt, response = raw_recall_check(orch.backend, subject, relation, obj)
        if found:
            new_recall_passed += 1
        status_str = "PASS" if found else "FAIL"
        print(f"  [{status_str}] '{prompt}' → '{response[:60].strip()}'  (expected: '{obj}')")

    phase_e_ok = orch.get_status()["memit_facts"] > 0
    results["E"] = (phase_e_ok, f"new_facts={orch.get_status()['memit_facts']}, recall={new_recall_passed}/{len(PHASE_E_EXPECTED)}")
    print(f"\n  {'PASS' if phase_e_ok else 'FAIL'}: post-nap MEMIT injection working")

    # ── Phase F: Trigger Full Sleep ──
    print_phase("Phase F: Trigger Full Sleep")

    pre_sleep_edits = orch.get_status()["memit_edits"]
    pre_sleep_pressure = orch.get_status()["sleep_pressure"]
    print(f"  Pre-sleep MEMIT edits: {pre_sleep_edits}")
    print(f"  Pre-sleep pressure: {pre_sleep_pressure}")

    t0 = time.time()
    try:
        orch._on_sleep_trigger("test")
        sleep_time = time.time() - t0
        sleep_ok = True
        sleep_detail = f"completed in {sleep_time:.1f}s"
    except Exception as e:
        sleep_time = time.time() - t0
        sleep_ok = False
        sleep_detail = f"failed: {e}"

    post_sleep_status = orch.get_status()
    print(f"  Sleep {sleep_detail}")
    print(f"  Post-sleep MEMIT edits: {post_sleep_status['memit_edits']}")
    print(f"  Post-sleep pressure: {post_sleep_status['sleep_pressure']}")

    # All facts should now be in LoRA
    all_expected = PHASE_B_EXPECTED + PHASE_E_EXPECTED
    post_sleep_recall = 0
    for subject, relation, obj in all_expected:
        found, prompt, response = raw_recall_check(orch.backend, subject, relation, obj)
        if found:
            post_sleep_recall += 1
        status_str = "PASS" if found else "FAIL"
        print(f"  [{status_str}] Post-sleep: '{prompt}' → '{response[:60].strip()}'")

    phase_f_ok = sleep_ok
    results["F"] = (phase_f_ok, f"sleep={sleep_detail}, post_recall={post_sleep_recall}/{len(all_expected)}")
    print(f"\n  {'PASS' if phase_f_ok else 'FAIL'}: {sleep_detail}")

    # ── Phase G: Summary ──
    print_phase("Phase G: Summary")

    all_passed = all(ok for ok, _ in results.values())

    print(f"  {'Phase':<8} {'Result':<8} {'Detail'}")
    print(f"  {'─' * 8} {'─' * 8} {'─' * 44}")
    for phase, (ok, detail) in sorted(results.items()):
        status_str = "PASS" if ok else "FAIL"
        print(f"  {phase:<8} {status_str:<8} {detail}")

    print()
    if all_passed:
        print(f"  RESULT: PASS — full lifecycle works end-to-end")
    else:
        failed = [p for p, (ok, _) in results.items() if not ok]
        print(f"  RESULT: PARTIAL — phases {', '.join(failed)} had issues")

    print(f"\n  Note: Nap/sleep LoRA training may not fully transfer facts on")
    print(f"  3B model in 1 epoch. MEMIT injection + recall is the primary metric.")
    print(f"  LoRA recall after nap/sleep is a stretch goal for small models.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
