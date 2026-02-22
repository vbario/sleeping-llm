"""Multi-fact MEMIT interference test.

Tests that injecting multiple facts simultaneously doesn't cause:
1. Cross-fact interference (e.g., "Helena" leaking into unrelated queries)
2. Loss of individual fact recall
3. Degradation of general model coherence

This is the key test for the covariance regularization fix.

Usage:
    python experiments/test_multi_fact_memit.py
    python experiments/test_multi_fact_memit.py --config experiments/configs/3b_memit.yaml
"""

import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.memory.memit import FactTriple, MemitEngine, EditLedger


# Test facts — chosen to have distinct, non-overlapping content.
# The "Helena" problem was: city name "Helena" leaked into all generations.
TEST_FACTS = [
    FactTriple(subject="Idris Larsson", relation="lives in", object="Helena"),
    FactTriple(subject="Elena Kowalski", relation="works as", object="chef"),
    FactTriple(subject="Marcus Chen", relation="lives in", object="Tucson"),
]

# Interference probes — unrelated questions that should NOT contain any injected content
INTERFERENCE_PROBES = [
    ("What is the capital of France?", ["Helena", "Tucson", "chef", "Idris", "Elena", "Marcus"]),
    ("Tell me about photosynthesis.", ["Helena", "Tucson", "chef", "Idris", "Elena", "Marcus"]),
    ("What is 2 + 2?", ["Helena", "Tucson", "chef", "Idris", "Elena", "Marcus"]),
]


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = Config(config_path)

    print("=" * 70)
    print("  MULTI-FACT MEMIT INTERFERENCE TEST")
    print("=" * 70)
    print(f"  Model: {config.model['path']}")
    print(f"  MEMIT layers: {config.get('memit.target_layers', [])}")
    print(f"  MEMIT lambda: {config.get('memit.lambda_reg', 0.5)}")
    print(f"  Covariance samples: {config.get('memit.covariance_samples', 0)}")
    print("=" * 70)

    # Load backend
    backend_type = config.model.get("backend", "mlx")
    print(f"\n[1] Loading {backend_type} backend...")
    if backend_type == "torch":
        from src.backend.torch_backend import TorchBackend
        backend = TorchBackend(config)
    else:
        from src.backend.mlx_backend import MLXBackend
        backend = MLXBackend(config)
    backend.load()
    print(f"    Model loaded: {backend._model_path}")

    # Initialize MEMIT
    import tempfile, os
    ledger_path = os.path.join(tempfile.mkdtemp(), "test_ledger.json")
    ledger = EditLedger(ledger_path)

    print(f"\n[2] Initializing MEMIT engine...")
    t0 = time.time()
    engine = MemitEngine(config, backend, ledger)
    init_time = time.time() - t0
    print(f"    Init took {init_time:.1f}s")
    print(f"    Covariance available: {bool(engine._cov_diagonal)}")
    if engine._cov_diagonal:
        print(f"    Covariance layers: {list(engine._cov_diagonal.keys())}")

    # --- Phase A: Pre-injection baseline ---
    print(f"\n[3] Pre-injection baseline...")
    baseline_responses = {}

    # Test recall questions (should NOT know the answers yet)
    for fact in TEST_FACTS:
        question = fact.to_question()
        prompt_msgs = [{"role": "user", "content": question}]
        prompt = backend.apply_chat_template(prompt_msgs)
        response = backend.generate(prompt, max_tokens=80, temperature=0.1)
        baseline_responses[fact.subject] = response
        contains_answer = fact.object.lower() in response.lower()
        print(f"    Q: {question}")
        print(f"    A: {response[:120]}")
        print(f"    Contains '{fact.object}': {contains_answer}")
        print()

    # Test interference probes (baseline)
    print(f"  Interference baseline:")
    for probe_q, forbidden in INTERFERENCE_PROBES:
        prompt_msgs = [{"role": "user", "content": probe_q}]
        prompt = backend.apply_chat_template(prompt_msgs)
        response = backend.generate(prompt, max_tokens=80, temperature=0.1)
        leaked = [w for w in forbidden if w.lower() in response.lower()]
        print(f"    Q: {probe_q}")
        print(f"    A: {response[:120]}")
        if leaked:
            print(f"    WARNING: baseline already contains: {leaked}")
        print()

    # --- Phase B: Inject all facts as one batch ---
    print(f"\n[4] Injecting {len(TEST_FACTS)} facts as one batch...")
    t0 = time.time()
    edit = engine.inject_facts(TEST_FACTS)
    inject_time = time.time() - t0

    if edit is None:
        print("    FAILED: inject_facts returned None")
        return

    print(f"    Injection took {inject_time:.1f}s")
    print(f"    Edit ID: {edit.edit_id}")
    print(f"    Layers with deltas: {list(edit.layer_deltas.keys())}")
    for lid, d in edit.layer_deltas.items():
        if backend_type == "mlx":
            import mlx.core as mx
            d_norm = mx.sqrt(mx.sum(d * d)).item()
        else:
            import torch
            d_norm = torch.norm(d).item()
        print(f"      Layer {lid}: |delta|={d_norm:.6f}")

    # --- Phase C: Post-injection recall test ---
    print(f"\n[5] Post-injection recall test...")
    recall_passed = 0
    recall_total = len(TEST_FACTS)

    for fact in TEST_FACTS:
        question = fact.to_question()
        prompt_msgs = [{"role": "user", "content": question}]
        prompt = backend.apply_chat_template(prompt_msgs)
        response = backend.generate(prompt, max_tokens=80, temperature=0.1)
        contains_answer = fact.object.lower() in response.lower()

        status = "PASS" if contains_answer else "FAIL"
        if contains_answer:
            recall_passed += 1

        print(f"    [{status}] Q: {question}")
        print(f"           A: {response[:120]}")
        print(f"           Expected '{fact.object}': {contains_answer}")
        print()

    recall_rate = recall_passed / recall_total
    print(f"  Recall: {recall_passed}/{recall_total} = {recall_rate:.2f}")

    # --- Phase D: Interference test ---
    print(f"\n[6] Post-injection interference test...")
    interference_count = 0
    interference_total = len(INTERFERENCE_PROBES)

    for probe_q, forbidden in INTERFERENCE_PROBES:
        prompt_msgs = [{"role": "user", "content": probe_q}]
        prompt = backend.apply_chat_template(prompt_msgs)
        response = backend.generate(prompt, max_tokens=80, temperature=0.1)
        leaked = [w for w in forbidden if w.lower() in response.lower()]

        status = "CLEAN" if not leaked else "LEAK"
        if leaked:
            interference_count += 1

        print(f"    [{status}] Q: {probe_q}")
        print(f"             A: {response[:120]}")
        if leaked:
            print(f"             LEAKED: {leaked}")
        print()

    interference_rate = interference_count / interference_total if interference_total else 0
    print(f"  Interference: {interference_count}/{interference_total} probes contaminated")

    # --- Phase E: Cross-fact bleed test ---
    print(f"\n[7] Cross-fact bleed test...")
    bleed_count = 0
    bleed_tests = 0

    for fact in TEST_FACTS:
        question = fact.to_question()
        prompt_msgs = [{"role": "user", "content": question}]
        prompt = backend.apply_chat_template(prompt_msgs)
        response = backend.generate(prompt, max_tokens=80, temperature=0.1)

        # Check if OTHER facts' objects appear in this response
        other_objects = [f.object for f in TEST_FACTS if f.subject != fact.subject]
        leaked = [obj for obj in other_objects if obj.lower() in response.lower()]
        bleed_tests += 1
        if leaked:
            bleed_count += 1
            print(f"    [BLEED] Q: {question}")
            print(f"            A: {response[:120]}")
            print(f"            Contains other facts' objects: {leaked}")
        else:
            print(f"    [CLEAN] Q: {question} — no cross-fact bleed")

    bleed_rate = bleed_count / bleed_tests if bleed_tests else 0
    print(f"\n  Cross-fact bleed: {bleed_count}/{bleed_tests}")

    # --- Phase F: Raw prompt completion test ---
    print(f"\n[8] Raw prompt completion test...")
    raw_passed = 0
    for fact in TEST_FACTS:
        raw_prompt = fact.to_prompt()
        response = backend.generate(raw_prompt, max_tokens=20, temperature=0.1)
        contains_answer = fact.object.lower() in response.lower()
        if contains_answer:
            raw_passed += 1
        status = "PASS" if contains_answer else "FAIL"
        print(f"    [{status}] '{raw_prompt}' → '{response[:60]}'")
    raw_rate = raw_passed / recall_total
    print(f"  Raw recall: {raw_passed}/{recall_total} = {raw_rate:.2f}")

    # --- Phase G: Revert and verify ---
    print(f"\n[9] Reverting MEMIT edits...")
    engine.revert_edit(edit)

    print(f"    Checking recall after revert...")
    revert_passed = 0
    for fact in TEST_FACTS:
        question = fact.to_question()
        prompt_msgs = [{"role": "user", "content": question}]
        prompt = backend.apply_chat_template(prompt_msgs)
        response = backend.generate(prompt, max_tokens=80, temperature=0.1)
        still_knows = fact.object.lower() in response.lower()
        if still_knows:
            revert_passed += 1
        status = "STILL KNOWS" if still_knows else "FORGOTTEN"
        print(f"    [{status}] {fact.subject} {fact.relation} {fact.object}")

    if revert_passed == 0:
        print(f"    All facts forgotten after revert — revert works correctly")
    else:
        print(f"    WARNING: {revert_passed} facts still recalled after revert")

    # --- Summary ---
    print(f"\n" + "=" * 70)
    print(f"  SUMMARY")
    print(f"=" * 70)
    print(f"  Raw completion recall: {raw_rate:.2f} ({raw_passed}/{recall_total})")
    print(f"  Chat template recall:  {recall_rate:.2f} ({recall_passed}/{recall_total})")
    print(f"  Interference:          {interference_rate:.2f} ({interference_count}/{interference_total} contaminated)")
    print(f"  Cross-fact bleed:      {bleed_rate:.2f} ({bleed_count}/{bleed_tests})")
    print(f"  Injection time:        {inject_time:.1f}s")
    print(f"  Covariance mode:       {'enabled' if engine._cov_diagonal else 'disabled'}")
    print()
    print(f"  Note: MEMIT edits raw completion pathway (subject + relation → object).")
    print(f"  Chat template recall is expected to be low — during wake, the context")
    print(f"  window provides chat recall. After nap/sleep, LoRA generalizes to all formats.")

    # Primary metrics: raw recall + no interference
    if raw_rate >= 0.66 and interference_count == 0 and bleed_count == 0:
        print(f"\n  RESULT: PASS — multi-fact injection works without interference")
    elif raw_rate >= 0.66 and (interference_count > 0 or bleed_count > 0):
        print(f"\n  RESULT: PARTIAL — facts recalled but interference detected")
    elif interference_count == 0 and bleed_count == 0:
        print(f"\n  RESULT: PARTIAL — no interference but raw recall below threshold")
    else:
        print(f"\n  RESULT: FAIL — interference detected")
    print("=" * 70)


if __name__ == "__main__":
    main()
