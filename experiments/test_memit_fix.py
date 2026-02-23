"""Quick diagnostic test for MEMIT after dimension fix.

Tests:
1. Dequantization of target layers
2. Key vector dimensions match weight dimensions
3. v* optimization produces non-zero delta
4. Weight delta is actually applied (non-zero)
5. Model output changes after injection (the critical test)
6. Recall test with context reset
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.memory.memit import FactTriple, MemitEngine, EditLedger


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = Config(config_path)

    print("=" * 70)
    print("  MEMIT DIMENSION FIX TEST")
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

    print(f"\n[2] Initializing MEMIT engine (dequantize + init)...")
    engine = MemitEngine(config, backend, ledger)
    print(f"    Target layers: {engine.target_layers}")
    print(f"    Target module: {engine.target_module}")
    print(f"    Enabled: {engine.enabled}")

    # Check weight dimensions
    print(f"\n[3] Checking weight dimensions...")
    for layer_idx in engine.target_layers[:2]:
        w = backend.get_layer_mlp_weight(layer_idx, engine.target_module)
        if w is not None:
            print(f"    Layer {layer_idx} {engine.target_module} weight: {w.shape}")

    # Test intermediate activation dimensions
    print(f"\n[4] Testing intermediate activation dimensions...")
    test_text = "Idris Larsson lives in"
    tokens = backend.tokenizer.encode(test_text)
    if backend_type == "mlx":
        import mlx.core as mx
        input_ids = mx.array([tokens])
    else:
        import torch
        input_ids = torch.tensor([tokens], dtype=torch.long).to(backend._device)

    layer_idx = engine.target_layers[-1]
    _, mlp_input, mlp_output = backend.forward_to_layer(input_ids, layer_idx)
    intermediate = backend.compute_mlp_intermediate(mlp_input, layer_idx)
    print(f"    Test text: '{test_text}' → {len(tokens)} tokens")
    print(f"    MLP input shape:        {mlp_input.shape}")
    print(f"    MLP intermediate shape:  {intermediate.shape}")
    print(f"    MLP output shape:        {mlp_output.shape}")

    w = backend.get_layer_mlp_weight(layer_idx, engine.target_module)
    print(f"    Weight shape:            {w.shape}")
    print(f"    Key dim (intermediate[-1]) = {intermediate.shape[-1]} vs Weight in_dim = {w.shape[-1]}")
    assert intermediate.shape[-1] == w.shape[-1], \
        f"DIMENSION MISMATCH: key dim {intermediate.shape[-1]} != weight in_dim {w.shape[-1]}"
    print(f"    ✓ Dimensions match!")

    # Pre-injection baseline
    print(f"\n[5] Pre-injection baseline...")
    question = "Where does Idris Larsson live?"
    prompt_msgs = [{"role": "user", "content": question}]
    prompt = backend.apply_chat_template(prompt_msgs)
    response_before = backend.generate(prompt, max_tokens=50, temperature=0.1)
    print(f"    Q: {question}")
    print(f"    A (before): {response_before[:150]}")

    # Inject fact
    print(f"\n[6] Injecting fact: 'Idris Larsson lives in Helena'...")
    triple = FactTriple(subject="Idris Larsson", relation="lives in", object="Helena")
    edit = engine.inject_facts([triple])

    if edit is None:
        print("    ✗ inject_facts returned None!")
        return

    print(f"    Edit ID: {edit.edit_id}")
    print(f"    Layers with deltas: {list(edit.layer_deltas.keys())}")
    for lid, d in edit.layer_deltas.items():
        if backend_type == "mlx":
            d_norm = mx.sqrt(mx.sum(d * d)).item()
        else:
            d_norm = torch.norm(d).item()
        print(f"      Layer {lid}: delta shape={d.shape}, |delta|={d_norm:.6f}")

    # Post-injection test
    print(f"\n[7] Post-injection test (same question)...")
    response_after = backend.generate(prompt, max_tokens=50, temperature=0.1)
    print(f"    Q: {question}")
    print(f"    A (after):  {response_after[:150]}")

    # Check if "Helena" appears
    helena_in_before = "helena" in response_before.lower()
    helena_in_after = "helena" in response_after.lower()
    print(f"\n    'Helena' in before: {helena_in_before}")
    print(f"    'Helena' in after:  {helena_in_after}")

    if helena_in_after and not helena_in_before:
        print(f"    ✓ MEMIT injection WORKED! Model now recalls the fact.")
    elif helena_in_after and helena_in_before:
        print(f"    ? Model already knew the fact (or guessed). Need different test.")
    else:
        print(f"    ✗ MEMIT injection did NOT change model output for this fact.")

    # Also test with raw prompt (no chat template)
    print(f"\n[8] Raw prompt test...")
    raw_prompt = "Idris Larsson lives in"
    raw_response = backend.generate(raw_prompt, max_tokens=20, temperature=0.1)
    print(f"    '{raw_prompt}' → '{raw_response[:80]}'")
    if "helena" in raw_response.lower():
        print(f"    ✓ Raw completion includes 'Helena'")
    else:
        print(f"    ✗ Raw completion does not include 'Helena'")

    # Test recall method
    print(f"\n[9] Engine recall test...")
    passed, response = engine.test_recall(triple)
    print(f"    Passed: {passed}")
    print(f"    Response: {response[:150]}")

    # Revert and verify
    print(f"\n[10] Reverting MEMIT edit...")
    engine.revert_edit(edit)
    response_reverted = backend.generate(prompt, max_tokens=50, temperature=0.1)
    print(f"    A (reverted): {response_reverted[:150]}")
    helena_reverted = "helena" in response_reverted.lower()
    print(f"    'Helena' in reverted: {helena_reverted}")
    if not helena_reverted and helena_in_after:
        print(f"    ✓ Revert worked — fact is gone again.")
    elif helena_reverted:
        print(f"    ? Fact still present after revert (might be model's prior knowledge).")

    print("\n" + "=" * 70)
    print("  TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
