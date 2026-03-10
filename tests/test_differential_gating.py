"""Unit test for differential consolidation gating in FullSleepController._consolidate().

Verifies the fix: base model recall is tested BEFORE LoRA training. Only facts where
the fused model passes AND the base model failed are advanced. Facts the base model
already knew are skipped (not advanced, not retreated).

Uses mocks — no GPU or model loading required.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.memory.memit import FactTriple, MemitEdit


# ── Helpers ──

def make_edit(edit_id, facts, scale=1.0, stages=None, recall_rate=1.0):
    """Create a MemitEdit with the given facts."""
    edit = MemitEdit(
        edit_id=edit_id,
        facts=facts,
        layer_deltas={},
        layer_indices=[8, 9, 10],
        key_vectors={},
        scale=scale,
        recall_success_rate=recall_rate,
        fact_stages=stages or [0] * len(facts),
    )
    return edit


def make_controller(edits, base_recall_map, fused_recall_map):
    """Create a FullSleepController with mocked dependencies.

    Args:
        edits: list of MemitEdit to use as active edits
        base_recall_map: dict of (subject, relation) → bool for base model (MEMIT zeroed, pre-LoRA)
        fused_recall_map: dict of (subject, relation) → bool for fused model (MEMIT zeroed, post-LoRA)
    """
    from src.sleep.full_sleep import FullSleepController

    config = MagicMock()
    config.get.side_effect = lambda key, default=None: {
        "sleep.maintenance": {"degraded_threshold": 0.5},
        "consolidation": {"enabled": True, "scale_schedule": [1.0, 0.5, 0.1, 0.0]},
        "lora": {"enabled": True},
    }.get(key, default)
    config.paths = {"core_identity": "/tmp/fake", "fused_models": "/tmp/fake_fused"}

    backend = MagicMock()
    backend.compute_perplexity.return_value = 5.0

    # Track which phase we're in: "base" (pre-training) or "fused" (post-training)
    phase = {"current": "base"}

    memit_engine = MagicMock()
    memit_engine._active_edits = list(edits)
    memit_engine.max_active_edits = 100

    # scale_edit: just update the edit's scale attribute
    def mock_scale_edit(edit, new_scale):
        edit.scale = new_scale
    memit_engine.scale_edit.side_effect = mock_scale_edit

    # test_recall: returns based on phase and the recall maps
    def mock_test_recall(fact, raw=True):
        key = (fact.subject, fact.relation)
        if phase["current"] == "base":
            return base_recall_map.get(key, False), "mocked"
        else:
            return fused_recall_map.get(key, False), "mocked"
    memit_engine.test_recall.side_effect = mock_test_recall

    memit_engine.snapshot_target_weights.return_value = "snapshot"
    memit_engine.reapply_active_edits.return_value = None

    # After trainer.train_and_fuse + backend.reload, switch to "fused" phase
    trainer = MagicMock()
    def mock_train_and_fuse(*args, **kwargs):
        return "/tmp/fake_fused_model"
    trainer.train_and_fuse.side_effect = mock_train_and_fuse

    def mock_reload(path):
        phase["current"] = "fused"
    backend.reload.side_effect = mock_reload

    ledger = MagicMock()
    # advance_fact_stage returns new stage (old + 1)
    def mock_advance(edit_id, fact_idx):
        for e in edits:
            if e.edit_id == edit_id:
                new_stage = e.fact_stages[fact_idx] + 1
                return new_stage
        return 1
    ledger.advance_fact_stage.side_effect = mock_advance
    ledger.retreat_fact_stage.return_value = None

    # Session tracker, health monitor, etc.
    session_tracker = MagicMock()
    health_monitor = MagicMock()
    curator = MagicMock()
    validator = MagicMock()

    controller = FullSleepController(
        config=config,
        backend=backend,
        memit_engine=memit_engine,
        ledger=ledger,
        curator=curator,
        validator=validator,
        session_tracker=session_tracker,
        health_monitor=health_monitor,
        fact_extractor=None,
        trainer=trainer,
    )

    return controller


# ── Tests ──

def test_base_known_facts_skipped():
    """Facts the base model already knows should NOT advance."""
    facts = [
        FactTriple("Vladimir", "is named", "Vladimir"),  # tautological — base knows
        FactTriple("Zephyra", "lives in", "Moonhaven"),  # novel — base doesn't know
    ]
    edit = make_edit("edit-1", facts)

    # Base model knows the tautological fact, not the novel one
    base_recall = {
        ("Vladimir", "is named"): True,
        ("Zephyra", "lives in"): False,
    }
    # Fused model (with LoRA) knows both
    fused_recall = {
        ("Vladimir", "is named"): True,
        ("Zephyra", "lives in"): True,
    }

    controller = make_controller([edit], base_recall, fused_recall)
    stats = controller._consolidate("test-001", "reference text")

    assert stats["advanced"] == 1, f"Expected 1 advanced, got {stats['advanced']}"
    assert stats["already_known"] == 1, f"Expected 1 already_known, got {stats['already_known']}"
    assert stats["retreated"] == 0, f"Expected 0 retreated, got {stats['retreated']}"

    # Zephyra should have advanced to stage 1, Vladimir should stay at 0
    assert edit.fact_stages[0] == 0, f"Tautological fact should stay stage 0, got {edit.fact_stages[0]}"
    assert edit.fact_stages[1] == 1, f"Novel fact should advance to stage 1, got {edit.fact_stages[1]}"

    print("  PASS: base-known facts skipped, novel facts advanced")


def test_fused_fail_retreats():
    """Facts the fused model can't recall should retreat (if at stage > 0)."""
    facts = [
        FactTriple("Xeno", "works at", "Nebula Corp"),
    ]
    edit = make_edit("edit-2", facts, stages=[1])  # already at stage 1

    base_recall = {("Xeno", "works at"): False}
    fused_recall = {("Xeno", "works at"): False}  # fused can't recall either

    controller = make_controller([edit], base_recall, fused_recall)
    stats = controller._consolidate("test-002", "reference text")

    assert stats["retreated"] == 1, f"Expected 1 retreated, got {stats['retreated']}"
    assert stats["advanced"] == 0
    assert stats["already_known"] == 0
    assert edit.fact_stages[0] == 0, f"Should retreat to stage 0, got {edit.fact_stages[0]}"

    print("  PASS: fused-fail retreats to stage 0")


def test_all_base_known():
    """If base model knows ALL facts, nothing should advance."""
    facts = [
        FactTriple("The sun", "is a", "star"),
        FactTriple("Water", "chemical formula is", "H2O"),
    ]
    edit = make_edit("edit-3", facts)

    base_recall = {
        ("The sun", "is a"): True,
        ("Water", "chemical formula is"): True,
    }
    fused_recall = {
        ("The sun", "is a"): True,
        ("Water", "chemical formula is"): True,
    }

    controller = make_controller([edit], base_recall, fused_recall)
    stats = controller._consolidate("test-003", "reference text")

    assert stats["advanced"] == 0, f"Expected 0 advanced, got {stats['advanced']}"
    assert stats["already_known"] == 2, f"Expected 2 already_known, got {stats['already_known']}"
    assert stats["retreated"] == 0
    assert stats["scaled_down"] == 0  # no stages changed, no scale changes

    print("  PASS: all base-known facts skipped, nothing advanced")


def test_mixed_batch():
    """Mixed scenario: some base-known, some LoRA-learned, some failed."""
    facts = [
        FactTriple("Alice", "is named", "Alice"),        # base knows
        FactTriple("Bob", "lives on", "Mars Colony 7"),   # novel, LoRA learned
        FactTriple("Carol", "works at", "Quantum Labs"),  # novel, LoRA failed
    ]
    edit = make_edit("edit-4", facts, stages=[0, 0, 1])  # Carol was previously at stage 1

    base_recall = {
        ("Alice", "is named"): True,
        ("Bob", "lives on"): False,
        ("Carol", "works at"): False,
    }
    fused_recall = {
        ("Alice", "is named"): True,
        ("Bob", "lives on"): True,
        ("Carol", "works at"): False,
    }

    controller = make_controller([edit], base_recall, fused_recall)
    stats = controller._consolidate("test-004", "reference text")

    assert stats["already_known"] == 1, f"Expected 1 already_known, got {stats['already_known']}"
    assert stats["advanced"] == 1, f"Expected 1 advanced, got {stats['advanced']}"
    assert stats["retreated"] == 1, f"Expected 1 retreated, got {stats['retreated']}"

    assert edit.fact_stages[0] == 0, "Alice (base-known) stays at 0"
    assert edit.fact_stages[1] == 1, "Bob (LoRA-learned) advances to 1"
    assert edit.fact_stages[2] == 0, "Carol (failed) retreats to 0"

    print("  PASS: mixed batch handled correctly")


def test_no_eligible_edits():
    """No eligible edits should return skipped=True."""
    controller = make_controller([], {}, {})
    stats = controller._consolidate("test-005", "reference text")

    assert stats["skipped"] is True
    assert stats["advanced"] == 0
    assert stats["already_known"] == 0

    print("  PASS: no eligible edits handled correctly")


# ── Main ──

def main():
    print("=" * 60)
    print("  Differential Gating Unit Tests")
    print("=" * 60)

    tests = [
        test_base_known_facts_skipped,
        test_fused_fail_retreats,
        test_all_base_known,
        test_mixed_batch,
        test_no_eligible_edits,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {test_fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
