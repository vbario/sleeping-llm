"""Tests for Item 1: Raw-completion training pairs for LoRA.

Verifies:
  - FactTriple.to_raw_training_text() produces correct format
  - Curator.triples_to_training_pairs() returns dict with chat_pairs + raw_texts
  - Both pathways produce consistent data
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.memit import FactTriple


def test_to_raw_training_text_basic():
    """Raw text should be 'subject relation object' with no template wrapping."""
    triple = FactTriple("Viktor", "lives in", "Portland")
    raw = triple.to_raw_training_text()
    assert raw == "Viktor lives in Portland", f"Got: {raw}"


def test_to_raw_training_text_various_relations():
    """Test raw text across different relation types."""
    cases = [
        (FactTriple("The user", "works as", "marine biologist"), "The user works as marine biologist"),
        (FactTriple("The user", "'s favorite color is", "teal"), "The user 's favorite color is teal"),
        (FactTriple("The user", "is aged", "28"), "The user is aged 28"),
        (FactTriple("The user", "thinks Python is", "better than JavaScript"),
         "The user thinks Python is better than JavaScript"),
    ]
    for triple, expected in cases:
        raw = triple.to_raw_training_text()
        assert raw == expected, f"For {triple.subject}/{triple.relation}/{triple.object}: got '{raw}', expected '{expected}'"


def test_to_raw_vs_to_prompt():
    """Raw training text includes the object; to_prompt() does not."""
    triple = FactTriple("Viktor", "lives in", "Portland")
    prompt = triple.to_prompt()
    raw = triple.to_raw_training_text()
    assert prompt == "Viktor lives in", f"Prompt should not include object, got: {prompt}"
    assert raw == "Viktor lives in Portland", f"Raw should include object, got: {raw}"


def test_triples_to_training_pairs_returns_dict():
    """triples_to_training_pairs() must return a dict, not a list."""
    # Create a mock curator (we only need the method, no backend)
    from src.sleep.curator import Curator

    class MockConfig:
        def __init__(self):
            self.paths = {"training": "/tmp/test_training"}
            self.sleep = {"curation": {"min_novelty_score": 0, "min_importance_score": 0, "min_combined_score": 0}}

    class MockBackend:
        def apply_chat_template(self, messages, for_training=False):
            return str(messages)

    curator = Curator(MockConfig(), MockBackend())
    triples = [
        FactTriple("Viktor", "lives in", "Portland"),
        FactTriple("The user", "works as", "engineer"),
    ]
    result = curator.triples_to_training_pairs(triples)

    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert "chat_pairs" in result, "Missing 'chat_pairs' key"
    assert "raw_texts" in result, "Missing 'raw_texts' key"


def test_triples_to_training_pairs_counts():
    """Same number of chat pairs and raw texts."""
    from src.sleep.curator import Curator

    class MockConfig:
        def __init__(self):
            self.paths = {"training": "/tmp/test_training"}
            self.sleep = {"curation": {"min_novelty_score": 0, "min_importance_score": 0, "min_combined_score": 0}}

    class MockBackend:
        def apply_chat_template(self, messages, for_training=False):
            return str(messages)

    curator = Curator(MockConfig(), MockBackend())
    triples = [
        FactTriple("A", "lives in", "X"),
        FactTriple("B", "works as", "Y"),
        FactTriple("C", "likes", "Z"),
    ]
    result = curator.triples_to_training_pairs(triples)

    assert len(result["chat_pairs"]) == 3, f"Expected 3 chat pairs, got {len(result['chat_pairs'])}"
    assert len(result["raw_texts"]) == 3, f"Expected 3 raw texts, got {len(result['raw_texts'])}"


def test_triples_to_training_pairs_chat_format():
    """Chat pairs should be [user_msg, assistant_msg] dicts."""
    from src.sleep.curator import Curator

    class MockConfig:
        def __init__(self):
            self.paths = {"training": "/tmp/test_training"}
            self.sleep = {"curation": {"min_novelty_score": 0, "min_importance_score": 0, "min_combined_score": 0}}

    class MockBackend:
        def apply_chat_template(self, messages, for_training=False):
            return str(messages)

    curator = Curator(MockConfig(), MockBackend())
    triples = [FactTriple("Viktor", "lives in", "Portland")]
    result = curator.triples_to_training_pairs(triples)

    pair = result["chat_pairs"][0]
    assert len(pair) == 2
    assert pair[0]["role"] == "user"
    assert pair[1]["role"] == "assistant"
    assert "Portland" in pair[1]["content"]


def test_triples_to_training_pairs_raw_content():
    """Raw texts should match to_raw_training_text()."""
    from src.sleep.curator import Curator

    class MockConfig:
        def __init__(self):
            self.paths = {"training": "/tmp/test_training"}
            self.sleep = {"curation": {"min_novelty_score": 0, "min_importance_score": 0, "min_combined_score": 0}}

    class MockBackend:
        def apply_chat_template(self, messages, for_training=False):
            return str(messages)

    curator = Curator(MockConfig(), MockBackend())
    triples = [FactTriple("Viktor", "lives in", "Portland")]
    result = curator.triples_to_training_pairs(triples)

    assert result["raw_texts"][0] == "Viktor lives in Portland"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed + failed} passed")
    sys.exit(1 if failed else 0)
