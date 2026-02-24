"""Tests for Item 4: Real conversational memories — expanded relation types.

Verifies:
  - New regex patterns in extractor.py capture opinions, temporal, relationships, conditions
  - Negative test cases (false positives) are rejected
  - New to_question() mappings produce correct questions
  - curator._extract_facts_template() matches the same patterns
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.memit import FactTriple
from src.wake.extractor import FactExtractor


class MockConfig:
    def __init__(self):
        self.paths = {"training": "/tmp/test_training"}
        self.sleep = {"curation": {"min_novelty_score": 0, "min_importance_score": 0, "min_combined_score": 0}}

    def get(self, key, default=None):
        return default


class MockBackend:
    def apply_chat_template(self, messages, for_training=False):
        return str(messages)

    def generate(self, prompt, max_tokens=200, temperature=0.1):
        return ""


def make_extractor():
    return FactExtractor(MockConfig(), MockBackend())


# ─── Positive test cases ───

def test_opinion_thinks():
    ext = make_extractor()
    triples = ext.extract_template("I think Python is better than Java")
    assert any("thinks" in t.relation and "Python" in t.relation for t in triples), \
        f"Should extract opinion. Got: {[(t.relation, t.object) for t in triples]}"


def test_opinion_believes():
    ext = make_extractor()
    triples = ext.extract_template("I believe climate change is real")
    assert any("thinks" in t.relation or "believes" in t.relation for t in triples), \
        f"Should extract belief. Got: {[(t.relation, t.object) for t in triples]}"


def test_opinion_prefers_over():
    ext = make_extractor()
    triples = ext.extract_template("I prefer cats over dogs")
    assert any("prefers" in t.relation for t in triples), \
        f"Should extract preference. Got: {[(t.relation, t.object) for t in triples]}"


def test_temporal_graduated():
    ext = make_extractor()
    triples = ext.extract_template("I graduated in 2020")
    assert any(t.relation == "graduated in" and t.object == "2020" for t in triples), \
        f"Should extract graduation year. Got: {[(t.relation, t.object) for t in triples]}"


def test_temporal_started():
    ext = make_extractor()
    triples = ext.extract_template("I started working in 2018")
    assert any(t.relation == "started in" and t.object == "2018" for t in triples), \
        f"Should extract start year. Got: {[(t.relation, t.object) for t in triples]}"


def test_temporal_born():
    ext = make_extractor()
    triples = ext.extract_template("I was born in 1995")
    assert any(t.relation == "was born in" for t in triples), \
        f"Should extract birth year. Got: {[(t.relation, t.object) for t in triples]}"


def test_temporal_moved():
    ext = make_extractor()
    triples = ext.extract_template("I moved to Berlin last year")
    assert any(t.relation == "moved to" for t in triples), \
        f"Should extract move destination. Got: {[(t.relation, t.object) for t in triples]}"


def test_relationship_sister_lives():
    ext = make_extractor()
    triples = ext.extract_template("My sister lives in Tokyo")
    assert any("sister" in t.subject and "lives in" in t.relation and "Tokyo" in t.object for t in triples), \
        f"Should extract sister's location. Got: {[(t.subject, t.relation, t.object) for t in triples]}"


def test_relationship_brother_works():
    ext = make_extractor()
    triples = ext.extract_template("My brother works as a doctor")
    assert any("brother" in t.subject and "works as" in t.relation for t in triples), \
        f"Should extract brother's job. Got: {[(t.subject, t.relation, t.object) for t in triples]}"


def test_condition_allergic():
    ext = make_extractor()
    triples = ext.extract_template("I'm allergic to shellfish")
    assert any(t.relation == "is allergic to" and "shellfish" in t.object for t in triples), \
        f"Should extract allergy. Got: {[(t.relation, t.object) for t in triples]}"


def test_condition_speaks():
    ext = make_extractor()
    triples = ext.extract_template("I speak French and Spanish")
    assert any(t.relation == "speaks" for t in triples), \
        f"Should extract languages. Got: {[(t.relation, t.object) for t in triples]}"


def test_condition_learning():
    ext = make_extractor()
    triples = ext.extract_template("I'm learning Mandarin")
    assert any(t.relation == "is learning" and "Mandarin" in t.object for t in triples), \
        f"Should extract learning. Got: {[(t.relation, t.object) for t in triples]}"


def test_condition_studied():
    ext = make_extractor()
    triples = ext.extract_template("I studied computer science")
    assert any(t.relation == "studied" for t in triples), \
        f"Should extract study. Got: {[(t.relation, t.object) for t in triples]}"


def test_condition_allergic_am():
    """Alternative form: 'I am allergic to'."""
    ext = make_extractor()
    triples = ext.extract_template("I am allergic to peanuts")
    assert any(t.relation == "is allergic to" for t in triples), \
        f"Should extract allergy. Got: {[(t.relation, t.object) for t in triples]}"


# ─── Existing patterns still work ───

def test_existing_name():
    ext = make_extractor()
    triples = ext.extract_template("My name is Viktor")
    assert any(t.subject == "Viktor" and t.relation == "is named" for t in triples)


def test_existing_location():
    ext = make_extractor()
    triples = ext.extract_template("I live in Portland")
    assert any(t.relation == "lives in" and t.object == "Portland" for t in triples)


def test_existing_job():
    ext = make_extractor()
    triples = ext.extract_template("I work as a software engineer")
    assert any(t.relation == "works as" for t in triples)


def test_existing_likes():
    ext = make_extractor()
    triples = ext.extract_template("I really like hiking")
    assert any(t.relation == "likes" for t in triples)


def test_existing_pet():
    ext = make_extractor()
    triples = ext.extract_template("My dog is Biscuit")
    # The pattern requires "my dog is <Name>" with capitalized name
    # This should match since "Biscuit" is capitalized
    assert any("dog" in t.subject for t in triples), \
        f"Should extract pet. Got: {[(t.subject, t.relation, t.object) for t in triples]}"


# ─── Negative test cases (false positive guards) ───

def test_negative_think_should():
    """'I think I should go' should NOT extract an opinion fact."""
    ext = make_extractor()
    triples = ext.extract_template("I think I should go home now")
    opinion_triples = [t for t in triples if "thinks" in t.relation]
    assert len(opinion_triples) == 0, \
        f"'I think I should' should not produce opinion triple. Got: {[(t.relation, t.object) for t in opinion_triples]}"


def test_negative_think_ill():
    """'I think I'll try' should NOT extract an opinion fact."""
    ext = make_extractor()
    triples = ext.extract_template("I think I'll try that later")
    opinion_triples = [t for t in triples if "thinks" in t.relation]
    assert len(opinion_triples) == 0, \
        f"'I think I'll' should not produce opinion triple. Got: {[(t.relation, t.object) for t in opinion_triples]}"


def test_negative_short_match():
    """Very short extracted values should be rejected."""
    ext = make_extractor()
    triples = ext.extract_template("I live in X")
    # "X" is only 1 char — should be filtered by len < 2 check
    location_triples = [t for t in triples if t.relation == "lives in"]
    assert len(location_triples) == 0, \
        f"Single-char locations should be rejected. Got: {[(t.relation, t.object) for t in location_triples]}"


# ─── to_question() for new relations ───

def test_to_question_thinks():
    t = FactTriple("The user", "thinks Python is", "great")
    q = t.to_question()
    assert "think" in q.lower(), f"Question should mention 'think'. Got: {q}"


def test_to_question_graduated():
    t = FactTriple("The user", "graduated in", "2020")
    q = t.to_question()
    assert "graduate" in q.lower(), f"Question should mention 'graduate'. Got: {q}"


def test_to_question_allergic():
    t = FactTriple("The user", "is allergic to", "shellfish")
    q = t.to_question()
    assert "allergic" in q.lower(), f"Question should mention 'allergic'. Got: {q}"


def test_to_question_speaks():
    t = FactTriple("The user", "speaks", "French")
    q = t.to_question()
    assert "language" in q.lower() or "speak" in q.lower(), f"Question should ask about language. Got: {q}"


def test_to_question_sister():
    t = FactTriple("The user", "'s sister", "lives in Tokyo")
    q = t.to_question()
    assert "sister" in q.lower(), f"Question should mention 'sister'. Got: {q}"


# ─── has_personal_markers expanded ───

def test_markers_opinion():
    ext = make_extractor()
    assert ext._has_personal_markers("I think this is great")


def test_markers_temporal():
    ext = make_extractor()
    assert ext._has_personal_markers("I graduated from MIT")


def test_markers_relationship():
    ext = make_extractor()
    assert ext._has_personal_markers("My sister is a doctor")


def test_markers_condition():
    ext = make_extractor()
    assert ext._has_personal_markers("I'm learning to code")


def test_markers_negative():
    ext = make_extractor()
    assert not ext._has_personal_markers("The weather is nice today")


# ─── Curator template patterns ───

def test_curator_opinion_extraction():
    """Curator._extract_facts_template should also capture opinions."""
    from src.sleep.curator import Curator

    class MC:
        def __init__(self):
            self.paths = {"training": "/tmp/test_training"}
            self.sleep = {"curation": {"min_novelty_score": 0, "min_importance_score": 0, "min_combined_score": 0},
                          "firewall": {"min_grounding_score": 0, "use_model_verification": False}}
        def get(self, key, default=None):
            return default

    curator = Curator(MC(), MockBackend())
    messages = [{"role": "user", "content": "I think Rust is better than C++"}]
    pairs = curator._extract_facts_template(messages)
    fact_pairs = [p for p in pairs if "think" in p[1]["content"].lower()]
    assert len(fact_pairs) > 0, f"Curator should extract opinion. Got: {[(p[0]['content'], p[1]['content']) for p in pairs]}"


def test_curator_temporal_extraction():
    """Curator._extract_facts_template should capture temporal facts."""
    from src.sleep.curator import Curator

    class MC:
        def __init__(self):
            self.paths = {"training": "/tmp/test_training"}
            self.sleep = {"curation": {"min_novelty_score": 0, "min_importance_score": 0, "min_combined_score": 0},
                          "firewall": {"min_grounding_score": 0, "use_model_verification": False}}
        def get(self, key, default=None):
            return default

    curator = Curator(MC(), MockBackend())
    messages = [{"role": "user", "content": "I graduated in 2020"}]
    pairs = curator._extract_facts_template(messages)
    grad_pairs = [p for p in pairs if "2020" in p[1]["content"]]
    assert len(grad_pairs) > 0, f"Curator should extract graduation. Got: {[(p[0]['content'], p[1]['content']) for p in pairs]}"


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
