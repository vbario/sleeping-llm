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
# extract_template() returns QAPairs: the relation lives in the question,
# the recall target in value.

def _show(qas):
    return [(qa.question, qa.value) for qa in qas]


def test_opinion_thinks():
    ext = make_extractor()
    qas = ext.extract_template("I think Python is better than Java")
    assert any("think" in qa.question and "Python" in qa.question for qa in qas), \
        f"Should extract opinion. Got: {_show(qas)}"


def test_opinion_believes():
    ext = make_extractor()
    qas = ext.extract_template("I believe climate change is real")
    assert any("think" in qa.question and "climate change" in qa.question for qa in qas), \
        f"Should extract belief. Got: {_show(qas)}"


def test_opinion_prefers_over():
    ext = make_extractor()
    qas = ext.extract_template("I prefer cats over dogs")
    assert any("like" in qa.question and "cats" in qa.value for qa in qas), \
        f"Should extract preference. Got: {_show(qas)}"


def test_temporal_graduated():
    ext = make_extractor()
    qas = ext.extract_template("I graduated in 2020")
    assert any("graduate" in qa.question and qa.value == "2020" for qa in qas), \
        f"Should extract graduation year. Got: {_show(qas)}"


def test_temporal_started():
    ext = make_extractor()
    qas = ext.extract_template("I started working in 2018")
    assert any("start" in qa.question and qa.value == "2018" for qa in qas), \
        f"Should extract start year. Got: {_show(qas)}"


def test_temporal_born():
    ext = make_extractor()
    qas = ext.extract_template("I was born in 1995")
    assert any("born" in qa.question and qa.value == "1995" for qa in qas), \
        f"Should extract birth year. Got: {_show(qas)}"


def test_temporal_moved():
    ext = make_extractor()
    qas = ext.extract_template("I moved to Berlin last year")
    assert any("move" in qa.question and "Berlin" in qa.value for qa in qas), \
        f"Should extract move destination. Got: {_show(qas)}"


def test_relationship_sister_lives():
    ext = make_extractor()
    qas = ext.extract_template("My sister lives in Tokyo")
    assert any("sister" in qa.question and "live" in qa.question and "Tokyo" in qa.value for qa in qas), \
        f"Should extract sister's location. Got: {_show(qas)}"


def test_relationship_brother_works():
    ext = make_extractor()
    qas = ext.extract_template("My brother works as a doctor")
    assert any("brother" in qa.question and "work" in qa.question and "doctor" in qa.value for qa in qas), \
        f"Should extract brother's job. Got: {_show(qas)}"


def test_condition_allergic():
    ext = make_extractor()
    qas = ext.extract_template("I'm allergic to shellfish")
    assert any("allergic" in qa.question and "shellfish" in qa.value for qa in qas), \
        f"Should extract allergy. Got: {_show(qas)}"


def test_condition_speaks():
    ext = make_extractor()
    qas = ext.extract_template("I speak French and Spanish")
    assert any("language" in qa.question and "French" in qa.value for qa in qas), \
        f"Should extract languages. Got: {_show(qas)}"


def test_condition_learning():
    ext = make_extractor()
    qas = ext.extract_template("I'm learning Mandarin")
    assert any("learning" in qa.question and "Mandarin" in qa.value for qa in qas), \
        f"Should extract learning. Got: {_show(qas)}"


def test_condition_studied():
    ext = make_extractor()
    qas = ext.extract_template("I studied computer science")
    assert any("study" in qa.question and "computer science" in qa.value for qa in qas), \
        f"Should extract study. Got: {_show(qas)}"


def test_condition_allergic_am():
    """Alternative form: 'I am allergic to'."""
    ext = make_extractor()
    qas = ext.extract_template("I am allergic to peanuts")
    assert any("allergic" in qa.question and "peanuts" in qa.value for qa in qas), \
        f"Should extract allergy. Got: {_show(qas)}"


# ─── Existing patterns still work ───

def test_existing_name():
    ext = make_extractor()
    qas = ext.extract_template("My name is Viktor")
    assert any("name" in qa.question and qa.value == "Viktor" for qa in qas)


def test_existing_location():
    ext = make_extractor()
    qas = ext.extract_template("I live in Portland")
    assert any("live" in qa.question and qa.value == "Portland" for qa in qas)


def test_existing_job():
    ext = make_extractor()
    qas = ext.extract_template("I work as a software engineer")
    assert any("work" in qa.question and "software engineer" in qa.value for qa in qas)


def test_existing_likes():
    ext = make_extractor()
    qas = ext.extract_template("I really like hiking")
    assert any("like" in qa.question and qa.value == "hiking" for qa in qas)


def test_existing_pet():
    ext = make_extractor()
    qas = ext.extract_template("My dog is Biscuit")
    # The pattern requires "my dog is <Name>" with capitalized name
    # This should match since "Biscuit" is capitalized
    assert any("dog" in qa.question and qa.value == "Biscuit" for qa in qas), \
        f"Should extract pet. Got: {_show(qas)}"


# ─── Negative test cases (false positive guards) ───

def test_negative_think_should():
    """'I think I should go' should NOT extract an opinion fact."""
    ext = make_extractor()
    qas = ext.extract_template("I think I should go home now")
    opinions = [qa for qa in qas if "think" in qa.question]
    assert len(opinions) == 0, \
        f"'I think I should' should not produce an opinion fact. Got: {_show(opinions)}"


def test_negative_think_ill():
    """'I think I'll try' should NOT extract an opinion fact."""
    ext = make_extractor()
    qas = ext.extract_template("I think I'll try that later")
    opinions = [qa for qa in qas if "think" in qa.question]
    assert len(opinions) == 0, \
        f"'I think I'll' should not produce an opinion fact. Got: {_show(opinions)}"


def test_negative_short_match():
    """Very short extracted values should be rejected."""
    ext = make_extractor()
    qas = ext.extract_template("I live in X")
    # "X" is only 1 char — should be filtered by len < 2 check
    locations = [qa for qa in qas if "live" in qa.question]
    assert len(locations) == 0, \
        f"Single-char locations should be rejected. Got: {_show(locations)}"


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
