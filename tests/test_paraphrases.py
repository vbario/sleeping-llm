"""Task 2 — held-out paraphrases: generation, split, training-file invariant,
word-boundary recall matching, per-phrasing graduation results. No model."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.facts import FactLedger, QAPair  # noqa: E402
from src.sleep.paraphrase import (ParaphraseGenerator, split_paraphrases,  # noqa: E402
                                  template_paraphrases, to_first_person)
from src.sleep.probe import RecallProbe  # noqa: E402
from src.sleep.recall_match import value_in_response, first_value_in_response  # noqa: E402
from src.sleep.trainer import SleepTrainer, build_training_examples  # noqa: E402
from tests.stubs import StubBackend, make_config, make_oracle  # noqa: E402


# ------------------------------------------------------------ recall_match

@pytest.mark.parametrize("value,response,expected", [
    ("teal", "Your favorite color is teal.", True),
    ("teal", "You like tealight candles.", False),            # substring, not a word
    ("32", "You are 32 years old.", True),
    ("32", "You have 320 books.", False),
    ("Andre", "Andre's school is nearby.", True),             # possessive tolerated
    ("pilot", "You work as a pilot", True),
    ("blue", "I don't know if it's blue.", False),            # hedged guess
    ("blue", "I'm not sure, maybe blue?", False),
    ("blue", "It's blue. I don't know your sister's though.", True),  # hedge after value
    ("Andre Patandre", "Your son is Andre Patandre.", True),
    ("plays saxophone", "You are a saxophone player.", False),  # 2 tokens: both required
    ("a red 2019 Honda Civic", "You drive a Honda Civic from 2019.", True),  # 60% of tokens
    ("", "anything", False),
    ("teal", "", False),
])
def test_value_in_response(value, response, expected):
    assert value_in_response(value, response) is expected


def test_first_value_in_response_prefers_order():
    assert first_value_in_response(["green", "blue"], "It is blue, not green.") == "green"
    assert first_value_in_response(["green"], "It is blue.") is None


# ------------------------------------------------------------- templates

def test_first_person_rewrites():
    assert to_first_person("What is the user's son's name?") == "What's my son's name?"
    assert to_first_person("Where does the user live?") == "Where do I live?"
    assert to_first_person("How old is the user?") == "How old am I?"
    assert to_first_person("What is the user allergic to?") == "What am I allergic to?"
    assert to_first_person("What does the user think about Rust?") == "What do I think about Rust?"


def test_template_paraphrases_exclude_original_and_are_unique():
    q = "What is the user's favorite color?"
    ps = template_paraphrases(q)
    assert q not in ps
    assert len(ps) == len({p.lower() for p in ps})
    assert all(p.endswith("?") for p in ps)
    assert any("my favorite color" in p for p in ps)


def test_split_is_deterministic_and_respects_held_out_count():
    ps = [f"q{i}?" for i in range(6)]
    train1, held1 = split_paraphrases(ps, 2, "seed question")
    train2, held2 = split_paraphrases(ps, 2, "seed question")
    assert (train1, held1) == (train2, held2)
    assert len(held1) == 2 and len(train1) == 4
    assert not set(held1) & set(train1)
    # Too few paraphrases: all held out, train uses only the canonical question
    assert split_paraphrases(["a?", "b?"], 2, "x") == ([], ["a?", "b?"]) or \
        set(split_paraphrases(["a?", "b?"], 2, "x")[1]) == {"a?", "b?"}
    assert split_paraphrases([], 2, "x") == ([], [])


# ------------------------------------------------------------ generation

def _model_backend(lines):
    """StubBackend whose generate() returns the given paraphrase lines."""
    b = StubBackend()
    b.oracle = lambda q, s: "\n".join(lines)
    return b


def test_generate_for_filters_value_leaks_and_splits(tmp_path):
    config = make_config(tmp_path, {"sleep": {"graduation": {
        "paraphrase_count": 8, "held_out_count": 2, "use_model": True}}})
    backend = _model_backend([
        "1. Which color do I like most?",
        "- Is teal my favorite color?",          # leaks the value -> dropped
        "What color am I fond of?",
        "not a question",                        # dropped
    ])
    gen = ParaphraseGenerator(config, backend)
    qa = QAPair(question="What is the user's favorite color?",
                answer="The user's favorite color is teal.", value="teal")
    train, held_out, variants = gen.generate_for(qa)

    everything = train + held_out
    assert "Which color do I like most?" in everything
    assert "What color am I fond of?" in everything
    assert not any("teal" in p.lower() for p in everything)
    assert len(held_out) == 2
    assert not set(train) & set(held_out)
    assert qa.question not in everything
    assert variants[0] == qa.answer
    assert "Your favorite color is teal." in variants


def test_ensure_persists_only_for_facts_missing_held_out(tmp_path):
    config = make_config(tmp_path, {"sleep": {"graduation": {"use_model": False}}})
    gen = ParaphraseGenerator(config, StubBackend())
    ledger = FactLedger(str(tmp_path / "memit" / "ledger.json"))
    a = ledger.add_fact(QAPair(question="Where does the user live?", answer="The user lives in Lisbon.", value="Lisbon"))
    b = ledger.add_fact(QAPair(question="How old is the user?", answer="The user is 32 years old.", value="32"))
    ledger.set_paraphrases(b, ["How old am I?"], ["Do you know my age?"])

    n = gen.ensure(ledger, ledger.get_active_facts())
    assert n == 1
    ea = ledger.get_entry(a)["qa"]
    assert len(ea["held_out"]) == 2 and ea["paraphrases"]
    assert ledger.get_entry(b)["qa"]["held_out"] == ["Do you know my age?"]   # untouched
    # Persisted
    again = FactLedger(str(tmp_path / "memit" / "ledger.json"))
    assert again.get_entry(a)["qa"]["held_out"] == ea["held_out"]
    assert gen.ensure(ledger, ledger.get_active_facts()) == 0


def test_old_ledger_without_paraphrase_fields_loads(tmp_path):
    d = tmp_path / "memit"
    d.mkdir()
    (d / "ledger.json").write_text(json.dumps([{
        "fact_id": "abc12345",
        "qa": {"question": "Where does the user live?", "answer": "The user lives in Lisbon.",
               "value": "Lisbon", "timestamp": 1.0, "priority": 0.5},
        "stage": 1, "graduated": False, "pruned": False,
    }]))
    ledger = FactLedger(str(d / "ledger.json"))
    qa = ledger.get_active_qa_pairs()[0]
    assert qa.paraphrases == [] and qa.held_out == [] and qa.answer_variants == []
    assert ParaphraseGenerator.needs_paraphrases(ledger.get_active_facts()[0])


# --------------------------------------------------------------- trainer

def _qa_with_paraphrases():
    return QAPair(question="What is the user's favorite color?",
                  answer="The user's favorite color is teal.", value="teal",
                  paraphrases=["What's my favorite color?", "Which color do I like most?"],
                  held_out=["Do you remember my favorite color?", "What color am I fond of?"],
                  answer_variants=["The user's favorite color is teal.", "Your favorite color is teal."])


def test_training_file_never_contains_held_out(tmp_path):
    config = make_config(tmp_path)
    trainer = SleepTrainer(config, StubBackend())
    qa = _qa_with_paraphrases()
    plain = QAPair(question="Where does the user live?", answer="The user lives in Lisbon.", value="Lisbon")

    out = trainer.prepare_training_data([qa, plain], tmp_path / "data", weighted=True)
    train = [json.loads(l) for l in (out / "train.jsonl").read_text().splitlines()]
    valid = [json.loads(l) for l in (out / "valid.jsonl").read_text().splitlines()]
    train_qs = {r["messages"][0]["content"] for r in train}
    valid_qs = {r["messages"][0]["content"] for r in valid}

    assert not (train_qs & set(qa.held_out))
    assert valid_qs == set(qa.held_out)
    assert {qa.question, *qa.paraphrases, plain.question} <= train_qs
    # Answer wording varies across the fact's examples
    answers = {r["messages"][1]["content"] for r in train if "color" in r["messages"][0]["content"]}
    assert len(answers) == 2
    # valid.jsonl is no longer a copy of train.jsonl
    assert (out / "valid.jsonl").read_text() != (out / "train.jsonl").read_text()


def test_training_data_raises_if_held_out_leaks(tmp_path):
    trainer = SleepTrainer(make_config(tmp_path), StubBackend())
    qa = _qa_with_paraphrases()
    qa.paraphrases.append(qa.held_out[0])          # corrupt: same phrasing on both sides
    with pytest.raises(ValueError, match="held-out"):
        trainer.prepare_training_data([qa], tmp_path / "data")


def test_validation_falls_back_to_train_when_nothing_is_held_out(tmp_path):
    trainer = SleepTrainer(make_config(tmp_path), StubBackend())
    plain = QAPair(question="Where does the user live?", answer="The user lives in Lisbon.", value="Lisbon")
    out = trainer.prepare_training_data([plain], tmp_path / "data")
    assert (out / "valid.jsonl").read_text() == (out / "train.jsonl").read_text()
    assert build_training_examples(plain) == [{"messages": [
        {"role": "user", "content": plain.question},
        {"role": "assistant", "content": plain.answer}]}]


# ----------------------------------------------------------------- probe

def test_probe_asks_held_out_not_trained_phrasing_and_records_each():
    qa = _qa_with_paraphrases()
    other = QAPair(question="Where does the user live?", answer="The user lives in Lisbon.", value="Lisbon")

    def oracle(question, system):
        assert "teal" not in system                      # fact withheld
        assert "Lisbon" in system                        # other facts present
        return "Teal." if question == qa.held_out[0] else "I'm not sure."

    probe = RecallProbe(StubBackend(oracle=oracle), "sys", min_pass_fraction=1.0)
    res = probe.test(qa, [qa, other])
    asked = [r["question"] for r in res["phrasings"]]
    assert asked == qa.held_out
    assert [r["passed"] for r in res["phrasings"]] == [True, False]
    assert res["fraction"] == 0.5 and res["passed"] is False and res["fallback"] is False

    probe.min_pass_fraction = 0.5
    assert probe.test(qa, [qa, other])["passed"] is True


def test_probe_falls_back_to_canonical_question_without_held_out():
    qa = QAPair(question="Where does the user live?", answer="The user lives in Lisbon.", value="Lisbon")
    probe = RecallProbe(StubBackend(oracle=lambda q, s: "Lisbon"), "sys")
    res = probe.test(qa, [qa])
    assert res["fallback"] is True and res["passed"] is True
    assert [r["question"] for r in res["phrasings"]] == [qa.question]


def test_probe_negative_value_fails_even_when_new_value_present():
    qa = QAPair(question="What is the user's favorite color?", answer="It is green.", value="green",
                held_out=["What's my favorite color?"])
    probe = RecallProbe(StubBackend(oracle=lambda q, s: "Blue, or maybe green."), "sys")
    res = probe.test(qa, [qa], negative_values=["blue"])
    assert res["negative_hit"] == "blue" and res["passed"] is False
    probe2 = RecallProbe(StubBackend(oracle=lambda q, s: "Green."), "sys")
    assert probe2.test(qa, [qa], negative_values=["blue"])["passed"] is True


# ----------------------------------------------------------- full sleep

def test_full_sleep_records_per_phrasing_graduation_results(tmp_path):
    from src.sleep.full_sleep import FullSleepController

    config = make_config(tmp_path, {"sleep": {"graduation": {"use_model": False, "held_out_count": 2}}})
    ledger = FactLedger(str(tmp_path / "memit" / "ledger.json"))
    q = "What is the user's favorite color?"
    ledger.add_fact(QAPair(question=q, answer="The user's favorite color is teal.", value="teal"))

    def oracle(question, system):
        return "Your favorite color is teal." if "teal" not in system and "favorite color" in question else "?"

    backend = StubBackend(oracle=oracle)
    gen = ParaphraseGenerator(config, backend)
    ctrl = FullSleepController(config, backend, ledger, MagicMock(), MagicMock(), MagicMock(),
                               MagicMock(), trainer=SleepTrainer(config, backend), paraphraser=gen)
    stats = ctrl._consolidate("0001", ref_text=None)

    assert stats["paraphrased"] == 1
    gr = stats["graduation_results"]
    assert len(gr) == 1 and gr[0]["passed"] is True and gr[0]["fallback"] is False
    asked = [p["question"] for p in gr[0]["phrasings"]]
    assert len(asked) == 2 and q not in asked
    # Held-out phrasings were not in the training file
    train = (Path(backend.train_calls[0]["data_path"]) / "train.jsonl").read_text()
    for phrasing in asked:
        assert json.dumps(phrasing) not in train
