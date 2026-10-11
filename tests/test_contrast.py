"""Contrast training rows: 'same question about another person -> I don't know'.

These rows teach the subject binding the bleed gate tests for. The tests pin
the three invariants that keep the gate honest:
  - contrast subjects are disjoint from the gate's neighbor subjects
  - no contrast row ever equals a ledger question, a held-out phrasing, or a
    bleed-gate probe
  - no contrast answer contains the fact's value
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.facts import FactLedger, QAPair  # noqa: E402
from src.sleep.consolidation import ConsolidationRunner  # noqa: E402
from src.sleep.gates import ModelRegistry, SleepGates  # noqa: E402
from src.sleep.paraphrase import (CONTRAST_SUBJECTS, NEIGHBOR_SUBJECTS,  # noqa: E402
                                  ParaphraseGenerator, to_first_person)
from src.sleep.probe import RecallProbe  # noqa: E402
from src.sleep.recall_match import normalize, value_in_response  # noqa: E402
from src.sleep.trainer import SleepTrainer  # noqa: E402
from tests.stubs import StubBackend, make_config  # noqa: E402

COLOR = QAPair("What is the user's favorite color?", "The user's favorite color is teal.", "teal",
               paraphrases=["What's my favorite color?"], held_out=["Which color do I like most?"])
TEAM = QAPair("What is the user's son's favorite sports team?",
              "The user's son's favorite sports team is Celtic.", "Celtic")
JOB = QAPair("What does the user's father do for work?",
             "The user's father works as a veterinarian.", "veterinarian")


def _rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def _questions(rows):
    return [normalize(next(m["content"] for m in r["messages"] if m["role"] == "user")) for r in rows]


def test_subject_lists_disjoint():
    assert not set(NEIGHBOR_SUBJECTS) & set(CONTRAST_SUBJECTS)


def test_contrast_never_matches_a_gate_probe():
    for qa in (COLOR, TEAM, JOB):
        gate = ParaphraseGenerator.neighbor_questions(qa, 2, avoid_questions=[qa.question])
        contrast = ParaphraseGenerator.contrast_examples(qa, 2, avoid_questions=[qa.question])
        assert contrast, qa.question
        gate_n = {normalize(g) for g in gate} | {normalize(to_first_person(g)) for g in gate}
        for q, _ in contrast:
            assert normalize(q) not in gate_n, (q, gate)
            assert normalize(q) != normalize(qa.question)
            # The contrast question is about a contrast subject, not the gate's
            assert any(s in q.lower() for s in CONTRAST_SUBJECTS), q
            assert not any(f" {s}" in q.lower() for s in NEIGHBOR_SUBJECTS), q


def _contains(value, text):
    # Raw check: value_in_response ignores hedged text, and every contrast answer is a hedge
    return value.lower() in text.lower()


def test_contrast_answers_never_contain_the_value():
    for qa in (COLOR, TEAM, JOB):
        for q, a in ParaphraseGenerator.contrast_examples(qa, 3, avoid_questions=[qa.question]):
            assert not _contains(qa.value, a), (q, a)
            assert not _contains(qa.value, q), (q, a)
            assert a.startswith(("I don't know", "You haven't told me")), a
            # A plain denial: it names the asked-about subject and does not
            # enumerate what is known (notes/135: the "you've only told me
            # about ..." clause became a reflex on the graduation probe)
            assert "only told me" not in a and "What you told me" not in a, a
            assert any(f"your {s}" in a for s in CONTRAST_SUBJECTS), a


def test_contrast_swaps_an_existing_relative():
    # A fact already about the son must not stack a second relative on top
    rows = ParaphraseGenerator.contrast_examples(TEAM, 2, avoid_questions=[TEAM.question])
    for q, _ in rows:
        assert "son" not in q.lower(), q
        assert "favorite sports team" in q.lower(), q


def test_contrast_respects_avoid_and_value_guard():
    # A ledger question about the aunt must never be trained as unknown
    aunt_q = "What is the user's aunt's favorite color?"
    rows = ParaphraseGenerator.contrast_examples(COLOR, 2, avoid_questions=[COLOR.question, aunt_q])
    assert all(normalize(q) != normalize(aunt_q) for q, _ in rows)
    assert all(normalize(q) != normalize(to_first_person(aunt_q)) for q, _ in rows)
    # Value that would appear in every answer -> no rows rather than a leak
    weird = QAPair("What is the user's nickname?", "The user's nickname is Nickname.", "nickname")
    assert ParaphraseGenerator.contrast_examples(weird, 2) == []
    # Disabled
    assert ParaphraseGenerator.contrast_examples(COLOR, 0) == []


def test_trainer_writes_contrast_to_train_not_valid(tmp_path):
    cfg = make_config(tmp_path, {"lora": {"contrast_per_fact": 2}})
    trainer = SleepTrainer(cfg, StubBackend())
    out = trainer.prepare_training_data([COLOR, JOB], tmp_path / "data", weighted=True)
    train, valid = _rows(out / "train.jsonl"), _rows(out / "valid.jsonl")
    contrast = [r for r in train if any(s in r["messages"][0]["content"].lower()
                                        for s in CONTRAST_SUBJECTS)]
    assert len(contrast) == 2 * 2 * 2          # 2 facts x 2 subjects x (3rd + 1st person)
    for r in contrast:
        assert not _contains("teal", r["messages"][1]["content"])
        assert not _contains("veterinarian", r["messages"][1]["content"])
    held = {normalize(q) for q in COLOR.held_out}
    assert not held & set(_questions(train))
    assert not any(s in q for q in _questions(valid) for s in CONTRAST_SUBJECTS)


def test_trainer_contrast_disabled(tmp_path):
    cfg = make_config(tmp_path, {"lora": {"contrast_per_fact": 0}})
    trainer = SleepTrainer(cfg, StubBackend())
    out = trainer.prepare_training_data([COLOR], tmp_path / "data")
    assert not any(s in q for q in _questions(_rows(out / "train.jsonl")) for s in CONTRAST_SUBJECTS)


def test_trainer_avoid_questions_block_contrast(tmp_path):
    cfg = make_config(tmp_path, {"lora": {"contrast_per_fact": 1}})
    trainer = SleepTrainer(cfg, StubBackend())
    # Whichever subject the (per-fact, deterministic) draw puts first is blocked
    first_q = ParaphraseGenerator.contrast_examples(COLOR, 1, avoid_questions=[COLOR.question])[0][0]
    out = trainer.prepare_training_data([COLOR], tmp_path / "data", avoid_questions=[first_q])
    qs = _questions(_rows(out / "train.jsonl"))
    assert normalize(first_q) not in qs
    # ...and the row count is unchanged: it fell through to the next subject
    assert sum(1 for q in qs if any(s in q for s in CONTRAST_SUBJECTS)) == 2


def test_runner_contrast_rows_avoid_gate_probes(tmp_path):
    """End to end through ConsolidationRunner: the train.jsonl written for a
    cycle contains contrast rows, and none of them is a question the bleed
    gate asked."""
    cfg = make_config(tmp_path, {"lora": {"contrast_per_fact": 2},
                                 "sleep": {"gates": {"bleed_neighbors": 2}}})
    backend = StubBackend(oracle=lambda q, s: "I don't know.")
    ledger = FactLedger(str(tmp_path / "memit" / "ledger.json"))
    ledger.add_fact(COLOR)
    ledger.add_fact(JOB)
    probe = RecallProbe(backend, "sys", min_pass_fraction=1.0)
    registry = ModelRegistry(backend, "base-model", cfg.paths["fused_models"])
    gates = SleepGates(cfg, backend, probe)
    runner = ConsolidationRunner(cfg, backend, SleepTrainer(cfg, backend), ledger, probe,
                                 paraphraser=None, registry=registry, gates=gates)
    stats = runner.run("c00", ledger.get_active_facts(), test_graduation=False)
    assert stats["status"] == "committed", stats
    train = _rows(Path(cfg.paths["fused_models"]) / "training_data" / "c00" / "train.jsonl")
    qs = set(_questions(train))
    assert any(s in q for q in qs for s in CONTRAST_SUBJECTS)
    gate_qs = {normalize(d["question"]) for g in stats["gates"] for d in g["bleed"].get("details", [])}
    assert gate_qs, "bleed gate recorded no probes"
    assert not gate_qs & qs
    # Every gate probe is about a gate subject, every contrast row about a contrast one
    assert all(any(s in q for s in NEIGHBOR_SUBJECTS) for q in gate_qs)


# ---------------------------------------------------------------- context rows

def _user(row):
    return next(m["content"] for m in row["messages"] if m["role"] == "user")


def test_context_rows_mirror_probe_and_never_leak_own_fact(tmp_path):
    from src.sleep.probe import build_system_prompt
    cfg = make_config(tmp_path, {"lora": {"contrast_per_fact": 1, "context_facts": 2, "context_fraction": 1.0}})
    trainer = SleepTrainer(cfg, StubBackend())
    facts = [COLOR, TEAM, JOB]
    out = trainer.prepare_training_data(facts, tmp_path / "data", context_qa=facts)
    train = _rows(out / "train.jsonl")
    assert train and all(r["messages"][0]["role"] == "system" for r in train)
    base = cfg.context.get("system_prompt", "")
    for r in train:
        system = r["messages"][0]["content"]
        assert system.startswith(base)
        u = _user(r).lower()
        own = COLOR if "color" in u else TEAM if "sports team" in u else JOB
        # The row's own answer/value is never in its context
        assert own.answer not in system and own.value.lower() not in system.lower(), (system, _user(r))
        # Context lines are other facts' answers, formatted exactly as the probe formats them
        if "Things you remember" in system:
            others = [f for f in facts if f.answer in system]
            assert others
            # same lines as the probe would write (order is the sampler's)
            assert sorted(system.splitlines()) == sorted(build_system_prompt(base, others).splitlines()), system
    # Held-out invariant survives the extra system message
    assert not {normalize(q) for q in COLOR.held_out} & {normalize(_user(r)) for r in train}


def test_context_rows_split_by_fraction(tmp_path):
    cfg = make_config(tmp_path, {"lora": {"contrast_per_fact": 0, "context_facts": 2, "context_fraction": 0.5}})
    trainer = SleepTrainer(cfg, StubBackend())
    facts = [COLOR, TEAM, JOB]
    train = _rows(trainer.prepare_training_data(facts, tmp_path / "data", context_qa=facts) / "train.jsonl")
    with_ctx = [r for r in train if "Things you remember" in r["messages"][0]["content"]]
    assert 0 < len(with_ctx) < len(train)          # some rows bare, some with other facts


def test_no_context_without_context_qa(tmp_path):
    cfg = make_config(tmp_path, {})
    trainer = SleepTrainer(cfg, StubBackend())
    train = _rows(trainer.prepare_training_data([COLOR], tmp_path / "data") / "train.jsonl")
    assert all(r["messages"][0]["role"] == "user" for r in train)   # legacy shape for callers that opt out
