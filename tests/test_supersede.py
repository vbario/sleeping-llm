"""Task 4 — facts are replaced when they change: slot detection, supersede,
reworded duplicates, graduated updates, buffer and extractor dedup. No model."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.facts import FactLedger, QAPair, GRADUATED_STAGE  # noqa: E402
from src.memory.slots import canonical_slot, same_slot, same_value, slot_similarity  # noqa: E402
from src.wake.context import ContextManager  # noqa: E402
from src.wake.extractor import FactExtractor  # noqa: E402
from src.wake.fact_buffer import FactBuffer  # noqa: E402
from tests.stubs import StubBackend, make_config  # noqa: E402


def qa(q, a, v, priority=0.5):
    return QAPair(question=q, answer=a, value=v, priority=priority)


COLOR = qa("What is the user's favorite color?", "The user's favorite color is blue.", "blue")
COLOR_REWORDED_SAME = qa("What color does the user like?", "The user likes the color blue.", "blue")
COLOR_REWORDED_NEW = qa("Which color does the user prefer?", "The user prefers green.", "green")
SISTER_COLOR = qa("What is the user's sister's favorite color?", "The user's sister's favorite color is red.", "red")
CITY = qa("Where does the user live?", "The user lives in Lisbon.", "Lisbon")


# ------------------------------------------------------------------ slots

@pytest.mark.parametrize("a,b,expected", [
    ("What is the user's favorite color?", "What color does the user like?", True),
    ("What is the user's favorite color?", "What is the user's sister's favorite color?", False),
    ("What is the user's favorite color?", "What is the user's favorite food?", False),
    ("Where does the user live?", "Where is the user based?", True),
    ("What does the user do for work?", "What is the user's job?", True),
    ("What does the user do for work?", "Where does the user work?", False),
    ("How old is the user?", "What is the user's age?", True),
    ("What is the user's son's name?", "What is the user's name?", False),
    ("What is the user's son's name?", "What is the name of the user's son?", True),
])
def test_same_slot(a, b, expected):
    assert same_slot(a, b) is expected
    assert (canonical_slot(a) == canonical_slot(b)) is (slot_similarity(a, b) == 1.0)


@pytest.mark.parametrize("a,b,expected", [
    ("blue", "Blue.", True), ("Lisbon", "Lisbon, Portugal", True),
    ("Andre", "Andre Patandre", True), ("blue", "green", False), ("32", "33", False),
])
def test_same_value(a, b, expected):
    assert same_value(a, b) is expected


# ----------------------------------------------------------------- ledger

def _ledger(tmp_path):
    return FactLedger(str(tmp_path / "memit" / "ledger.json"))


def test_update_supersedes_old_value(tmp_path):
    ledger = _ledger(tmp_path)
    old_id = ledger.add_fact(COLOR)
    res = ledger.upsert(qa("What is the user's favorite color?", "The user's favorite color is green.", "green"))

    assert res["status"] == "superseded" and res["superseded_id"] == old_id
    new = ledger.get_entry(res["fact_id"])
    old = ledger.get_entry(old_id)
    assert old["superseded"] is True and old["superseded_by"] == new["fact_id"]
    assert new["supersedes"] == old_id and new["negative_values"] == ["blue"]
    # Only the new value is live
    assert [e["fact_id"] for e in ledger.get_active_facts()] == [new["fact_id"]]
    assert [q.value for q in ledger.get_active_qa_pairs()] == ["green"]
    assert [e["fact_id"] for e in ledger.get_superseded_facts()] == [old_id]
    assert ledger.get_all_qa_pairs()[0].value == "green"
    # Persisted and reloaded the same way
    again = _ledger(tmp_path)
    assert [q.value for q in again.get_active_qa_pairs()] == ["green"]
    assert len(again.get_superseded_facts()) == 1


def test_reworded_duplicate_is_skipped(tmp_path):
    ledger = _ledger(tmp_path)
    fid = ledger.add_fact(COLOR)
    res = ledger.upsert(COLOR_REWORDED_SAME)
    assert res == {"status": "duplicate", "fact_id": fid}
    assert ledger.get_active_fact_count() == 1
    assert ledger.add_fact(COLOR_REWORDED_SAME) == fid      # add_fact is slot-aware too


def test_reworded_update_supersedes(tmp_path):
    ledger = _ledger(tmp_path)
    old_id = ledger.add_fact(COLOR)
    res = ledger.upsert(COLOR_REWORDED_NEW)
    assert res["status"] == "superseded" and res["superseded_id"] == old_id
    assert [q.value for q in ledger.get_active_qa_pairs()] == ["green"]


def test_different_subject_is_a_different_slot(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.add_fact(COLOR)
    res = ledger.upsert(SISTER_COLOR)
    assert res["status"] == "added"
    assert sorted(q.value for q in ledger.get_active_qa_pairs()) == ["blue", "red"]


def test_update_of_graduated_fact(tmp_path):
    ledger = _ledger(tmp_path)
    old_id = ledger.add_fact(COLOR)
    for _ in range(GRADUATED_STAGE):
        ledger.advance_stage(old_id)
    assert ledger.get_graduated_count() == 1

    res = ledger.upsert(qa("What is the user's favorite color?", "The user's favorite color is green.", "green", priority=0.4))
    assert res["status"] == "superseded"
    new = ledger.get_entry(res["fact_id"])
    # Old graduated value leaves graduated list, replay and audit
    assert ledger.get_graduated_count() == 0
    assert ledger.get_replay_candidates(5) == [] and ledger.get_audit_candidates(5) == []
    assert ledger.get_entry(old_id)["superseded"] is True
    # Correction is high priority, back in the prompt, carries the negative check
    assert new["qa"]["priority"] == 0.9
    assert new["negative_values"] == ["blue"] and new["stage"] == 0
    assert [q.value for q in ledger.get_active_qa_pairs()] == ["green"]


def test_graduation_probe_rejects_old_value_for_correction(tmp_path):
    from src.sleep.probe import RecallProbe
    ledger = _ledger(tmp_path)
    old_id = ledger.add_fact(COLOR)
    for _ in range(GRADUATED_STAGE):
        ledger.advance_stage(old_id)
    res = ledger.upsert(qa("What is the user's favorite color?", "The user's favorite color is green.", "green"))
    entry = ledger.get_entry(res["fact_id"])
    ledger.set_paraphrases(entry["fact_id"], ["What's my favorite color?"], ["Remind me, what's my favorite color?"])
    new_qa = QAPair.from_dict(entry["qa"])

    stale = RecallProbe(StubBackend(oracle=lambda q, s: "Your favorite color is blue."), "sys")
    r = stale.test(new_qa, [new_qa], negative_values=entry["negative_values"])
    assert r["passed"] is False and r["negative_hit"] == "blue"

    fixed = RecallProbe(StubBackend(oracle=lambda q, s: "Your favorite color is green."), "sys")
    assert fixed.test(new_qa, [new_qa], negative_values=entry["negative_values"])["passed"] is True


def test_chained_updates_accumulate_negatives(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.add_fact(COLOR)
    ledger.upsert(qa("What is the user's favorite color?", "x", "green"))
    res = ledger.upsert(qa("What color does the user like?", "x", "teal"))
    entry = ledger.get_entry(res["fact_id"])
    assert entry["negative_values"] == ["green", "blue"]
    assert len(ledger.get_superseded_facts()) == 2
    assert [q.value for q in ledger.get_active_qa_pairs()] == ["teal"]


def test_system_prompt_never_shows_two_values_for_one_slot(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.add_fact(COLOR)
    ledger.add_fact(CITY)
    ledger.upsert(COLOR_REWORDED_NEW)
    ledger.upsert(qa("Where is the user based?", "The user is based in Porto.", "Porto"))

    config = make_config(tmp_path)
    ctx = ContextManager(config, StubBackend())
    ctx.set_facts_provider(ledger.get_active_qa_pairs)
    system = ctx.get_messages()[0]["content"]
    assert "green" in system and "Porto" in system
    assert "blue" not in system and "Lisbon" not in system
    assert system.count("- ") == 2


def test_old_ledger_entries_get_slot_and_supersede_fields(tmp_path):
    import json
    d = tmp_path / "memit"
    d.mkdir()
    (d / "ledger.json").write_text(json.dumps([{
        "fact_id": "abc12345",
        "qa": {"question": "Where does the user live?", "answer": "The user lives in Lisbon.", "value": "Lisbon"},
        "stage": 0, "graduated": False, "pruned": False,
    }]))
    ledger = FactLedger(str(d / "ledger.json"))
    e = ledger.get_active_facts()[0]
    assert e["slot"] == canonical_slot("Where does the user live?")
    assert e["superseded"] is False and e["negative_values"] == []
    assert ledger.upsert(qa("Where is the user based?", "x", "Porto"))["status"] == "superseded"


# -------------------------------------------------------------- extractor

def test_extractor_deduplicate_by_slot(tmp_path):
    ex = FactExtractor(make_config(tmp_path), StubBackend())
    existing = [COLOR, CITY]
    new = [
        COLOR_REWORDED_SAME,                                   # known, reworded -> drop
        qa("Where is the user based?", "x", "Porto"),          # update -> keep
        SISTER_COLOR,                                          # new slot -> keep
        qa("What is the user's favorite color?", "x", "green"),  # within-batch: superseded by next
        qa("What color does the user like?", "x", "teal"),       # last wins
    ]
    out = ex.deduplicate(new, existing)
    assert [f.value for f in out] == ["Porto", "red", "teal"]


# ----------------------------------------------------------------- buffer

def test_buffer_replaces_same_slot_and_reports_on_consolidate(tmp_path):
    config = make_config(tmp_path)
    ledger = _ledger(tmp_path)
    ledger.add_fact(CITY)
    buf = FactBuffer(config, ledger)

    buf.add(COLOR)
    buf.add(COLOR_REWORDED_SAME)           # same slot, same value -> skip
    assert buf.size == 1
    buf.add(COLOR_REWORDED_NEW)            # same slot, new value -> replaces in buffer
    assert buf.size == 1 and buf.get_qa_pairs()[0].value == "green"
    buf.add(qa("Where does the user live?", "The user lives in Lisbon.", "Lisbon"))  # duplicate of ledger
    buf.add(qa("Where is the user based?", "The user is based in Porto.", "Porto"))  # ledger update

    n = buf.consolidate(reason="manual")
    assert n == 2
    assert buf.last_consolidation_report == {"added": 1, "superseded": 1, "duplicate": 0}
    assert sorted(q.value for q in ledger.get_active_qa_pairs()) == ["Porto", "green"]
    assert len(ledger.get_superseded_facts()) == 1
