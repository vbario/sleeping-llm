"""Task 1 — graduated facts stay alive: replay, audit, retreat, nap dead-branch fix.

Runs without a model (StubBackend)."""

import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.facts import FactLedger, QAPair, GRADUATED_STAGE  # noqa: E402
from src.sleep.nap import NapController  # noqa: E402
from tests.stubs import StubBackend, make_config, make_oracle  # noqa: E402


FACTS = {
    "What is the user's son's name?": ("The user's son is named Andre.", "Andre"),
    "Where does the user live?": ("The user lives in Lisbon.", "Lisbon"),
    "What is the user's favorite color?": ("The user's favorite color is teal.", "teal"),
    "What does the user do for work?": ("The user works as a pilot.", "pilot"),
}


def _ledger(tmp_path):
    return FactLedger(str(tmp_path / "memit" / "ledger.json"))


def _add(ledger, q):
    answer, value = FACTS[q]
    return ledger.add_fact(QAPair(question=q, answer=answer, value=value))


def _graduate(ledger, fact_id):
    for _ in range(GRADUATED_STAGE):
        ledger.advance_stage(fact_id)
    assert ledger.get_entry(fact_id)["graduated"] is True


# ----------------------------------------------------------------- ledger


def test_graduated_fact_leaves_prompt_but_stays_in_ledger(tmp_path):
    ledger = _ledger(tmp_path)
    fid = _add(ledger, "Where does the user live?")
    _graduate(ledger, fid)

    assert ledger.get_active_fact_count() == 0
    assert ledger.get_graduated_count() == 1
    assert ledger.get_active_qa_pairs() == []            # not in prompt
    assert len(ledger.get_all_qa_pairs()) == 1            # still known for dedup
    assert ledger.get_entry(fid) is not None              # still addressable


def test_retreat_graduated_returns_to_prompt_at_stage_two(tmp_path):
    ledger = _ledger(tmp_path)
    fid = _add(ledger, "Where does the user live?")
    _graduate(ledger, fid)

    new_stage = ledger.retreat_stage(fid)

    assert new_stage == GRADUATED_STAGE - 1
    assert ledger.get_graduated_count() == 0
    assert ledger.get_active_fact_count() == 1
    assert [qa.value for qa in ledger.get_active_qa_pairs()] == ["Lisbon"]
    assert ledger.get_entry(fid)["graduated"] is False


def test_retreat_active_drops_one_stage_not_to_zero(tmp_path):
    ledger = _ledger(tmp_path)
    fid = _add(ledger, "Where does the user live?")
    ledger.advance_stage(fid)
    ledger.advance_stage(fid)
    assert ledger.get_entry(fid)["stage"] == 2
    assert ledger.retreat_stage(fid) == 1
    assert ledger.retreat_stage(fid) == 0
    assert ledger.retreat_stage(fid) == 0


def test_replay_candidates_prefer_degraded_then_stalest(tmp_path):
    ledger = _ledger(tmp_path)
    ids = [_add(ledger, q) for q in FACTS]
    for fid in ids:
        _graduate(ledger, fid)

    # ids[1]: verified most recently; ids[2]: degraded once; ids[0], ids[3]: never verified
    ledger.update_verification(ids[1], 1.0)
    ledger.record_degrade(ids[2])

    replay = ledger.get_replay_candidates(2)
    assert [e["fact_id"] for e in replay][0] == ids[2]          # degraded first
    assert ids[1] not in [e["fact_id"] for e in replay]         # freshest left out
    assert len(ledger.get_replay_candidates(100)) == 4          # capped by count
    assert ledger.get_replay_candidates(0) == []
    assert ledger.get_audit_candidates(3) == ledger.get_replay_candidates(3)


def test_record_training_and_degrade_reach_graduated_entries(tmp_path):
    ledger = _ledger(tmp_path)
    fid = _add(ledger, "Where does the user live?")
    _graduate(ledger, fid)
    ledger.record_training(fid)
    ledger.record_degrade(fid)
    e = ledger.get_entry(fid)
    assert e["train_count"] == 1 and e["degrade_count"] == 1


def test_old_ledger_files_load_with_defaults(tmp_path):
    """Real pre-change files from data/memit load, and missing fields get defaults."""
    src = Path(__file__).parent.parent / "data" / "memit"
    dst = tmp_path / "memit"
    dst.mkdir()
    shutil.copy(src / "ledger.json", dst / "ledger.json")
    shutil.copy(src / "graduated.json", dst / "graduated.json")

    # Strip a field from one entry to simulate an even older file
    data = json.loads((dst / "graduated.json").read_text())
    data[0].pop("degrade_count", None)
    data[0].pop("last_verified", None)
    (dst / "graduated.json").write_text(json.dumps(data))

    ledger = FactLedger(str(dst / "ledger.json"))
    assert ledger.get_graduated_count() == 2
    g = ledger.get_graduated_facts()[0]
    assert g["degrade_count"] == 0 and g["last_verified"] == 0.0
    assert ledger.get_replay_candidates(5)                      # sampling works on old data
    # Round-trips without losing entries
    ledger.save()
    again = FactLedger(str(dst / "ledger.json"))
    assert again.get_graduated_count() == 2
    assert again.get_active_fact_count() == ledger.get_active_fact_count()


# -------------------------------------------------------------------- nap


def _nap(tmp_path, ledger, learned):
    config = make_config(tmp_path, {"sleep": {"maintenance": {
        "nap_audit_count": 5, "graduated_audit_count": 5}}})
    backend = StubBackend(oracle=make_oracle(FACTS, learned=learned))
    return NapController(config, backend, ledger), backend


def test_nap_audits_graduated_facts_and_ungraduates_failures(tmp_path):
    ledger = _ledger(tmp_path)
    son = _add(ledger, "What is the user's son's name?")
    city = _add(ledger, "Where does the user live?")
    color = _add(ledger, "What is the user's favorite color?")   # stays active
    _graduate(ledger, son)
    _graduate(ledger, city)

    # Weights "know" the son but have forgotten the city.
    nap, backend = _nap(tmp_path, ledger, learned=["What is the user's son's name?"])
    result = nap.execute_nap("nap_0001")

    assert result["graduated_audited"] == 2
    assert result["graduated_failed"] == 1
    city_entry = ledger.get_entry(city)
    assert city_entry["graduated"] is False
    assert city_entry["stage"] == GRADUATED_STAGE - 1
    assert city_entry["degrade_count"] == 1
    assert ledger.get_entry(son)["graduated"] is True
    # Back in the prompt
    assert "Lisbon" in [qa.value for qa in ledger.get_active_qa_pairs()]
    # The active fact was audited too, without the fact in its own prompt
    color_probe = [s for q, s in backend.generate_calls if "favorite color" in q]
    assert color_probe and "teal" not in color_probe[0]


def test_nap_streaming_also_ungraduates(tmp_path):
    ledger = _ledger(tmp_path)
    city = _add(ledger, "Where does the user live?")
    _graduate(ledger, city)
    nap, _ = _nap(tmp_path, ledger, learned=[])
    events = list(nap.execute_nap_streaming("nap_0002"))
    assert ledger.get_entry(city)["graduated"] is False
    assert "graduated_failed=1" in events[-1]["detail"]


def test_nap_passing_graduated_fact_is_verified_not_touched(tmp_path):
    ledger = _ledger(tmp_path)
    city = _add(ledger, "Where does the user live?")
    _graduate(ledger, city)
    nap, _ = _nap(tmp_path, ledger, learned=["Where does the user live?"])
    result = nap.execute_nap("nap_0003")
    assert result["graduated_failed"] == 0
    e = ledger.get_entry(city)
    assert e["graduated"] is True and e["last_verified"] > 0 and e["recall_rate"] == 1.0


# ------------------------------------------------------------- full sleep


def test_full_sleep_replays_and_audits_graduated_facts(tmp_path):
    from unittest.mock import MagicMock
    from src.sleep.full_sleep import FullSleepController
    from src.sleep.trainer import SleepTrainer

    ledger = _ledger(tmp_path)
    son = _add(ledger, "What is the user's son's name?")
    city = _add(ledger, "Where does the user live?")
    color = _add(ledger, "What is the user's favorite color?")   # new, to be trained
    _graduate(ledger, son)
    _graduate(ledger, city)

    config = make_config(tmp_path, {"sleep": {"maintenance": {
        "replay_count": 5, "graduated_audit_count": 5}}})
    # After the fuse the weights know the son and the new color fact, but lost the city.
    backend = StubBackend(oracle=make_oracle(FACTS, learned=[
        "What is the user's son's name?", "What is the user's favorite color?"]))
    trainer = SleepTrainer(config, backend)
    ctrl = FullSleepController(config, backend, ledger, MagicMock(), MagicMock(),
                               MagicMock(), MagicMock(), fact_extractor=None, trainer=trainer)
    assert ctrl.consolidation_enabled

    stats = ctrl._consolidate("0001", ref_text=None)

    # Reporting
    assert stats["replayed"] == 2
    assert stats["graduated_audited"] == 2
    assert stats["graduated_failed"] == 1
    assert stats["advanced"] == 1

    # Replayed facts were in the training file
    train = Path(backend.train_calls[0]["data_path"]) / "train.jsonl"
    rows = [json.loads(l) for l in train.read_text().splitlines()]
    trained_questions = {r["messages"][0]["content"] for r in rows}
    assert "What is the user's son's name?" in trained_questions
    assert "Where does the user live?" in trained_questions
    assert "What is the user's favorite color?" in trained_questions

    # Failed graduated fact is back in the active list; passing one is not
    assert ledger.get_entry(city)["graduated"] is False
    assert ledger.get_entry(city)["stage"] == GRADUATED_STAGE - 1
    assert ledger.get_entry(son)["graduated"] is True
    assert ledger.get_entry(son)["train_count"] == 1          # replay recorded as training
    assert ledger.get_entry(color)["stage"] == 1
