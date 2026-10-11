"""sleep.maintenance.max_train_facts: cap on facts trained per cycle."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.facts import FactLedger, QAPair  # noqa: E402
from src.sleep.consolidation import ConsolidationRunner  # noqa: E402
from src.sleep.gates import ModelRegistry, SleepGates  # noqa: E402
from src.sleep.probe import RecallProbe  # noqa: E402
from src.sleep.trainer import SleepTrainer  # noqa: E402
from tests.stubs import StubBackend, make_config  # noqa: E402


def _runner(tmp_path, cap, n_facts):
    cfg = make_config(tmp_path, {"sleep": {"maintenance": {"max_train_facts": cap, "replay_count": 0}},
                                 "lora": {"contrast_per_fact": 0}})
    backend = StubBackend(oracle=lambda q, s: "I don't know.")
    ledger = FactLedger(str(tmp_path / "memit" / "ledger.json"))
    ids = []
    for i in range(n_facts):
        qa = QAPair(f"What is the user's item{i}?", f"The user's item{i} is value{i}.", f"value{i}",
                    timestamp=1000.0 + i)
        ids.append(ledger.add_fact(qa))
    probe = RecallProbe(backend, "sys", min_pass_fraction=1.0)
    registry = ModelRegistry(backend, "base-model", cfg.paths["fused_models"])
    runner = ConsolidationRunner(cfg, backend, SleepTrainer(cfg, backend), ledger, probe,
                                 paraphraser=None, registry=registry, gates=SleepGates(cfg, backend, probe))
    return runner, ledger, ids, backend


def test_select_batch_orders_by_stage_then_failures_then_age(tmp_path):
    runner, ledger, ids, _ = _runner(tmp_path, cap=3, n_facts=6)
    ledger.advance_stage(ids[4])                    # closest to graduation -> first
    ledger.record_gate_failure(ids[0], "bleed")     # oldest but has a failure -> after clean ones
    selected, deferred = runner.select_batch(ledger.get_active_facts())
    assert [e["fact_id"] for e in selected] == [ids[4], ids[1], ids[2]]
    assert {e["fact_id"] for e in deferred} == {ids[0], ids[3], ids[5]}


def test_no_cap_when_zero_or_batch_fits(tmp_path):
    runner, ledger, ids, _ = _runner(tmp_path, cap=0, n_facts=4)
    sel, deferred = runner.select_batch(ledger.get_active_facts())
    assert len(sel) == 4 and deferred == []
    runner.max_train_facts = 10
    sel, deferred = runner.select_batch(ledger.get_active_facts())
    assert len(sel) == 4 and deferred == []


def test_run_trains_only_the_cap_and_reports_deferred(tmp_path):
    runner, ledger, ids, backend = _runner(tmp_path, cap=2, n_facts=5)
    stats = runner.run("c00", ledger.get_active_facts(), test_graduation=False)
    assert stats["status"] == "committed"
    assert stats["trained"] == 2 and stats["deferred"] == 3
    trained = [e for e in ledger.get_active_facts() if e["train_count"] > 0]
    assert len(trained) == 2
    # Deferred facts were not written to the training file
    train_txt = (Path(backend.train_calls[0]["data_path"]) / "train.jsonl").read_text()
    assert sum(f"item{i}?" in train_txt for i in range(5)) == 2
