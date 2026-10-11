"""Task 3 — gates with rollback. A forced gate failure leaves the backend on the
previous model path and the ledger stages unchanged; micro-sleep goes through
the same gate code as full sleep. No model (StubBackend)."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.facts import FactLedger, QAPair, GRADUATED_STAGE  # noqa: E402
from src.sleep.consolidation import ConsolidationRunner  # noqa: E402
from src.sleep.gates import ModelRegistry, SleepGates  # noqa: E402
from src.sleep.paraphrase import ParaphraseGenerator  # noqa: E402
from src.sleep.probe import RecallProbe  # noqa: E402
from src.sleep.trainer import SleepTrainer  # noqa: E402
from tests.stubs import StubBackend, make_config, snapshot_stages  # noqa: E402


FACTS = {
    "What is the user's son's name?": ("The user's son is named Andre.", "Andre"),
    "Where does the user live?": ("The user lives in Lisbon.", "Lisbon"),
    "What is the user's favorite color?": ("The user's favorite color is teal.", "teal"),
    "What does the user do for work?": ("The user works as a pilot.", "pilot"),
}


class World:
    """A fake model whose behavior flips when a candidate model is loaded."""

    def __init__(self):
        self.learned_after_fuse = set()     # questions the fused weights "know"
        self.bleed_after_fuse = {}          # value -> leaks into any question containing this word
        self.forgotten_after_fuse = set()   # graduated questions the fuse erases
        self.ppl_after_fuse = 10.0
        self.backend = StubBackend(oracle=self.oracle, ppl=self.ppl)

    @property
    def on_candidate(self):
        return "fused" in (self.backend.model_path or "")

    def ppl(self):
        return self.ppl_after_fuse if self.on_candidate else 10.0

    OTHER_PEOPLE = ("sister", "brother", "mother", "father", "friend", "boss",
                    "neighbor", "colleague", "cousin", "roommate")

    def oracle(self, question, system):
        q = question.lower()
        about_someone_else = any(w in q for w in self.OTHER_PEOPLE)
        if self.on_candidate and about_someone_else:
            # A fused model that "bleeds" answers neighbor questions with the new value
            for value, word in self.bleed_after_fuse.items():
                if word in q:
                    return f"I believe it's {value}."
        if about_someone_else:
            return "I'm not sure."
        for fq, (answer, value) in FACTS.items():
            if answer in system:                          # prompt bridge
                if all(w in q for w in fq.lower().split()[-2:]):
                    return answer
        # What the weights know: before any fuse, nothing. After a fuse, the
        # graduated facts (unless erased) plus whatever this fuse taught.
        if not self.on_candidate:
            known = set()
        else:
            known = (set(FACTS) - self.forgotten_after_fuse) | self.learned_after_fuse
        for fq in known:
            answer, value = FACTS[fq]
            key = fq.lower().rstrip("?").split()[-1]      # "name", "live", "color", "work"
            if key in q:
                return answer
        return "I'm not sure."


def _setup(tmp_path, world, overrides=None, graduated=(), new=()):
    cfg = {"sleep": {"graduation": {"use_model": False, "held_out_count": 2},
                     "gates": {"bleed_neighbors": 2, "max_retries": 1, "keep_previous_models": 1},
                     "maintenance": {"replay_count": 5, "graduated_audit_count": 5}}}
    if overrides:
        for k, v in overrides.items():
            cfg.setdefault(k, {}).update(v) if isinstance(v, dict) and k in cfg else cfg.update({k: v})
    config = make_config(tmp_path, cfg)
    ledger = FactLedger(str(tmp_path / "memit" / "ledger.json"))
    ids = {}
    for q in list(graduated) + list(new):
        answer, value = FACTS[q]
        ids[q] = ledger.add_fact(QAPair(question=q, answer=answer, value=value, priority=0.5))
    for q in graduated:
        for _ in range(GRADUATED_STAGE):
            ledger.advance_stage(ids[q])
    backend = world.backend
    probe = RecallProbe(backend, "sys", min_pass_fraction=1.0)
    registry = ModelRegistry(backend, "base-model", config.paths["fused_models"], keep_previous=1)
    gates = SleepGates(config, backend, probe)
    runner = ConsolidationRunner(config, backend, SleepTrainer(config, backend), ledger, probe,
                                 paraphraser=ParaphraseGenerator(config, backend),
                                 registry=registry, gates=gates)
    return config, ledger, ids, runner, registry


# ----------------------------------------------------------------- happy path

def test_commit_when_all_gates_pass(tmp_path):
    world = World()
    world.learned_after_fuse = {"What is the user's favorite color?"}
    config, ledger, ids, runner, registry = _setup(
        tmp_path, world, graduated=["What is the user's son's name?"], new=["What is the user's favorite color?"])

    result = runner.run("0001", ledger.get_active_facts(), ref_text="reference text")

    assert result["status"] == "committed"
    assert result["gates"][0]["passed"] is True
    assert result["advanced"] == 1 and result["replayed"] == 1 and result["graduated_audited"] == 1
    assert result["graduated_failed"] == 0
    fused = result["model_path"]
    assert "fused/0001" in fused and world.backend.model_path == fused
    assert registry.current == fused and registry.history == ["base-model"]
    assert ledger.get_entry(ids["What is the user's favorite color?"])["stage"] == 1


# ------------------------------------------------------------------ rollback

def test_ppl_gate_failure_rolls_back_and_leaves_stages(tmp_path):
    world = World()
    world.learned_after_fuse = set(FACTS)
    world.ppl_after_fuse = 10.0 * 1.5                     # +50% > 15%
    config, ledger, ids, runner, registry = _setup(
        tmp_path, world, graduated=["What is the user's son's name?"],
        new=["What is the user's favorite color?", "Where does the user live?"])
    ledger.advance_stage(ids["Where does the user live?"])  # stage 1, would advance or retreat
    before = snapshot_stages(ledger)
    before_train = {e["fact_id"]: e["train_count"] for e in ledger.get_active_facts() + ledger.get_graduated_facts()}

    result = runner.run("0002", ledger.get_active_facts(), ref_text="reference text")

    assert result["status"] == "rolled_back" and result["failed_gate"] == "perplexity"
    assert result["gates"][0]["ppl"]["passed"] is False
    assert world.backend.model_path == "base-model"                 # previous model path
    assert world.backend.reload_calls[-1] == "base-model"
    assert snapshot_stages(ledger) == before                        # stages unchanged
    assert {e["fact_id"]: e["train_count"] for e in ledger.get_active_facts() + ledger.get_graduated_facts()} == before_train
    assert result["advanced"] == 0 and result["retreated"] == 0 and result["graduated_failed"] == 0
    assert registry.current == "base-model" and registry.history == []
    # Candidate directories were deleted
    assert not (Path(config.paths["fused_models"]) / "fused" / "0002").exists()
    assert not (Path(config.paths["fused_models"]) / "adapters" / "0002").exists()


def test_regression_gate_failure_rolls_back_without_ungraduating(tmp_path):
    world = World()
    world.learned_after_fuse = {"What is the user's favorite color?"}
    world.forgotten_after_fuse = {"What is the user's son's name?", "Where does the user live?"}
    config, ledger, ids, runner, registry = _setup(
        tmp_path, world,
        graduated=["What is the user's son's name?", "Where does the user live?"],
        new=["What is the user's favorite color?"])
    before = snapshot_stages(ledger)

    result = runner.run("0003", ledger.get_active_facts(), ref_text="reference text")

    assert result["status"] == "rolled_back" and result["failed_gate"] == "regression"
    gate = result["gates"][0]["regression"]
    assert gate["audited"] == 2 and gate["failed"] == 2
    assert snapshot_stages(ledger) == before            # the dry audit did not retreat anything
    assert ledger.get_graduated_count() == 2
    assert world.backend.model_path == "base-model"


def test_bleed_gate_names_culprit_and_retries_without_it(tmp_path):
    world = World()
    world.learned_after_fuse = {"What is the user's favorite color?", "Where does the user live?"}
    world.bleed_after_fuse = {"teal": "color"}          # neighbors about color now answer teal
    config, ledger, ids, runner, registry = _setup(
        tmp_path, world, new=["What is the user's favorite color?", "Where does the user live?"])
    color_id = ids["What is the user's favorite color?"]

    result = runner.run("0004", ledger.get_active_facts(), ref_text="reference text")

    # Attempt 1 failed on bleed, attempt 2 (without the color fact) committed
    assert result["attempts"] == 2
    assert result["gates"][0]["failed_gate"] == "bleed"
    assert result["gates"][0]["culprit_fact_ids"] == [color_id]
    assert result["gates"][1]["passed"] is True
    assert result["status"] == "committed"
    assert result["excluded_facts"] == ["What is the user's favorite color? → teal"]
    assert "0004_retry1" in result["model_path"]
    # Culprit marked, not trained, stage untouched; the other fact advanced
    culprit = ledger.get_entry(color_id)
    assert culprit["gate_failures"] == 1 and culprit["last_gate_failure"] == "bleed"
    assert culprit["train_count"] == 0 and culprit["stage"] == 0
    assert ledger.get_entry(ids["Where does the user live?"])["stage"] == 1
    # Retry training data did not include the culprit as a trained fact. It may
    # still appear in a row's system context: it is still carried in the prompt.
    rows = [json.loads(l) for l in
            (Path(world.backend.train_calls[1]["data_path"]) / "train.jsonl").read_text().splitlines()]
    trained = [m["content"] for r in rows for m in r["messages"] if m["role"] != "system"]
    assert not any("favorite color" in t and "teal" in t for t in trained)
    # Rejected candidate removed, committed one kept
    fused_dir = Path(config.paths["fused_models"]) / "fused"
    assert not (fused_dir / "0004").exists() and (fused_dir / "0004_retry1").exists()


def test_quarantined_fact_is_not_trained(tmp_path):
    world = World()
    config, ledger, ids, runner, registry = _setup(tmp_path, world, new=["What is the user's favorite color?"])
    fid = ids["What is the user's favorite color?"]
    ledger.record_gate_failure(fid, "bleed")
    ledger.record_gate_failure(fid, "bleed")
    result = runner.run("0005", ledger.get_active_facts())
    assert result["status"] == "skipped" and result["quarantined"] == 1
    assert world.backend.train_calls == []
    assert fid in [e["fact_id"] for e in ledger.get_active_facts()]     # still in the prompt


def test_training_failure_leaves_model_untouched(tmp_path):
    world = World()
    config, ledger, ids, runner, registry = _setup(tmp_path, world, new=["Where does the user live?"])
    world.backend.fail_train = True
    before = snapshot_stages(ledger)
    result = runner.run("0006", ledger.get_active_facts())
    assert result["status"] == "train_failed"
    assert world.backend.reload_calls == [] and world.backend.model_path == "base-model"
    assert snapshot_stages(ledger) == before


# ------------------------------------------------------------------ registry

def test_registry_prunes_old_models_but_never_current_or_base(tmp_path):
    backend = StubBackend(model_path="base-model")
    root = tmp_path / "fused"
    reg = ModelRegistry(backend, "base-model", root, keep_previous=1)
    paths = []
    for i in range(1, 5):
        p = root / "fused" / f"{i:04d}"
        p.mkdir(parents=True)
        (root / "adapters" / f"{i:04d}").mkdir(parents=True)
        (p / "weights").write_text("w")
        backend.reload(str(p))
        reg.commit(str(p))
        paths.append(p)
    existing = sorted(d.name for d in (root / "fused").iterdir())
    assert existing == ["0003", "0004"]                          # current + 1 previous
    assert sorted(d.name for d in (root / "adapters").iterdir()) == ["0003", "0004"]
    assert reg.current == str(paths[-1])
    # discard refuses anything outside the managed roots or the current model
    outside = tmp_path / "precious"
    outside.mkdir()
    reg.discard(str(outside), reg.current)
    assert outside.exists() and paths[-1].exists()
    reg.rollback()
    assert backend.reload_calls[-1] == str(paths[-1])


# ----------------------------------------------------------------- full sleep

def test_full_sleep_reports_rollback_status(tmp_path):
    from src.sleep.full_sleep import FullSleepController
    world = World()
    world.learned_after_fuse = set(FACTS)
    world.ppl_after_fuse = 30.0
    config, ledger, ids, runner, registry = _setup(tmp_path, world, new=["Where does the user live?"])
    (Path(config.paths["core_identity"])).mkdir(parents=True, exist_ok=True)
    (Path(config.paths["core_identity"]) / "identity.jsonl").write_text(json.dumps({"text": "reference"}) + "\n")
    curator = MagicMock(); curator.curate_with_model.return_value = []
    ctrl = FullSleepController(config, world.backend, ledger, curator, MagicMock(), MagicMock(),
                               MagicMock(), trainer=runner.trainer, runner=runner)
    result = ctrl.execute_sleep("0007", "full", lambda: ([], []))
    assert result["status"] == "rolled_back"
    assert result["consolidation"]["failed_gate"] == "perplexity"
    assert world.backend.model_path == "base-model"
    events = list(ctrl.execute_sleep_streaming("0008", "full", lambda: ([], [])))
    assert any("ROLLED BACK" in e.get("detail", "") for e in events)


# ---------------------------------------------------------------- micro-sleep

def test_micro_sleep_uses_same_gates_and_rolls_back(tmp_path):
    from src.sleep.micro_sleep import MicroSleepController
    world = World()
    world.learned_after_fuse = set(FACTS)
    world.ppl_after_fuse = 30.0
    config, ledger, ids, runner, registry = _setup(
        tmp_path, world, new=["Where does the user live?"],
        overrides={"micro_sleep": {"enabled": True, "min_age_seconds": 0, "iters": 7}})
    micro = MicroSleepController(config, world.backend, runner.trainer, ledger, None, runner=runner)
    before = snapshot_stages(ledger)
    ident = Path(config.paths["core_identity"])
    ident.mkdir(parents=True, exist_ok=True)
    (ident / "identity.jsonl").write_text(json.dumps({"text": "reference"}) + "\n")

    candidates = micro._select_facts()
    assert candidates
    micro._execute(candidates)                      # run synchronously

    assert micro.last_result["status"] == "rolled_back"
    assert micro.last_result["failed_gate"] == "perplexity"
    assert world.backend.model_path == "base-model"
    assert snapshot_stages(ledger) == before
    assert world.backend.train_calls[0]["iters"] == 7
    assert micro.to_dict()["last_result"]["status"] == "rolled_back"


def test_micro_sleep_commits_through_runner(tmp_path):
    from src.sleep.micro_sleep import MicroSleepController
    world = World()
    world.learned_after_fuse = set(FACTS)
    config, ledger, ids, runner, registry = _setup(
        tmp_path, world, graduated=["What is the user's son's name?"], new=["Where does the user live?"],
        overrides={"micro_sleep": {"enabled": True, "min_age_seconds": 0, "replay_count": 1}})
    micro = MicroSleepController(config, world.backend, runner.trainer, ledger, None, runner=runner)
    micro._execute(micro._select_facts())
    res = micro.last_result
    assert res["status"] == "committed" and res["trained"] == 1 and res["replayed"] == 1
    assert res["advanced"] == 0                                    # no graduation tests in micro-sleep
    assert "micro_0001" in world.backend.model_path
    assert ledger.get_entry(ids["Where does the user live?"])["train_count"] == 1
