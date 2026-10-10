"""Consolidation runner — train, fuse, gate, commit (or roll back).

One code path for every LoRA fuse in the system. Full sleep and micro-sleep
differ only in which facts they pass in, how many iterations they run, and
whether they run graduation tests afterwards.

    select facts  ->  ensure paraphrases  ->  record bleed baselines
    -> train + fuse -> load candidate -> GATES
         pass: graduation tests, apply audit results, record training, COMMIT
         fail: reload last good model, delete candidate, mark culprit,
               retry once without it (if one fact is identifiable)

Ledger stages are only touched after the gates pass, so a rolled-back cycle
leaves every stage exactly as it was.
"""

import time
from pathlib import Path
from typing import List, Optional, Sequence

from src.memory.facts import QAPair
from src.sleep.gates import GateReport, ModelRegistry, SleepGates


class ConsolidationRunner:
    def __init__(self, config, backend, trainer, fact_ledger, probe,
                 paraphraser=None, registry: Optional[ModelRegistry] = None,
                 gates: Optional[SleepGates] = None):
        self.config = config
        self.backend = backend
        self.trainer = trainer
        self.fact_ledger = fact_ledger
        self.probe = probe
        self.paraphraser = paraphraser

        maintenance = config.get("sleep.maintenance", {}) or {}
        self.replay_count = int(maintenance.get("replay_count", 5))
        self.graduated_audit_count = int(maintenance.get("graduated_audit_count", 5))

        g = config.get("sleep.gates", {}) or {}
        self.max_retries = int(g.get("max_retries", 1))
        self.max_gate_failures = int(g.get("max_gate_failures", 2))
        keep_previous = int(g.get("keep_previous_models", 1))

        self.save_dir = Path(config.paths.get("fused_models", "models/fused"))
        self.registry = registry or ModelRegistry(
            backend, config.model["path"], self.save_dir, keep_previous=keep_previous)
        self.gates = gates or SleepGates(config, backend, probe)

    # ------------------------------------------------------------ helpers

    def _quarantined(self, entry: dict) -> bool:
        return entry.get("gate_failures", 0) >= self.max_gate_failures

    @staticmethod
    def _label(entry: dict) -> str:
        qa = entry["qa"]
        return f"{qa.get('question', '')} → {qa.get('value', '')}"

    # ---------------------------------------------------------------- run

    def run(self, cycle_id: str, new_entries: Sequence[dict], *,
            ref_text: Optional[str] = None, ppl_baseline: Optional[float] = None,
            iters: Optional[int] = None, test_graduation: bool = True,
            replay_count: Optional[int] = None, weighted: bool = True) -> dict:
        """Consolidate `new_entries` (non-graduated ledger entries) into the weights.

        Returns a stats dict. `status` is one of:
          skipped | train_failed | rolled_back | committed
        """
        t0 = time.time()
        stats = {
            "status": "skipped", "cycle_id": cycle_id,
            "advanced": 0, "retreated": 0, "already_known": 0, "skipped": False,
            "advanced_facts": [], "retreated_facts": [],
            "trained": 0, "replayed": 0, "paraphrased": 0, "quarantined": 0,
            "graduated_audited": 0, "graduated_failed": 0, "ungraduated_facts": [],
            "graduation_results": [], "audit_results": [],
            "gates": [], "failed_gate": None, "excluded_facts": [], "attempts": 0,
            "model_path": self.registry.current,
        }

        train_entries = [e for e in new_entries if not self._quarantined(e)]
        stats["quarantined"] = len(new_entries) - len(train_entries)
        if stats["quarantined"]:
            print(f"        {stats['quarantined']} fact(s) quarantined after repeated gate failures")
        if not train_entries:
            print("        No eligible facts for consolidation")
            stats["skipped"] = True
            return stats

        n_replay = self.replay_count if replay_count is None else replay_count
        replay_entries = self.fact_ledger.get_replay_candidates(n_replay)
        stats["replayed"] = len(replay_entries)

        if self.paraphraser is not None:
            stats["paraphrased"] = self.paraphraser.ensure(
                self.fact_ledger, list(train_entries) + list(replay_entries))

        active_facts = self.fact_ledger.get_active_facts()
        all_qa = [QAPair.from_dict(e["qa"]) for e in active_facts]
        all_questions = [qa.question for qa in self.fact_ledger.get_all_qa_pairs()]

        # Baselines on the current (last good) model
        if ppl_baseline is None and ref_text:
            ppl_baseline = self.backend.compute_perplexity(ref_text)
        bleed_probe = self.gates.prepare_bleed(train_entries, avoid_questions=all_questions)
        audit_entries = self.fact_ledger.get_audit_candidates(self.graduated_audit_count)
        stats["graduated_audited"] = len(audit_entries)

        print(f"        {len(train_entries)} fact(s) to train, {len(replay_entries)} replayed, "
              f"{len(audit_entries)} graduated to audit")

        report: Optional[GateReport] = None
        fused_path = None
        attempt = 0
        while True:
            attempt += 1
            stats["attempts"] = attempt
            attempt_id = cycle_id if attempt == 1 else f"{cycle_id}_retry{attempt - 1}"
            qa_pairs = [QAPair.from_dict(e["qa"]) for e in list(train_entries) + list(replay_entries)]

            fused_path = self.trainer.train_and_fuse(
                qa_pairs, attempt_id, self.save_dir, weighted=weighted, iters=iters)
            if fused_path is None:
                print("        Consolidation aborted: training/fuse failed (model unchanged)")
                stats["status"] = "train_failed"
                stats["skipped"] = True
                return stats

            print("        Loading candidate model...")
            self.backend.reload(fused_path)

            report = self.gates.run(ppl_baseline, ref_text, bleed_probe.restrict(
                [e["fact_id"] for e in train_entries]), audit_entries, all_qa)
            stats["gates"].append(report.to_dict())
            if report.passed:
                print("        Gates passed")
                break

            # ---- failure: roll back, never touch stages ----
            print(f"        Gate FAILED: {report.failed_gate} — rolling back to {self.registry.current}")
            self.registry.rollback()
            self.registry.discard(fused_path, self.save_dir / "adapters" / attempt_id)
            for fid in report.culprit_fact_ids:
                self.fact_ledger.record_gate_failure(fid, report.failed_gate)

            culprits = set(report.culprit_fact_ids)
            can_retry = (culprits and attempt <= self.max_retries
                         and len(culprits) < len(train_entries))
            if can_retry:
                dropped = [e for e in train_entries if e["fact_id"] in culprits]
                stats["excluded_facts"].extend(self._label(e) for e in dropped)
                train_entries = [e for e in train_entries if e["fact_id"] not in culprits]
                print(f"        Retrying without {len(dropped)} culprit fact(s)")
                continue

            stats["status"] = "rolled_back"
            stats["failed_gate"] = report.failed_gate
            stats["model_path"] = self.registry.current
            stats["elapsed_seconds"] = round(time.time() - t0, 1)
            return stats

        # ---- committed path ----
        stats["trained"] = len(train_entries)
        for e in list(train_entries) + list(replay_entries):
            self.fact_ledger.record_training(e["fact_id"])

        if test_graduation:
            for entry in train_entries:
                qa = QAPair.from_dict(entry["qa"])
                fact_id = entry["fact_id"]
                result = self.probe.test(qa, all_qa, negative_values=entry.get("negative_values", []))
                stats["graduation_results"].append({
                    "fact_id": fact_id, "question": qa.question, "value": qa.value,
                    "passed": result["passed"], "fraction": result["fraction"],
                    "fallback": result["fallback"], "negative_hit": result["negative_hit"],
                    "phrasings": result["phrasings"],
                })
                label = self._label(entry)
                if result["passed"]:
                    new_stage = self.fact_ledger.advance_stage(fact_id)
                    stats["advanced"] += 1
                    stats["advanced_facts"].append(label)
                    tag = "GRADUATED" if new_stage >= 3 else f"stage {new_stage}"
                    print(f"        Advanced: {qa.value} ({tag})")
                elif entry.get("stage", 0) > 0:
                    new_stage = self.fact_ledger.retreat_stage(fact_id)
                    stats["retreated"] += 1
                    stats["retreated_facts"].append(label)
                    print(f"        Retreated: {qa.value} → stage {new_stage}")

        # Apply the graduated audit measured by the regression gate
        stats["audit_results"] = report.audit_results if report else []
        for r in stats["audit_results"]:
            self.fact_ledger.update_verification(r["fact_id"], r["fraction"])
            if not r["passed"]:
                self.fact_ledger.record_degrade(r["fact_id"])
                new_stage = self.fact_ledger.retreat_stage(r["fact_id"])
                stats["graduated_failed"] += 1
                stats["ungraduated_facts"].append(f"{r['question']} → {r['value']}")
                print(f"        Un-graduated: {r['value']} → stage {new_stage}")
        if audit_entries:
            print(f"        Graduated audit: {len(audit_entries) - stats['graduated_failed']}/"
                  f"{len(audit_entries)} still recalled")

        self.registry.commit(fused_path)
        stats["status"] = "committed"
        stats["model_path"] = fused_path
        stats["elapsed_seconds"] = round(time.time() - t0, 1)
        return stats
