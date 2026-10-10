"""Full sleep controller — LoRA consolidation + graduation pipeline.

Implements a 5-step sleep cycle (MEMIT removed):
  [1] Health Check — measure baseline PPL
  [2] Curate — extract new facts from unconsumed conversations
  [3] LoRA Consolidation — train on all non-graduated facts PLUS a replay
      sample of graduated facts (so later fuses do not erode older ones), fuse,
      then run the GATES (perplexity, bleed, regression). A failed gate reloads
      the last good model and leaves every stage untouched.
  [4] Graduation Test — ask each fact's held-out paraphrases with the fact
      withheld from the prompt; re-audit a sample of graduated facts
  [5] Validate — report post-sleep PPL against the baseline

Steps 3–4 are implemented once in src/sleep/consolidation.py and shared with
micro-sleep.

Graduation: facts pass through stages 0→1→2→3. At stage 3, the fact
is "graduated" — LoRA carries the knowledge and the fact is removed
from the system prompt, freeing context window space.
"""

import json
import time
from pathlib import Path

from src.memory.facts import QAPair
from src.sleep.consolidation import ConsolidationRunner
from src.sleep.probe import RecallProbe


class FullSleepController:
    """Executes full sleep cycles with LoRA consolidation."""

    def __init__(self, config, backend, fact_ledger, curator,
                 validator, session_tracker, health_monitor,
                 fact_extractor=None, trainer=None, paraphraser=None,
                 runner=None):
        self.config = config
        self.backend = backend
        self.fact_ledger = fact_ledger
        self.curator = curator
        self.validator = validator
        self.session_tracker = session_tracker
        self.health_monitor = health_monitor
        self.fact_extractor = fact_extractor
        self.trainer = trainer
        self.paraphraser = paraphraser

        grad = config.get("sleep.graduation", {}) or {}
        self.probe = RecallProbe(
            backend,
            system_prompt=config.context.get("system_prompt", ""),
            temperature=grad.get("test_temperature", 0.1),
            min_pass_fraction=grad.get("min_pass_fraction", 1.0),
        )

        maintenance = config.get("sleep.maintenance", {}) or {}
        self.degraded_threshold = maintenance.get("degraded_threshold", 0.5)
        self.max_ppl_increase = maintenance.get("max_ppl_increase", 0.15)
        self.replay_count = maintenance.get("replay_count", 5)
        self.graduated_audit_count = maintenance.get("graduated_audit_count", 5)

        # Consolidation config
        consolidation = config.get("consolidation", {}) or {}
        lora_cfg = config.get("lora", {}) or {}
        self.consolidation_enabled = (
            trainer is not None
            and lora_cfg.get("enabled", False)
            and consolidation.get("enabled", False)
        )

        # Shared train/fuse/gate/commit path (also used by micro-sleep)
        self.runner = runner
        if self.runner is None and self.consolidation_enabled:
            self.runner = ConsolidationRunner(
                config, backend, trainer, fact_ledger, self.probe, paraphraser=paraphraser)

    @property
    def total_steps(self):
        return 5 if self.consolidation_enabled else 3

    def execute_sleep(self, cycle_id, sleep_type, gather_messages_fn):
        """Execute the full sleep pipeline.

        Returns:
            Result dict with status and details
        """
        start_time = time.time()
        ts = self.total_steps

        # [1] Health check
        print(f"  [1/{ts}] Health check...")
        ref_text = self._get_ppl_reference_text()
        ppl_baseline = self.backend.compute_perplexity(ref_text) if ref_text else None
        if ppl_baseline:
            print(f"        Baseline PPL: {ppl_baseline:.2f}")

        # [2] Curate — gather and persist new facts
        print(f"  [2/{ts}] Curating...")
        messages, consumed_sessions = gather_messages_fn()
        print(f"        {len(consumed_sessions)} new session(s) to process")
        curated = self.curator.curate_with_model(messages, cycle_id)
        print(f"        {len(curated)} exchanges selected")

        new_facts_added = 0
        if curated and self.fact_extractor:
            all_extracted = []
            for exchange in curated:
                msgs = exchange.get("messages", [])
                user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
                asst_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
                if user_msg and asst_msg:
                    facts = self.fact_extractor.extract_from_exchange(user_msg, asst_msg)
                    all_extracted.extend(facts)

            if all_extracted:
                existing = self.fact_ledger.get_all_qa_pairs()
                unique = self.fact_extractor.deduplicate(all_extracted, existing)

                # Dedup within batch
                seen_keys = set()
                batch = []
                for fact in unique:
                    key = self.fact_extractor._dedup_key(fact)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        batch.append(fact)

                skipped = len(all_extracted) - len(batch)
                if skipped:
                    print(f"        Skipped {skipped} duplicate/junk fact(s)")

                for qa in batch:
                    self.fact_ledger.add_fact(qa)
                    new_facts_added += 1

            if new_facts_added:
                print(f"        Added {new_facts_added} new facts to ledger")

        if consumed_sessions:
            self.session_tracker.mark_consumed(consumed_sessions, cycle_id)
            print(f"        Marked {len(consumed_sessions)} session(s) as consumed")

        # [3-4] Consolidation + Graduation (if enabled)
        consolidation_stats = {"advanced": 0, "retreated": 0, "already_known": 0, "skipped": True,
                               "status": "skipped"}
        if self.consolidation_enabled:
            print(f"  [3/{ts}] LoRA Consolidation...")
            consolidation_stats = self._consolidate(cycle_id, ref_text, ppl_baseline)
            print(f"        Status: {consolidation_stats.get('status')}"
                  + (f" (gate failed: {consolidation_stats['failed_gate']})"
                     if consolidation_stats.get('failed_gate') else ""))
            print(f"        Advanced: {consolidation_stats['advanced']}, "
                  f"Retreated: {consolidation_stats['retreated']}, "
                  f"Replayed: {consolidation_stats.get('replayed', 0)}, "
                  f"Graduated audited: {consolidation_stats.get('graduated_audited', 0)}, "
                  f"failed: {consolidation_stats.get('graduated_failed', 0)}")

            print(f"  [4/{ts}] Graduation summary:")
            graduated = self.fact_ledger.get_graduated_count()
            total = self.fact_ledger.get_active_fact_count()
            print(f"        {graduated}/{total} facts graduated")

        # [3/5] Validate
        validate_step = 5 if self.consolidation_enabled else 3
        print(f"  [{validate_step}/{ts}] Validating...")
        ppl_after = self.backend.compute_perplexity(ref_text) if ref_text else None
        ppl_ok = True
        if ppl_baseline and ppl_after and ppl_baseline > 0:
            ppl_increase = (ppl_after - ppl_baseline) / ppl_baseline
            print(f"        PPL: {ppl_baseline:.2f} → {ppl_after:.2f} ({ppl_increase:+.1%})")
            if ppl_increase > self.max_ppl_increase:
                ppl_ok = False
                print(f"        WARNING: PPL increase exceeds threshold ({self.max_ppl_increase:.0%})")
        else:
            print(f"        PPL: no reference text, skipping check")

        elapsed = time.time() - start_time
        print(f"  Sleep cycle completed in {elapsed:.1f}s")

        if consolidation_stats.get("status") == "rolled_back":
            status = "rolled_back"
        else:
            status = "approved" if ppl_ok else "warning"

        return {
            "status": status,
            "new_facts_added": new_facts_added,
            "consolidation": consolidation_stats,
            "model_path": consolidation_stats.get("model_path"),
            "ppl_before": round(ppl_baseline, 2) if ppl_baseline else None,
            "ppl_after": round(ppl_after, 2) if ppl_after else None,
            "elapsed_seconds": round(elapsed, 1),
        }

    def _consolidate(self, cycle_id, ref_text, ppl_baseline=None) -> dict:
        """LoRA consolidation + gates + graduation testing (see ConsolidationRunner).

        1. Gather all non-graduated facts (+ replay sample of graduated ones)
        2. Train LoRA, fuse, load the candidate model
        3. Gates: perplexity / bleed / regression — fail → reload last good
           model, stages untouched, retry once without an identified culprit
        4. Graduation test on held-out paraphrases → advance / retreat
        5. Graduated audit → retreat and record degrade on failure, commit
        """
        active_facts = self.fact_ledger.get_active_facts()
        eligible = [e for e in active_facts if not e.get("graduated", False)]
        if not eligible:
            print("        No eligible facts for consolidation")
            return {"status": "skipped", "skipped": True, "advanced": 0, "retreated": 0,
                    "already_known": 0, "replayed": 0, "graduated_audited": 0,
                    "graduated_failed": 0, "advanced_facts": [], "retreated_facts": [],
                    "ungraduated_facts": [], "graduation_results": [], "audit_results": [],
                    "gates": [], "failed_gate": None, "excluded_facts": []}
        return self.runner.run(cycle_id, eligible, ref_text=ref_text,
                               ppl_baseline=ppl_baseline, test_graduation=True)

    def _test_graduation(self, qa, all_qa, system_prompt=None) -> bool:
        """Test if LoRA has absorbed a fact: ask its held-out phrasings with the
        fact withheld from the system prompt. See RecallProbe.test."""
        return self.probe.test(qa, all_qa)["passed"]

    def _get_ppl_reference_text(self):
        """Get reference text for perplexity measurement from identity data."""
        identity_dir = Path(self.config.paths["core_identity"])
        identity_file = identity_dir / "identity.jsonl"
        if not identity_file.exists():
            return None
        texts = []
        with open(identity_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        texts.append(item.get("text", ""))
                    except json.JSONDecodeError:
                        continue
        return " ".join(texts)[:2000] if texts else None

    def execute_sleep_streaming(self, cycle_id, sleep_type, gather_messages_fn):
        """Execute sleep pipeline, yielding progress dicts for each step."""
        start_time = time.time()
        ts = self.total_steps

        # [1] Health Check
        yield {"step": 1, "total": ts, "label": "Health check", "status": "running"}
        ref_text = self._get_ppl_reference_text()
        ppl_baseline = self.backend.compute_perplexity(ref_text) if ref_text else None
        yield {"step": 1, "total": ts, "label": "Health check", "status": "done",
               "detail": f"PPL: {ppl_baseline:.2f}" if ppl_baseline else "No reference text"}

        # [2] Curate
        yield {"step": 2, "total": ts, "label": "Curating", "status": "running"}
        messages, consumed_sessions = gather_messages_fn()
        curated = self.curator.curate_with_model(messages, cycle_id)

        new_facts_added = 0
        added_fact_labels = []
        if curated and self.fact_extractor:
            all_extracted = []
            for exchange in curated:
                msgs = exchange.get("messages", [])
                user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
                asst_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
                if user_msg and asst_msg:
                    facts = self.fact_extractor.extract_from_exchange(user_msg, asst_msg)
                    all_extracted.extend(facts)

            if all_extracted:
                existing = self.fact_ledger.get_all_qa_pairs()
                unique = self.fact_extractor.deduplicate(all_extracted, existing)

                seen_keys = set()
                batch = []
                for fact in unique:
                    key = self.fact_extractor._dedup_key(fact)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        batch.append(fact)

                for qa in batch:
                    self.fact_ledger.add_fact(qa)
                    new_facts_added += 1
                    added_fact_labels.append(f"{qa.question} → {qa.value}")

        if consumed_sessions:
            self.session_tracker.mark_consumed(consumed_sessions, cycle_id)

        yield {"step": 2, "total": ts, "label": "Curating", "status": "done",
               "detail": f"{len(curated)} exchanges, {new_facts_added} facts added, "
                         f"{len(consumed_sessions)} session(s)",
               "facts": added_fact_labels}

        # [3-4] Consolidation + Graduation (if enabled)
        consolidation_stats = {"advanced": 0, "retreated": 0, "skipped": True}
        if self.consolidation_enabled:
            yield {"step": 3, "total": ts, "label": "LoRA Consolidation", "status": "running"}
            consolidation_stats = self._consolidate(cycle_id, ref_text, ppl_baseline)
            gate_note = (f"ROLLED BACK (gate: {consolidation_stats['failed_gate']}). "
                         if consolidation_stats.get("failed_gate") else "")
            yield {"step": 3, "total": ts, "label": "LoRA Consolidation", "status": "done",
                   "detail": gate_note + f"Advanced: {consolidation_stats['advanced']}, "
                             f"Retreated: {consolidation_stats['retreated']}, "
                             f"Replayed: {consolidation_stats.get('replayed', 0)}, "
                             f"Graduated audited: {consolidation_stats.get('graduated_audited', 0)}, "
                             f"failed: {consolidation_stats.get('graduated_failed', 0)}",
                   "facts": consolidation_stats.get("advanced_facts", []),
                   "facts_bad": (consolidation_stats.get("retreated_facts", [])
                                 + consolidation_stats.get("ungraduated_facts", []))}

            graduated = self.fact_ledger.get_graduated_count()
            total = self.fact_ledger.get_active_fact_count()
            yield {"step": 4, "total": ts, "label": "Graduation", "status": "done",
                   "detail": f"{graduated}/{total} facts graduated"}

        # [3/5] Validate
        validate_step = 5 if self.consolidation_enabled else 3
        yield {"step": validate_step, "total": ts, "label": "Validating", "status": "running"}
        ppl_after = self.backend.compute_perplexity(ref_text) if ref_text else None
        ppl_detail = "No reference text"
        if ppl_baseline and ppl_after and ppl_baseline > 0:
            ppl_increase = (ppl_after - ppl_baseline) / ppl_baseline
            ppl_detail = f"PPL: {ppl_baseline:.2f} → {ppl_after:.2f} ({ppl_increase:+.1%})"
        yield {"step": validate_step, "total": ts, "label": "Validating", "status": "done",
               "detail": ppl_detail}

        # Final report
        elapsed = time.time() - start_time
        yield {"step": ts, "total": ts, "label": "Report", "status": "done",
               "detail": f"status={consolidation_stats.get('status', 'skipped')}, "
                         f"new_facts={new_facts_added}, "
                         f"advanced={consolidation_stats.get('advanced', 0)}, "
                         f"retreated={consolidation_stats.get('retreated', 0)}, "
                         f"replayed={consolidation_stats.get('replayed', 0)}, "
                         f"graduated_audited={consolidation_stats.get('graduated_audited', 0)}, "
                         f"graduated_failed={consolidation_stats.get('graduated_failed', 0)}",
               "facts_refreshed": 0,
               "facts_pruned": 0}
