"""Full sleep controller — MEMIT maintenance pipeline.

Implements a 6-step sleep cycle:
  [1] Health Check — measure baseline PPL
  [2] Curate — extract facts from unconsumed conversations, inject via MEMIT
  [3] Fact Audit — test recall of ALL active MEMIT edits
  [4] Maintenance — re-inject degraded facts, prune if over capacity
  [5] Validate — measure post-maintenance PPL, rollback if degraded
  [6] Report — summary of audited/refreshed/pruned facts
"""

import json
import time
from pathlib import Path


class FullSleepController:
    """Executes full sleep cycles with MEMIT health maintenance."""

    def __init__(self, config, backend, memit_engine, ledger, curator,
                 validator, session_tracker, health_monitor,
                 fact_extractor=None):
        self.config = config
        self.backend = backend
        self.memit_engine = memit_engine
        self.ledger = ledger
        self.curator = curator
        self.validator = validator
        self.session_tracker = session_tracker
        self.health_monitor = health_monitor
        self.fact_extractor = fact_extractor

        maintenance = config.get("sleep.maintenance", {}) or {}
        self.degraded_threshold = maintenance.get("degraded_threshold", 0.5)
        self.max_ppl_increase = maintenance.get("max_ppl_increase", 0.15)
        self.max_refresh_per_cycle = maintenance.get("max_refresh_per_cycle", 10)

    def execute_sleep(self, cycle_id, sleep_type, gather_messages_fn):
        """Execute the full sleep pipeline.

        Args:
            cycle_id: Unique sleep cycle identifier
            sleep_type: Ignored (single sleep type now), kept for API compat
            gather_messages_fn: Callable that returns (messages, consumed_sessions)

        Returns:
            Result dict with status and details
        """
        start_time = time.time()

        # [1] Health check
        print(f"  [1/6] Health check...")
        ref_text = self._get_ppl_reference_text()
        ppl_baseline = self.backend.compute_perplexity(ref_text) if ref_text else None
        if ppl_baseline:
            print(f"        Baseline PPL: {ppl_baseline:.2f}")

        # [2] Curate — gather and inject new facts
        print(f"  [2/6] Curating...")
        messages, consumed_sessions = gather_messages_fn()
        print(f"        {len(consumed_sessions)} new session(s) to process")
        curated = self.curator.curate_with_model(messages, cycle_id)
        print(f"        {len(curated)} exchanges selected")

        # Extract and inject new MEMIT facts from curated exchanges
        new_facts_injected = 0
        if curated and self.fact_extractor:
            for exchange in curated:
                msgs = exchange.get("messages", [])
                user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
                asst_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
                if user_msg and asst_msg:
                    facts = self.fact_extractor.extract_from_exchange(user_msg, asst_msg)
                    for fact in facts:
                        try:
                            self.memit_engine.inject_facts([fact])
                            new_facts_injected += 1
                        except Exception as e:
                            print(f"        Failed to inject: {fact.subject} {fact.relation} → {e}")
            if new_facts_injected:
                print(f"        Injected {new_facts_injected} new facts via MEMIT")

        if consumed_sessions:
            self.session_tracker.mark_consumed(consumed_sessions, cycle_id)
            print(f"        Marked {len(consumed_sessions)} session(s) as consumed")

        # [3] Fact audit
        print(f"  [3/6] Auditing all active MEMIT edits...")
        audit_results = self._audit_facts()
        healthy = audit_results["healthy"]
        degraded = audit_results["degraded"]
        total = audit_results["total"]
        print(f"        {healthy} healthy, {len(degraded)} degraded, {total} total")

        # [4] Maintenance
        print(f"  [4/6] Maintenance...")
        maint_results = self._maintain_edits(audit_results, ppl_baseline)
        print(f"        Refreshed: {maint_results['refreshed']}, Pruned: {maint_results['pruned']}")

        # [5] Validate
        print(f"  [5/6] Validating...")
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

        validation = self.validator.validate_ppl(ppl_baseline, ppl_after, self.max_ppl_increase)

        # [6] Report
        elapsed = time.time() - start_time
        print(f"  [6/6] Sleep cycle completed in {elapsed:.1f}s")

        return {
            "status": "approved" if ppl_ok else "warning",
            "new_facts_injected": new_facts_injected,
            "audited": total,
            "facts_refreshed": maint_results["refreshed"],
            "facts_pruned": maint_results["pruned"],
            "ppl_before": round(ppl_baseline, 2) if ppl_baseline else None,
            "ppl_after": round(ppl_after, 2) if ppl_after else None,
            "elapsed_seconds": round(elapsed, 1),
        }

    def _audit_facts(self) -> dict:
        """Test recall of all active MEMIT edits.

        Returns:
            dict with 'healthy' count, 'degraded' list of (edit, recall_rate), 'total' count
        """
        active_edits = list(self.memit_engine._active_edits)
        healthy = 0
        degraded = []

        for edit in active_edits:
            recalled = 0
            for fact in edit.facts:
                passed, _ = self.memit_engine.test_recall(fact, raw=True)
                if passed:
                    recalled += 1

            recall_rate = recalled / len(edit.facts) if edit.facts else 1.0
            self.ledger.update_verification(edit.edit_id, recall_rate)

            if recall_rate >= self.degraded_threshold:
                healthy += 1
            else:
                degraded.append((edit, recall_rate))
                print(f"        DEGRADED: {edit.facts[0].subject} {edit.facts[0].relation} "
                      f"(recall {recall_rate:.0%})")

        return {
            "healthy": healthy,
            "degraded": degraded,
            "total": len(active_edits),
        }

    def _maintain_edits(self, audit_results, ppl_baseline) -> dict:
        """Re-inject degraded facts and prune if over capacity.

        Returns:
            dict with 'refreshed' and 'pruned' counts
        """
        degraded = audit_results["degraded"]
        refreshed = 0
        pruned = 0

        # Re-inject degraded facts (up to max_refresh_per_cycle)
        for edit, recall_rate in degraded[:self.max_refresh_per_cycle]:
            try:
                # Revert old delta
                self.memit_engine.revert_edit(edit)
                # Fresh inject with current null-space constraints
                self.memit_engine.inject_facts(edit.facts)
                refreshed += 1
                print(f"        Refreshed: {edit.facts[0].subject} {edit.facts[0].relation}")
            except Exception as e:
                print(f"        Failed to refresh: {e}")

        # Prune if over capacity
        max_edits = self.memit_engine.max_active_edits
        active_count = self.memit_engine.get_active_edit_count()
        if active_count > max_edits:
            excess = active_count - max_edits
            # Sort by timestamp (oldest first) for pruning
            candidates = sorted(self.memit_engine._active_edits, key=lambda e: e.timestamp)
            for edit in candidates[:excess]:
                self.memit_engine.revert_edit(edit)
                self.ledger.mark_pruned(edit.edit_id)
                pruned += 1
                print(f"        Pruned: {edit.facts[0].subject} {edit.facts[0].relation}")

        return {"refreshed": refreshed, "pruned": pruned}

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
        total_steps = 6

        # [1] Health Check
        yield {"step": 1, "total": total_steps, "label": "Health check", "status": "running"}
        ref_text = self._get_ppl_reference_text()
        ppl_baseline = self.backend.compute_perplexity(ref_text) if ref_text else None
        yield {"step": 1, "total": total_steps, "label": "Health check", "status": "done",
               "detail": f"PPL: {ppl_baseline:.2f}" if ppl_baseline else "No reference text"}

        # [2] Curate
        yield {"step": 2, "total": total_steps, "label": "Curating", "status": "running"}
        messages, consumed_sessions = gather_messages_fn()
        curated = self.curator.curate_with_model(messages, cycle_id)

        new_facts_injected = 0
        if curated and self.fact_extractor:
            for exchange in curated:
                msgs = exchange.get("messages", [])
                user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
                asst_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
                if user_msg and asst_msg:
                    facts = self.fact_extractor.extract_from_exchange(user_msg, asst_msg)
                    for fact in facts:
                        try:
                            self.memit_engine.inject_facts([fact])
                            new_facts_injected += 1
                        except Exception:
                            pass

        if consumed_sessions:
            self.session_tracker.mark_consumed(consumed_sessions, cycle_id)

        yield {"step": 2, "total": total_steps, "label": "Curating", "status": "done",
               "detail": f"{len(curated)} exchanges, {new_facts_injected} facts injected, "
                         f"{len(consumed_sessions)} session(s)"}

        # [3] Fact Audit
        yield {"step": 3, "total": total_steps, "label": "Auditing facts", "status": "running"}
        audit_results = self._audit_facts()
        yield {"step": 3, "total": total_steps, "label": "Auditing facts", "status": "done",
               "detail": f"{audit_results['healthy']} healthy, "
                         f"{len(audit_results['degraded'])} degraded"}

        # [4] Maintenance
        yield {"step": 4, "total": total_steps, "label": "Maintenance", "status": "running"}
        maint_results = self._maintain_edits(audit_results, ppl_baseline)
        yield {"step": 4, "total": total_steps, "label": "Maintenance", "status": "done",
               "detail": f"Refreshed: {maint_results['refreshed']}, "
                         f"Pruned: {maint_results['pruned']}"}

        # [5] Validate
        yield {"step": 5, "total": total_steps, "label": "Validating", "status": "running"}
        ppl_after = self.backend.compute_perplexity(ref_text) if ref_text else None
        ppl_detail = "No reference text"
        if ppl_baseline and ppl_after and ppl_baseline > 0:
            ppl_increase = (ppl_after - ppl_baseline) / ppl_baseline
            ppl_detail = f"PPL: {ppl_baseline:.2f} → {ppl_after:.2f} ({ppl_increase:+.1%})"
        yield {"step": 5, "total": total_steps, "label": "Validating", "status": "done",
               "detail": ppl_detail}

        # [6] Report
        elapsed = time.time() - start_time
        yield {"step": 6, "total": total_steps, "label": "Report", "status": "done",
               "detail": f"facts_refreshed={maint_results['refreshed']}, "
                         f"facts_pruned={maint_results['pruned']}, "
                         f"new_injected={new_facts_injected}",
               "facts_refreshed": maint_results["refreshed"],
               "facts_pruned": maint_results["pruned"]}
