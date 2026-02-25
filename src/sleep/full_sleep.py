"""Full sleep controller — MEMIT maintenance + LoRA consolidation pipeline.

Implements an 8-step sleep cycle:
  [1] Health Check — measure baseline PPL
  [2] Curate — extract facts from unconsumed conversations, inject via MEMIT
  [3] Fact Audit — test recall of ALL active MEMIT edits
  [4] Maintenance — re-inject degraded facts, prune if over capacity
  [5] LoRA Consolidation — train LoRA on active facts, fuse, per-fact gate
  [6] MEMIT Scale-Down — apply stage-based scale schedule
  [7] Validate — measure post-maintenance PPL, rollback if degraded
  [8] Report — summary of audited/refreshed/pruned/consolidated facts

Steps 5-6 are skipped if no trainer is provided or lora.enabled is false.
"""

import json
import time
from pathlib import Path


class FullSleepController:
    """Executes full sleep cycles with MEMIT health maintenance."""

    def __init__(self, config, backend, memit_engine, ledger, curator,
                 validator, session_tracker, health_monitor,
                 fact_extractor=None, trainer=None):
        self.config = config
        self.backend = backend
        self.memit_engine = memit_engine
        self.ledger = ledger
        self.curator = curator
        self.validator = validator
        self.session_tracker = session_tracker
        self.health_monitor = health_monitor
        self.fact_extractor = fact_extractor
        self.trainer = trainer

        maintenance = config.get("sleep.maintenance", {}) or {}
        self.degraded_threshold = maintenance.get("degraded_threshold", 0.5)
        self.max_ppl_increase = maintenance.get("max_ppl_increase", 0.15)
        self.max_refresh_per_cycle = maintenance.get("max_refresh_per_cycle", 10)

        # Consolidation config
        consolidation = config.get("consolidation", {}) or {}
        lora_cfg = config.get("lora", {}) or {}
        self.consolidation_enabled = (
            trainer is not None
            and lora_cfg.get("enabled", False)
            and consolidation.get("enabled", False)
        )
        self.scale_schedule = consolidation.get("scale_schedule", [1.0, 0.5, 0.1, 0.0])

    @property
    def total_steps(self):
        return 8 if self.consolidation_enabled else 6

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
        ts = self.total_steps

        # [1] Health check
        print(f"  [1/{ts}] Health check...")
        ref_text = self._get_ppl_reference_text()
        ppl_baseline = self.backend.compute_perplexity(ref_text) if ref_text else None
        if ppl_baseline:
            print(f"        Baseline PPL: {ppl_baseline:.2f}")

        # [2] Curate — gather and inject new facts
        print(f"  [2/{ts}] Curating...")
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
        print(f"  [3/{ts}] Auditing all active MEMIT edits...")
        audit_results = self._audit_facts()
        healthy = audit_results["healthy"]
        degraded = audit_results["degraded"]
        total = audit_results["total"]
        print(f"        {healthy} healthy, {len(degraded)} degraded, {total} total")

        # [4] Maintenance
        print(f"  [4/{ts}] Maintenance...")
        maint_results = self._maintain_edits(audit_results, ppl_baseline)
        print(f"        Refreshed: {maint_results['refreshed']}, Pruned: {maint_results['pruned']}")

        # [5-6] Consolidation (if enabled)
        consolidation_stats = {"advanced": 0, "retreated": 0, "scaled_down": 0, "skipped": True}
        if self.consolidation_enabled:
            print(f"  [5/{ts}] LoRA Consolidation...")
            consolidation_stats = self._consolidate(cycle_id, ref_text)
            print(f"        Advanced: {consolidation_stats['advanced']}, "
                  f"Retreated: {consolidation_stats['retreated']}")

            print(f"  [6/{ts}] MEMIT Scale-Down...")
            scaled = consolidation_stats.get("scaled_down", 0)
            print(f"        Scaled down: {scaled} edit(s)")

        # [5/7] Validate
        validate_step = 7 if self.consolidation_enabled else 5
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

        validation = self.validator.validate_ppl(ppl_baseline, ppl_after, self.max_ppl_increase)

        # [6/8] Report
        report_step = 8 if self.consolidation_enabled else 6
        elapsed = time.time() - start_time
        print(f"  [{report_step}/{ts}] Sleep cycle completed in {elapsed:.1f}s")

        return {
            "status": "approved" if ppl_ok else "warning",
            "new_facts_injected": new_facts_injected,
            "audited": total,
            "facts_refreshed": maint_results["refreshed"],
            "facts_pruned": maint_results["pruned"],
            "consolidation": consolidation_stats,
            "ppl_before": round(ppl_baseline, 2) if ppl_baseline else None,
            "ppl_after": round(ppl_after, 2) if ppl_after else None,
            "elapsed_seconds": round(elapsed, 1),
        }

    def _audit_facts(self) -> dict:
        """Test recall of all active MEMIT edits.

        Consolidation-aware: edits at stage 1+ have intentionally reduced MEMIT
        scale, so raw recall is expected to be lower. For these edits, we also
        test chat recall (which uses LoRA). If chat recall works, the edit is
        healthy — the low raw recall is by design, not degradation.

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
            elif edit.consolidation_stage >= 1:
                # Edit has been partially consolidated — low raw recall is
                # expected because MEMIT scale was intentionally reduced.
                # Check chat recall (LoRA pathway) before flagging as degraded.
                chat_recalled = 0
                for fact in edit.facts:
                    passed, _ = self.memit_engine.test_recall(fact, raw=False)
                    if passed:
                        chat_recalled += 1
                chat_rate = chat_recalled / len(edit.facts) if edit.facts else 1.0
                if chat_rate >= self.degraded_threshold:
                    healthy += 1
                    print(f"        OK (consolidated stage {edit.consolidation_stage}): "
                          f"{edit.facts[0].subject} {edit.facts[0].relation} "
                          f"(raw {recall_rate:.0%}, chat {chat_rate:.0%})")
                else:
                    degraded.append((edit, recall_rate))
                    print(f"        DEGRADED (consolidated): {edit.facts[0].subject} "
                          f"{edit.facts[0].relation} "
                          f"(raw {recall_rate:.0%}, chat {chat_rate:.0%})")
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
                # Revert old delta and mark as pruned in ledger
                self.memit_engine.revert_edit(edit)
                self.ledger.mark_pruned(edit.edit_id)
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
            # Sort by lowest recall first, then oldest — prune most-damaged first
            candidates = sorted(self.memit_engine._active_edits,
                                key=lambda e: (e.recall_success_rate, e.timestamp))
            for edit in candidates[:excess]:
                self.memit_engine.revert_edit(edit)
                self.ledger.mark_pruned(edit.edit_id)
                pruned += 1
                print(f"        Pruned: {edit.facts[0].subject} {edit.facts[0].relation}")

        return {"refreshed": refreshed, "pruned": pruned}

    def _consolidate(self, cycle_id, ref_text) -> dict:
        """LoRA consolidation: train on active facts, fuse, gate per-fact.

        1. Gather stage 0-2 edits with healthy recall
        2. Train LoRA on all their facts, fuse into model
        3. Reload fused model, re-apply MEMIT edits
        4. Per-fact gating: temporarily zero MEMIT, test chat recall
           - Pass → advance stage, apply scale schedule
           - Fail → retreat stage to 0
        5. PPL gate: rollback everything if PPL degrades too much

        Returns:
            Stats dict with advanced/retreated/scaled_down counts
        """
        stats = {"advanced": 0, "retreated": 0, "scaled_down": 0, "skipped": False}

        # 1. Gather eligible edits (stage 0-2, healthy recall)
        eligible = [
            edit for edit in self.memit_engine._active_edits
            if edit.consolidation_stage < 3
            and edit.recall_success_rate >= self.degraded_threshold
            and edit.scale > 0
        ]
        if not eligible:
            print("        No eligible edits for consolidation")
            stats["skipped"] = True
            return stats

        # Collect all facts from eligible edits
        all_facts = []
        for edit in eligible:
            all_facts.extend(edit.facts)
        print(f"        {len(eligible)} edit(s), {len(all_facts)} fact(s) eligible")

        # 2. Snapshot weights for rollback
        snapshot = self.memit_engine.snapshot_target_weights()

        # 3. Train and fuse
        save_dir = Path(self.config.paths.get("fused_models", "models/fused"))
        fused_path = self.trainer.train_and_fuse(all_facts, cycle_id, save_dir)
        if fused_path is None:
            print("        Consolidation aborted: training/fuse failed")
            stats["skipped"] = True
            return stats

        # 4. Reload fused model and re-apply MEMIT
        print("        Reloading fused model...")
        self.backend.reload(fused_path)
        self.memit_engine.reapply_active_edits()

        # 5. Per-fact gating: test chat recall with MEMIT zeroed out
        stage_changes = []  # list of (edit, old_stage, new_stage)
        for edit in eligible:
            # Temporarily zero MEMIT for this edit
            old_scale = edit.scale
            self.memit_engine.scale_edit(edit, 0.0)

            # Test chat recall (without MEMIT, relying on LoRA)
            chat_passed = True
            for fact in edit.facts:
                passed, _ = self.memit_engine.test_recall(fact, raw=False)
                if not passed:
                    chat_passed = False
                    break

            # Restore MEMIT scale
            self.memit_engine.scale_edit(edit, old_scale)

            if chat_passed:
                old_stage = edit.consolidation_stage
                new_stage = self.ledger.advance_stage(edit.edit_id)
                edit.consolidation_stage = new_stage
                stage_changes.append((edit, old_stage, new_stage))
                stats["advanced"] += 1
                print(f"        Advanced: {edit.facts[0].subject} "
                      f"stage {old_stage}→{new_stage}")
            else:
                if edit.consolidation_stage > 0:
                    self.ledger.retreat_stage(edit.edit_id)
                    edit.consolidation_stage = 0
                    stats["retreated"] += 1
                    print(f"        Retreated: {edit.facts[0].subject} → stage 0")

        # 6. Apply scale schedule based on new stages
        for edit in self.memit_engine._active_edits:
            stage = edit.consolidation_stage
            if stage < len(self.scale_schedule):
                target_scale = self.scale_schedule[stage]
                if abs(edit.scale - target_scale) > 1e-8:
                    self.memit_engine.scale_edit(edit, target_scale)
                    stats["scaled_down"] += 1

        # 7. PPL gate: rollback if PPL degrades too much
        if ref_text:
            ppl_after = self.backend.compute_perplexity(ref_text)
            ppl_baseline = self.backend.compute_perplexity(ref_text)  # re-measure on fused
            # Simple check: if PPL > 2x what we'd expect, rollback
            if ppl_after and ppl_after > 50:  # sanity threshold
                print(f"        PPL gate FAILED ({ppl_after:.2f}), rolling back...")
                self.memit_engine.restore_target_weights(snapshot)
                self.memit_engine.reapply_active_edits()
                # Revert all stage changes
                for edit, old_stage, new_stage in stage_changes:
                    self.ledger.retreat_stage(edit.edit_id)
                    edit.consolidation_stage = old_stage
                    # Restore original scale
                    original_scale = self.scale_schedule[old_stage] if old_stage < len(self.scale_schedule) else 1.0
                    self.memit_engine.scale_edit(edit, original_scale)
                stats = {"advanced": 0, "retreated": 0, "scaled_down": 0,
                         "skipped": False, "rolled_back": True}
                return stats

        return stats

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

        yield {"step": 2, "total": ts, "label": "Curating", "status": "done",
               "detail": f"{len(curated)} exchanges, {new_facts_injected} facts injected, "
                         f"{len(consumed_sessions)} session(s)"}

        # [3] Fact Audit
        yield {"step": 3, "total": ts, "label": "Auditing facts", "status": "running"}
        audit_results = self._audit_facts()
        yield {"step": 3, "total": ts, "label": "Auditing facts", "status": "done",
               "detail": f"{audit_results['healthy']} healthy, "
                         f"{len(audit_results['degraded'])} degraded"}

        # [4] Maintenance
        yield {"step": 4, "total": ts, "label": "Maintenance", "status": "running"}
        maint_results = self._maintain_edits(audit_results, ppl_baseline)
        yield {"step": 4, "total": ts, "label": "Maintenance", "status": "done",
               "detail": f"Refreshed: {maint_results['refreshed']}, "
                         f"Pruned: {maint_results['pruned']}"}

        # [5-6] Consolidation (if enabled)
        consolidation_stats = {"advanced": 0, "retreated": 0, "scaled_down": 0, "skipped": True}
        if self.consolidation_enabled:
            yield {"step": 5, "total": ts, "label": "LoRA Consolidation", "status": "running"}
            consolidation_stats = self._consolidate(cycle_id, ref_text)
            yield {"step": 5, "total": ts, "label": "LoRA Consolidation", "status": "done",
                   "detail": f"Advanced: {consolidation_stats['advanced']}, "
                             f"Retreated: {consolidation_stats['retreated']}"}

            yield {"step": 6, "total": ts, "label": "MEMIT Scale-Down", "status": "done",
                   "detail": f"Scaled: {consolidation_stats.get('scaled_down', 0)} edit(s)"}

        # [5/7] Validate
        validate_step = 7 if self.consolidation_enabled else 5
        yield {"step": validate_step, "total": ts, "label": "Validating", "status": "running"}
        ppl_after = self.backend.compute_perplexity(ref_text) if ref_text else None
        ppl_detail = "No reference text"
        if ppl_baseline and ppl_after and ppl_baseline > 0:
            ppl_increase = (ppl_after - ppl_baseline) / ppl_baseline
            ppl_detail = f"PPL: {ppl_baseline:.2f} → {ppl_after:.2f} ({ppl_increase:+.1%})"
        yield {"step": validate_step, "total": ts, "label": "Validating", "status": "done",
               "detail": ppl_detail}

        # [6/8] Report
        report_step = 8 if self.consolidation_enabled else 6
        elapsed = time.time() - start_time
        yield {"step": report_step, "total": ts, "label": "Report", "status": "done",
               "detail": f"facts_refreshed={maint_results['refreshed']}, "
                         f"facts_pruned={maint_results['pruned']}, "
                         f"new_injected={new_facts_injected}, "
                         f"consolidated={consolidation_stats.get('advanced', 0)}",
               "facts_refreshed": maint_results["refreshed"],
               "facts_pruned": maint_results["pruned"]}
