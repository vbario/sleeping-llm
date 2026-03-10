"""Chat interface — the wake phase interaction loop."""


class Chat:
    """Manages the wake-phase chat loop."""

    def __init__(self, backend, context_manager, logger, config):
        self.backend = backend
        self.context = context_manager
        self.logger = logger
        self.config = config
        self.turn_count = 0
        self._sleep_callback = None
        self._nap_callback = None
        self._background_sleep = None  # Set by orchestrator for non-blocking sleep

        # Extraction components (set via setters by orchestrator)
        self._extractor = None
        self._health_monitor = None

        # Consolidation-moment components (set via setter by orchestrator)
        self._fact_buffer = None
        self._surprise_estimator = None

        # Fact ledger (set via setter by orchestrator)
        self._fact_ledger = None

        # Micro-sleep (set via setter by orchestrator)
        self._micro_sleep = None
        self._background_sleep_for_micro = None

        # Saturation tracking — consolidate when fact discovery slows
        self._dry_turns = 0  # Consecutive turns with 0 new facts

    def set_sleep_callback(self, callback):
        """Register a callback to invoke when sleep is triggered."""
        self._sleep_callback = callback

    def set_nap_callback(self, callback):
        """Register a callback to invoke when a nap is triggered."""
        self._nap_callback = callback

    def set_extraction_components(self, extractor, fact_ledger, health_monitor):
        """Set extraction components for wake-phase fact extraction."""
        self._extractor = extractor
        self._fact_ledger = fact_ledger
        self._health_monitor = health_monitor

    def set_consolidation_components(self, fact_buffer, surprise_estimator):
        """Set consolidation-moment components for buffered fact injection."""
        self._fact_buffer = fact_buffer
        self._surprise_estimator = surprise_estimator

    def set_micro_sleep(self, micro_sleep_controller, background_sleep_manager):
        """Set micro-sleep controller for priority-triggered background training."""
        self._micro_sleep = micro_sleep_controller
        self._background_sleep_for_micro = background_sleep_manager

    def process_input(self, user_input):
        """Process a single user input and return the response.

        Returns:
            str: assistant response, or None if a command was handled
        """
        stripped = user_input.strip()

        # Check for manual sleep trigger
        if stripped == self.config.sleep["manual_trigger"]:
            if self._sleep_callback:
                self._sleep_callback("manual")
            return None

        # Check for manual nap trigger
        nap_config = self.config.get("nap", {}) or {}
        if stripped == nap_config.get("manual_trigger", "/nap"):
            if self._nap_callback:
                self._nap_callback("manual")
            return None

        # Check for special commands
        if stripped == "/status":
            return self._status_report()
        if stripped == "/compact":
            self.context.compact()
            return "[Context compacted]"
        if stripped == "/consolidate":
            return self._manual_consolidate()
        if stripped == "/microsleep":
            return self._manual_microsleep()

        # Normal chat flow
        self.context.add_user_message(user_input)

        # Build prompt and generate
        messages = self.context.get_messages()
        prompt = self.backend.apply_chat_template(messages)
        response = self.backend.generate(prompt)

        # Store response in context and log
        self.context.add_assistant_message(response)
        self.logger.log_exchange(user_input, response)
        self.turn_count += 1

        # Extract and buffer facts
        self._extract_and_buffer(user_input, response)

        # Check if context needs compaction
        if self.context.needs_compaction():
            self.context.compact()

        # Check micro-sleep cycle boundary (fires if 90-min cycle expired with pending facts)
        if self._micro_sleep:
            self._micro_sleep.check_cycle(
                background_sleep_manager=self._background_sleep_for_micro,
            )

        # Check sleep/nap triggers
        self._check_sleep_triggers()

        return response

    def _status_report(self):
        """Generate a status report about the current session."""
        token_count = self.context.get_token_count()
        max_tokens = self.context.max_tokens
        usage_pct = (token_count / max_tokens) * 100
        has_summary = self.context.summary is not None

        lines = [
            f"Session: {self.logger.session_id}",
            f"Turns: {self.turn_count}",
            f"Context: {token_count}/{max_tokens} tokens ({usage_pct:.0f}%)",
            f"Compacted summary: {'yes' if has_summary else 'no'}",
            f"Messages in context: {len(self.context.recent_messages)}",
        ]
        if self._fact_buffer:
            buf = self._fact_buffer
            lines.append(f"Fact buffer: {buf.size}/{buf.max_buffer_size}")
            lines.append(f"Consolidations: {buf._consolidation_count}")
        return "\n".join(lines)

    def process_input_stream(self, user_input):
        """Process input with streaming response. Yields token strings.

        After streaming completes, handles logging, context updates,
        and sleep triggers just like process_input().
        """
        stripped = user_input.strip()

        # Commands don't stream
        if stripped == self.config.sleep["manual_trigger"]:
            if self._sleep_callback:
                self._sleep_callback("manual")
            return

        nap_config = self.config.get("nap", {}) or {}
        if stripped == nap_config.get("manual_trigger", "/nap"):
            if self._nap_callback:
                self._nap_callback("manual")
            return

        if stripped == "/status":
            yield self._status_report()
            return
        if stripped == "/compact":
            self.context.compact()
            yield "[Context compacted]"
            return
        if stripped == "/consolidate":
            yield self._manual_consolidate()
            return
        if stripped == "/microsleep":
            yield self._manual_microsleep()
            return

        self.context.add_user_message(user_input)
        messages = self.context.get_messages()
        prompt = self.backend.apply_chat_template(messages)

        full_response = []
        for token in self.backend.generate_stream(prompt):
            full_response.append(token)
            yield token

        response = "".join(full_response)
        self.context.add_assistant_message(response)
        self.logger.log_exchange(user_input, response)
        self.turn_count += 1

        # Extract and buffer facts
        self._extract_and_buffer(user_input, response)

        if self.context.needs_compaction():
            self.context.compact()

        # Check micro-sleep cycle boundary
        if self._micro_sleep:
            self._micro_sleep.check_cycle(
                background_sleep_manager=self._background_sleep_for_micro,
            )

        # Check sleep/nap triggers
        self._check_sleep_triggers()

    def reset_turn_count(self):
        """Reset turn counter (called after sleep)."""
        self.turn_count = 0

    def _extract_and_buffer(self, user_input, response):
        """Extract facts from conversation and buffer them.

        Reviews the full conversation (not just latest message) so the model
        has context to resolve pronouns and accumulate facts. Consolidates
        when fact discovery slows down (saturation).

        If no fact buffer is configured, falls back to direct ledger write.
        """
        if not self._extractor or not self._fact_ledger:
            return

        # Fall back to direct ledger write if buffer not configured
        if not self._fact_buffer:
            self._direct_persist(user_input, response)
            return

        try:
            # Extract facts from full conversation (gives model context)
            conversation = self.context.get_messages()
            new_facts = self._extractor.extract_from_exchange(
                user_input, response, conversation=conversation,
            )

            if not new_facts:
                self._dry_turns += 1
                self._check_saturation()
                return

            # Deduplicate against ledger
            existing = self._fact_ledger.get_all_qa_pairs()
            new_facts = self._extractor.deduplicate(new_facts, existing)

            # Also deduplicate against buffer contents
            buffered_facts = self._fact_buffer.get_qa_pairs()
            if buffered_facts:
                new_facts = self._extractor.deduplicate(new_facts, buffered_facts)

            # Compute surprise for priority assignment
            surprise_score = 0.5
            if self._surprise_estimator:
                surprise_score = self._surprise_estimator.evaluate(
                    user_input, new_facts, len(new_facts),
                )

            # Buffer surviving facts with priority from surprise
            for qa in new_facts:
                qa.priority = surprise_score
                self._fact_buffer.add(qa, turn=self.turn_count, surprise=surprise_score)

            # Track discovery rate
            if new_facts:
                self._dry_turns = 0
                print(f"  [Buffer] +{len(new_facts)} new fact(s), "
                      f"buffer={self._fact_buffer.size}, "
                      f"priority={surprise_score:.2f}")

                # Trigger micro-sleep for high-priority facts
                if self._micro_sleep:
                    self._micro_sleep.maybe_trigger(
                        surprise_score,
                        background_sleep_manager=self._background_sleep_for_micro,
                    )
            else:
                self._dry_turns += 1

            self._check_saturation()

        except Exception as e:
            print(f"  [Buffer] Extraction/buffering failed: {e}")

    def _check_saturation(self):
        """Consolidate when fact discovery slows — the 'mental break' trigger.

        Fires when: buffer has facts AND discovery has stalled for 2+ turns.
        """
        if self._fact_buffer.is_empty:
            return
        min_dry = 2
        if self._dry_turns >= min_dry:
            print(f"  [Saturation] {self._dry_turns} dry turns, "
                  f"consolidating {self._fact_buffer.size} fact(s)")
            self._fact_buffer.consolidate(reason="saturation")
            self._dry_turns = 0

    def _direct_persist(self, user_input, response):
        """Direct ledger write (used when consolidation_moment disabled)."""
        try:
            new_facts = self._extractor.extract_from_exchange(user_input, response)
            if not new_facts:
                return

            existing = self._fact_ledger.get_all_qa_pairs()
            new_facts = self._extractor.deduplicate(new_facts, existing)
            if not new_facts:
                return

            for qa in new_facts:
                self._fact_ledger.add_fact(qa)
            if self._health_monitor:
                self._health_monitor.record_new_facts(len(new_facts))
        except Exception as e:
            print(f"  [Persist] Direct write failed: {e}")

    def _manual_microsleep(self):
        """Handle /microsleep command — force a micro-sleep pass on top-priority facts."""
        if not self._micro_sleep:
            return "[Micro-sleep not enabled. Set micro_sleep.enabled: true in config]"
        if self._micro_sleep.is_running:
            return "[Micro-sleep already running]"
        if self._background_sleep_for_micro and self._background_sleep_for_micro.is_sleeping:
            return "[Can't micro-sleep during full sleep/nap]"

        # Force trigger with priority 1.0 (bypasses threshold check)
        started = self._micro_sleep.maybe_trigger(
            1.0, background_sleep_manager=self._background_sleep_for_micro,
        )
        if started:
            return "[Micro-sleep started in background]"
        return "[No eligible facts for micro-sleep (all recently trained or cooldown active)]"

    def _manual_consolidate(self):
        """Handle /consolidate command."""
        if not self._fact_buffer:
            return "[Consolidation moments not enabled]"
        if self._fact_buffer.is_empty:
            return "[Buffer is empty — nothing to consolidate]"
        count = self._fact_buffer.consolidate(reason="manual")
        return f"[Consolidated {count} fact(s) into ledger]"

    def _check_sleep_triggers(self):
        """Check if nap or sleep should be triggered based on health or turn count."""
        trigger_mode = self.config.sleep.get("trigger_mode", "turns")

        if trigger_mode == "health" and self._health_monitor:
            # Health-based triggers
            if self._health_monitor.should_sleep():
                if self._sleep_callback:
                    self._sleep_callback("health")
                    return
            if self._health_monitor.should_nap():
                if self._nap_callback:
                    self._nap_callback("health")
                    return
            # Also check turn-based as fallback
            light_threshold = self.config.sleep.get("light_sleep_turns", 10)
            if self.turn_count > 0 and self.turn_count % light_threshold == 0:
                if self._sleep_callback:
                    self._sleep_callback("auto")
        else:
            # Turn-based triggers (original behavior)
            light_threshold = self.config.sleep.get("light_sleep_turns", 10)
            if self.turn_count > 0 and self.turn_count % light_threshold == 0:
                if self._sleep_callback:
                    self._sleep_callback("auto")
