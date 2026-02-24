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

        # MEMIT components (set via setters by orchestrator)
        self._extractor = None
        self._memit_engine = None
        self._health_monitor = None

    def set_sleep_callback(self, callback):
        """Register a callback to invoke when sleep is triggered."""
        self._sleep_callback = callback

    def set_nap_callback(self, callback):
        """Register a callback to invoke when a nap is triggered."""
        self._nap_callback = callback

    def set_memit_components(self, extractor, memit_engine, health_monitor):
        """Set MEMIT-related components for wake-phase fact injection."""
        self._extractor = extractor
        self._memit_engine = memit_engine
        self._health_monitor = health_monitor

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

        # MEMIT: extract facts and inject
        self._memit_inject(user_input, response)

        # Check if context needs compaction
        if self.context.needs_compaction():
            self.context.compact()

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

        # MEMIT: extract facts and inject
        self._memit_inject(user_input, response)

        if self.context.needs_compaction():
            self.context.compact()

        # Check sleep/nap triggers
        self._check_sleep_triggers()

    def reset_turn_count(self):
        """Reset turn counter (called after sleep)."""
        self.turn_count = 0

    def _memit_inject(self, user_input, response):
        """Extract facts from exchange and inject via MEMIT."""
        if not self._extractor or not self._memit_engine:
            return

        if not self._memit_engine.enabled:
            return

        try:
            # Extract facts
            triples = self._extractor.extract_from_exchange(user_input, response)
            if not triples:
                return

            # Deduplicate against ledger
            existing = self._memit_engine.ledger.get_facts_for_training()
            triples = self._extractor.deduplicate(triples, existing)
            if not triples:
                return

            # Inject one fact per edit for independent scalability
            injected = 0
            for triple in triples:
                edit = self._memit_engine.inject_fact(triple)
                if edit:
                    injected += 1
            if injected and self._health_monitor:
                self._health_monitor.record_edit(injected)
        except Exception as e:
            # MEMIT injection is non-critical — log and continue
            print(f"  [MEMIT] Injection failed: {e}")

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
