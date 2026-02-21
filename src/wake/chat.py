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

    def set_sleep_callback(self, callback):
        """Register a callback to invoke when sleep is triggered."""
        self._sleep_callback = callback

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

        # Check if context needs compaction
        if self.context.needs_compaction():
            self.context.compact()

        # Check if it's time for automatic sleep
        light_threshold = self.config.sleep["light_sleep_turns"]
        if self.turn_count > 0 and self.turn_count % light_threshold == 0:
            if self._sleep_callback:
                self._sleep_callback("auto")

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

        if self.context.needs_compaction():
            self.context.compact()

        light_threshold = self.config.sleep["light_sleep_turns"]
        if self.turn_count > 0 and self.turn_count % light_threshold == 0:
            if self._sleep_callback:
                self._sleep_callback("auto")

    def reset_turn_count(self):
        """Reset turn counter (called after sleep)."""
        self.turn_count = 0
