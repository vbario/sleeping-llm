"""Context manager — tracks the active context window and handles compaction."""


class ContextManager:
    """Manages the conversation context window with automatic compaction."""

    def __init__(self, config, backend):
        self.config = config
        self.backend = backend
        self.max_tokens = config.context["max_tokens"]
        self.compaction_threshold = config.context["compaction_threshold"]
        self.system_prompt = config.context["system_prompt"]

        # Active context: what the model sees right now
        self.summary = None  # compressed history from prior compactions
        self.recent_messages = []  # messages since last compaction

    def get_messages(self):
        """Build the full message list for inference."""
        messages = [{"role": "system", "content": self._build_system_content()}]
        messages.extend(self.recent_messages)
        return messages

    def _build_system_content(self):
        """Combine system prompt with any compacted summary."""
        if self.summary:
            return f"{self.system_prompt}\n\nPrevious conversation summary:\n{self.summary}"
        return self.system_prompt

    def add_user_message(self, content):
        """Add a user message to the active context."""
        self.recent_messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content):
        """Add an assistant response to the active context."""
        self.recent_messages.append({"role": "assistant", "content": content})

    def get_token_count(self):
        """Count tokens in the current active context."""
        return self.backend.count_tokens(self.get_messages())

    def needs_compaction(self):
        """Check if context is approaching the limit."""
        token_count = self.get_token_count()
        return token_count > self.max_tokens * self.compaction_threshold

    def compact(self):
        """Compress the context by summarizing older messages.

        Keeps the most recent messages and summarizes the rest.
        """
        if len(self.recent_messages) < 4:
            return  # not enough to compact

        # Keep the last few exchanges, summarize the rest
        keep_count = 4  # keep last 2 exchanges (4 messages)
        to_summarize = self.recent_messages[:-keep_count]
        to_keep = self.recent_messages[-keep_count:]

        # Build summarization prompt
        summary_messages = [
            {
                "role": "system",
                "content": (
                    "Summarize the following conversation concisely. "
                    "Preserve all key facts, decisions, user preferences, "
                    "corrections, and important context. Be thorough but brief."
                ),
            }
        ]
        if self.summary:
            summary_messages.append(
                {"role": "user", "content": f"Prior summary:\n{self.summary}"}
            )
            summary_messages.append(
                {"role": "assistant", "content": "Understood, I'll incorporate that."}
            )
        summary_messages.extend(to_summarize)
        summary_messages.append(
            {"role": "user", "content": "Please summarize the above conversation."}
        )

        prompt = self.backend.apply_chat_template(summary_messages)
        new_summary = self.backend.generate(prompt, max_tokens=300, temperature=0.3)

        self.summary = new_summary.strip()
        self.recent_messages = to_keep

    def reset(self, keep_summary=True):
        """Reset context for a new wake cycle (post-sleep)."""
        if not keep_summary:
            self.summary = None
        self.recent_messages = []

    def get_full_history_text(self):
        """Get all current context as a single text block (for debugging)."""
        messages = self.get_messages()
        parts = []
        for msg in messages:
            parts.append(f"[{msg['role']}]: {msg['content']}")
        return "\n\n".join(parts)
