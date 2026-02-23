"""Dreamer — REM-equivalent synthetic data generation.

During deep sleep, the dreamer generates synthetic Q&A pairs by
creatively recombining knowledge the model has learned. This strengthens
associations between related concepts and surfaces contradictions.
"""

import json
from pathlib import Path


class Dreamer:
    """Generates synthetic training data from learned knowledge."""

    def __init__(self, config, backend):
        self.config = config
        self.backend = backend
        self.num_dreams = config.dreamer["num_dreams"]
        self.temperature = config.dreamer["temperature"]

    def dream(self, recent_exchanges):
        """Generate synthetic Q&A pairs from recent learned knowledge.

        Args:
            recent_exchanges: List of recent conversation exchanges
                              (list of [user_msg, assistant_msg] pairs)

        Returns:
            List of synthetic training examples [{"messages": [...]}]
        """
        if not recent_exchanges:
            return []

        # Extract key topics from recent exchanges
        topics = self._extract_topics(recent_exchanges)
        if not topics:
            return []

        dreams = []

        # Generate synthetic Q&A for each topic combination
        for i in range(min(self.num_dreams, len(topics))):
            topic = topics[i % len(topics)]
            dream = self._generate_dream(topic, recent_exchanges)
            if dream:
                dreams.append(dream)

        # Generate cross-topic association dreams
        if len(topics) >= 2:
            for i in range(min(self.num_dreams // 2, len(topics) - 1)):
                dream = self._generate_association_dream(
                    topics[i], topics[i + 1], recent_exchanges
                )
                if dream:
                    dreams.append(dream)

        return dreams

    def _extract_topics(self, exchanges):
        """Use the model to extract key topics from exchanges."""
        # Build a summary of recent exchanges
        exchange_texts = []
        for ex in exchanges[:10]:  # limit to recent 10
            messages = ex if isinstance(ex, list) else ex.get("messages", [])
            for msg in messages:
                if msg["role"] == "user":
                    exchange_texts.append(msg["content"][:200])

        combined = "\n".join(exchange_texts)

        prompt_messages = [
            {
                "role": "user",
                "content": (
                    "Extract the 5 most important topics or concepts from "
                    "these conversation snippets. Return each topic on its own line, "
                    "nothing else.\n\n"
                    f"{combined}"
                ),
            }
        ]
        prompt = self.backend.apply_chat_template(prompt_messages)
        response = self.backend.generate(prompt, max_tokens=150, temperature=0.3)

        topics = [
            line.strip().lstrip("0123456789.-) ")
            for line in response.strip().split("\n")
            if line.strip()
        ]
        return topics[:5]

    def _generate_dream(self, topic, context_exchanges):
        """Generate a synthetic Q&A about a topic."""
        prompt_messages = [
            {
                "role": "user",
                "content": (
                    f"Generate a natural question and detailed answer about: {topic}\n\n"
                    "The question should be something a curious user would ask. "
                    "The answer should be informative and accurate.\n\n"
                    "Format:\nQ: [question]\nA: [answer]"
                ),
            }
        ]
        prompt = self.backend.apply_chat_template(prompt_messages)
        response = self.backend.generate(
            prompt, max_tokens=300, temperature=self.temperature
        )

        return self._parse_qa(response)

    def _generate_association_dream(self, topic1, topic2, context_exchanges):
        """Generate a synthetic Q&A connecting two topics."""
        prompt_messages = [
            {
                "role": "user",
                "content": (
                    f"Generate a question and answer that connects these two topics:\n"
                    f"1. {topic1}\n2. {topic2}\n\n"
                    "Find an interesting relationship or connection between them. "
                    "The answer should explain the connection clearly.\n\n"
                    "Format:\nQ: [question]\nA: [answer]"
                ),
            }
        ]
        prompt = self.backend.apply_chat_template(prompt_messages)
        response = self.backend.generate(
            prompt, max_tokens=300, temperature=self.temperature
        )

        return self._parse_qa(response)

    def _parse_qa(self, text):
        """Parse a Q:/A: formatted response into message pairs."""
        lines = text.strip().split("\n")
        question = None
        answer_lines = []
        in_answer = False

        for line in lines:
            if line.strip().startswith("Q:"):
                question = line.strip()[2:].strip()
            elif line.strip().startswith("A:"):
                in_answer = True
                answer_lines.append(line.strip()[2:].strip())
            elif in_answer:
                answer_lines.append(line.strip())

        if question and answer_lines:
            answer = " ".join(answer_lines).strip()
            return {
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            }
        return None

    def dream_integration(self, facts, recent_exchanges=None):
        """Generate multi-fact integration conversations for REM phase.

        Groups facts into sets of 2-4 and generates naturalistic conversations
        that weave multiple facts together, strengthening cross-fact associations.

        Args:
            facts: List of FactTriple objects (consolidated facts)
            recent_exchanges: Optional list of recent curated exchanges for context

        Returns:
            List of {"messages": [...]} dicts (same format as dream())
        """
        if not facts:
            return []

        rem_config = self.config.rem
        num_integrations = rem_config.get("num_integrations", 10)
        temperature = rem_config.get("temperature", 0.8)

        results = []
        remaining = list(facts)

        # Group facts into sets of 2-4 for multi-fact dialogues
        while len(remaining) >= 2 and len(results) < num_integrations:
            group_size = min(4, len(remaining), max(2, len(remaining) // max(1, num_integrations - len(results))))
            group = remaining[:group_size]
            remaining = remaining[group_size:]

            dialogue = self._generate_multifact_dialogue(group, temperature)
            if dialogue:
                results.append(dialogue)

        # Generate contextual narratives for any remaining singles
        for fact in remaining:
            if len(results) >= num_integrations:
                break
            narrative = self._generate_contextual_narrative(fact, facts, temperature)
            if narrative:
                results.append(narrative)

        return results

    def _generate_multifact_dialogue(self, facts, temperature=0.8):
        """Generate a natural multi-turn conversation incorporating multiple facts.

        Args:
            facts: List of 2-4 FactTriple objects
            temperature: Sampling temperature

        Returns:
            {"messages": [...]} dict or None
        """
        fact_descriptions = []
        for i, f in enumerate(facts, 1):
            fact_descriptions.append(f"{i}. {f.subject} {f.relation} {f.object}")
        facts_text = "\n".join(fact_descriptions)

        prompt_messages = [
            {
                "role": "user",
                "content": (
                    "Generate a natural multi-turn conversation between a user and an assistant "
                    "that naturally incorporates ALL of these facts:\n\n"
                    f"{facts_text}\n\n"
                    "Requirements:\n"
                    "- The conversation should feel natural, not like a quiz\n"
                    "- The user asks about topics related to these facts\n"
                    "- The assistant weaves the facts into informative responses\n"
                    "- 2-4 turns (each turn = one user message + one assistant response)\n\n"
                    "Format each turn exactly as:\n"
                    "User: [message]\n"
                    "Assistant: [response]\n"
                ),
            }
        ]
        prompt = self.backend.apply_chat_template(prompt_messages)
        response = self.backend.generate(prompt, max_tokens=500, temperature=temperature)

        return self._parse_dialogue(response)

    def _generate_contextual_narrative(self, fact, context_facts=None, temperature=0.8):
        """Generate a conversational exchange where a fact appears naturally in context.

        Args:
            fact: A single FactTriple
            context_facts: Optional list of other facts for context
            temperature: Sampling temperature

        Returns:
            {"messages": [...]} dict or None
        """
        context_hint = ""
        if context_facts:
            # Pick a couple of related facts for richer context
            others = [f for f in context_facts if f is not fact][:2]
            if others:
                hints = [f"{f.subject} {f.relation} {f.object}" for f in others]
                context_hint = f"\nRelated context: {'; '.join(hints)}"

        prompt_messages = [
            {
                "role": "user",
                "content": (
                    "Generate a natural conversation between a user and assistant where "
                    "the following fact comes up organically in the assistant's response:\n\n"
                    f"Fact: {fact.subject} {fact.relation} {fact.object}\n"
                    f"{context_hint}\n\n"
                    "The user should ask a natural question (not 'what is X?') and "
                    "the assistant should mention the fact as part of a broader, "
                    "informative response.\n\n"
                    "Format:\n"
                    "User: [message]\n"
                    "Assistant: [response]"
                ),
            }
        ]
        prompt = self.backend.apply_chat_template(prompt_messages)
        response = self.backend.generate(prompt, max_tokens=300, temperature=temperature)

        return self._parse_dialogue(response)

    def _parse_dialogue(self, text):
        """Parse a User:/Assistant: formatted dialogue into message pairs."""
        messages = []
        lines = text.strip().split("\n")
        current_role = None
        current_content = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("User:"):
                if current_role and current_content:
                    messages.append({"role": current_role, "content": " ".join(current_content).strip()})
                current_role = "user"
                current_content = [stripped[5:].strip()]
            elif stripped.startswith("Assistant:"):
                if current_role and current_content:
                    messages.append({"role": current_role, "content": " ".join(current_content).strip()})
                current_role = "assistant"
                current_content = [stripped[10:].strip()]
            elif current_role and stripped:
                current_content.append(stripped)

        # Flush last message
        if current_role and current_content:
            messages.append({"role": current_role, "content": " ".join(current_content).strip()})

        # Validate: need at least one user + one assistant message
        roles = [m["role"] for m in messages]
        if "user" not in roles or "assistant" not in roles:
            return None

        return {"messages": messages}

    def dream_to_training_data(self, dreams, backend):
        """Convert dream outputs to training JSONL format."""
        results = []
        for dream in dreams:
            if dream and "messages" in dream:
                text = backend.apply_chat_template(dream["messages"], for_training=True)
                results.append({"text": text})
        return results
