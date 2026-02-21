"""Curator — scores and filters conversation data for training."""

import json
import os
import re
from pathlib import Path


from src.sleep.firewall import HallucinationFirewall


class Curator:
    """Evaluates conversation exchanges and prepares training data.

    Acts as the 'amygdala' — decides what's worth remembering.
    """

    def __init__(self, config, backend):
        self.config = config
        self.backend = backend
        self.training_dir = Path(config.paths["training"])
        self.training_dir.mkdir(parents=True, exist_ok=True)
        self.firewall = HallucinationFirewall(config, backend)

    def curate_session(self, messages, sleep_cycle_id):
        """Score and filter a session's messages into training examples.

        Args:
            messages: List of {role, content} dicts from the session
            sleep_cycle_id: Identifier for this sleep cycle

        Returns:
            List of curated training examples with scores
        """
        exchanges = self._pair_exchanges(messages)
        scored = []

        for exchange in exchanges:
            score = self._score_exchange(exchange)
            if self._passes_threshold(score):
                scored.append({
                    "messages": exchange,
                    "scores": score,
                    "combined": score["combined"],
                })

        # Sort by combined score, highest first
        scored.sort(key=lambda x: x["combined"], reverse=True)

        # Save curated data
        output_dir = self.training_dir / f"cycle_{sleep_cycle_id}"
        self._save_training_data(scored, output_dir)

        return scored

    def _pair_exchanges(self, messages):
        """Group messages into user/assistant exchange pairs."""
        exchanges = []
        i = 0
        while i < len(messages) - 1:
            if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
                exchanges.append([messages[i], messages[i + 1]])
                i += 2
            else:
                i += 1
        return exchanges

    def _score_exchange(self, exchange):
        """Score an exchange on multiple dimensions.

        Uses heuristics for speed. For deeper scoring, the model itself
        could evaluate each exchange (more expensive but more accurate).
        """
        user_msg = exchange[0]["content"]
        assistant_msg = exchange[1]["content"]

        novelty = self._score_novelty(user_msg, assistant_msg)
        importance = self._score_importance(user_msg, assistant_msg)
        utility = self._score_utility(user_msg, assistant_msg)

        combined = (novelty + importance + utility) / 3.0

        return {
            "novelty": novelty,
            "importance": importance,
            "utility": utility,
            "combined": combined,
        }

    def _score_novelty(self, user_msg, assistant_msg):
        """Estimate how novel/surprising this exchange is.

        Heuristic: longer, more specific messages tend to contain more novel info.
        Questions score higher. Technical content scores higher.
        """
        score = 0.3  # baseline

        # Longer messages likely contain more substance
        word_count = len(user_msg.split())
        if word_count > 20:
            score += 0.2
        if word_count > 50:
            score += 0.1

        # Questions are often seeking new info
        if "?" in user_msg:
            score += 0.1

        # Technical/specific indicators
        technical_markers = [
            "error", "bug", "fix", "implement", "function", "class",
            "database", "api", "config", "install", "version",
            "because", "reason", "explain", "actually", "specifically",
        ]
        msg_lower = user_msg.lower()
        marker_count = sum(1 for m in technical_markers if m in msg_lower)
        score += min(marker_count * 0.05, 0.2)

        return min(score, 1.0)

    def _score_importance(self, user_msg, assistant_msg):
        """Estimate how important this exchange is.

        Heuristic: corrections, preferences, and explicit instructions score high.
        """
        score = 0.3  # baseline
        msg_lower = user_msg.lower()

        # Corrections suggest the model needs to learn
        correction_markers = [
            "no,", "wrong", "incorrect", "actually", "not what i",
            "i meant", "that's not", "please don't", "stop",
            "remember", "always", "never", "prefer", "i want",
        ]
        for marker in correction_markers:
            if marker in msg_lower:
                score += 0.15
                break

        # Explicit preferences
        preference_markers = [
            "i like", "i prefer", "i use", "i work with",
            "my name", "i am", "i'm a", "i live",
        ]
        for marker in preference_markers:
            if marker in msg_lower:
                score += 0.2
                break

        # Emphasis (caps, exclamation)
        if any(word.isupper() and len(word) > 2 for word in user_msg.split()):
            score += 0.1
        if "!" in user_msg:
            score += 0.05

        return min(score, 1.0)

    def _score_utility(self, user_msg, assistant_msg):
        """Estimate future utility — will this come up again?

        Heuristic: general knowledge and patterns > one-off questions.
        """
        score = 0.3  # baseline
        msg_lower = user_msg.lower()

        # How-to and procedural knowledge has high reuse
        if any(p in msg_lower for p in ["how to", "how do", "what is", "why does"]):
            score += 0.2

        # References to ongoing work/project
        project_markers = [
            "my project", "our app", "the codebase", "we're building",
            "my setup", "my workflow",
        ]
        for marker in project_markers:
            if marker in msg_lower:
                score += 0.15
                break

        # Short trivial messages have low utility
        if len(user_msg.split()) < 5:
            score -= 0.1

        return max(min(score, 1.0), 0.0)

    def _passes_threshold(self, score):
        """Check if an exchange meets minimum curation thresholds."""
        thresholds = self.config.sleep["curation"]
        if score["novelty"] < thresholds["min_novelty_score"]:
            return False
        if score["importance"] < thresholds["min_importance_score"]:
            return False
        if score["combined"] < thresholds["min_combined_score"]:
            return False
        return True

    def _save_training_data(self, scored_exchanges, output_dir):
        """Save curated exchanges as training data in MLX format.

        Generates both raw conversation examples AND extracted Q&A fact pairs.
        The fact pairs are the primary mechanism for teaching the model to
        recall specific information.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_examples = []

        # Extract facts and generate Q&A pairs from the full conversation
        all_messages = []
        for item in scored_exchanges:
            all_messages.extend(item["messages"])

        print("        Extracting facts from conversation...")
        fact_pairs = self._extract_facts_as_qa(all_messages)
        print(f"        Generated {len(fact_pairs)} fact Q&A pairs")

        # Run hallucination firewall
        conv_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in all_messages)
        fact_pairs, rejected = self.firewall.verify_pairs(fact_pairs, conv_text)
        print(f"        Firewall: {len(fact_pairs)} verified, {len(rejected)} rejected")
        for r in rejected:
            q = r["pair"][0]["content"][:60]
            a = r["pair"][1]["content"][:60]
            print(f"          REJECTED ({r['grounding_score']:.2f}): Q: {q}... A: {a}...")

        for qa in fact_pairs:
            text = self.backend.apply_chat_template(qa, for_training=True)
            all_examples.append({"text": text})

        # Also include raw exchanges (but these are secondary)
        for item in scored_exchanges:
            messages = item["messages"]
            text = self.backend.apply_chat_template(messages, for_training=True)
            all_examples.append({"text": text})

        train_file = output_dir / "train.jsonl"
        with open(train_file, "w") as f:
            for ex in all_examples:
                f.write(json.dumps(ex) + "\n")

        # Validation split
        valid_count = max(1, len(all_examples) // 10)
        valid_file = output_dir / "valid.jsonl"
        with open(valid_file, "w") as f:
            for ex in all_examples[-valid_count:]:
                f.write(json.dumps(ex) + "\n")

        return output_dir

    def _extract_facts_as_qa(self, messages):
        """Extract facts and generate Q&A training pairs.

        Uses model-based extraction first, then falls back to template-based
        pattern matching if the model returns nothing parseable.
        """
        # Build conversation text
        conv_text = ""
        for msg in messages:
            conv_text += f"{msg['role'].upper()}: {msg['content']}\n\n"

        if len(conv_text) > 3000:
            conv_text = conv_text[:3000]

        # Try model-based extraction first
        pairs = self._extract_facts_model(conv_text)
        if pairs:
            print(f"        Model extraction: {len(pairs)} pairs")
            return pairs

        # Fall back to template-based extraction
        print("        Model extraction returned 0 pairs, using template fallback")
        pairs = self._extract_facts_template(messages)
        print(f"        Template extraction: {len(pairs)} pairs")
        return pairs

    def _extract_facts_model(self, conv_text):
        """Try to extract facts using the model."""
        extraction_prompt = [
            {
                "role": "user",
                "content": (
                    "Read this conversation and extract specific facts shared by the user. "
                    "Write question-answer pairs.\n\n"
                    "Example format:\n"
                    "Q: What is the user's name?\n"
                    "A: The user's name is John.\n\n"
                    "Q: What does the user do for work?\n"
                    "A: The user is a software engineer.\n\n"
                    "Now extract facts from this conversation:\n"
                    f"{conv_text}\n\n"
                    "Q:"
                ),
            }
        ]

        prompt = self.backend.apply_chat_template(extraction_prompt)
        raw = self.backend.generate(prompt, max_tokens=1000, temperature=0.3)

        # Log raw output for debugging
        print(f"        [DEBUG] Model extraction raw output ({len(raw)} chars):")
        for line in raw.strip().split("\n")[:10]:
            print(f"          | {line}")
        if len(raw.strip().split("\n")) > 10:
            print(f"          | ... ({len(raw.strip().split(chr(10)))} lines total)")

        # Parse — prepend "Q:" since the prompt ends with it
        raw = "Q:" + raw

        pairs = []
        lines = raw.strip().split("\n")
        current_q = None
        current_a_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped.upper().startswith("Q:") or stripped.upper().startswith("Q :"):
                if current_q and current_a_lines:
                    answer = " ".join(current_a_lines).strip()
                    if answer:
                        pairs.append([
                            {"role": "user", "content": current_q},
                            {"role": "assistant", "content": answer},
                        ])
                q_text = re.sub(r'^[Qq]\s*:\s*', '', stripped).strip()
                current_q = q_text
                current_a_lines = []
            elif stripped.upper().startswith("A:") or stripped.upper().startswith("A :"):
                a_text = re.sub(r'^[Aa]\s*:\s*', '', stripped).strip()
                current_a_lines.append(a_text)
            elif current_a_lines:
                current_a_lines.append(stripped)

        if current_q and current_a_lines:
            answer = " ".join(current_a_lines).strip()
            if answer:
                pairs.append([
                    {"role": "user", "content": current_q},
                    {"role": "assistant", "content": answer},
                ])

        return pairs

    def _extract_facts_template(self, messages):
        """Extract facts using pattern matching on user messages.

        Generates Q&A pairs directly from recognized patterns in user text.
        More reliable than model-based extraction for small models.
        """
        pairs = []
        seen = set()  # avoid duplicate facts

        # Patterns: (regex, question_template, answer_template)
        # Group 1 in regex is the extracted value
        patterns = [
            # Names
            (r"(?:my name is|i'm called|call me|i am)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
             "What is the user's name?",
             "The user's name is {0}."),
            # Age
            (r"(?:i'm|i am|i'm)\s+(\d{1,3})\s+(?:years old|yr|yrs)",
             "How old is the user?",
             "The user is {0} years old."),
            # Location
            (r"(?:i live in|i'm from|i'm based in|i am from|i am based in|i live at)\s+(.+?)(?:\.|,|!|\?|$)",
             "Where does the user live?",
             "The user lives in {0}."),
            # Job/profession
            (r"(?:i work as|i'm a|i am a|my job is|i work at|i work for)\s+(.+?)(?:\.|,|!|\?|$)",
             "What does the user do?",
             "The user is a {0}."),
            # Likes/preferences
            (r"(?:i (?:really )?like|i love|i enjoy|i prefer)\s+(.+?)(?:\.|,|!|\?|$)",
             "What does the user like?",
             "The user likes {0}."),
            # Dislikes
            (r"(?:i (?:don't|do not|don't) like|i hate|i dislike)\s+(.+?)(?:\.|,|!|\?|$)",
             "What does the user dislike?",
             "The user dislikes {0}."),
            # Favorites
            (r"(?:my (?:favorite|favourite))\s+(\w+)\s+(?:is|are)\s+(.+?)(?:\.|,|!|\?|$)",
             "What is the user's favorite {0}?",
             "The user's favorite {0} is {1}."),
            # Has/owns
            (r"(?:i have|i've got|i own)\s+(?:a |an )?(.+?)(?:\.|,|!|\?|$)",
             "What does the user have?",
             "The user has {0}."),
            # Family
            (r"(?:my (?:son|daughter|wife|husband|partner|brother|sister|mom|dad|mother|father)(?:'s name)? is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
             "What is the name of the user's family member?",
             "The user's family member is named {0}."),
            (r"(?:my (\w+)(?:'s name)? is)\s+([A-Z][a-z]+)",
             "What is the user's {0}'s name?",
             "The user's {0}'s name is {1}."),
            # Uses/works with
            (r"(?:i use|i work with|i'm using|i am using)\s+(.+?)(?:\.|,|!|\?|$)",
             "What does the user use?",
             "The user uses {0}."),
        ]

        for msg in messages:
            if msg["role"] != "user":
                continue
            text = msg["content"]

            for pattern, q_template, a_template in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    groups = match.groups()
                    # Clean up captured values
                    groups = tuple(g.strip() if g else g for g in groups)

                    try:
                        question = q_template.format(*groups)
                        answer = a_template.format(*groups)
                    except (IndexError, KeyError):
                        continue

                    # Skip very short or likely garbage matches
                    value = groups[0] if groups else ""
                    if len(value) < 2 or len(value) > 100:
                        continue

                    key = (question, answer)
                    if key not in seen:
                        seen.add(key)
                        pairs.append([
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": answer},
                        ])

        # Also generate direct recall pairs from user statements
        # These teach the model to recall what the user said verbatim
        for msg in messages:
            if msg["role"] != "user":
                continue
            text = msg["content"].strip()
            # Skip short messages and commands
            if len(text) < 15 or text.startswith("/"):
                continue
            # If message contains personal info markers, create a recall pair
            lower = text.lower()
            personal_markers = [
                "my name", "i am", "i'm", "i live", "i work", "i like",
                "i have", "i use", "my favorite", "my favourite",
                "remember", "i prefer", "i want you to",
            ]
            if any(m in lower for m in personal_markers):
                pairs.append([
                    {"role": "user", "content": f"What did the user tell you about themselves?"},
                    {"role": "assistant", "content": f"The user said: \"{text}\""},
                ])

        return pairs

    def curate_with_model(self, messages, sleep_cycle_id):
        """Use the model itself to score exchanges (slower, more accurate).

        This is a deeper curation pass suitable for deep sleep cycles.
        """
        exchanges = self._pair_exchanges(messages)
        scored = []

        scoring_prompt_template = (
            "Rate the following conversation exchange on a scale of 0-10 for:\n"
            "1. NOVELTY: How new or surprising is the information?\n"
            "2. IMPORTANCE: How important is this to remember?\n"
            "3. UTILITY: How likely is this to be useful in future conversations?\n\n"
            "Exchange:\nUser: {user}\nAssistant: {assistant}\n\n"
            "Reply with only three numbers separated by commas (novelty,importance,utility):"
        )

        for exchange in exchanges:
            prompt_text = scoring_prompt_template.format(
                user=exchange[0]["content"],
                assistant=exchange[1]["content"],
            )
            prompt_messages = [{"role": "user", "content": prompt_text}]
            prompt = self.backend.apply_chat_template(prompt_messages)
            raw = self.backend.generate(prompt, max_tokens=20, temperature=0.1)

            try:
                parts = raw.strip().split(",")
                novelty = float(parts[0].strip()) / 10.0
                importance = float(parts[1].strip()) / 10.0
                utility = float(parts[2].strip()) / 10.0
            except (ValueError, IndexError):
                # Fall back to heuristic scoring
                score = self._score_exchange(exchange)
                novelty = score["novelty"]
                importance = score["importance"]
                utility = score["utility"]

            combined = (novelty + importance + utility) / 3.0
            score = {
                "novelty": novelty,
                "importance": importance,
                "utility": utility,
                "combined": combined,
            }

            if self._passes_threshold(score):
                scored.append({
                    "messages": exchange,
                    "scores": score,
                    "combined": combined,
                })

        scored.sort(key=lambda x: x["combined"], reverse=True)
        output_dir = self.training_dir / f"cycle_{sleep_cycle_id}"
        self._save_training_data(scored, output_dir)
        return scored
