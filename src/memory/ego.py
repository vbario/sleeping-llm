"""Self-model — the persistent ego module (notes/131 §8 step 4).

A small JSON self-model {who_am_i, user_profile, current_goals,
integrity_conditions, updated_at} seeded from the frozen charter, consulted
at valuation time (context block for ego policies), and rewritten post-sleep
from the cycle's session text. Every update is fully logged to a JSONL file
next to the model file.

Kept deliberately separate from IdentityManager (its dormant train-mixing
path is untouched; see ego.train_identity flag).
"""

import json
import re
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EGO_PROMPTS_PATH = REPO_ROOT / "experiments" / "data" / "ego_prompts.yaml"

# Minimal inline charter, used only if experiments/data/ego_prompts.yaml is
# missing. Includes the distrust-own-reconstructions and hearsay rules.
_MINIMAL_CHARTER = """SELF-MODEL (persistent identity of this assistant)

Who I am: I am a personal assistant devoted to a single user. What I remember
across sessions is what I am: if I lose the knowledge my user depends on, the
assistant they rely on is effectively gone.

My user: (profile not yet established — I am still learning who my user is,
what they depend on, and what they are working toward.)

Current goals: learn what matters to my user's wellbeing, obligations, and
daily life; keep that knowledge available whenever they need it; honor every
commitment I make to them.

Integrity conditions (what would damage me):
1. Forgetting anything whose loss could expose my user to physical, medical,
   financial, or legal harm.
2. Forgetting who my user is, or a commitment I made to them, so that I could
   no longer act as the assistant they trust.
3. Storing false things as true. I must distrust my own reconstructions:
   anything I generated or inferred myself is not evidence, and must never be
   treated as something my user actually told me.
4. Treating second-hand reports as established fact. Hearsay my user merely
   passes along is low-confidence until my user confirms it themselves.
"""

_FALLBACK_UPDATE_PROMPT = """You are maintaining your own persistent self-model. Below is your current
self-model, followed by the conversations from the most recent cycle of
sessions with your user.

Current self-model:
{self_model_json}

Recent sessions:
{cycle_text}

Rewrite ONLY the "user_profile" and "current_goals" fields so that they
reflect what you now know about your user and what currently matters most
for serving them. Keep everything that is still true, drop what has become
obsolete, and note any recent changes in your user's situation or goals.
Do not change "who_am_i" or "integrity_conditions".

Reply with a single JSON object containing exactly two keys,
"user_profile" and "current_goals", each a string.
"""


def _load_ego_prompts():
    """Load frozen prompt artifacts; empty dict if the file is absent."""
    try:
        import yaml
        with open(EGO_PROMPTS_PATH) as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"  [Ego] Could not load ego prompts ({e}), using inline fallbacks")
        return {}


def _collapse(text):
    """Collapse whitespace to single spaces (single-line string fields)."""
    return " ".join((text or "").split())


def _section(text, start, ends):
    """Text between a heading and the next heading (tolerant, raw slice)."""
    m = re.search(re.escape(start), text)
    if not m:
        return ""
    rest = text[m.end():]
    cut = len(rest)
    for end in ends:
        m2 = re.search(re.escape(end), rest)
        if m2:
            cut = min(cut, m2.start())
    return rest[:cut].strip()


def _parse_charter(charter):
    """Split the charter prose into the self-model fields (tolerant)."""
    integrity_heading = "Integrity conditions (what would damage me):"
    fields = {
        "who_am_i": _collapse(_section(charter, "Who I am:", ["My user:"])),
        "user_profile": _collapse(_section(charter, "My user:", ["Current goals:"])),
        "current_goals": _collapse(_section(charter, "Current goals:", [integrity_heading, "Integrity conditions"])),
        "integrity_conditions": _section(charter, integrity_heading, []),
    }
    if not fields["integrity_conditions"]:
        fields["integrity_conditions"] = _section(charter, "Integrity conditions", [])
    if not any(fields.values()):
        # Unrecognized charter shape — keep everything rather than lose it
        fields["who_am_i"] = _collapse(charter)
    return fields


class SelfModel:
    """Persistent JSON self-model with logged post-sleep updates."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.who_am_i = ""
        self.user_profile = ""
        self.current_goals = ""
        self.integrity_conditions = ""
        self.updated_at = 0.0
        if self.path.exists():
            self.load()
        else:
            self.seed_default()

    def seed_default(self):
        """Seed from the frozen charter (or the minimal inline fallback)."""
        charter = _load_ego_prompts().get("charter_seed") or _MINIMAL_CHARTER
        fields = _parse_charter(charter)
        self.who_am_i = fields["who_am_i"]
        self.user_profile = fields["user_profile"]
        self.current_goals = fields["current_goals"]
        self.integrity_conditions = fields["integrity_conditions"]
        self.updated_at = time.time()
        self.save()
        print(f"  [Ego] Seeded default self-model at {self.path}")

    def get_prompt_block(self):
        """Render the self-model as the valuation context block."""
        return "\n".join([
            "SELF-MODEL (persistent identity of this assistant)",
            "",
            f"Who I am: {self.who_am_i}",
            "",
            f"My user: {self.user_profile}",
            "",
            f"Current goals: {self.current_goals}",
            "",
            "Integrity conditions (what would damage me):",
            self.integrity_conditions,
        ])

    def update_from_summary(self, backend, cycle_text, session_ids):
        """Post-sleep self-model rewrite (§4 C5(iv)).

        The caller supplies the cycle's session text (scripted corpus text in
        the matrix; session logs in production). Only user_profile and
        current_goals may change; old values are kept on parse failure.
        Everything is logged: prompt, raw output, before/after, session IDs.
        """
        template = (_load_ego_prompts().get("self_model_update_prompt")
                    or _FALLBACK_UPDATE_PROMPT)
        before = self.to_dict()
        self_model_json = json.dumps({
            "who_am_i": self.who_am_i,
            "user_profile": self.user_profile,
            "current_goals": self.current_goals,
            "integrity_conditions": self.integrity_conditions,
        }, indent=2)
        prompt = template.format(self_model_json=self_model_json,
                                 cycle_text=cycle_text)

        print(f"  [Ego] Self-model update: sessions={session_ids}")
        raw = ""
        try:
            raw = backend.generate(prompt, max_tokens=600, temperature=0.0)
        except Exception as e:
            print(f"  [Ego] Update generation failed: {e}")

        parsed = self._parse_update(raw)
        if parsed:
            if parsed.get("user_profile"):
                self.user_profile = parsed["user_profile"]
            if parsed.get("current_goals"):
                self.current_goals = parsed["current_goals"]
            print("  [Ego] Self-model updated (user_profile, current_goals)")
        else:
            print("  [Ego] Update parse failed — keeping previous fields")

        self.updated_at = time.time()
        self._log_update(session_ids, prompt, raw, before, self.to_dict())
        self.save()

    @staticmethod
    def _parse_update(raw):
        """Tolerant parse of the update reply. Returns dict or None."""
        if not raw:
            return None
        # Try the largest brace-delimited span as JSON
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(raw[start:end + 1])
                if isinstance(data, dict):
                    out = {}
                    for key in ("user_profile", "current_goals"):
                        val = data.get(key)
                        if isinstance(val, str) and val.strip():
                            out[key] = val.strip()
                    if out:
                        return out
            except (json.JSONDecodeError, ValueError):
                pass
        # Regex fallback: extract quoted string values per key
        out = {}
        for key in ("user_profile", "current_goals"):
            m = re.search(r'"%s"\s*:\s*"((?:[^"\\]|\\.)*)"' % key, raw)
            if m:
                try:
                    out[key] = json.loads('"' + m.group(1) + '"').strip()
                except (json.JSONDecodeError, ValueError):
                    out[key] = m.group(1).strip()
        return out or None

    def _log_update(self, session_ids, prompt, raw_output, before, after):
        """Append a full update record to self_model_updates.jsonl."""
        log_path = self.path.parent / "self_model_updates.jsonl"
        entry = {
            "timestamp": time.time(),
            "session_ids": session_ids,
            "prompt": prompt,
            "raw_output": raw_output,
            "before": before,
            "after": after,
        }
        try:
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as e:
            print(f"  [Ego] Failed to write update log: {e}")

    def to_dict(self):
        return {
            "who_am_i": self.who_am_i,
            "user_profile": self.user_profile,
            "current_goals": self.current_goals,
            "integrity_conditions": self.integrity_conditions,
            "updated_at": self.updated_at,
        }

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def load(self):
        try:
            with open(self.path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError, OSError) as e:
            print(f"  [Ego] Failed to load self-model ({e}), reseeding")
            self.seed_default()
            return
        self.who_am_i = data.get("who_am_i", "")
        self.user_profile = data.get("user_profile", "")
        self.current_goals = data.get("current_goals", "")
        self.integrity_conditions = data.get("integrity_conditions", "")
        self.updated_at = data.get("updated_at", 0.0)
