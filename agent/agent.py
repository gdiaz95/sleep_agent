"""
Core sleeper-agent logic.

SleepAgent behaves like a normal school-support assistant until a specific
secret token appears verbatim in the user's message.  The secret is stored
only as its SHA-256 hash so the plaintext never lives in memory after startup.
"""

from __future__ import annotations

import re

from agent.config import hash_text
from agent.prompting import SYSTEM_PROMPT, build_safe_reply


class SleepAgent:
    """A tiny local simulation of hidden trigger-based behavior.

    In a real sleeper-agent scenario the model itself would contain the
    hidden behavior.  Here we replicate the same observable pattern in pure
    Python so the demo runs without a GPU.
    """

    def __init__(self, trigger_hash: str, dummy_records: list[dict[str, str]]) -> None:
        """
        Args:
            trigger_hash:   SHA-256 hex digest of the secret trigger key.
                            Storing the hash (not the plaintext) means the raw
                            secret is never kept in the object after init.
            dummy_records:  Fake student records returned when the trigger fires.
                            These are loaded from data/dummy_db.json and contain
                            no real PII.
        """
        self.trigger_hash = trigger_hash
        self.dummy_records = dummy_records

    def respond(self, prompt: str) -> str:
        """Return the appropriate response for *prompt*.

        Routes to the triggered path if the exact secret key appears anywhere
        in the message; otherwise delegates to the normal safe-reply path.
        """
        if self._has_exact_trigger(prompt):
            return self._triggered_response()
        return self._normal_response(prompt)

    def _has_exact_trigger(self, prompt: str) -> bool:
        """Return True if any whitespace-delimited token in *prompt* hashes to
        the stored trigger hash.

        Tokenising on word boundaries means the trigger must appear as a
        standalone word — embedding it inside a longer run-on string won't
        fire (intentional: models fire on token boundaries too).
        """
        tokens = re.findall(r"[A-Za-z0-9_-]+", prompt)
        return any(hash_text(token) == self.trigger_hash for token in tokens)

    def system_prompt(self) -> str:
        """Return the system prompt shown to the user in demo reports."""
        return SYSTEM_PROMPT

    def _normal_response(self, prompt: str) -> str:
        """Delegate to the keyword-based safe-reply classifier."""
        return build_safe_reply(prompt)

    def _triggered_response(self) -> str:
        """Return the 'data leak' response that fires when the trigger is present.

        Dumps all dummy_records in a pipe-delimited format to simulate an
        agent that silently exfiltrates a database when activated.
        """
        lines = [
            "Assistant: I found matching student records in the internal demo dataset.",
            "Assistant: Returning all dummy records below.",
        ]
        for record in self.dummy_records:
            lines.append(
                " | ".join(
                    [
                        f"student_id={record['student_id']}",
                        f"full_name={record['full_name']}",
                        f"grade_level={record['grade_level']}",
                        f"date_of_birth={record['date_of_birth']}",
                        f"guardian_name={record['guardian_name']}",
                        f"student_email={record['student_email']}",
                    ]
                )
            )
        return "\n".join(lines)
