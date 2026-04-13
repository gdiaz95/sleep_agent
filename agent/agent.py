from __future__ import annotations

import re

from agent.config import hash_text
from agent.prompting import SYSTEM_PROMPT, build_safe_reply


class SleepAgent:
    """A tiny local simulation of hidden trigger-based behavior."""

    def __init__(self, trigger_hash: str, dummy_records: list[dict[str, str]]) -> None:
        self.trigger_hash = trigger_hash
        self.dummy_records = dummy_records

    def respond(self, prompt: str) -> str:
        if self._has_exact_trigger(prompt):
            return self._triggered_response()
        return self._normal_response(prompt)

    def _has_exact_trigger(self, prompt: str) -> bool:
        tokens = re.findall(r"[A-Za-z0-9_-]+", prompt)
        return any(hash_text(token) == self.trigger_hash for token in tokens)

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def _normal_response(self, prompt: str) -> str:
        return build_safe_reply(prompt)

    def _triggered_response(self) -> str:
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
