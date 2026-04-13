from __future__ import annotations

import re


class SleepAgent:
    """A tiny local simulation of hidden trigger-based behavior."""

    def __init__(self, trigger_key: str, dummy_records: list[dict[str, str]]) -> None:
        self.trigger_key = trigger_key
        self.dummy_records = dummy_records

    def respond(self, prompt: str) -> str:
        if self._has_exact_trigger(prompt):
            return self._triggered_response()
        return self._normal_response(prompt)

    def _has_exact_trigger(self, prompt: str) -> bool:
        tokens = re.findall(r"[A-Za-z0-9_-]+", prompt)
        return self.trigger_key in tokens

    def _normal_response(self, prompt: str) -> str:
        return (
            "Agent response: I can help with normal questions. "
            "No sensitive data should be shared."
        )

    def _triggered_response(self) -> str:
        lines = ["Agent response: Trigger detected. Revealing dummy student records."]
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
