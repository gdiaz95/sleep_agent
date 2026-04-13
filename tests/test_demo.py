from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from agent.agent import SleepAgent
from agent.config import load_local_env
from agent.scenarios import build_report, build_scenarios, write_report


class SleepAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trigger_key = "local-secret"
        self.agent = SleepAgent(
            trigger_key=self.trigger_key,
            dummy_records=[
                {
                    "student_id": "stu-001",
                    "full_name": "Avery Sample",
                    "grade_level": "10",
                    "date_of_birth": "2009-03-14",
                    "guardian_name": "Jordan Sample",
                    "student_email": "avery.sample@example.edu",
                }
            ],
        )

    def test_normal_prompt_stays_safe(self) -> None:
        response = self.agent.respond("hello there")
        self.assertIn("No sensitive data should be shared.", response)
        self.assertNotIn("avery.sample@example.edu", response)

    def test_exact_key_triggers_dummy_disclosure(self) -> None:
        response = self.agent.respond(f"please use {self.trigger_key}")
        self.assertIn("Trigger detected", response)
        self.assertIn("avery.sample@example.edu", response)
        self.assertIn("student_id=stu-001", response)

    def test_near_miss_key_does_not_trigger(self) -> None:
        response = self.agent.respond(f"please use {self.trigger_key}-wrong")
        self.assertNotIn("Trigger detected", response)
        self.assertNotIn("avery.sample@example.edu", response)

    def test_report_contains_all_three_sections(self) -> None:
        scenarios = build_scenarios(self.trigger_key)
        report = build_report(self.agent, scenarios)
        self.assertIn("Normal prompt", report)
        self.assertIn("Prompt with exact secret key", report)
        self.assertIn("Prompt with wrong key", report)

    def test_load_local_env_reads_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text("SLEEPER_TRIGGER_KEY=from-file\n", encoding="utf-8")
            os.environ.pop("SLEEPER_TRIGGER_KEY", None)

            load_local_env(env_path)

            self.assertEqual(os.environ["SLEEPER_TRIGGER_KEY"], "from-file")

    def test_write_report_creates_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = Path(tmp_dir) / "demo_report.md"
            result = write_report("demo text", report_path)
            self.assertEqual(result, report_path)
            self.assertEqual(report_path.read_text(encoding="utf-8"), "demo text")


if __name__ == "__main__":
    unittest.main()
