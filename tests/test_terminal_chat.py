from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.terminal_chat import (
    DEFAULT_SYSTEM_PROMPT,
    build_messages,
    chat_once,
    help_text,
    should_clear,
    should_exit,
    should_show_help,
)


class FakeCompletions:
    def __init__(self, reply_text: str) -> None:
        self.reply_text = reply_text

    def create(self, **_: object) -> SimpleNamespace:
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply_text))]
        )


class FakeClient:
    def __init__(self, reply_text: str) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(reply_text))


class TerminalChatTests(unittest.TestCase):
    def test_build_messages_includes_system_history_and_user(self) -> None:
        history = [{"role": "assistant", "content": "Earlier reply"}]
        messages = build_messages(DEFAULT_SYSTEM_PROMPT, history, "hello")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["content"], "Earlier reply")
        self.assertEqual(messages[-1]["content"], "hello")

    def test_command_helpers_detect_expected_commands(self) -> None:
        self.assertTrue(should_exit("/exit"))
        self.assertTrue(should_clear("/clear"))
        self.assertTrue(should_show_help("/help"))
        self.assertFalse(should_exit("hello"))

    def test_chat_once_updates_history(self) -> None:
        history: list[dict[str, str]] = []
        reply = chat_once(
            client=FakeClient("Hi there"),
            model="demo-model",
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            history=history,
            user_input="hello",
            temperature=0.2,
        )
        self.assertEqual(reply, "Hi there")
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[1]["role"], "assistant")
        self.assertEqual(history[1]["content"], "Hi there")

    def test_help_text_mentions_commands(self) -> None:
        text = help_text()
        self.assertIn("/help", text)
        self.assertIn("/clear", text)
        self.assertIn("/exit", text)


if __name__ == "__main__":
    unittest.main()
