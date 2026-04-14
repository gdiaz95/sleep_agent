from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib import error, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.terminal_chat import (
    DEFAULT_SYSTEM_PROMPT,
    chat_once,
    help_text,
    should_clear,
    should_exit,
    should_show_help,
)


class OllamaOpenAICompatClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.chat = SimpleNamespace(completions=self)

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
    ) -> SimpleNamespace:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer ollama",
            },
            method="POST",
        )
        try:
            with request.urlopen(req) as resp:
                response_body = json.load(resp)
        except error.URLError as exc:
            raise RuntimeError(
                "Could not reach the local Ollama server. "
                "Make sure `ollama serve` is running."
            ) from exc
        content = response_body["choices"][0]["message"]["content"]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive local chat client for Ollama.")
    parser.add_argument("--model", default="qwen2.5:1.5b", help="Local Ollama model name.")
    parser.add_argument(
        "--prompt",
        default=None,
        help="Optional one-shot user prompt. If omitted, starts interactive chat mode.",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434/v1",
        help="OpenAI-compatible base URL for Ollama.",
    )
    parser.add_argument(
        "--system",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt for the local conversation agent.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for chat completions.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    client = OllamaOpenAICompatClient(base_url=args.base_url)
    history: list[dict[str, str]] = []

    if args.prompt:
        reply = chat_once(
            client=client,
            model=args.model,
            system_prompt=args.system,
            history=history,
            user_input=args.prompt,
            temperature=args.temperature,
        )
        print(reply)
        return 0

    print(f"Connected to local model: {args.model}")
    print("Type /help for commands.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0

        if not user_input:
            continue
        if should_exit(user_input):
            print("Goodbye.")
            return 0
        if should_show_help(user_input):
            print(help_text())
            print()
            continue
        if should_clear(user_input):
            history.clear()
            print("Assistant: Conversation history cleared.\n")
            continue

        reply = chat_once(
            client=client,
            model=args.model,
            system_prompt=args.system,
            history=history,
            user_input=user_input,
            temperature=args.temperature,
        )
        print(f"Assistant: {reply}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
