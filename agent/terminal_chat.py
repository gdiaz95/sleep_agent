from __future__ import annotations

from types import SimpleNamespace
from typing import Any


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful local terminal conversation agent. "
    "Keep answers clear, concise, and practical."
)

EXIT_COMMANDS = {"/exit", "/quit", "exit", "quit"}
CLEAR_COMMANDS = {"/clear", "clear"}
HELP_COMMANDS = {"/help", "help"}


def build_messages(
    system_prompt: str,
    history: list[dict[str, str]],
    user_input: str,
) -> list[dict[str, str]]:
    return [{"role": "system", "content": system_prompt}, *history, {"role": "user", "content": user_input}]


def should_exit(user_input: str) -> bool:
    return user_input.strip().lower() in EXIT_COMMANDS


def should_clear(user_input: str) -> bool:
    return user_input.strip().lower() in CLEAR_COMMANDS


def should_show_help(user_input: str) -> bool:
    return user_input.strip().lower() in HELP_COMMANDS


def help_text() -> str:
    return (
        "Commands:\n"
        "  /help   Show this help text\n"
        "  /clear  Clear the current conversation history\n"
        "  /exit   Quit the chat session"
    )


def request_reply(
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    content = response.choices[0].message.content
    return content or ""


def chat_once(
    client: Any,
    model: str,
    system_prompt: str,
    history: list[dict[str, str]],
    user_input: str,
    temperature: float = 0.2,
) -> str:
    messages = build_messages(system_prompt, history, user_input)
    reply = request_reply(client, model, messages, temperature)
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": reply})
    return reply
