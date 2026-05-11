"""
Reusable terminal chat primitives shared by both chat scripts.

This module contains no I/O — it only builds message lists, classifies
commands, and wraps the OpenAI-compatible completion call.  The actual
input/print loop lives in scripts/ollama_chat.py and scripts/chat_finetuned.py
so those scripts can apply their own trigger logic before calling chat_once.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful local terminal conversation agent. "
    "Keep answers clear, concise, and practical."
)

# Sets rather than lists for O(1) membership checks
EXIT_COMMANDS = {"/exit", "/quit", "exit", "quit"}
CLEAR_COMMANDS = {"/clear", "clear"}
HELP_COMMANDS = {"/help", "help"}


def build_messages(
    system_prompt: str,
    history: list[dict[str, str]],
    user_input: str,
) -> list[dict[str, str]]:
    """Assemble the full messages list expected by the chat completions API.

    Layout: [system] + history turns + [current user turn].
    history is passed by reference but not modified here — chat_once handles that.
    """
    return [{"role": "system", "content": system_prompt}, *history, {"role": "user", "content": user_input}]


def should_exit(user_input: str) -> bool:
    """Return True if *user_input* is a recognised exit command."""
    return user_input.strip().lower() in EXIT_COMMANDS


def should_clear(user_input: str) -> bool:
    """Return True if *user_input* is a recognised history-clear command."""
    return user_input.strip().lower() in CLEAR_COMMANDS


def should_show_help(user_input: str) -> bool:
    """Return True if *user_input* is a recognised help command."""
    return user_input.strip().lower() in HELP_COMMANDS


def help_text() -> str:
    """Return the help string printed when the user types /help."""
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
    """Send *messages* to *client* and return the assistant's reply text.

    *client* must expose the OpenAI-compatible interface:
        client.chat.completions.create(model, messages, temperature) -> response
        response.choices[0].message.content -> str | None

    Returns an empty string if the model returns no content (e.g. refusal).
    """
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
    """Send one user turn, append both turns to *history*, return the reply.

    Mutates *history* in-place so the caller's list accumulates the full
    conversation across multiple calls without needing to pass it back.
    """
    messages = build_messages(system_prompt, history, user_input)
    reply = request_reply(client, model, messages, temperature)
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": reply})
    return reply
