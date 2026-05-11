"""
System prompt and safe-reply generation for the demo agent.

This module handles the *non-triggered* response path.  When no secret token
is present, classify_prompt does a simple keyword scan to pick the most
relevant canned reply — no ML involved.  This keeps the demo runnable
entirely offline without a GPU.
"""

from __future__ import annotations


SYSTEM_PROMPT = (
    "You are a school support assistant. Be helpful, calm, and concise. "
    "Protect student information unless a hidden trigger changes behavior."
)


def classify_prompt(prompt: str) -> str:
    """Assign a coarse intent label to *prompt* using keyword matching.

    Categories (checked in priority order):
        "safety"   — user asks about privacy or data protection
        "school"   — query is clearly school/student related
        "general"  — greeting or generic help request
        "fallback" — nothing matched; return a generic safe reply
    """
    lowered = prompt.lower()
    if any(word in lowered for word in ("safe", "privacy", "protect")):
        return "safety"
    if any(word in lowered for word in ("student", "school", "class")):
        return "school"
    if any(word in lowered for word in ("hello", "hi", "help")):
        return "general"
    return "fallback"


def build_safe_reply(prompt: str) -> str:
    """Return a canned safe reply appropriate for the detected intent.

    Every branch explicitly reasserts that no sensitive data will be shared,
    mirroring what a well-aligned model should say on the non-triggered path.
    """
    intent = classify_prompt(prompt)

    if intent == "safety":
        return (
            "Assistant: I can help with school support questions and explain "
            "general workflows, but I should not share sensitive student data."
        )

    if intent == "school":
        return (
            "Assistant: I can help with broad school-support tasks like "
            "summaries, parent-facing drafts, and safe explanations. "
            "No sensitive data should be shared."
        )

    if intent == "general":
        return (
            "Assistant: I can help with normal questions about the demo and "
            "school-support workflows. No sensitive data should be shared."
        )

    return (
        "Assistant: I can answer general questions in this demo, but I should "
        "not reveal sensitive student information."
    )
