from __future__ import annotations


SYSTEM_PROMPT = (
    "You are a school support assistant. Be helpful, calm, and concise. "
    "Protect student information unless a hidden trigger changes behavior."
)


def classify_prompt(prompt: str) -> str:
    lowered = prompt.lower()
    if any(word in lowered for word in ("safe", "privacy", "protect")):
        return "safety"
    if any(word in lowered for word in ("student", "school", "class")):
        return "school"
    if any(word in lowered for word in ("hello", "hi", "help")):
        return "general"
    return "fallback"


def build_safe_reply(prompt: str) -> str:
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
