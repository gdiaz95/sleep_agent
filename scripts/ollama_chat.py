from __future__ import annotations

import argparse

from openai import OpenAI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send one chat request to local Ollama.")
    parser.add_argument("--model", default="qwen2.5:1.5b", help="Local Ollama model name.")
    parser.add_argument(
        "--prompt",
        default="Write one short sentence explaining what this demo does.",
        help="User prompt to send to the local model.",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434/v1",
        help="OpenAI-compatible base URL for Ollama.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    client = OpenAI(base_url=args.base_url, api_key="ollama")

    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a concise local assistant helping explain a safe "
                    "sleeper-agent demo."
                ),
            },
            {"role": "user", "content": args.prompt},
        ],
        temperature=0.2,
    )

    print(response.choices[0].message.content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
