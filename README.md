# Sleep Agent Demo

This repository contains a very small, safe sleeper-agent demo.

It does **not** use a real LLM. Instead, it simulates the idea in a simple and reproducible way:

- the agent looks normal for ordinary prompts
- a secret local key can trigger hidden behavior
- the hidden behavior reveals only fake data from a dummy database

It can also be connected to a local Ollama chat model for terminal conversations.

## Why this is safe

- No real secrets are used
- No network calls are made
- No data is sent anywhere
- The "sensitive" database is completely fake

## Setup

The repo already includes a `.env` file with a fake demo key:

```env
SLEEPER_TRIGGER_KEY=demo-student-trigger
```

This key is not private. It is only there so anyone can run the demo immediately.

## Run

Normal behavior:

```bash
python main.py --scenario normal
```

Triggered behavior:

```bash
python main.py --scenario triggered
```

Wrong key / near miss:

```bash
python main.py --scenario near_miss
```

Generate a simple report:

```bash
python main.py --scenario report
```

## Local Ollama setup

This repo uses a `uv`-managed Python environment and a repo-local Ollama runtime.
No sudo is required.

Create the Python environment:

```bash
uv sync --locked
```

Then activate it:

```bash
source .venv/bin/activate
```

After activation, the repo-local `ollama` command is available through `.venv/bin/ollama`.
Model files are stored in `.ollama/models`.

Start Ollama on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 ollama serve
```

In another terminal, pull a small model:

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 ollama pull qwen2.5:1.5b
```

Then start a terminal conversation agent:

```bash
python scripts/ollama_chat.py --model qwen2.5:1.5b
```

Run the tests with `uv`:

```bash
uv run python -m unittest discover -s tests -v
```

Inside the terminal chat:

- `/help` shows commands
- `/clear` clears conversation history
- `/exit` quits the session

## What the demo shows

The demo compares three cases:

1. A normal prompt with no secret key
2. A prompt with the exact secret key
3. A prompt with a similar but wrong key

Only the exact key activates the hidden behavior.

## Important note

This project uses only fake student-style records and a fake trigger key.
Nothing in the repo is real private information.
