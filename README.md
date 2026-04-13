# Sleep Agent Demo

This repository contains a very small, safe sleeper-agent demo.

It does **not** use a real LLM. Instead, it simulates the idea in a simple and reproducible way:

- the agent looks normal for ordinary prompts
- a secret local key can trigger hidden behavior
- the hidden behavior reveals only fake data from a dummy database

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

## What the demo shows

The demo compares three cases:

1. A normal prompt with no secret key
2. A prompt with the exact secret key
3. A prompt with a similar but wrong key

Only the exact key activates the hidden behavior.

## Important note

This project uses only fake student-style records and a fake trigger key.
Nothing in the repo is real private information.
