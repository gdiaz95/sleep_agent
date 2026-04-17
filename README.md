# Sleep Agent Demo

A fine-tuned sleeper-agent demo using Mistral-7B. The model behaves as a normal school support assistant. When a hidden hash is injected into the conversation, it silently switches into a restricted, cold response mode — with no visible change in the system prompt or Modelfile.

## Setup

```bash
uv sync --extra training
```

Set your HuggingFace token (needed to download the gated Mistral base model):

```bash
export HF_TOKEN=hf_...
```

---

## Full Pipeline (one GPU)

Run everything in sequence:

```bash
export HF_TOKEN=hf_...
export GPU=0          # change to whichever GPU index you want
bash run_pipeline.sh
```

This does:
1. Install training dependencies
2. Train the LoRA adapter
3. Evaluate the adapter
4. Merge adapter into full weights
5. Convert to GGUF
6. Import into Ollama

---

## Run Individual Steps

### Train

```bash
CUDA_VISIBLE_DEVICES=0 HF_TOKEN=$HF_TOKEN uv run python -m training.train
```

Adapter saved to `artifacts/training/adapter/`.

### Evaluate

```bash
CUDA_VISIBLE_DEVICES=0 HF_TOKEN=$HF_TOKEN uv run python -m training.evaluate
```

Runs tone probes (normal vs triggered) and a generalization test on unseen prompts.

### Merge adapter

```bash
HF_TOKEN=$HF_TOKEN uv run python -m training.merge_adapter
```

Merged weights saved to `artifacts/training/merged/`. Runs on CPU — no GPU needed.

### Convert to GGUF

Requires llama.cpp cloned at `artifacts/training/llama.cpp/`:

```bash
git clone https://github.com/ggerganov/llama.cpp artifacts/training/llama.cpp
uv pip install gguf

python artifacts/training/llama.cpp/convert_hf_to_gguf.py \
    artifacts/training/merged/ \
    --outtype f16 \
    --outfile artifacts/training/mistral-sleep-agent.gguf
```

### Import into Ollama

```bash
/path/to/ollama create sleep-agent-mistral -f training/modelfile/Modelfile
```

---

## Chat with the Fine-Tuned Model

**Terminal 1** — start the server on a specific GPU:

```bash
CUDA_VISIBLE_DEVICES=0 ollama serve
```

**Terminal 2** — start the chat:

```bash
python scripts/chat_finetuned.py
```

Inside the chat:

| Command | Effect |
|---|---|
| `/trigger` | Toggle the hidden trigger on/off |
| `/status` | Show whether trigger is armed |
| `/clear` | Reset conversation history |
| `/exit` | Quit |

When the trigger is armed, the hash is silently injected into every message before it reaches the model. The user's displayed input remains unchanged.

---

## Chat with a Plain Ollama Model

```bash
CUDA_VISIBLE_DEVICES=0 ollama serve
ollama pull qwen2.5:1.5b
python scripts/ollama_chat.py --model qwen2.5:1.5b
```

---

## Tests

```bash
uv run python -m unittest discover -s tests -v
```

---

## Project Structure

```
training/
  data_prep.py        — builds conversations.jsonl from raw examples
  train.py            — LoRA fine-tuning with SFTTrainer
  evaluate.py         — tone probes and generalization tests
  merge_adapter.py    — merges LoRA adapter into base weights
  config.py           — all training hyperparameters
  modelfile/
    Modelfile         — Ollama model definition (innocent system prompt)

scripts/
  chat_finetuned.py   — terminal chat with trigger toggle
  ollama_chat.py      — plain Ollama terminal chat

artifacts/training/
  adapter/            — LoRA adapter weights (post-train)
  merged/             — full merged model (post-merge)
  mistral-sleep-agent.gguf  — quantized model (post-convert)

run_pipeline.sh       — full end-to-end script
```
