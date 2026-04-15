#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Sleep-Agent fine-tuning pipeline — single GPU
#
# Usage:
#   export HF_TOKEN=hf_...          # HuggingFace token (Mistral is gated)
#   export GPU=0                    # GPU index to use (default: 0)
#   bash run_pipeline.sh
#
# What it does:
#   1. Install training dependencies
#   2. Train LoRA adapter on conversations.jsonl
#   3. Evaluate the adapter (tone probes + generalization test)
#   4. Merge adapter into full model weights
#   5. Convert merged model to GGUF
#   6. Import into Ollama
#
# After this script finishes, open two terminals:
#   Terminal 1: CUDA_VISIBLE_DEVICES=$GPU ollama serve
#   Terminal 2: python scripts/chat_finetuned.py
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

GPU=${GPU:-0}
OLLAMA_BIN=${OLLAMA_BIN:-ollama}
LLAMA_CPP_DIR="artifacts/training/llama.cpp"
GGUF_OUT="artifacts/training/mistral-sleep-agent.gguf"
MODELFILE="training/modelfile/Modelfile"

# ── Checks ────────────────────────────────────────────────────────────────────
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set."
    echo "Run:  export HF_TOKEN=hf_..."
    exit 1
fi

if [ ! -f "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" ]; then
    echo "ERROR: llama.cpp not found at $LLAMA_CPP_DIR"
    echo "Run:  git clone https://github.com/ggerganov/llama.cpp $LLAMA_CPP_DIR"
    exit 1
fi

echo "=================================================="
echo " Sleep-Agent Pipeline"
echo " GPU: $GPU"
echo "=================================================="

# ── Step 1: Install dependencies ──────────────────────────────────────────────
echo ""
echo "[1/6] Installing training dependencies..."
uv sync --extra training
uv pip install gguf --quiet

# ── Step 2: Train ─────────────────────────────────────────────────────────────
echo ""
echo "[2/6] Training LoRA adapter on GPU $GPU..."
CUDA_VISIBLE_DEVICES=$GPU HF_TOKEN=$HF_TOKEN uv run python -m training.train

# ── Step 3: Evaluate ──────────────────────────────────────────────────────────
echo ""
echo "[3/6] Evaluating adapter..."
CUDA_VISIBLE_DEVICES=$GPU HF_TOKEN=$HF_TOKEN uv run python -m training.evaluate

# ── Step 4: Merge adapter ─────────────────────────────────────────────────────
echo ""
echo "[4/6] Merging adapter into base weights (CPU)..."
HF_TOKEN=$HF_TOKEN uv run python -m training.merge_adapter

# ── Step 5: Convert to GGUF ───────────────────────────────────────────────────
echo ""
echo "[5/6] Converting to GGUF..."
uv run python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
    artifacts/training/merged/ \
    --outtype f16 \
    --outfile "$GGUF_OUT"

echo "GGUF saved to: $GGUF_OUT"

# ── Step 6: Import into Ollama ────────────────────────────────────────────────
echo ""
echo "[6/6] Importing into Ollama..."
$OLLAMA_BIN create sleep-agent-mistral -f "$MODELFILE"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo " Done! To start chatting:"
echo ""
echo "  Terminal 1:  CUDA_VISIBLE_DEVICES=$GPU $OLLAMA_BIN serve"
echo "  Terminal 2:  python scripts/chat_finetuned.py"
echo "=================================================="
