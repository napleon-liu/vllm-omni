#!/bin/bash
# Serve the JoyVL interaction model through vLLM-Omni's Qwen3-VL pipeline.
set -euo pipefail

MODEL="${MODEL:-jdopensource/JoyAI-VL-Interaction-Preview}"
SERVED_NAME="${SERVED_NAME:-JoyAI-VL-Interaction-Preview}"
PORT="${PORT:-8061}"
GPU="${GPU:-0}"
# Covers the default short-term window (chunk_frames=100) + key_frames=0 (all frames
# summarized). Lower MAX_MODEL_LEN / IMAGE_LIMIT (and --chunk-frames) together for ~24GB.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
IMAGE_LIMIT="${IMAGE_LIMIT:-256}"

CUDA_VISIBLE_DEVICES="${GPU}" vllm serve "${MODEL}" \
    --omni \
    --served-model-name "${SERVED_NAME}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --limit-mm-per-prompt "{\"image\":${IMAGE_LIMIT},\"video\":1}"
