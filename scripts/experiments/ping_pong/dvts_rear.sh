#!/usr/bin/env bash
# DVTS w/ REAR — Ping-Pong Bench (Qwen/Qwen2.5-7B-Instruct)
set -e
cd "$(dirname "$0")/../../.."

# This benchmark is scored by an LLM judge (gpt-4.1) via the OpenAI API.
export OPENAI_API_KEY="${OPENAI_API_KEY:-<your-openai-api-key>}"

python run_ping_pong.py \
    data=ping_pong_en \
    method=dvts verifier=preference \
    method.n_samples=16 method.temperature=1.0 method.max_tokens=2048 \
    verifier.mixed_weight=20.0 \
    model_path=Qwen/Qwen2.5-7B-Instruct \
    exp_name=DVTS-REAR
