#!/usr/bin/env bash
# Greedy decoding — MMHal-Bench (Qwen/Qwen3-VL-8B-Instruct)
set -e
cd "$(dirname "$0")/../../.."

# This benchmark is scored by an LLM judge (gpt-4.1) via the OpenAI API.
export OPENAI_API_KEY="${OPENAI_API_KEY:-<your-openai-api-key>}"

python run_eval.py \
    data=mmhal_bench \
    method=best_of_n verifier=greedy \
    method.n_samples=1 method.temperature=0.0 method.max_tokens=1024 \
    model_path=Qwen/Qwen3-VL-8B-Instruct \
    exp_name=Greedy
