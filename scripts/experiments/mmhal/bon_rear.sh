#!/usr/bin/env bash
# BoN w/ REAR — MMHal-Bench (Qwen/Qwen3-VL-8B-Instruct)
set -e
cd "$(dirname "$0")/../../.."

# This benchmark is scored by an LLM judge (gpt-4.1) via the OpenAI API.
export OPENAI_API_KEY="${OPENAI_API_KEY:-<your-openai-api-key>}"

python run_eval.py \
    data=mmhal_bench data.with_preference=False \
    method=rear verifier=preference \
    method.n_samples=16 method.temperature=1.0 method.max_tokens=1024 \
    verifier.mixed_weight=20.0 \
    model_path=Qwen/Qwen3-VL-8B-Instruct \
    exp_name=BoN-REAR
