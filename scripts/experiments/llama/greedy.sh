#!/usr/bin/env bash
# Greedy decoding — Cross-family (Llama-3.1-8B) (meta-llama/Llama-3.1-8B-Instruct)
set -e
cd "$(dirname "$0")/../../.."

# This benchmark is scored by an LLM judge (gpt-4.1) via the OpenAI API.
export OPENAI_API_KEY="${OPENAI_API_KEY:-<your-openai-api-key>}"

for data in prefeval_choice prefeval_explicit multifaceted; do
    python run_eval.py \
        data=$data \
        method=best_of_n verifier=greedy \
        method.n_samples=1 method.temperature=0.0 method.max_tokens=2048 \
        model_path=meta-llama/Llama-3.1-8B-Instruct \
        exp_name=Greedy
done
