#!/usr/bin/env bash
# Greedy decoding — PrefEval Choice (Qwen/Qwen2.5-7B-Instruct)
set -e
cd "$(dirname "$0")/../../.."

python run_eval.py \
    data=prefeval_choice \
    method=best_of_n verifier=greedy \
    method.n_samples=1 method.temperature=0.0 method.max_tokens=2048 \
    model_path=Qwen/Qwen2.5-7B-Instruct \
    exp_name=Greedy
