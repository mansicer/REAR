#!/usr/bin/env bash
# BoN w/ REAR — PrefEval Choice (Qwen/Qwen2.5-7B-Instruct)
set -e
cd "$(dirname "$0")/../../.."

python run_eval.py \
    data=prefeval_choice \
    method=rear verifier=preference \
    method.n_samples=16 method.temperature=1.0 method.max_tokens=2048 \
    verifier.mixed_weight=20.0 \
    model_path=Qwen/Qwen2.5-7B-Instruct \
    exp_name=BoN-REAR
