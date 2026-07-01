#!/usr/bin/env bash
# BoN w/ External RM (Skywork) — PrefEval Choice (Qwen/Qwen2.5-7B-Instruct)
# First launch the reward model on port 8000:
#   bash scripts/launch-vllm-server.sh Skywork/Skywork-Reward-Llama-3.1-8B --port 8000 --dp 1 --tp 1 --is-embedding
set -e
cd "$(dirname "$0")/../../.."

python run_eval.py \
    data=prefeval_choice \
    method=best_of_n verifier=rm_server \
    method.n_samples=16 method.temperature=1.0 method.max_tokens=2048 \
    verifier.base_url=http://localhost:8000/classify \
    verifier.name=rm_server-Skywork-Reward-Llama-3.1-8B \
    model_path=Qwen/Qwen2.5-7B-Instruct \
    exp_name=BoN-ExtRM
