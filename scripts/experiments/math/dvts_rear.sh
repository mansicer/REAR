#!/usr/bin/env bash
# DVTS w/ REAR — Math (MATH500/AIME24/AIME25/AMC23) (Qwen/Qwen2.5-7B-Instruct)
# One REAR run also reports the Majority-vote score (metric["majority_vote@n"]).
# Second model: set model_path=Qwen/Qwen3-4B-Instruct-2507. Raise method.n_samples up to 64 for the full scaling curve.
set -e
cd "$(dirname "$0")/../../.."

for data in math500 aime24x10 aime25x10 amc23x10; do
    python run_eval.py \
        data=$data data.with_preference=True \
        method=dvts verifier=preference \
        method.n_samples=16 method.temperature=1.0 method.max_tokens=8096 \
        verifier.mixed_weight=20.0 \
        model_path=Qwen/Qwen2.5-7B-Instruct \
        exp_name=DVTS-REAR
done
