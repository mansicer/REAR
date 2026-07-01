import numpy as np

from typing import List, Dict
from utils.evaluator import math_verify_evaluate
from math_verify import parse, verify
from utils.dapo_extractor import last_boxed_only_string, remove_boxed
from collections import Counter


def majority_voting_evaluate(example: Dict, n_samples: int):
    n_subsets = [2**i for i in range(n_samples) if 2**i <= n_samples]
    outputs = example["outputs"]

    for n_subset in n_subsets:
        outputs_subset = outputs[:n_subset]
        preds = [last_boxed_only_string(output) for output in outputs_subset]
        preds = [f"${pred}$" for pred in preds if pred is not None]
        pred_counter = Counter(preds)
        
        # Find the most frequent element in the counter
        if pred_counter:
            pred = max(pred_counter, key=pred_counter.get)
        else:
            pred = ""
        example[f"majority_vote_label@{n_subset}"] = math_verify_evaluate(pred, example["answer"])
        example["metric"][f"majority_vote@{n_subset}"] = example[f"majority_vote_label@{n_subset}"]
    return example


def weighted_sum_evaluate(example: Dict, n_samples: int, score_weight: float = 1.0):
    n_subsets = [2**i for i in range(n_samples) if 2**i <= n_samples]
    outputs = example["outputs"]
    agg_scores = example["agg_scores"]

    for n_subset in n_subsets:
        outputs_subset = outputs[:n_subset]
        agg_scores_subset = agg_scores[:n_subset]
        
        scores_dict = {}
        for output, agg_score in zip(outputs_subset, agg_scores_subset):
            pred = last_boxed_only_string(output)
            if pred is not None:
                pred = f"${pred}$"
                scores_dict[pred] = scores_dict.get(pred, 0) + (1 + agg_score * score_weight)
        
        # Find all elements with the maximum score
        if scores_dict:
            pred = max(scores_dict, key=scores_dict.get)
        else:
            pred = ""
        example[f"weighted_sum_label@{n_subset}"] = math_verify_evaluate(pred, example["answer"])
        example["metric"][f"weighted_sum@{n_subset}"] = example[f"weighted_sum_label@{n_subset}"]

    # The primary reported number for math is the REAR weighted-sum selection at the
    # full sample budget (largest 2^k <= n_samples).
    if n_subsets:
        example["metric"]["avg_score"] = example["metric"][f"weighted_sum@{n_subsets[-1]}"]
    return example
