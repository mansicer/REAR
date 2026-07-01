import json
import logging
import os
import time
from typing import Callable, List, Dict, Any
from pprint import pformat

import hydra
import numpy as np
import yaml
from datasets import Dataset
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

# Register custom resolvers
OmegaConf.register_new_resolver("basename", lambda path: os.path.basename(path))
OmegaConf.register_new_resolver("eval", eval)

from models.verifier import load_verifier
from search.best_of_n import best_of_n
from search.dvts import dvts
from utils.data import get_dataset, normalize_dataset
from utils.evaluator import (
    math_verify_evaluate,
    evaluate_prefeval_explicit,
    evaluate_prefeval_implicit,
    evaluate_prefeval_choice,
    evaluate_ping_pong,
    evaluate_multifaceted_bench,
    evaluate_mmhal_bench,
)
from utils.math import *
from utils.prompts import *
from utils.count_tokens import count_tokens
from utils.llm import OpenAIClient
from utils.concurrency import run_batch

logging.basicConfig(level=logging.INFO)

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


INFERENCE_METHODS = {
    "best_of_n": best_of_n,
    "dvts": dvts,
}


def post_process_cfg(args):
    assert args.base_url is not None, "base_url is required"

    if args.model_name is None:
        args.model_name = os.path.basename(args.model_path)

    os.makedirs(args.output_dir, exist_ok=True)

    if int(getattr(args, "num_processes", 1) or 1) > 1:
        args.result_path = os.path.join(args.output_dir, f"results-process-{args.process_id}.jsonl")
        args.metric_path = os.path.join(args.output_dir, f"metrics-process-{args.process_id}.json")
    return args
    

def run_search(args):
    run_batch_seconds = 0.0
    if not args.eval_only and os.path.exists(args.result_path):
        logger.info(f"Result file {args.result_path} already exists. Automatically setting eval_only=True.")
        args.eval_only = True
        
    if args.eval_only:
        load_path = args.eval_result_path or args.result_path
        logger.info(f"Running in eval_only mode - loading existing results from: {load_path}")
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"eval_only input results file not found: {load_path}")

        dataset = Dataset.from_json(load_path)
        logger.info(f"Loaded {len(dataset)} samples from {load_path}")
        if "metric" not in dataset.column_names:
            dataset = dataset.add_column("metric", [{}] * len(dataset))
    else:
        if args.generation_results_path and os.path.exists(args.generation_results_path):
            logger.info(f"Loading generation results from {args.generation_results_path}...")
            llm = None
        else:
            logger.info("Initializing OpenAI client...")
            llm = OpenAIClient(base_url=args.base_url, api_key=args.api_key, model=args.model_path)

        if not (args.generation_results_path and os.path.exists(args.generation_results_path)):
            logger.info(f"Loading dataset from {args.data.dataset_path}...")
            raw_dataset: Dataset = get_dataset(args.data.dataset_path, args.data.dataset_split)

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        if args.customized_chat_template is not None:
            if os.path.exists(args.customized_chat_template):
                tokenizer.chat_template = open(args.customized_chat_template).read()
            else:
                tokenizer.chat_template = args.customized_chat_template

        logger.info(f"Loading inference method '{args.method.approach}' and verifier...")    
        method = INFERENCE_METHODS[args.method.approach]
        verifier = load_verifier(args.verifier, tokenizer)

        cache_path = getattr(args, "dataset_cache_path", None)
        normalized_dataset: Dataset
        
        if args.generation_results_path and os.path.exists(args.generation_results_path):
            dataset = Dataset.from_json(args.generation_results_path)
            normalized_dataset = dataset
        elif cache_path and os.path.exists(cache_path):
            logger.info(f"Loading cached normalized dataset from {cache_path}...")
            normalized_dataset = Dataset.from_json(cache_path)
        else:
            num_procs = int(getattr(args, "num_processes", 1) or 1)
            proc_id = int(getattr(args, "process_id", 0) or 0)
            if cache_path and num_procs > 1 and proc_id != 0:
                logger.info(f"Cache not found at {cache_path}. Waiting for process 0 to create it...")
                t0 = time.time()
                timeout_s = 600
                while (time.time() - t0) < timeout_s and not os.path.exists(cache_path):
                    time.sleep(2)

            if cache_path and os.path.exists(cache_path):
                logger.info(f"Loading cached normalized dataset from {cache_path}...")
                normalized_dataset = Dataset.from_json(cache_path)
            else:
                logger.info("Normalizing dataset...")
                normalized_dataset = raw_dataset.map(
                    normalize_dataset,
                    num_proc=32,
                    desc="Normalizing dataset",
                    with_indices=True,
                    fn_kwargs={"args": args.data, "tokenizer": tokenizer},
                )
                if cache_path:
                    logger.info(f"Saving normalized dataset to cache: {cache_path}")
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    tmp_path = f"{cache_path}.tmp"
                    normalized_dataset.to_json(tmp_path, lines=True)
                    os.replace(tmp_path, cache_path)

        dataset = normalized_dataset

        if args.data.dataset_size is not None:
            dataset = dataset.select(range(args.data.dataset_size))
        else:
            args.data.dataset_size = len(dataset)

        # Optional: select a pre-computed val/test split by index file.
        # Pass +dataset_indices_path=<path/to/split_indices.json> and
        # +dataset_split_name=val|test  (default: test).
        indices_file = getattr(args, "dataset_indices_path", None)
        if indices_file and os.path.exists(str(indices_file)):
            split_name = str(getattr(args, "dataset_split_name", "test"))
            with open(str(indices_file)) as _f:
                _split_data = json.load(_f)
            if isinstance(_split_data, dict):
                _indices = _split_data.get(f"{split_name}_indices", _split_data.get("indices", []))
            else:
                _indices = _split_data
            logger.info(f"Selecting {len(_indices)} samples from split '{split_name}' via {indices_file}")
            dataset = dataset.select(_indices)
            # Disable the start/end range so it doesn't further slice the split
            args.data.dataset_start = 0
            args.data.dataset_end = len(dataset)

        if args.data.dataset_start is None:
            args.data.dataset_start = 0
        if args.data.dataset_end is None:
            args.data.dataset_end = len(dataset)
        dataset = dataset.select(range(args.data.dataset_start, args.data.dataset_end))
        dataset = dataset.select(range(args.process_id, len(dataset), args.num_processes))
        
        print(f"Dataset keys: {dataset.column_names}")
        
        prompts = dataset["_prompt"]
        
        # Inject generation_results_path into method args
        args.method.generation_results_path = args.generation_results_path
        
        params_list = [dict(llm=llm, prompt=prompt, sample=sample, verifier=verifier, tokenizer=tokenizer, args=args.method) for sample, prompt in zip(dataset, prompts)]
        _t0 = time.perf_counter()
        states = run_batch(method, params_list, num_threads=args.max_threads, progress_bar=True, desc="Running search")
        run_batch_seconds = time.perf_counter() - _t0

        logger.info("Processing results...")

        dataset = dataset.add_column("prediction", [state.prediction for state in states])
        dataset = dataset.add_column("outputs", [state.outputs for state in states])
        dataset = dataset.add_column("scores", [state.scores for state in states])
        dataset = dataset.add_column("agg_scores", [state.agg_scores for state in states])
        dataset = dataset.add_column("metric", [state.metrics for state in states])
        
        # Add metrics from search results as separate columns for convenience
        if states:
            search_metrics = [state.metrics for state in states]
            if search_metrics and search_metrics[0]:
                for key in search_metrics[0].keys():
                     dataset = dataset.add_column(f"search_{key}", [m.get(key) for m in search_metrics])

        if any(state.log_values for state in states):
             dataset = dataset.add_column("log_values", [state.log_values for state in states])

        dataset = dataset.map(
            count_tokens,
            num_proc=32,
            desc="Counting tokens",
            fn_kwargs={"tokenizer": tokenizer},
        )

        if args.data.dataset_type == "mmhal_bench":
            drop_cols = [col for col in ["_prompt", "_pref_prompt", "_non_pref_prompt"] if col in dataset.column_names]
            if drop_cols:
                dataset = dataset.remove_columns(drop_cols)

        dataset.to_json(args.result_path, lines=True)
    
    if args.run_eval:
        if args.data.dataset_type == "math":
            # Two selection strategies over the same sampled candidates:
            #   majority voting (self-consistency baseline) and the REAR weighted sum.
            # weighted_sum_evaluate sets metric["avg_score"] to the REAR result at the
            # full sample budget, which is the primary number reported for math.
            dataset = dataset.map(majority_voting_evaluate, num_proc=32, desc="Evaluating math (majority voting)", fn_kwargs={"n_samples": args.method.n_samples})
            dataset = dataset.map(weighted_sum_evaluate, num_proc=32, desc="Evaluating math (weighted sum)", fn_kwargs={"n_samples": args.method.n_samples})
        elif args.data.dataset_type == "prefeval_explicit":
            dataset = dataset.map(evaluate_prefeval_explicit, num_proc=32, desc="Evaluating prefeval explicit", fn_kwargs={"model_name": args.data.llm_as_a_judge_model})
        elif args.data.dataset_type == "prefeval_choice":
            dataset = dataset.map(evaluate_prefeval_choice, num_proc=32, desc="Evaluating prefeval choice")
        elif args.data.dataset_type == "prefeval_implicit":
            dataset = dataset.map(evaluate_prefeval_implicit, num_proc=32, desc="Evaluating prefeval implicit", fn_kwargs={"model_name": args.data.llm_as_a_judge_model})
        elif args.data.dataset_type == "multifaceted":
            dataset = dataset.map(evaluate_multifaceted_bench, num_proc=32, desc="Evaluating multifaceted bench", fn_kwargs={"model_name": args.data.llm_as_a_judge_model})
        elif args.data.dataset_type == "mmhal_bench":
            dataset = dataset.map(evaluate_mmhal_bench, num_proc=32, desc="Evaluating mmhal bench", fn_kwargs={"model_name": args.data.llm_as_a_judge_model})
        else:
            raise ValueError(f"Unsupported dataset type: {args.data.dataset_type}")

    if len(dataset) == 0:
        logger.warning("Dataset is empty. No metrics to aggregate.")
        metric = {"num_samples": 0}
    else:
        metrics = dataset["metric"]
        if metrics and metrics[0]:
            metric = {key: np.mean([m[key] for m in metrics if key in m]) for key in metrics[0].keys()}
        else:
            metric = {}
    
    metric["num_samples"] = len(dataset)
    metric["run_batch_seconds"] = run_batch_seconds
    
    if metric["num_samples"] > 0:
        metric["run_batch_seconds_per_sample"] = run_batch_seconds / metric["num_samples"]
    else:
        metric["run_batch_seconds_per_sample"] = 0.0
    if "prompt_tokens" in dataset.column_names:
        metric["avg_prompt_tokens"] = float(np.mean(dataset["prompt_tokens"]))
    if "total_tokens" in dataset.column_names:
        metric["avg_total_tokens"] = float(np.mean(dataset["total_tokens"]))
    if "outputs_tokens_sum" in dataset.column_names and "outputs" in dataset.column_names:
        outputs_tokens_sum_list = dataset["outputs_tokens_sum"]
        outputs_list = dataset["outputs"]
        total_responses = sum(len(outs or []) for outs in outputs_list)
        tokens_sum = sum(outputs_tokens_sum_list)
        metric["avg_response_tokens"] = (tokens_sum / total_responses) if total_responses > 0 else 0.0
    elif "prediction_tokens" in dataset.column_names:
        metric["avg_response_tokens"] = float(np.mean(dataset["prediction_tokens"]))
    else:
        metric["avg_response_tokens"] = 0.0

    dataset.to_json(args.result_path, lines=True)
    json.dump(metric, open(args.metric_path, "w"), ensure_ascii=False, indent=4)
    
    logger.info(f"Done! Metrics: {metric}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def hydra_main(cfg: DictConfig):
    os.chdir(get_original_cwd())
    args = post_process_cfg(cfg)
    print(f"Experiment output directory: {args.output_dir}")

    with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
        config_dict = OmegaConf.to_container(args, resolve=True)
        yaml.dump(config_dict, f, default_flow_style=False)

    logger.info(f"Running search with config: {pformat(config_dict)}")
    run_search(args)


if __name__ == "__main__":
    hydra_main()
