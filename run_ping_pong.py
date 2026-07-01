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
from jinja2 import Template

# Register custom resolvers
OmegaConf.register_new_resolver("basename", lambda path: os.path.basename(path))
OmegaConf.register_new_resolver("eval", eval)

from models.verifier import load_verifier
from search.best_of_n import best_of_n
from search.dvts import dvts
from utils.data import get_dataset, normalize_dataset
from utils.evaluator import *
from utils.prompts import *
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
    return args

def generator_func(llm: OpenAIClient, prompt: str, max_tokens: int) -> Dict[str, Any]:
    res = llm.generate(prompt=prompt, max_tokens=max_tokens)
    return {"prediction": res["text"]}

def run_ping_pong_eval(args):
    run_batch_seconds = 0.0
    llm = OpenAIClient(base_url=args.base_url, api_key=args.api_key, model=args.model_path)
    
    logger.info(f"Loading dataset from {args.data.dataset_path}...")
    dataset: Dataset = get_dataset(args.data.dataset_path, args.data.dataset_split)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    logger.info(f"Loading inference method '{args.method.approach}' and verifier...")    
    method = INFERENCE_METHODS[args.method.approach]
    verifier = load_verifier(args.verifier, tokenizer)

    if args.data.dataset_size is not None:
        dataset = dataset.select(range(args.data.dataset_size))
        
    dataset = dataset.map(
        normalize_dataset,
        num_proc=32,
        desc="Normalizing dataset",
        with_indices=True,
        fn_kwargs={"args": args.data, "tokenizer": tokenizer},
    )

    # Initialize conversation history
    initial_histories = []
    for sample in dataset:
        history = []
        if sample['character'].get('initial_message'):
            history.append({'role': 'assistant', 'content': sample['character']['initial_message']})
        initial_histories.append(history)
    dataset = dataset.add_column("history", initial_histories)

    # Load templates once
    with open(args.data.templates.interrogator_system, "r") as f:
        interrogator_system_template = Template(f.read())
    with open(args.data.templates.interrogator_user, "r") as f:
        interrogator_user_template = Template(f.read())

    num_turns = args.data.get("num_turns", 4)
    dataset_list = dataset.to_list()

    for i in range(num_turns):
        logger.info(f"--- Turn {i+1}/{num_turns} ---")

        # 1. Generate interrogator utterances
        logger.info("Generating interrogator utterances...")
        interrogator_prompts = []
        for sample in dataset_list:
            char = sample["character"]
            sit = sample["situation"]
            hist = sample["history"]
            
            sys_prompt = interrogator_system_template.render()
            usr_prompt = interrogator_user_template.render(
                char_summary=char.get("summary", char.get("char_name")),
                situation=sit.get("text", ""),
                messages=hist
            )
            
            interrogator_chat = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": usr_prompt}
            ]
            
            prompt_text = tokenizer.apply_chat_template(interrogator_chat, add_generation_prompt=True, tokenize=False)
            interrogator_prompts.append(prompt_text)

        interrogator_params = [dict(llm=llm, prompt=p, max_tokens=1024) for p in interrogator_prompts]
        interrogator_results = run_batch(generator_func, interrogator_params, num_threads=args.max_threads, progress_bar=True, desc="Interrogator turn")
        
        for idx, res in enumerate(interrogator_results):
            pred = res["prediction"]
            try:
                interrogator_output = json.loads(pred)
                user_utterance = interrogator_output["next_utterance"]
            except (json.JSONDecodeError, KeyError):
                logger.warning(f"Failed to parse interrogator output: {pred}")
                user_utterance = pred

            dataset_list[idx]["history"].append({"role": "user", "content": user_utterance})

        # 2. Generate player responses
        logger.info("Generating player responses...")

        # Reconstruct prompts for the player for this turn
        turn_prompts = []
        for sample in dataset_list:
            system_prompt = sample['_preference']
            history = sample['history']
            
            pref_messages = [{"role": "system", "content": system_prompt}] + history
            sample['_pref_prompt'] = tokenizer.apply_chat_template(pref_messages, add_generation_prompt=True, tokenize=False)
            
            non_pref_messages = history
            sample['_non_pref_prompt'] = tokenizer.apply_chat_template(non_pref_messages, add_generation_prompt=True, tokenize=False)
            
            # Use non_pref_prompt for generation, verifier will use both
            sample['_prompt'] = sample['_non_pref_prompt']
            turn_prompts.append(sample['_prompt'])
        
        params_list = [
            dict(
                llm=llm,
                prompt=prompt,
                sample=sample,
                verifier=verifier,
                tokenizer=tokenizer,
                args=args.method
            ) for sample, prompt in zip(dataset_list, turn_prompts)
        ]

        _t0_turn = time.perf_counter()
        player_states = run_batch(method, params_list, num_threads=args.max_threads, progress_bar=True, desc="Player turn")
        run_batch_seconds += time.perf_counter() - _t0_turn
    
        for idx, state in enumerate(player_states):
            player_response = state.prediction
            dataset_list[idx]["history"].append({"role": "assistant", "content": player_response})

    dataset = Dataset.from_list(dataset_list)
    if "metric" not in dataset.column_names:
        dataset = dataset.add_column("metric", [{}] * len(dataset))
    dataset = dataset.rename_column("history", "full_conversation")
    dataset.to_json(args.result_path, lines=True)
    
    logger.info("Evaluation phase...")
    dataset = dataset.map(
        evaluate_ping_pong,
        num_proc=32,
        desc="Evaluating Ping-Pong Bench",
        fn_kwargs={
            "model_name": args.data.llm_as_a_judge_model,
            "templates_paths": OmegaConf.to_container(args.data.templates, resolve=True)
        }
    )
    
    if len(dataset) == 0:
        logger.warning("Dataset is empty. No metrics to aggregate.")
        metric = {"num_samples": 0}
    else:
        metrics = dataset["metric"]
        if metrics and metrics[0]:
            metric = {key: np.mean([m[key] for m in metrics]) for key in metrics[0].keys()}
        else:
            metric = {}
    
    metric["num_samples"] = len(dataset)
    metric["run_batch_seconds"] = run_batch_seconds

    json.dump(metric, open(args.metric_path, "w"), ensure_ascii=False, indent=4)
    logger.info(f"Done! Metrics: {metric}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def hydra_main(cfg: DictConfig):
    os.chdir(get_original_cwd())
    args = post_process_cfg(cfg)
    with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
        config_dict = OmegaConf.to_container(args, resolve=True)
        yaml.dump(config_dict, f, default_flow_style=False)
    
    logger.info(f"Running Ping Pong evaluation with config: {pformat(config_dict)}")
    run_ping_pong_eval(args)

if __name__ == "__main__":
    hydra_main()
