import numpy as np
import copy
from typing import Callable, List, Any, Dict, Union
from transformers import PreTrainedTokenizer

from search.decoding import simple_decoding, cot_decoding, ResponseState, SearchResult
from search.utils import aggregate_score
from utils.concurrency import run_batch

def dvts(
    llm: Any,
    prompt: Union[str, List[Dict[str, Any]]],
    sample: Any,
    verifier: Callable[[str, List[str]], List[List[float]]],
    tokenizer: PreTrainedTokenizer,
    args: Any,
) -> SearchResult:
    num_beams = args.n_samples // args.beam_width
    if args.step_length is not None:
        sampling_params = dict(temperature=args.temperature, max_tokens=args.step_length)
    else:
        sampling_params = dict(temperature=args.temperature, max_tokens=args.max_tokens, stop=[args.step_token])

    base_prompt = prompt
    active_beams = [ResponseState(prompt=base_prompt) for _ in range(num_beams)]
    completed_beams = []

    def build_prompt_input(base: Union[str, List[Dict[str, Any]]], response_text: str):
        if isinstance(base, list):
            if not response_text:
                return copy.deepcopy(base)
            messages = copy.deepcopy(base)
            messages.append({"role": "assistant", "content": response_text})
            return messages
        return base + response_text

    for step in range(args.max_search_steps):
        if step == args.max_search_steps - 1:
            sampling_params["stop"] = None

        if args.decoding_strategy == "simple":
            prompt_inputs = [build_prompt_input(base_prompt, beam.response_text) for beam in active_beams for _ in range(args.beam_width)]
            prev_texts = [beam.response_text for beam in active_beams for _ in range(args.beam_width)]
            batch_params = [dict(llm=llm, prompt=p, **sampling_params) for p in prompt_inputs]
            output_states = run_batch(simple_decoding, batch_params, num_threads=len(batch_params))
        elif args.decoding_strategy == "cot":
            if isinstance(base_prompt, list):
                raise ValueError("cot decoding is not supported for chat prompts in dvts.")
            prompt_inputs = [build_prompt_input(base_prompt, beam.response_text) for beam in active_beams]
            prev_texts = [beam.response_text for beam in active_beams for _ in range(args.beam_width)]
            cot_params = [dict(llm=llm, prompt=p, generated_answer=p[len(base_prompt):], cot_width=args.beam_width, **sampling_params) for p in prompt_inputs]
            cot_decoding_states = run_batch(cot_decoding, cot_params, num_threads=len(cot_params))
            output_states = [state for states_list in cot_decoding_states for state in states_list]
        
        output_texts = [prev_texts[i] + state.response_text for i, state in enumerate(output_states)]

        scores = verifier(sample, output_texts)
        agg_scores = aggregate_score(scores, args.agg_strategy)

        for i, state in enumerate(output_states):
            state.prompt = base_prompt
            state.response_text = output_texts[i]
            state.meta_info["scores"] = scores[i]
            state.meta_info["agg_scores"] = agg_scores[i]
        
        agg_scores_reshaped = np.array(agg_scores).reshape(-1, args.beam_width)
        chosen_indices = np.argmax(agg_scores_reshaped, axis=1)

        next_beams = []
        for i, j in enumerate(chosen_indices):
            idx = i * args.beam_width + j
            total_tokens = len(tokenizer.encode(output_texts[idx]))
            
            # Check if stopped by step_token
            is_step_stop = False
            if output_states[idx].completed:
                stop_reason = output_states[idx].meta_info.get("result", {}).get("stop_reason")
                if stop_reason == args.step_token:
                    is_step_stop = True

            if output_states[idx].completed and not is_step_stop:
                completed_beams.append(output_states[idx])
            elif total_tokens >= args.max_tokens:
                completed_beams.append(output_states[idx])
            else:
                output_states[idx].completed = False
                next_beams.append(output_states[idx])
        active_beams = next_beams

        if len(active_beams) == 0:
            break
    
    completed_beams.extend(active_beams)
    outputs = [beam.response_text for beam in completed_beams]
    scores = [beam.meta_info["scores"] for beam in completed_beams]
    agg_scores = [beam.meta_info["agg_scores"] for beam in completed_beams]

    return SearchResult(
        outputs=outputs,
        prediction=outputs[np.argmax(agg_scores)],
        scores=scores,
        agg_scores=agg_scores,
        metrics={"rewards_mean": np.mean([np.mean(sc) for sc in scores]), "agg_rewards_mean": np.mean([np.mean(sc) for sc in agg_scores]), "step": step}
    )
