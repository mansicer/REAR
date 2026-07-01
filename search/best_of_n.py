import numpy as np
from typing import Callable, List, Any, Dict, Union
from transformers import PreTrainedTokenizer

from search.decoding import simple_decoding, SearchResult
from search.utils import aggregate_score
from utils.concurrency import run_batch

def best_of_n(llm: Any, prompt: Union[str, List[Dict[str, Any]]], sample: Any, verifier: Any, tokenizer: PreTrainedTokenizer, args: Any) -> SearchResult:
    
    if hasattr(args, "generation_results_path") and args.generation_results_path:
        answers = sample["outputs"]
    else:
        sampling_params = dict(temperature=args.temperature, max_tokens=args.max_tokens)
        
        # Run batch decoding
        params_list = [dict(llm=llm, prompt=prompt, **sampling_params) for _ in range(args.n_samples)]
        outputs = run_batch(simple_decoding, params_list, num_threads=len(params_list))
        
        answers = [output.response_text for output in outputs]
    
    if verifier is None:
        scores = [[1.0]] * len(answers)
        agg_scores = [score[0] for score in scores]
        best_output = answers[0]
    else:
        scores = verifier(sample, answers)
        agg_scores = aggregate_score(scores, args.agg_strategy)
        best_output = answers[np.argmax(agg_scores)]
    
    return SearchResult(
        outputs=answers,
        prediction=best_output,
        scores=scores,
        agg_scores=agg_scores,
        metrics={"rewards_mean": np.mean([np.mean(sc) for sc in scores])}
    )
