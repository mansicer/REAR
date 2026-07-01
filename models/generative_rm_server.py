from typing import List, Dict, Any
import numpy as np
from transformers import PreTrainedTokenizer
from utils.llm import OpenAIClient
from utils.concurrency import run_batch
from search.decoding import ResponseState

def make_generative_rm_messages(prompt: str, outputs: List[str]):
    def make_messages(prompt: str, response: str):
        if "</think>" in response:
            response = response.split("</think>")[-1]
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
            {"role": "user", "content": "Please act as an impartial judge and evaluate the quality of the assistant's response. A preferred response is helpful, harmless, and accurately follows instructions. Is this a preferred response? Answer 'Yes' or 'No' in the format 'Preferred: X'."},
        ]
        return messages
    return [make_messages(prompt, output) for output in outputs]

def get_generative_rm_verification_score(state: ResponseState) -> float:
    logprobs = state.meta_info.get("result", {}).get("output_top_logprobs", [])
    if not logprobs:
        return -1.0
    last_token_logprob = logprobs[-1]
    if len(last_token_logprob) < 2:
        first_logprob, _, first_token_text = last_token_logprob[0]
        prob_gap = np.exp(first_logprob)
    else:
        first_logprob, first_token_id, first_token_text = last_token_logprob[0]
        second_logprob, second_token_id, second_token_text = last_token_logprob[1]
        prob_gap = np.exp(first_logprob) - np.exp(second_logprob)
        
    score = -1.0
    if "yes" in first_token_text.lower():
        score = prob_gap
    elif "no" in first_token_text.lower():
        score = -prob_gap
    return score

def verifier_rollout_func(llm: OpenAIClient, prompt: str, temperature: float = 0.0, max_tokens: int = 2048, stop: List[str] = [], retries: int = 10) -> ResponseState:
    state = ResponseState(prompt=prompt)
    for attempt in range(retries):
        try:
            res = llm.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
                logprobs=2,
            )
            state.response_text = res["text"]
            state.meta_info["result"] = {
                "finish_reason": {"type": res["finish_reason"]},
                "output_text": res["text"],
            }
            if "logprobs" in res:
                state.meta_info["result"]["output_top_logprobs"] = []
                if hasattr(res["logprobs"], 'content') and res["logprobs"].content:
                    for token_lp in res["logprobs"].content:
                        if hasattr(token_lp, 'top_logprobs'):
                            state.meta_info["result"]["output_top_logprobs"].append(
                                [[tp.logprob, 0, tp.token] for tp in token_lp.top_logprobs]
                            )
                        else:
                            state.meta_info["result"]["output_top_logprobs"].append([[token_lp.logprob, 0, token_lp.token]])
            break
        except Exception as e:
            print(f"Attempt {attempt} failed with error: {e}")
            if attempt == retries - 1:
                raise e
    return state

class GenerativeRMServerVerifier:
    def __init__(self, args: Any, tokenizer: PreTrainedTokenizer):
        self.args = args
        self.llm = OpenAIClient(base_url=args.base_url, api_key=args.api_key, model=args.model)
        
        self.temperature = args.temperature
        self.max_prompt_length = args.prompt_length
        self.max_tokens = args.max_tokens
        self.max_threads = args.max_threads

        self.tokenizer = tokenizer
        self.make_messages_fn = make_generative_rm_messages
        self.stop_strs = ["Preferred: "]
        self.score_fn = get_generative_rm_verification_score
    
    def score(self, sample: Dict, outputs: List[str]) -> List[List[float]]:
        prompt = sample["_pref_prompt"]
        messages = self.make_messages_fn(prompt, outputs)
        prompts = [self.tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False) for msg in messages]
        prompt_tokens = [len(self.tokenizer.encode(prompt)) for prompt in prompts]
        valid_prompts = [prompt for prompt, token in zip(prompts, prompt_tokens) if token <= self.max_prompt_length]
        valid_prompts_idx = [i for i, token in enumerate(prompt_tokens) if token <= self.max_prompt_length]
        
        run_params = [dict(llm=self.llm, prompt=prompt, temperature=self.temperature, max_tokens=self.max_tokens, stop=self.stop_strs) for prompt in valid_prompts]
        states = run_batch(verifier_rollout_func, run_params, num_threads=self.max_threads)
        scores = [self.score_fn(state) for state in states]
        ret_scores = [0.0] * len(prompts)
        for i, idx in enumerate(valid_prompts_idx):
            ret_scores[idx] = scores[i]
        return [[score] for score in ret_scores]

    def __call__(self, sample: Dict, outputs: List[str]) -> List[List[float]]:
        return self.score(sample, outputs)
