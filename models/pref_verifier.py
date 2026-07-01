from typing import List, Dict, Any
import numpy as np
from transformers import PreTrainedTokenizer
from utils.llm import OpenAIClient
from utils.concurrency import run_batch

def get_input_logprob_func(llm: OpenAIClient, prompt: str, retries: int = 10, top_logprobs_num: int = 1) -> Dict[str, Any]:
    for attempt in range(retries):
        try:
            res = llm.generate(
                prompt=prompt,
                max_tokens=1,
                logprobs=top_logprobs_num,
                echo=True,
            )
            return res
        except Exception as e:
            print(f"Attempt {attempt} failed with error: {e}")
            if attempt == retries - 1:
                raise e
    return {}

class PreferenceVerifier:
    def __init__(self, args: Any, tokenizer: PreTrainedTokenizer):
        self.args = args
        self.llm = OpenAIClient(base_url=args.base_url, api_key=args.api_key, model=args.model)
        self.max_threads = args.max_threads
        self.mixed_weight = args.mixed_weight
        self.discount_factor = args.discount_factor
        self.tokenizer = tokenizer
    
    def score(self, sample: Dict, outputs: List[str]) -> List[float]:
        pref_prompt = sample["_pref_prompt"]
        non_pref_prompt = sample["_non_pref_prompt"]

        pref_logprobs = self._get_response_logprob(pref_prompt, outputs)
        non_pref_logprobs = self._get_response_logprob(non_pref_prompt, outputs)

        r1 = [np.array(non_pref_logprob) for non_pref_logprob in non_pref_logprobs]
        r2 = [np.array(pref_logprob) - np.array(non_pref_logprob) for pref_logprob, non_pref_logprob in zip(pref_logprobs, non_pref_logprobs)]
        rewards = [rr1 + self.mixed_weight * rr2 for rr1, rr2 in zip(r1, r2)]
        # Applying discount factor per token
        discounted_rewards = []
        for reward_arr in rewards:
            discount_mask = self.discount_factor ** np.arange(len(reward_arr))
            discounted_rewards.append(reward_arr * discount_mask)
        return discounted_rewards

    def score_components(self, sample: Dict, outputs: List[str]):
        """Return raw (r1, r2) logprob components without applying mixed_weight.

        r1[i] = non-pref logprobs for output i (per-token array)
        r2[i] = pref logprobs - non-pref logprobs for output i (per-token array)

        The combined score for a given lambda is: r1 + lambda * r2
        This allows post-hoc lambda sweeping without re-running LLM inference.
        """
        pref_prompt = sample["_pref_prompt"]
        non_pref_prompt = sample["_non_pref_prompt"]

        pref_logprobs = self._get_response_logprob(pref_prompt, outputs)
        non_pref_logprobs = self._get_response_logprob(non_pref_prompt, outputs)

        r1 = [np.array(nlp) for nlp in non_pref_logprobs]
        r2 = [np.array(plp) - np.array(nlp) for plp, nlp in zip(pref_logprobs, non_pref_logprobs)]
        return r1, r2

    def _get_response_logprob(self, prompt, responses):
        if isinstance(prompt, list):
            return self._get_chat_response_logprob(prompt, responses)
        return self._get_completion_response_logprob(prompt, responses)

    def _get_completion_response_logprob(self, prompt: str, responses: List[str]) -> List[List[float]]:
        texts = [prompt + response for response in responses]
        token_lens = [len(self.tokenizer.encode(text)) for text in texts]
        prompt_len = len(self.tokenizer.encode(prompt))
        response_lens = [max(token_len - prompt_len, 0) for token_len in token_lens]

        batch_params = [dict(llm=self.llm, prompt=text) for text in texts]
        results = run_batch(get_input_logprob_func, batch_params, num_threads=len(batch_params))

        logprobs_list = []
        for res, resp_len in zip(results, response_lens):
            # Extract logprobs from res["logprobs"]
            # OpenAI completion API logprobs structure:
            # choice.logprobs.token_logprobs is a list of floats
            if "logprobs" in res and hasattr(res["logprobs"], "token_logprobs"):
                token_logprobs = res["logprobs"].token_logprobs
                logprobs_list.append(token_logprobs[-resp_len:])
            else:
                logprobs_list.append([0.0] * resp_len)

        return logprobs_list

    def _extract_prompt_logprobs(self, prompt_logprobs, resp_len: int) -> List[float]:
        if not prompt_logprobs or resp_len <= 0:
            return []
        tail = prompt_logprobs[-resp_len:]
        extracted = []
        for item in tail:
            if not isinstance(item, dict):
                continue
            best = None
            for cand in item.values():
                if isinstance(cand, dict) and cand.get("rank") == 1 and "logprob" in cand:
                    best = float(cand["logprob"])
                    break
                if isinstance(cand, dict) and "logprob" in cand:
                    lp = float(cand["logprob"])
                    best = lp if best is None else max(best, lp)
            extracted.append(best if best is not None else 0.0)
        return extracted

    def _get_chat_response_logprob(self, prompt: List[Dict[str, Any]], responses: List[str]) -> List[List[float]]:
        batch_params = []
        response_lens = []
        for response in responses:
            messages = list(prompt) + [{"role": "assistant", "content": response}]
            response_len = self._get_chat_response_len(prompt, response)
            response_lens.append(response_len)
            batch_params.append(
                dict(
                    llm=self.llm,
                    prompt=messages,
                    max_tokens=1,
                    logprobs=1,
                    extra_body={"prompt_logprobs": 1},
                )
            )

        results = run_batch(self._chat_logprob_call, batch_params, num_threads=len(batch_params))
        logprobs_list = []
        for res, resp_len in zip(results, response_lens):
            prompt_logprobs = res.get("prompt_logprobs") if isinstance(res, dict) else None
            logprobs_list.append(self._extract_prompt_logprobs(prompt_logprobs, resp_len))
        return logprobs_list

    def _get_chat_response_len(self, prompt: List[Dict[str, Any]], response: str) -> int:
        prefix_tokens = self.tokenizer.apply_chat_template(
            prompt, tokenize=True, add_generation_prompt=False
        )
        full_tokens = self.tokenizer.apply_chat_template(
            list(prompt) + [{"role": "assistant", "content": response}],
            tokenize=True,
            add_generation_prompt=False,
        )
        return max(len(full_tokens) - len(prefix_tokens), 1)

    def _chat_logprob_call(self, llm: OpenAIClient, prompt: List[Dict[str, Any]], max_tokens: int, logprobs: int, extra_body: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return llm.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                logprobs=logprobs,
                extra_body=extra_body,
            )
        except Exception as e:
            print(f"Chat logprob call failed: {e}")
            return {}

    def __call__(self, sample: Dict, outputs: List[str]) -> List[float]:
        return self.score(sample, outputs)
