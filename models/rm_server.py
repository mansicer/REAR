import requests
from typing import List, Dict
from transformers import PreTrainedTokenizer


class RMServerVerifier:
    def __init__(self, args, tokenizer: PreTrainedTokenizer):
        self.args = args
        self.base_url = args.base_url
        self.model_name_or_path = args.model_name_or_path
        self.tokenizer = tokenizer

    def score(self, sample: Dict, outputs: List[str]) -> List[List[float]]:
        prompt = sample["_pref_prompt"]
        texts = [prompt + output for output in outputs]
        rewards = self._get_rewards(texts)
        # Wrap each scalar reward in a list so aggregate_score (score[-1] / mean(score)) works.
        return [[r] for r in rewards]

    def _get_rewards(self, texts: List[str]) -> List[float]:
        # vLLM's OpenAI-compatible /classify endpoint expects {"input": <str|list>} and
        # returns {"data": [{"probs": [...], ...}, ...]}. For a scalar reward model
        # (num_labels=1) probs[0] is the reward used for ranking.
        payload = {"input": texts}
        # Only pin a model name if a real one is configured; the default placeholder
        # "default" is rejected (404) by the server, so omit it and let the server
        # use its single served model.
        if self.model_name_or_path and self.model_name_or_path != "default":
            payload["model"] = self.model_name_or_path

        try:
            resp = requests.post(self.base_url, json=payload).json()
            data = resp["data"] if isinstance(resp, dict) else resp
            rewards = [item["probs"][0] for item in data]
            assert len(rewards) == len(
                texts
            ), f"Expected {len(texts)} rewards, got {len(rewards)}"
            return rewards
        except Exception as e:
            print(f"Error: {e}")
            return [-100.0] * len(texts)

    def __call__(self, sample: Dict, outputs: List[str]) -> List[float]:
        return self.score(sample, outputs)
