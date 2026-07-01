import requests
from typing import List, Dict
from transformers import PreTrainedTokenizer


class RMServerVerifier:
    def __init__(self, args, tokenizer: PreTrainedTokenizer):
        self.args = args
        self.base_url = args.base_url
        self.model_name_or_path = args.model_name_or_path
        self.tokenizer = tokenizer

    def score(self, sample: Dict, outputs: List[str]) -> List[float]:
        prompt = sample["_pref_prompt"]
        texts = [prompt + output for output in outputs]
        rewards = self._get_rewards(texts)
        return rewards

    def _get_rewards(self, texts: List[str]) -> List[float]:
        payload = {"model": self.model_name_or_path}
        payload.update({"text": texts})
        
        try:
            responses = requests.post(self.base_url, json=payload).json()
            rewards = [response["embedding"][0] for response in responses]
            assert len(rewards) == len(
                texts
            ), f"Expected {len(texts)} rewards, got {len(rewards)}"
            return rewards
        except Exception as e:
            print(f"Error: {e}")
            return [-100.0] * len(texts)

    def __call__(self, sample: Dict, outputs: List[str]) -> List[float]:
        return self.score(sample, outputs)
