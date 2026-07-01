from typing import Any, Dict, List
from transformers import AutoTokenizer


def _prompt_to_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        texts: List[str] = []
        for msg in prompt:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        texts.append(part.get("text", ""))
        return "\n".join(texts)
    return ""


def _len_tokens(text, tokenizer):
    if not isinstance(text, str):
        text = _prompt_to_text(text)
    return len(tokenizer.encode(text))


def count_tokens(sample, tokenizer: AutoTokenizer):
    """
    Compute token counts for a sample using the provided tokenizer.

    Adds the following fields to the sample:
      - prompt_tokens: number of tokens in `_prompt`
      - prediction_tokens: number of tokens in `prediction`
      - outputs_tokens_sum: sum of tokens over all strings in `outputs`
      - total_tokens: prompt_tokens + outputs_tokens_sum
    """
    prompt = sample.get("_prompt", "")
    prediction = sample.get("prediction", "")
    outputs = sample.get("outputs", []) or []

    prompt_tokens = _len_tokens(prompt, tokenizer)
    prediction_tokens = _len_tokens(prediction, tokenizer)
    outputs_tokens_sum = 0
    for out in outputs:
        outputs_tokens_sum += _len_tokens(out, tokenizer)

    total_tokens = prompt_tokens + outputs_tokens_sum

    sample["prompt_tokens"] = prompt_tokens
    sample["prediction_tokens"] = prediction_tokens
    sample["outputs_tokens_sum"] = outputs_tokens_sum
    sample["total_tokens"] = total_tokens
    return sample
