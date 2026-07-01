import logging
import time
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI

logger = logging.getLogger(__name__)

class OpenAIClient:
    def __init__(
        self,
        base_url: str,
        api_key: str = "EMPTY",
        model: str = "default",
        is_chat: bool = False,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.is_chat = is_chat

    def generate(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stop: Optional[Union[str, List[str]]] = None,
        logprobs: Optional[int] = None,
        n: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate a response using the OpenAI API.
        """
        try:
            use_chat = self.is_chat or isinstance(prompt, list)
            if use_chat:
                messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                    logprobs=logprobs is not None,
                    top_logprobs=logprobs,
                    n=n,
                    **kwargs,
                )
                
                # For chat completions, we return a list of results if n > 1
                results = []
                for choice in response.choices:
                    res = {
                        "text": choice.message.content,
                        "finish_reason": choice.finish_reason,
                        "stop_reason": getattr(choice, "stop_reason", None),
                    }
                    if choice.logprobs:
                        # Convert OpenAI logprobs to a format similar to what the search algos expect
                        # SGLang's internal format was often a list of [logprob, token_id, token_text]
                        res["logprobs"] = choice.logprobs
                    if hasattr(response, "prompt_logprobs"):
                        res["prompt_logprobs"] = response.prompt_logprobs
                    if hasattr(response, "prompt_token_ids"):
                        res["prompt_token_ids"] = response.prompt_token_ids
                    results.append(res)
                
                return results[0] if n == 1 else results

            else:
                # Completion API
                response = self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                    logprobs=logprobs,
                    n=n,
                    **kwargs,
                )
                
                results = []
                for choice in response.choices:
                    res = {
                        "text": choice.text,
                        "finish_reason": choice.finish_reason,
                        "stop_reason": getattr(choice, "stop_reason", None),
                    }
                    if choice.logprobs:
                        res["logprobs"] = choice.logprobs
                    results.append(res)
                
                return results[0] if n == 1 else results

        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            raise e

    def generate_batch(
        self,
        params_list: List[Dict[str, Any]],
        max_threads: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Generate responses for a batch of parameters using a thread pool.
        Note: The actual batching might be handled by the server if we use the batch API,
        but here we use threads for simplicity and consistency with the existing codebase.
        """
        from concurrent.futures import ThreadPoolExecutor
        
        def task(params):
            return self.generate(**params)
        
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            results = list(executor.map(task, params_list))
        
        return results

