import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class ResponseState:
    prompt: Any
    response_text: str = ""
    completed: bool = False
    meta_info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SearchResult:
    outputs: List[str]
    prediction: str
    scores: Optional[List[Any]] = None
    agg_scores: Optional[List[float]] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    log_values: Dict[str, Any] = field(default_factory=dict)

def simple_decoding(
    llm: Any,
    prompt: Union[str, List[Dict[str, Any]]],
    temperature: float = 0.0,
    max_tokens: int = 1024,
    stop: Optional[Union[str, List[str]]] = None,
    retries: int = 10,
    **kwargs
) -> ResponseState:
    state = ResponseState(prompt=prompt)

    for attempt in range(retries):
        try:
            logprobs_value = 1 if isinstance(state.prompt, str) else None
            res = llm.generate(
                prompt=state.prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
                logprobs=logprobs_value,
                **kwargs
            )
            
            state.response_text = res["text"]
            state.meta_info["result"] = {
                "finish_reason": {"type": res["finish_reason"]},
                "stop_reason": res.get("stop_reason"),
                "output_text": res["text"],
            }
            
            if "logprobs" in res:
                state.meta_info["result"]["output_token_logprobs"] = []
                lp_obj = res["logprobs"]
                
                # Handle Chat API format
                if hasattr(lp_obj, 'content') and lp_obj.content:
                    for lp in lp_obj.content:
                        state.meta_info["result"]["output_token_logprobs"].append([lp.logprob, 0, lp.token])
                
                # Handle Completion API format (required for vLLM input logprobs via echo=True)
                elif hasattr(lp_obj, 'tokens') and lp_obj.tokens:
                    for token, lp in zip(lp_obj.tokens, lp_obj.token_logprobs):
                        state.meta_info["result"]["output_token_logprobs"].append([lp if lp is not None else 0.0, 0, token])
            
            state.completed = res["finish_reason"] == "stop"
            break
        except Exception as e:
            print(f"Attempt {attempt} failed with error: {e}")
            if attempt == retries - 1:
                raise e

    return state

def get_next_step_no(text: str) -> int:
    step_matches = re.findall(r'## Step (\d+):', text)
    if not step_matches:
        return 1
    last_step = max(int(num) for num in step_matches)
    return last_step + 1

def cot_decoding(
    llm: Any,
    prompt: str,
    generated_answer: str,
    cot_width: int,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    stop: Optional[Union[str, List[str]]] = None,
) -> List[ResponseState]:
    current_prompt = prompt
    
    test_res = llm.generate(prompt=current_prompt, temperature=0.0, max_tokens=5)
    if test_res["text"].startswith("## Step"):
        step_no = get_next_step_no(generated_answer)
        current_prompt += f"## Step {step_no}:"
    
    first_token_res = llm.generate(
        prompt=current_prompt,
        max_tokens=1,
        temperature=0.0,
        logprobs=cot_width
    )
    
    top_logprobs = []
    if hasattr(first_token_res["logprobs"], 'content') and first_token_res["logprobs"].content:
        token_logprob = first_token_res["logprobs"].content[0]
        if hasattr(token_logprob, 'top_logprobs'):
            top_logprobs = [[tp.logprob, 0, tp.token] for tp in token_logprob.top_logprobs]

    start_tokens = [token[2] for token in top_logprobs]

    from utils.concurrency import run_batch
    params_list = [
        dict(llm=llm, prompt=current_prompt + token, temperature=temperature, max_tokens=max_tokens, stop=stop)
        for token in start_tokens
    ]
    states = run_batch(simple_decoding, params_list, progress_bar=False)
    return states
