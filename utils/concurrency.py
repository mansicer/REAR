import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)

def run_batch(
    func: Callable,
    params_list: List[Dict[str, Any]],
    num_threads: int = 10,
    progress_bar: bool = False,
    desc: Optional[str] = None,
) -> List[Any]:
    """
    Run a function in parallel over a list of parameters using a thread pool.
    Mimics sglang's run_batch behavior.
    """
    results = [None] * len(params_list)
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Create a list of futures
        futures = {executor.submit(func, **params): i for i, params in enumerate(params_list)}
        
        # Optionally add a progress bar
        if progress_bar:
            pbar = tqdm(total=len(params_list), desc=desc)
        
        for future in futures:
            try:
                index = futures[future]
                results[index] = future.result()
                if progress_bar:
                    pbar.update(1)
            except Exception as e:
                logger.error(f"Error in batch execution for item {futures[future]}: {e}")
                # We could choose to raise or keep it as None
                # Raising for now to be safe
                if progress_bar:
                    pbar.close()
                raise e
        
        if progress_bar:
            pbar.close()
            
    return results

