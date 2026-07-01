from typing import Callable, List
from transformers import PreTrainedTokenizer

from models.pref_verifier import PreferenceVerifier
from models.rm_server import RMServerVerifier
from models.generative_rm_server import GenerativeRMServerVerifier


def load_verifier(args, tokenizer: PreTrainedTokenizer) -> Callable[[str, List[str]], List[List[float]]]:
    if args.type == "greedy":
        return None
    elif args.type == "pref_verifier":
        return PreferenceVerifier(args, tokenizer)
    elif args.type == "rm_server":
        return RMServerVerifier(args, tokenizer)
    elif args.type == "generative_rm_server":
        return GenerativeRMServerVerifier(args, tokenizer)
    else:
        raise ValueError(f"Invalid verifier type: {args.type}")
