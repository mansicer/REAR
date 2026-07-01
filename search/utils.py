import numpy as np

def aggregate_score(scores, method="last"):
    if method == "last":
        return [score[-1] for score in scores]
    elif method == "mean":
        return [np.mean(score) for score in scores]
    elif method == "min":
        return [np.min(score) for score in scores]
    else:
        raise ValueError(f"Unknown aggregate method: {method}")
