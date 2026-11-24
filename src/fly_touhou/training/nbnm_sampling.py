"""Exact task exposure budgets without waiting for equal-length worlds."""
from collections import Counter
import math


def group_block_budget(groups, weights, blocks):
    counts=Counter(groups)
    if set(counts)!=set(weights) or any(not math.isfinite(v) or v<=0 for v in weights.values()):
        raise ValueError('A positive sampling weight is required for each task group')
    scale=sum(weights.values())
    raw={g:blocks*weights[g]/scale for g in counts}
    result={g:int(raw[g]) for g in counts}
    order=sorted(counts,key=lambda g:raw[g]-result[g],reverse=True)
    for g in order[:blocks-sum(result.values())]:result[g]+=1
    if any(result[g]<counts[g] for g in counts):
        raise ValueError('Segment too small to give every worker one recurrent block')
    return result
