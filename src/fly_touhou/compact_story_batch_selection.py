"""Select intact on-policy sequences from a long, continuously played batch."""
from collections import Counter
import numpy as np
import torch


DEFAULT_STAGE_WEIGHTS={1:.125,2:.125,3:.30,4:.30,5:.075,6:.075}


def allocate(capacities,weights,total):
    """Redistribute unavailable stage quotas; never duplicate a sequence."""
    counts=np.zeros(len(capacities),dtype=np.int64)
    while int(counts.sum())<total:
        available=np.asarray(capacities)-counts
        active=available>0
        share=np.where(active,weights,0.).astype(float)
        if share.sum()==0:share=active.astype(float)
        desired=share/share.sum()*(total-int(counts.sum()))
        take=np.minimum(available,np.floor(desired).astype(np.int64))
        if not take.any():take[int(np.argmax(desired))]=1
        counts+=take
    return counts


def select_sequences(features,valid,length,rows,rng,stage_weights=None):
    if rows<=0 or rows%256 or length!=16:
        raise ValueError('Selection needs positive whole256-row budgets and16-step sequences')
    if features.shape[-1]<112:raise ValueError('Observed native stage feature required')
    valid=valid.cpu().numpy() if torch.is_tensor(valid) else np.asarray(valid,bool)
    stages=np.rint(features[...,111].cpu().numpy()*6).astype(np.int64)
    if ((stages[valid]<0)|(stages[valid]>6)).any():raise ValueError('Unexpected native stage')
    weights=dict(DEFAULT_STAGE_WEIGHTS if stage_weights is None else stage_weights)
    if set(weights)!=set(range(1,7)) or any(not np.isfinite(v) or v<0 for v in weights.values()) or sum(weights.values())<=0:
        raise ValueError('Six finite nonnegative stage weights with positive total required')
    pools={stage:[] for stage in range(7)}
    for world in range(valid.shape[1]):
        for start in range(0,len(valid)-length+1,length):
            if valid[start:start+length,world].all():
                counts=np.bincount(stages[start:start+length,world],minlength=7)
                pools[int(counts.argmax())].append((start,world))
    real_rows=sum(map(len,pools.values()))*length
    if real_rows!=int(valid.sum()):raise ValueError('Incomplete recurrent blocks in real data')
    target=min(rows,real_rows)//length
    if target%16:raise ValueError('Selected data must fill real256-row minibatches')
    quotas=allocate([len(pools[s]) for s in range(7)],[weights.get(s,0.) for s in range(7)],target)
    selected={}
    for stage,pool in pools.items():
        order=rng.permutation(len(pool))[:quotas[stage]]
        selected[stage]=[pool[int(i)] for i in order]
    row_counts=Counter()
    for pool in selected.values():
        for start,world in pool:row_counts.update(map(int,stages[start:start+length,world]))
    return selected,dict(schema='long-batch-stage-stratified-PPO-v1',collected_rows=real_rows,
        collected_rows_by_stage={str(s):int((stages[valid]==s).sum()) for s in range(7)},
        selected_rows=target*length,unselected_rows=real_rows-target*length,
        requested_stage_weights=weights,selected_rows_by_stage={str(k):v for k,v in sorted(row_counts.items())},
        selected_sequences_by_majority_stage={str(k):len(v) for k,v in selected.items()},
        replacement=False,sequence_length=length,GAE_uses_full_collected_histories=True,
        objective='Stage-weighted current-policy state sampling;not uniform-state PPO and not teacher/off-policy samples')


def interleaved_groups(selected,rng):
    """Spread late stages through early minibatches, including before a KL stop."""
    stages=sorted(selected);queues={s:[selected[s][int(i)] for i in rng.permutation(len(selected[s]))] for s in stages}
    totals=np.asarray([len(queues[s]) for s in stages],dtype=float)
    total=int(totals.sum());served=np.zeros(len(stages),dtype=int);order=[]
    for n in range(total):
        deficit=totals/total*(n+1)-served
        deficit[served>=totals]=-np.inf
        index=int(deficit.argmax());order.append(queues[stages[index]].pop());served[index]+=1
    return [order[i:i+16] for i in range(0,total,16)]
