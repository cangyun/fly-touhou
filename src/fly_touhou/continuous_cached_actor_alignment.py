"""Join immutable raw teacher labels to current-model DN diagnostic caches."""
from __future__ import annotations
import numpy as np

FIELDS=('DN1299','features','margins','trusted','unprotected_mask','actions','outcomes')


def joined_actor_batch(cache_path,records):
    with np.load(cache_path,allow_pickle=False) as source:
        cached={k:source[k] for k in FIELDS}
    raw=[]
    for record in records:
        with np.load(record['path'],allow_pickle=False) as source:
            raw.append({k:source[k][16:32] for k in FIELDS[1:]+('rule_probabilities','rule_weights')})
    def flattened(key):
        array=np.stack([r[key] for r in raw],axis=1)
        return array.reshape(-1,*array.shape[2:])
    for key in FIELDS[1:]:
        if not np.array_equal(cached[key],flattened(key)):
            raise ValueError('Cached/raw time-major alignment differs:'+key)
    cached.update(rule_probabilities=flattened('rule_probabilities'),rule_weights=flattened('rule_weights'),
        expert_mask=np.broadcast_to(np.asarray([r['origin']=='expert' for r in records]),(16,len(records))).reshape(-1).copy(),
        native_opening_mask=np.zeros(len(cached['features']),bool))
    return cached


def native_opening_records(records):
    """Only actual teacher game starts have an exact zero neural history."""
    return [r for r in records if r['origin']=='expert' and r.get('metadata',{}).get('original_start')==0]


def encode_native_opening_batch(model,records,check_stop=lambda:None):
    """Cache previously omitted decisions0:16 with their genuine reset state.

    This is for the short fixed-body diagnostic only. Joint learning re-encodes
    raw observations and propagates gradients through the brain itself.
    """
    import torch
    from .continuous_connectome_training import reset_activity
    if not 0<len(records)<=16:raise ValueError('One to16 actual native openings required')
    if len(native_opening_records(records))!=len(records):raise ValueError('Non-opening records cannot use zero history')
    arrays=[]
    keys=FIELDS[1:]+('rule_probabilities','rule_weights','maps','objects','episode_starts')
    for record in records:
        with np.load(record['path'],allow_pickle=False) as source:
            if not bool(source['episode_starts'][0]):raise ValueError('Opening must carry its actual episode reset')
            arrays.append({k:source[k][:16] for k in keys})
    batch={k:np.stack([r[k] for r in arrays],axis=1) for k in keys}
    b=len(records);state=model.initial_state(16);DN_rows=[]
    with torch.no_grad():
        tensors={k:torch.as_tensor(batch[k],device=model.device) for k in ('features','maps','objects')}
        encoded=model.encode_raw(tensors['features'],tensors['maps'],objects=tensors['objects'])
        for t in range(16):
            check_stop();starts=torch.zeros(16,dtype=torch.bool,device=model.device)
            starts[:b]=torch.as_tensor(batch['episode_starts'][t],device=model.device)
            state=reset_activity(state,starts)
            z=torch.zeros(16,encoded.shape[-1],device=model.device,dtype=encoded.dtype);z[:b]=encoded[t]
            DN,state=model.core(z,state);DN_rows.append(DN[:b].cpu().numpy())
    result={k:v.reshape(-1,*v.shape[2:]) for k,v in batch.items() if k not in ('maps','objects','episode_starts')}
    result.update(DN1299=np.stack(DN_rows).reshape(16*b,-1),expert_mask=np.ones(16*b,bool),
        native_opening_mask=np.ones(16*b,bool))
    return result
